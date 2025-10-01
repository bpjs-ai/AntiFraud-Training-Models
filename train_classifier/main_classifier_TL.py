"""
Transfer Learning Pipeline for Hybrid OCR Classifier
==================================================

This script provides a training pipeline tailored for transfer learning when
adding new classes to an existing Hybrid OCR classifier.  It reuses a
pre‑trained model, loads its state dictionary, and fine‑tunes (or trains
only the classification head) on a new dataset without training from
scratch.

Key features:

* **OCR Preprocessing** – Uses the provided ``ocr_paddle.py`` to run
  PaddleOCR over your image dataset and produce a CSV of filenames,
  labels and extracted text.  This step is optional if you already have
  a prepared CSV.
* **Flexible Freezing** – By default the BERT and ResNet backbones are
  frozen so only the classification head and attention fusion layers are
  updated.  You can override this behaviour with command line flags.
* **State Dict Loading** – Loads as much of the pre‑trained model as
  possible.  Layers whose shapes differ (such as the final classifier
  when the number of classes changes) are left randomly initialised.

Usage example:

.. code-block:: bash

    python transfer_learning.py \
        --state_dict_path outputs/classifier/state_dict.pt \
        --image_root_dir dataset_new \
        --data_csv_path dataset_new.csv \
        --output_dir outputs_transfer \
        --freeze_backbone

Assumptions:

* Your new dataset is arranged in ``image_root_dir/class_name/filename.png``.
* Each image will be assigned a ``label`` equal to its immediate parent
  directory name.  Multiple classes are supported.
* The script will generate a ``dataset_new.csv`` file if one does not
  already exist.

This file resides at project root and can be run directly with
``python transfer_learning.py``.  It depends on ``dataset.py``,
``model.py``, ``train_utils.py`` and ``ocr_paddle.py`` located in the
same repository.
"""

import argparse
import os
import json
from datetime import datetime
from typing import Dict

import torch
import torch.nn as nn

from dataset import (
    build_dataloaders,
    build_tokenizer,
    build_transforms,
    load_and_prepare_dataframe,
    split_dataframe,
)
from model import HybridModel
from train_utils import (
    build_optimizer,
    build_scheduler,
    compute_weights,
    evaluate,
    save_label_mapping,
    train_one_epoch,
)

# We import the OCR helper to generate CSVs from raw images.  This
# function is defined in ocr_paddle.py and wrapped here for convenience.
from ocr_paddle import process_folder_multiprocessing


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the transfer learning pipeline."""
    parser = argparse.ArgumentParser(
        description="Fine tune a pre‑trained Hybrid OCR classifier on new data."
    )

    # Required arguments
    parser.add_argument(
        "--state_dict_path",
        type=str,
        required=True,
        help="Path to the pre‑trained model's state_dict.pt file",
    )
    parser.add_argument(
        "--image_root_dir",
        type=str,
        required=True,
        help="Root directory of the new dataset. Images should be in subfolders named after their labels.",
    )

    # Optional arguments
    parser.add_argument(
        "--data_csv_path",
        type=str,
        default="dataset_new.csv",
        help="CSV file containing filename, label and text columns. If the file does not exist, it will be generated using OCR.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_transfer",
        help="Directory where checkpoints, label maps and ONNX models will be saved.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of fine tuning epochs (default: 10).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Mini‑batch size for training and validation (default: 16).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for PyTorch DataLoader and OCR (default: 4).",
    )
    parser.add_argument(
        "--bert_model_name",
        type=str,
        default="indobenchmark/indobert-base-p2",
        help="Name or path of the BERT model to use (default: IndoBERT base).",
    )
    parser.add_argument(
        "--lr_bert",
        type=float,
        default=0.0,
        help="Learning rate for BERT parameters. Set to 0 to freeze BERT (default: 0).",
    )
    parser.add_argument(
        "--lr_resnet",
        type=float,
        default=0.0,
        help="Learning rate for ResNet parameters. Set to 0 to freeze ResNet (default: 0).",
    )
    parser.add_argument(
        "--lr_heads",
        type=float,
        default=1e-4,
        help="Learning rate for classification head and cross attention layers (default: 1e-4).",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="Fraction of the dataset to use for validation (default: 0.2).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for the tokenizer (default: 512).",
    )
    parser.add_argument(
        "--image_size",
        type=str,
        default="224,224",
        help="Image size as 'height,width' (default: 224,224).",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=5,
        help="Number of epochs with no improvement after which training will be stopped (default: 5).",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze BERT and ResNet entirely by zeroing their learning rates and disabling gradients.",
    )
    parser.add_argument(
        "--export_onnx",
        action="store_true",
        help="If set, export the best model to ONNX format at the end of training.",
    )

    return parser.parse_args()


def load_pretrained_weights(model: nn.Module, state_dict_path: str) -> None:
    """Load a state dictionary into ``model``, ignoring mismatched shapes.

    Parameters
    ----------
    model: nn.Module
        The model into which weights will be loaded.
    state_dict_path: str
        Path to the ``.pt`` file containing the saved state dictionary.
    """
    checkpoint = torch.load(state_dict_path, map_location="cpu")
    if isinstance(checkpoint, Dict) and "state_dict" in checkpoint:
        # Some checkpoints may store weights under a 'state_dict' key
        checkpoint = checkpoint["state_dict"]
    model_dict = model.state_dict()
    pretrained_dict = {}
    for k, v in checkpoint.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            pretrained_dict[k] = v
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def run_ocr(image_root_dir: str, csv_path: str, num_workers: int) -> None:
    """Run OCR on the dataset folder if the CSV file does not exist.

    Parameters
    ----------
    image_root_dir: str
        Path to the root of the dataset where images are stored in
        ``label/filename`` structure.
    csv_path: str
        Path to the CSV file that will contain three columns: filename,
        label and text.  If this file already exists, OCR will not be
        executed.
    num_workers: int
        Number of worker processes to use for OCR extraction.
    """
    if os.path.exists(csv_path):
        print(f"[INFO] Found existing CSV at {csv_path}, skipping OCR.")
        return
    print(f"[INFO] Running OCR on {image_root_dir} to generate {csv_path}…")
    process_folder_multiprocessing(image_root_dir, csv_path, num_workers=num_workers)


def main() -> None:
    args = parse_arguments()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Convert image size string to tuple
    try:
        h_str, w_str = args.image_size.split(",")
        image_size = (int(h_str), int(w_str))
    except Exception:
        raise ValueError("--image_size should be formatted as 'height,width', e.g., '224,224'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Prepare data CSV via OCR if needed
    run_ocr(args.image_root_dir, args.data_csv_path, num_workers=args.num_workers)

    # Step 2: Load dataframe and create label mapping for new dataset
    df, label_to_id = load_and_prepare_dataframe(
        data_csv_path=args.data_csv_path,
        image_root_dir=args.image_root_dir,
        filename_col="filename",
        label_col="label",
        text_col="text",
        drop_empty_text=True,
    )

    # Step 3: Split into train/val
    train_df, val_df = split_dataframe(
        df, stratify_col="label", test_size=args.val_size, random_state=42
    )

    # Step 4: Build tokenizer and transforms
    tokenizer = build_tokenizer(args.bert_model_name)
    transform = build_transforms(image_size)

    train_loader, val_loader = build_dataloaders(
        train_df=train_df,
        val_df=val_df,
        tokenizer=tokenizer,
        transform=transform,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    num_classes = len(label_to_id)
    id_to_label = {v: k for k, v in label_to_id.items()}
    label_names = [id_to_label[i] for i in range(num_classes)]

    # Step 5: Initialise model with new number of classes
    model = HybridModel(num_classes=num_classes, bert_model_name=args.bert_model_name).to(device)

    # Step 6: Load pre‑trained weights (skipping mismatched shapes)
    load_pretrained_weights(model, args.state_dict_path)

    # Step 7: Optionally freeze the backbone
    if args.freeze_backbone:
        print("[INFO] Freezing BERT and ResNet backbones. Only head and attention layers will be trained.")
        for param in model.bert.parameters():
            param.requires_grad = False
        for param in model.resnet.parameters():
            param.requires_grad = False

    # Step 8: Compute class weights to handle imbalance
    weights_tensor = compute_weights(
        df["label_id"].values, num_classes=num_classes, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # Step 9: Build optimizer and scheduler
    optimizer = build_optimizer(
        model,
        lr_bert=args.lr_bert,
        lr_resnet=args.lr_resnet,
        lr_heads=args.lr_heads,
        weight_decay=0.01,
    )
    scheduler = build_scheduler(optimizer, patience=args.early_stop_patience)

    best_val_loss = float("inf")
    early_stop_counter = 0
    best_state_dict_path = os.path.join(args.output_dir, "state_dict_transfer.pt")

    # Step 10: Training loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc, train_report = train_one_epoch(
            model, train_loader, criterion, optimizer, device, label_names
        )
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(train_report)

        val_loss, val_acc, val_report, p, r, f1 = evaluate(
            model, val_loader, criterion, device, label_names
        )
        print(
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro P/R/F1: {p:.4f}/{r:.4f}/{f1:.4f}"
        )
        print(val_report)

        # Step scheduler on validation loss
        scheduler.step(val_loss)

        # Early stopping + save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), best_state_dict_path)
            print(f"[INFO] Saved improved model to {best_state_dict_path}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.early_stop_patience:
                print("[INFO] Early stopping triggered.")
                break

    # Save the label mapping for the new dataset
    label_map_path = os.path.join(args.output_dir, "label2id_transfer.json")
    save_label_mapping(label_to_id, label_map_path)
    print(f"[INFO] Label mapping saved to {label_map_path}")

    print("[INFO] Training completed.")

    # Step 11: (Optional) Export to ONNX
    if args.export_onnx:
        # Reload the best checkpoint to ensure ONNX export uses best weights
        if os.path.exists(best_state_dict_path):
            model.load_state_dict(torch.load(best_state_dict_path, map_location=device))
            model.eval()
        else:
            print(f"[WARNING] Best model not found at {best_state_dict_path}. Skipping ONNX export.")
            return
        # Retrieve an example batch for export
        try:
            example_batch = next(iter(val_loader))
        except StopIteration:
            try:
                example_batch = next(iter(train_loader))
            except StopIteration:
                example_batch = None
        if example_batch is None:
            print("[WARNING] Unable to get an example batch for ONNX export. Skipping export.")
            return
        example_input = (
            example_batch["input_ids"].to(device),
            example_batch["attention_mask"].to(device),
            example_batch["pixel_values"].to(device),
        )
        # Timestamped ONNX filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        onnx_path = os.path.join(
            args.output_dir, f"hybrid_model_transfer_{timestamp}.onnx"
        )
        torch.onnx.export(
            model,
            example_input,
            onnx_path,
            input_names=["input_ids", "attention_mask", "pixel_values"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "pixel_values": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=18,
        )
        print(f"[INFO] ONNX model exported to {onnx_path}")


if __name__ == "__main__":
    main()
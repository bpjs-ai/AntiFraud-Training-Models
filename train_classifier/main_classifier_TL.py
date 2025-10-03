"""Transfer learning pipeline for the Hybrid OCR classifier.

This script fine-tunes a previously trained model on a new dataset while keeping
existing classes intact. It can add new classes on top of the previous label
set and will reuse the pretrained weights downloaded from S3 by default.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight

from utils.s3_utils import S3Handler
from train_classifier.dataset import (
    build_dataloaders,
    build_tokenizer,
    build_transforms,
    load_and_prepare_dataframe,
    split_dataframe,
)
from train_classifier.model import HybridModel
import subprocess
from train_classifier.train_utils import (
    build_optimizer,
    build_scheduler,
    evaluate,
    save_label_mapping,
    train_one_epoch,
)
from utils.s3_utils import S3Handler


CLASSIFIER_HEAD_WEIGHT_KEY = "fusion.8.weight"
CLASSIFIER_HEAD_BIAS_KEY = "fusion.8.bias"


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the transfer learning pipeline."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained Hybrid OCR classifier on new data."
    )

    parser.add_argument(
        "--state_dict_path",
        type=str,
        default=None,
        help="Local path to a pretrained state_dict. If omitted the artifact will be downloaded from S3.",
    )
    parser.add_argument(
        "--state_dict_s3_key",
        type=str,
        default="outputs_training/state_dict.pt",
        help="S3 key to download the pretrained state_dict when --state_dict_path is not provided.",
    )
    parser.add_argument(
        "--previous_label_map_path",
        type=str,
        default=None,
        help="Optional local path to the previous label2id mapping. Overrides the S3 download if provided.",
    )
    parser.add_argument(
        "--label_map_s3_key",
        type=str,
        default="outputs_training/label2id.json",
        help="S3 key to download the previous label2id mapping when no local path is provided.",
    )
    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="pretrained_artifacts",
        help="Directory used to cache artifacts downloaded from S3.",
    )
    parser.add_argument(
        "--image_root_dir",
        type=str,
        default="dataset",
        help="Root directory of the transfer learning dataset (default: dataset/dataset_TL).",
    )
    parser.add_argument(
        "--data_csv_path",
        type=str,
        default="dataset/dataset_TL.csv",
        help="CSV file containing filename, label and text columns. Will be generated via OCR when missing.",
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
        help="Number of fine-tuning epochs (default: 10).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Mini-batch size for training and validation (default: 16).",
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
        help="Learning rate for BERT parameters. Set to 0 to keep BERT frozen (default: 0).",
    )
    parser.add_argument(
        "--lr_resnet",
        type=float,
        default=0.0,
        help="Learning rate for ResNet parameters. Set to 0 to keep ResNet frozen (default: 0).",
    )
    parser.add_argument(
        "--lr_heads",
        type=float,
        default=1e-4,
        help="Learning rate for classification head and cross-attention layers (default: 1e-4).",
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
        help="Number of epochs without improvement before early stopping (default: 5).",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze BERT and ResNet by disabling gradients for those modules.",
    )
    parser.add_argument(
        "--export_onnx",
        action="store_true",
        help="Export the best checkpoint to ONNX once training finishes.",
    )

    return parser.parse_args()


def download_artifact(
    s3_handler: Optional[S3Handler],
    s3_key: Optional[str],
    destination_dir: str,
    description: str,
) -> Tuple[Optional[S3Handler], Optional[str]]:
    """Download an artifact from S3 and return the updated handler and local path."""
    if not s3_key:
        return s3_handler, None

    filename = os.path.basename(s3_key.rstrip("/"))
    if not filename:
        raise ValueError(f"Invalid S3 key '{s3_key}' for {description}.")

    if s3_handler is None:
        s3_handler = S3Handler()

    os.makedirs(destination_dir, exist_ok=True)
    local_path = os.path.join(destination_dir, filename)
    success = s3_handler.download_file(s3_key, local_path)
    if not success:
        raise FileNotFoundError(
            f"Unable to download {description} from s3://{s3_handler.bucket_name}/{s3_key}."
        )

    print(
        f"[INFO] Downloaded {description} to {local_path} "
        f"from s3://{s3_handler.bucket_name}/{s3_key}"
    )
    return s3_handler, local_path


def load_label_mapping(path: str) -> Dict[str, int]:
    """Load a label2id mapping from disk."""
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return {str(label): int(idx) for label, idx in data.items()}


def merge_label_mappings(
    new_labels: List[str],
    previous_mapping: Dict[str, int],
) -> Dict[str, int]:
    """Merge new labels into the previous mapping while keeping old indices."""
    merged: Dict[str, int] = {}
    max_existing_id = -1

    if previous_mapping:
        for label, idx in sorted(previous_mapping.items(), key=lambda item: item[1]):
            merged[str(label)] = int(idx)
            max_existing_id = max(max_existing_id, int(idx))

    next_id = max_existing_id + 1
    for label in sorted({str(label) for label in new_labels}):
        if label not in merged:
            merged[label] = next_id
            next_id += 1

    return merged


def build_class_weights(labels: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    """Compute balanced class weights and fill missing classes with weight 1."""
    weights = torch.ones(num_classes, dtype=torch.float32, device=device)
    if labels.size == 0:
        return weights

    unique_labels = np.unique(labels)
    class_weights = compute_class_weight(class_weight="balanced", classes=unique_labels, y=labels)
    for label_idx, class_weight in zip(unique_labels, class_weights):
        weights[int(label_idx)] = float(class_weight)
    return weights


def load_pretrained_weights(
    model: nn.Module,
    state_dict_path: str,
    previous_mapping: Dict[str, int],
    new_mapping: Dict[str, int],
) -> None:
    """Load pretrained weights while preserving logits for existing classes."""
    checkpoint = torch.load(state_dict_path, map_location="cpu")
    if isinstance(checkpoint, Dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model_dict = model.state_dict()

    for key, value in checkpoint.items():
        if key in (CLASSIFIER_HEAD_WEIGHT_KEY, CLASSIFIER_HEAD_BIAS_KEY):
            continue
        if key in model_dict and model_dict[key].shape == value.shape:
            model_dict[key] = value

    # Preserve classifier weights for existing labels
    if CLASSIFIER_HEAD_WEIGHT_KEY in checkpoint and CLASSIFIER_HEAD_WEIGHT_KEY in model_dict:
        old_weight = checkpoint[CLASSIFIER_HEAD_WEIGHT_KEY]
        new_weight = model_dict[CLASSIFIER_HEAD_WEIGHT_KEY]
        for label, old_idx in previous_mapping.items():
            label_str = str(label)
            if label_str not in new_mapping:
                continue
            new_idx = new_mapping[label_str]
            if old_idx < old_weight.shape[0] and new_idx < new_weight.shape[0]:
                new_weight[new_idx] = old_weight[old_idx]
        model_dict[CLASSIFIER_HEAD_WEIGHT_KEY] = new_weight

    if CLASSIFIER_HEAD_BIAS_KEY in checkpoint and CLASSIFIER_HEAD_BIAS_KEY in model_dict:
        old_bias = checkpoint[CLASSIFIER_HEAD_BIAS_KEY]
        new_bias = model_dict[CLASSIFIER_HEAD_BIAS_KEY]
        for label, old_idx in previous_mapping.items():
            label_str = str(label)
            if label_str not in new_mapping:
                continue
            new_idx = new_mapping[label_str]
            if old_idx < old_bias.shape[0] and new_idx < new_bias.shape[0]:
                new_bias[new_idx] = old_bias[old_idx]
        model_dict[CLASSIFIER_HEAD_BIAS_KEY] = new_bias

    model.load_state_dict(model_dict)
    print(f"[INFO] Loaded pretrained weights from {state_dict_path}")


def main() -> None:
    args = parse_arguments()
    s3_handler = S3Handler()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        height_str, width_str = args.image_size.split(",")
        image_size = (int(height_str), int(width_str))
    except Exception as exc:  # noqa: BLE001
        raise ValueError("--image_size should be formatted as 'height,width', e.g., '224,224'") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    s3_handler: Optional[S3Handler] = None

    state_dict_path = args.state_dict_path
    if state_dict_path:
        if not os.path.exists(state_dict_path):
            raise FileNotFoundError(f"state_dict not found at {state_dict_path}")
    else:
        s3_handler, state_dict_path = download_artifact(
            s3_handler,
            args.state_dict_s3_key,
            args.artifact_dir,
            "pretrained state_dict",
        )
        if state_dict_path is None:
            raise FileNotFoundError("Unable to obtain pretrained state_dict.")

    previous_mapping: Dict[str, int] = {}
    label_map_source = None
    if args.previous_label_map_path:
        if not os.path.exists(args.previous_label_map_path):
            raise FileNotFoundError(
                f"Previous label map not found at {args.previous_label_map_path}."
            )
        label_map_source = args.previous_label_map_path
    elif args.label_map_s3_key:
        try:
            s3_handler, label_map_source = download_artifact(
                s3_handler,
                args.label_map_s3_key,
                args.artifact_dir,
                "previous label mapping",
            )
        except FileNotFoundError:
            print(
                f"[WARNING] Failed to download label map from S3 key '{args.label_map_s3_key}'. "
                "Proceeding without the previous mapping."
            )
            label_map_source = None

    if label_map_source and os.path.exists(label_map_source):
        previous_mapping = load_label_mapping(label_map_source)
        print(f"[INFO] Loaded previous label map containing {len(previous_mapping)} classes.")
    else:
        print("[WARNING] Previous label map could not be loaded. The model will be fine-tuned only on the new dataset classes.")

    subprocess.run([
        "python", "train_classifier/ocr_paddle.py",
        "--input_folder", args.image_root_dir,
        "--output_csv", args.data_csv_path,
        "--num_workers", "50"
    ])
    
    df, _ = load_and_prepare_dataframe(
        data_csv_path=args.data_csv_path,
        image_root_dir=args.image_root_dir,
        filename_col="filename",
        label_col="label",
        text_col="text",
        drop_empty_text=True,
    )
    df["label"] = df["label"].astype(str)

    merged_mapping = merge_label_mappings(df["label"].tolist(), previous_mapping)
    df["label_id"] = df["label"].map(merged_mapping)

    added_classes = sorted(set(merged_mapping) - set(previous_mapping))
    if added_classes:
        print(f"[INFO] Added classes: {added_classes}")

    active_label_ids = sorted(df["label_id"].unique())
    metric_label_names = [label for label, idx in sorted(
        ((label, idx) for label, idx in merged_mapping.items() if idx in active_label_ids),
        key=lambda item: item[1],
    )]

    train_df, val_df = split_dataframe(
        df,
        stratify_col="label",
        test_size=args.val_size,
        random_state=42,
    )

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

    num_classes = len(merged_mapping)

    model = HybridModel(num_classes=num_classes, bert_model_name=args.bert_model_name).to(device)

    load_pretrained_weights(model, state_dict_path, previous_mapping, merged_mapping)

    if args.freeze_backbone:
        print("[INFO] Freezing BERT and ResNet backbones. Only head and attention layers will be trained.")
        for param in model.bert.parameters():
            param.requires_grad = False
        for param in model.resnet.parameters():
            param.requires_grad = False

    weights_tensor = build_class_weights(df["label_id"].values, num_classes=num_classes, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

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

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc, train_report = train_one_epoch(
            model, train_loader, criterion, optimizer, device, metric_label_names
        )
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(train_report)

        val_loss, val_acc, val_report, precision, recall, f1_score = evaluate(
            model, val_loader, criterion, device, metric_label_names
        )
        print(
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro P/R/F1: "
            f"{precision:.4f}/{recall:.4f}/{f1_score:.4f}"
        )
        print(val_report)

        scheduler.step(val_loss)

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

    label_map_path = os.path.join(args.output_dir, "label2id_transfer.json")
    save_label_mapping(merged_mapping, label_map_path)
    print(f"[INFO] Label mapping saved to {label_map_path}")

    print("[INFO] Training completed.")

    if args.export_onnx:
        if os.path.exists(best_state_dict_path):
            model.load_state_dict(torch.load(best_state_dict_path, map_location=device))
            model.eval()
        else:
            print(f"[WARNING] Best model not found at {best_state_dict_path}. Skipping ONNX export.")
            return

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

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        onnx_path = os.path.join(args.output_dir, f"hybrid_model_transfer_{timestamp}.onnx")
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
        s3_handler.upload_file(cfg.best_onnx_model_path, f"outputs_training/{os.path.basename(cfg.best_onnx_model_path)}")
        s3_handler.upload_file(cfg.label_map_path, f"outputs_training/{os.path.basename(cfg.label_map_path)}")
        print(f"✅ ONNX model saved at {cfg.best_onnx_model_path}")
        print(f"[INFO] ONNX model exported to {onnx_path}")


if __name__ == "__main__":
    main()

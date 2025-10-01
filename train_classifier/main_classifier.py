import argparse
import os
import json
import torch
import torch.nn as nn
from datetime import datetime

from utils.s3_utils import S3Handler
from train_classifier.config import TrainConfig
from train_classifier.dataset import (
    build_dataloaders,
    build_tokenizer,
    build_transforms,
    load_and_prepare_dataframe,
    split_dataframe,
)
from train_classifier.model import HybridModel
from train_classifier.train_utils import (
    build_optimizer,
    build_scheduler,
    compute_weights,
    evaluate,
    save_label_mapping,
    train_one_epoch,
)
import subprocess
    


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid OCR Training Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")

    # Allow override via CLI
    parser.add_argument("--data_csv_path", type=str, default="dataset/classifier/data.csv")
    parser.add_argument("--image_root_dir", type=str, default="dataset/classifier")
    parser.add_argument("--val_size", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lr_bert", type=float, default=None)
    parser.add_argument("--lr_resnet", type=float, default=None)
    parser.add_argument("--lr_heads", type=float, default=None)
    parser.add_argument("--bert_model_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def load_config(args) -> TrainConfig:
    cfg = TrainConfig()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    # CLI overrides
    for field in [
        "data_csv_path",
        "image_root_dir",
        "val_size",
        "epochs",
        "batch_size",
        "num_workers",
        "lr_bert",
        "lr_resnet",
        "lr_heads",
        "bert_model_name",
        "output_dir",
    ]:
        arg_val = getattr(args, field)
        if arg_val is not None:
            setattr(cfg, field, arg_val)

    return cfg


def main():
    args = parse_args()
    cfg = load_config(args)
    s3_handler = S3Handler()

    os.makedirs(cfg.output_dir, exist_ok=True)
    # Backward compatibility for older configs
    if not hasattr(cfg, "best_state_dict_path"):
        legacy_best = getattr(cfg, "best_model_path", os.path.join(cfg.output_dir, "best_model.pt"))
        cfg.best_state_dict_path = legacy_best

    if not hasattr(cfg, "best_onnx_model_path"):
        cfg.best_onnx_model_path = os.path.join(cfg.output_dir, "best_model.onnx")

    if not os.path.isabs(cfg.best_state_dict_path):
        cfg.best_state_dict_path = os.path.join(
            cfg.output_dir, os.path.basename(cfg.best_state_dict_path)
        )
    if not os.path.isabs(cfg.label_map_path):
        cfg.label_map_path = os.path.join(cfg.output_dir, os.path.basename(cfg.label_map_path))

    best_onnx_raw_path = getattr(cfg, "best_onnx_model_path", "full_model.onnx")
    best_onnx_name = os.path.basename(best_onnx_raw_path)
    if os.path.isabs(best_onnx_raw_path):
        onnx_dir = os.path.dirname(best_onnx_raw_path) or cfg.output_dir
    else:
        onnx_dir = cfg.output_dir

    timestamp = datetime.now().strftime("%Y%m%d%H")
    stem, ext = os.path.splitext(best_onnx_name)
    if not stem:
        stem = "full_model"
    if not ext:
        ext = ".onnx"

    os.makedirs(onnx_dir, exist_ok=True)
    cfg.best_onnx_model_path = os.path.join(onnx_dir, f"{stem}_{timestamp}{ext}")

    print(torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_folder=cfg.image_root_dir
    output_csv=cfg.data_csv_path

    

    subprocess.run([
        "python", "train_classifier/ocr_paddle.py",
        "--input_folder", cfg.image_root_dir,
        "--output_csv", cfg.data_csv_path,
        "--num_workers", "50"
    ])

    # Dataframe & label map
    df, label_to_id = load_and_prepare_dataframe(
        data_csv_path=cfg.data_csv_path,
        image_root_dir=cfg.image_root_dir,
        filename_col=cfg.filename_col,
        label_col=cfg.label_col,
        text_col=cfg.text_col,
        drop_empty_text=cfg.drop_empty_text,
    )

    train_df, val_df = split_dataframe(df, stratify_col=cfg.label_col, test_size=cfg.val_size, random_state=cfg.seed)

    tokenizer = build_tokenizer(cfg.bert_model_name)
    transform = build_transforms(cfg.image_size)

    train_loader, val_loader = build_dataloaders(
        train_df=train_df,
        val_df=val_df,
        tokenizer=tokenizer,
        transform=transform,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    num_classes = len(label_to_id)
    id_to_label = {v: k for k, v in label_to_id.items()}
    label_names = [id_to_label[i] for i in range(num_classes)]

    model = HybridModel(num_classes=num_classes, bert_model_name=cfg.bert_model_name).to(device)

    # Class weights
    weights = compute_weights(df["label_id"].values, num_classes=num_classes, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = build_optimizer(
        model,
        lr_bert=cfg.lr_bert,
        lr_resnet=cfg.lr_resnet,
        lr_heads=cfg.lr_heads,
        weight_decay=cfg.weight_decay,
    )
    scheduler = build_scheduler(optimizer, patience=cfg.scheduler_patience)

    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(cfg.epochs):
        print(f"Epoch {epoch + 1}/{cfg.epochs}")
        train_loss, train_acc, train_report = train_one_epoch(
            model, train_loader, criterion, optimizer, device, label_names
        )
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(train_report)

        val_loss, val_acc, val_report, p, r, f1 = evaluate(
            model, val_loader, criterion, device, label_names
        )
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro P/R/F1: {p:.4f}/{r:.4f}/{f1:.4f}")
        print(val_report)

        # Scheduler step
        scheduler.step(val_loss)

        # Early stopping + checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), cfg.best_state_dict_path)
            print(f"Saved improved state dict to {cfg.best_state_dict_path}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= cfg.early_stop_patience:
                print("Early stopping triggered")
                break

    # Save label mapping
    save_label_mapping(label_to_id, cfg.label_map_path)

    print("Training completed.")

    # Reload the best checkpoint before exporting to ensure ONNX uses the best weights
    if os.path.exists(cfg.best_state_dict_path):
        model.load_state_dict(torch.load(cfg.best_state_dict_path, map_location=device))
        model.to(device)
    else:
        print(
            f"Warning: best model checkpoint not found at {cfg.best_state_dict_path}. "
            "Skipping ONNX export."
        )
        return

    model.eval()

    # Try to grab an example batch from the validation loader, fallback to the training loader
    example_batch = None
    for loader in (val_loader, train_loader):
        if loader is None:
            continue
        try:
            example_batch = next(iter(loader))
            break
        except StopIteration:
            continue

    if example_batch is None:
        print("Warning: Unable to retrieve a batch for ONNX export. Skipping export.")
        return

    with torch.no_grad():
        example_input = (
            example_batch["input_ids"].to(device),
            example_batch["attention_mask"].to(device),
            example_batch["pixel_values"].to(device)
        )

        torch.onnx.export(
            model,
            example_input,
            cfg.best_onnx_model_path,
            input_names=["input_ids", "attention_mask", "pixel_values"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "pixel_values": {0: "batch_size"},
                "logits": {0: "batch_size"}
            },
            opset_version=18,
        )
    s3_handler.upload_file(cfg.best_onnx_model_path, f"outputs_training/{os.path.basename(cfg.best_onnx_model_path)}")
    s3_handler.upload_file(cfg.label_map_path, f"outputs_training/{os.path.basename(cfg.label_map_path)}")
    s3_handler.upload_file(cfg.best_state_dict_path, f"outputs_training/{os.path.basename(cfg.best_state_dict_path)}")
    print(f"✅ ONNX model saved at {cfg.best_onnx_model_path}")



if __name__ == "__main__":
    main()



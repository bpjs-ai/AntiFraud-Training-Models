import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import torch.nn as nn

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
import warnings
warnings.filterwarnings("ignore")
UTC_PLUS_7 = timezone(timedelta(hours=7))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid OCR Training Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")

    defaults = TrainConfig()
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--num_workers", type=int, default=defaults.num_workers)
    parser.add_argument("--lr_bert", type=float, default=defaults.lr_bert)
    parser.add_argument("--lr_resnet", type=float, default=defaults.lr_resnet)
    parser.add_argument("--lr_heads", type=float, default=defaults.lr_heads)
    parser.add_argument("--early_stop_patience", type=int, default=defaults.early_stop_patience)

    return parser.parse_args()


def now_in_gmt7() -> datetime:
    """Return the current time localized to UTC+7."""
    return datetime.now(timezone.utc).astimezone(UTC_PLUS_7)


def load_config(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

    for field in ("epochs", "batch_size", "num_workers", "lr_bert", "lr_resnet", "lr_heads", "early_stop_patience"):
        override = getattr(args, field, None)
        if override is not None:
            setattr(cfg, field, override)

    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Backward compatibility for legacy configs
    if not hasattr(cfg, "best_state_dict_path"):
        legacy_best = getattr(cfg, "best_model_path", os.path.join(cfg.output_dir, "best_model.pt"))
        cfg.best_state_dict_path = legacy_best

    if not hasattr(cfg, "best_onnx_model_path"):
        cfg.best_onnx_model_path = os.path.join(cfg.output_dir, "best_model.onnx")

    def resolve_output_path(path_value: str, default_value: str) -> Path:
        """Resolve path relative to the output directory and ensure parent exists."""
        candidate = Path(path_value or default_value)
        if not candidate.is_absolute():
            candidate = output_dir / candidate
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    def timestamped_path(template: Path, timestamp: str) -> Path:
        """Append a timestamp to the template path before its suffix."""
        if template.suffix:
            new_name = f"{template.stem}_{timestamp}{template.suffix}"
        else:
            new_name = f"{template.name}_{timestamp}"
        return template.with_name(new_name)

    state_dict_template = resolve_output_path(getattr(cfg, "best_state_dict_path", ""), "state_dict/best_state_dict.pt")
    label_map_template = resolve_output_path(getattr(cfg, "label_map_path", ""), "label2id/label2id.json")
    result_template = resolve_output_path(getattr(cfg, "result_path", ""), "result/training_report.json")
    onnx_template = resolve_output_path(getattr(cfg, "best_onnx_model_path", ""), "model/full_model.onnx")

    run_timestamp = now_in_gmt7().strftime("%Y%m%d%H%M%S")
    state_dict_path = timestamped_path(state_dict_template, run_timestamp)
    label_map_path = timestamped_path(label_map_template, run_timestamp)
    result_path = timestamped_path(result_template, run_timestamp)
    onnx_path = timestamped_path(onnx_template, run_timestamp)

    for path in (state_dict_path, label_map_path, result_path, onnx_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    cfg.best_state_dict_path = str(state_dict_path)
    cfg.label_map_path = str(label_map_path)
    cfg.result_path = str(result_path)
    cfg.best_onnx_model_path = str(onnx_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df, label_to_id = load_and_prepare_dataframe(
        data_csv_path=cfg.data_csv_path,
        image_root_dir=cfg.image_root_dir,
        filename_col=cfg.filename_col,
        label_col=cfg.label_col,
        text_col=cfg.text_col,
        drop_empty_text=cfg.drop_empty_text,
    )

    train_df, val_df = split_dataframe(
        df,
        stratify_col=cfg.label_col,
        test_size=cfg.val_size,
        random_state=cfg.seed,
    )

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
    best_metrics = {}
    last_metrics = {}

    for epoch in range(cfg.epochs):
        print(f"Epoch {epoch + 1}/{cfg.epochs}")
        train_loss, train_acc, train_report = train_one_epoch(
            model, train_loader, criterion, optimizer, device, label_names
        )
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(train_report)

        val_loss, val_acc, val_report, precision, recall, f1, per_class_metrics = evaluate(
            model, val_loader, criterion, device, label_names
        )
        print(
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro P/R/F1: "
            f"{precision:.4f}/{recall:.4f}/{f1:.4f}"
        )
        print(val_report)

        scheduler.step(val_loss)

        current_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "per_class": per_class_metrics,
        }
        last_metrics = current_metrics

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), cfg.best_state_dict_path)
            print(f"Saved improved state dict to {cfg.best_state_dict_path}")
            best_metrics = current_metrics
        else:
            early_stop_counter += 1
            if early_stop_counter >= cfg.early_stop_patience:
                print("Early stopping triggered")
                break

    save_label_mapping(label_to_id, cfg.label_map_path)
    print("Training completed.")

    summary_metrics = best_metrics or last_metrics
    per_class_summary = summary_metrics.get("per_class", {})
    aggregate_metrics = {k: v for k, v in summary_metrics.items() if k != "per_class"}
    
    # Convert metrics to percentage format 
    def format_metric(value, is_loss=False):
        if isinstance(value, float):
            if is_loss:
                return f"{value:.3f}"
            return f"{value * 100:.1f}%"
        return value
    
    def format_metrics_dict(metrics):
        """Recursively format all float values in a metrics dictionary to percentage."""
        formatted = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                formatted[key] = format_metrics_dict(value)
            elif key in ("train_loss", "val_loss"):
                formatted[key] = format_metric(value, is_loss=True)
            elif key in ("train_accuracy", "val_accuracy", 
                         "precision", "recall", "f1", "f1_score"):
                formatted[key] = format_metric(value, is_loss=False)
            else:
                formatted[key] = value
        return formatted
    
    # Format aggregate metrics
    formatted_aggregate_metrics = format_metrics_dict(aggregate_metrics)
    
    # Format per-class metrics
    formatted_per_class_summary = {}
    for class_name, class_metrics in per_class_summary.items():
        formatted_per_class_summary[class_name] = format_metrics_dict(class_metrics)
    
    hyperparameters = {
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "lr_bert": cfg.lr_bert,
        "lr_resnet": cfg.lr_resnet,
        "lr_heads": cfg.lr_heads,
        "weight_decay": cfg.weight_decay,
        "scheduler_patience": cfg.scheduler_patience,
        "early_stop_patience": cfg.early_stop_patience,
    }
    trained_state_dict_key = f"outputs_training/full/state_dicts/{Path(cfg.best_state_dict_path).name}"
    trained_state_dict_source = trained_state_dict_key
    training_summary = {
        "mode": "full",
        "timestamp": now_in_gmt7().isoformat(),
        "num_classes": num_classes,
        "hyperparameters": hyperparameters,
        "metrics": formatted_aggregate_metrics,
        "per_class_metrics": formatted_per_class_summary,
        "trained_model_path": trained_state_dict_source,
    }
    with open(cfg.result_path, "w", encoding="utf-8") as report_file:
        json.dump(training_summary, report_file, indent=2)

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

    example_input = (
        example_batch["input_ids"].to(device),
        example_batch["attention_mask"].to(device),
        example_batch["pixel_values"].to(device),
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
            "logits": {0: "batch_size"},
        },
        opset_version=18,
    )

    s3_handler = S3Handler()
    onnx_key = f"outputs_training/full/models/{os.path.basename(cfg.best_onnx_model_path)}"
    label_map_key = f"outputs_training/full/labelmaps/{os.path.basename(cfg.label_map_path)}"
    result_key = f"outputs_training/full/results/{os.path.basename(cfg.result_path)}"

    s3_handler.upload_file(cfg.best_onnx_model_path, onnx_key)
    s3_handler.upload_file(cfg.label_map_path, label_map_key)
    s3_handler.upload_file(cfg.result_path, result_key)
    s3_handler.upload_file(cfg.best_state_dict_path, trained_state_dict_key)

    report_filename = os.path.basename(result_key)

    # print(f"✅ ONNX model saved at {cfg.best_onnx_model_path}")
    print(f"[INFO] Uploaded ONNX model to {onnx_key}")
    print(f"[INFO] uploaded Labelmap to {label_map_key}")
    print(f"[INFO] Uploaded Report to {result_key}")
    
    # Extract numeric value for display
    val_acc = summary_metrics.get('val_accuracy', 'N/A')
    if isinstance(val_acc, str) and '%' in val_acc:
        val_acc_display = val_acc
    elif isinstance(val_acc, float):
        val_acc_display = f"{val_acc * 100:.1f}%"
    else:
        val_acc_display = str(val_acc)
    print(f"[INFO] Models Accuracy: {val_acc_display}")
    print(f"[INFO] Hyperparameters: {hyperparameters}")
    print(f"[INFO] Per Class Accuracy: {formatted_per_class_summary}")

if __name__ == "__main__":
    main()

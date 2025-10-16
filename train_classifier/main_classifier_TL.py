import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight

from train_classifier.dataset import (
    build_dataloaders,
    build_tokenizer,
    build_transforms,
    load_and_prepare_dataframe,
    split_dataframe,
)
from train_classifier.config import TransferConfig
from train_classifier.model import HybridModel
from train_classifier.train_utils import (
    build_optimizer,
    build_scheduler,
    evaluate,
    save_label_mapping,
    train_one_epoch,
)
from utils.s3_utils import S3Handler

import warnings
warnings.filterwarnings("ignore")

CLASSIFIER_HEAD_WEIGHT_KEY = "fusion.8.weight"
CLASSIFIER_HEAD_BIAS_KEY = "fusion.8.bias"
UTC_PLUS_7 = timezone(timedelta(hours=7))


def now_in_gmt7() -> datetime:
    """Return the current time localized to UTC+7."""
    return datetime.now(timezone.utc).astimezone(UTC_PLUS_7)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the transfer learning pipeline."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained Hybrid OCR classifier on new data."
    )
    defaults = TransferConfig()
    parser.add_argument(
        "--state_dict_s3_key",
        type=str,
        default=defaults.state_dict_s3_key,
        help="S3 key to download the pretrained state_dict.",
    )
    parser.add_argument(
        "--label_map_s3_key",
        type=str,
        default=defaults.label_map_s3_key,
        help="S3 key to download the previous label2id mapping.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=defaults.epochs,
        help="Number of fine-tuning epochs (default: 50).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=defaults.batch_size,
        help="Mini-batch size for training and validation (default: 16).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=defaults.num_workers,
        help="Number of worker processes for PyTorch DataLoader and OCR (default: 8).",
    )
    parser.add_argument(
        "--lr_bert",
        type=float,
        default=defaults.lr_bert,
        help="Learning rate for BERT parameters. Set to 0 to keep BERT frozen (default: 2e-5).",
    )
    parser.add_argument(
        "--lr_resnet",
        type=float,
        default=defaults.lr_resnet,
        help="Learning rate for ResNet parameters. Set to 0 to keep ResNet frozen (default: 1e-4).",
    )
    parser.add_argument(
        "--lr_heads",
        type=float,
        default=defaults.lr_heads,
        help="Learning rate for classification head and cross-attention layers (default: 1e-4).",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=defaults.early_stop_patience,
        help="Number of epochs without improvement before early stopping (default: 15).",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze BERT and ResNet by disabling gradients for those modules.",
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
    defaults = TransferConfig()

    output_dir = Path(defaults.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def resolve_output_path(path_value: str, default_value: str) -> Path:
        """Resolve a relative output path against the pipeline output directory."""
        candidate = Path(path_value or default_value)
        if not candidate.is_absolute():
            candidate = output_dir / candidate
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    def timestamped_path(template: Path, timestamp: str) -> Path:
        """Append a timestamp before the suffix of the provided template path."""
        if template.suffix:
            new_name = f"{template.stem}_{timestamp}{template.suffix}"
        else:
            new_name = f"{template.name}_{timestamp}"
        return template.with_name(new_name)

    best_state_dict_template = resolve_output_path(getattr(defaults, "best_state_dict_path", ""), "state_dicts/state_dict_transfer.pt")
    label_map_template = resolve_output_path(getattr(defaults, "label_map_path", ""), "labelmaps/label2id_transfer.json")
    result_template = resolve_output_path(getattr(defaults, "result_path", ""), "results/training_report.json")
    onnx_template = resolve_output_path(getattr(defaults, "best_onnx_model_path", ""), "models/hybrid_model_transfer.onnx")

    run_timestamp = now_in_gmt7().strftime("%Y%m%d%H%M%S")
    best_state_dict_path = timestamped_path(best_state_dict_template, run_timestamp)
    label_map_path = timestamped_path(label_map_template, run_timestamp)
    result_path = timestamped_path(result_template, run_timestamp)
    onnx_path = timestamped_path(onnx_template, run_timestamp)

    for path in (best_state_dict_path, label_map_path, result_path, onnx_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    defaults.best_state_dict_path = str(best_state_dict_path)
    defaults.label_map_path = str(label_map_path)
    defaults.result_path = str(result_path)
    defaults.best_onnx_model_path = str(onnx_path)

    image_size = defaults.image_size
    if len(image_size) != 2:
        raise ValueError("defaults.image_size must contain two entries.")
    image_size = (int(image_size[0]), int(image_size[1]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    s3_handler: Optional[S3Handler] = None

    state_dict_s3_key = args.state_dict_s3_key or defaults.state_dict_s3_key
    if not state_dict_s3_key:
        raise ValueError("state_dict_s3_key must be provided to download the pretrained weights.")
    s3_handler, state_dict_path = download_artifact(
        s3_handler,
        state_dict_s3_key,
        defaults.artifact_dir,
        "pretrained state_dict",
    )

    previous_mapping: Dict[str, int] = {}
    label_map_s3_key = args.label_map_s3_key or defaults.label_map_s3_key
    if not label_map_s3_key:
        raise ValueError("label_map_s3_key must be provided to download the previous label mapping.")
    s3_handler, label_map_source = download_artifact(
        s3_handler,
        label_map_s3_key,
        defaults.artifact_dir,
        "previous label mapping",
    )

    if label_map_source and os.path.exists(label_map_source):
        previous_mapping = load_label_mapping(label_map_source)
        print(f"[INFO] Loaded previous label map containing {len(previous_mapping)} classes.")
    else:
        print("[WARNING] Previous label map could not be loaded. The model will be fine-tuned only on the new dataset classes.")


    df, _ = load_and_prepare_dataframe(
        data_csv_path=defaults.data_csv_path,
        image_root_dir=defaults.image_root_dir,
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
        test_size=defaults.val_size,
        random_state=42,
    )

    tokenizer = build_tokenizer(defaults.bert_model_name)
    transform = build_transforms(image_size)

    train_loader, val_loader = build_dataloaders(
        train_df=train_df,
        val_df=val_df,
        tokenizer=tokenizer,
        transform=transform,
        max_length=defaults.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    num_classes = len(merged_mapping)

    model = HybridModel(num_classes=num_classes, bert_model_name=defaults.bert_model_name).to(device)

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
    best_metrics: Dict[str, Any] = {}
    last_metrics: Dict[str, Any] = {}

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc, train_report = train_one_epoch(
            model, train_loader, criterion, optimizer, device, metric_label_names
        )
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(train_report)

        val_loss, val_acc, val_report, precision, recall, f1_score, per_class_metrics = evaluate(
            model, val_loader, criterion, device, metric_label_names
        )
        print(
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro P/R/F1: "
            f"{precision:.4f}/{recall:.4f}/{f1_score:.4f}"
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
            "f1": f1_score,
            "per_class": per_class_metrics,
        }
        last_metrics = current_metrics

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), str(best_state_dict_path))
            print(f"[INFO] Saved improved model to {best_state_dict_path}")
            best_metrics = current_metrics
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.early_stop_patience:
                print("[INFO] Early stopping triggered.")
                break

    save_label_mapping(merged_mapping, str(label_map_path))
    print(f"[INFO] Label mapping saved to {label_map_path}")

    print("[INFO] Training completed.")

    summary_metrics = best_metrics or last_metrics
    per_class_summary = summary_metrics.get("per_class", {})
    aggregate_metrics = {k: v for k, v in summary_metrics.items() if k != "per_class"}
    hyperparameters = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr_bert": args.lr_bert,
        "lr_resnet": args.lr_resnet,
        "lr_heads": args.lr_heads,
        "early_stop_patience": args.early_stop_patience,
        "freeze_backbone": args.freeze_backbone,
    }
    trained_state_dict_key = f"outputs_training/transfer/state_dicts/{best_state_dict_path.name}"
    trained_state_dict_source = trained_state_dict_key
    if trained_state_dict_source and s3_handler and getattr(s3_handler, "bucket_name", None):
        trained_state_dict_source = f"s3://{s3_handler.bucket_name}/{trained_state_dict_source.lstrip('/')}"
    pretrained_state_dict_source = state_dict_s3_key
    if pretrained_state_dict_source and not pretrained_state_dict_source.startswith("s3://"):
        if s3_handler and getattr(s3_handler, "bucket_name", None):
            s3_key = pretrained_state_dict_source.lstrip("/")
            pretrained_state_dict_source = f"s3://{s3_handler.bucket_name}/{s3_key}"
    training_summary = {
        "mode": "transfer",
        "timestamp": now_in_gmt7().isoformat(),
        "num_classes": num_classes,
        "added_classes": added_classes,
        "hyperparameters": hyperparameters,
        "metrics": aggregate_metrics,
        "per_class_metrics": per_class_summary,
        "pretrained_state_dict": pretrained_state_dict_source or state_dict_path,
        "trained_model_path": trained_state_dict_source,
    }
    with open(result_path, "w", encoding="utf-8") as report_file:
        json.dump(training_summary, report_file, indent=2)

    if defaults.export_onnx:
        if best_state_dict_path.exists():
            model.load_state_dict(torch.load(str(best_state_dict_path), map_location=device))
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

        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            model,
            example_input,
            str(onnx_path),
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
        upload_handler = s3_handler or S3Handler()
        onnx_key = f"outputs_training/transfer/models/{onnx_path.name}"
        label_map_key = f"outputs_training/transfer/labelmaps/{label_map_path.name}"
        result_key = f"outputs_training/transfer/results/{result_path.name}"
        state_dict_key = trained_state_dict_key

        upload_handler.upload_file(str(onnx_path), onnx_key)
        upload_handler.upload_file(str(label_map_path), label_map_key)
        upload_handler.upload_file(str(result_path), result_key)
        upload_handler.upload_file(str(best_state_dict_path), state_dict_key)

        report_filename = os.path.basename(result_key)

        # print(f"✅ ONNX model saved at {onnx_path}")
        print(f"[INFO] Uploaded ONNX model to {onnx_key}")
        print(f"[INFO] Uploaded Labelmap to {label_map_key}")
        print(f"[INFO] Uploaded Report to {result_key}")
        
        print(f"[INFO] Pretrained State Dict used: {state_dict_s3_key}")
        print(f"[INFO] Models Accuracy: {summary_metrics.get('val_accuracy', 'N/A'):.4f}")

if __name__ == "__main__":
    main()

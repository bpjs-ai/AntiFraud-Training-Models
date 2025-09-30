import argparse
import os
import json
import torch
import torch.nn as nn

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
    parser.add_argument("--data_csv_path", type=str, default=None)
    parser.add_argument("--image_root_dir", type=str, default=None)
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

    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_folder=cfg.image_root_dir
    output_csv=cfg.data_csv_path

    

    subprocess.run([
        "python", "train_classifier/ocr_paddle.py",
        "--input_folder", cfg.image_root_dir,
        "--output_csv", cfg.data_csv_path,
        "--num_workers", "60"
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
            torch.save(model.state_dict(), cfg.best_model_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= cfg.early_stop_patience:
                print("Early stopping triggered")
                break

    # Save label mapping
    save_label_mapping(label_to_id, cfg.label_map_path)

    print("Training completed.")
    # Export ke ONNX pakai batch asli dari val_loader
    model.eval()
    with torch.no_grad():
        example_batch = next(iter(val_loader))

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
            output_names=["output"],
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "pixel_values": {0: "batch_size"},
                "output": {0: "batch_size"}
            },
            opset_version=17
        )
    print(f"✅ ONNX model saved at {cfg.best_onnx_model_path}")



if __name__ == "__main__":
    main()



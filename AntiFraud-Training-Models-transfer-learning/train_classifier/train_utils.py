import json
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


def build_optimizer(model: nn.Module, lr_bert: float, lr_resnet: float, lr_heads: float, weight_decay: float = 0.01):
    optimizer = AdamW(
        [
            {"params": model.bert.parameters(), "lr": lr_bert},
            {"params": model.resnet.parameters(), "lr": lr_resnet},
            {"params": model.text_linear.parameters(), "lr": lr_heads},
            {"params": model.image_linear.parameters(), "lr": lr_heads},
            {"params": model.fusion.parameters(), "lr": lr_heads * 10},
        ],
        weight_decay=weight_decay,
    )
    return optimizer


def build_scheduler(optimizer, patience: int = 5):
    return ReduceLROnPlateau(optimizer, mode="min", patience=patience)


def compute_weights(labels: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    classes = np.arange(num_classes)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return torch.tensor(class_weights, dtype=torch.float, device=device)


def train_one_epoch(model, loader, criterion, optimizer, device, label_names: List[str]):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Train"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, pixel_values)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)

    report = classification_report(all_labels, all_preds, target_names=label_names, zero_division=0)
    return avg_loss, acc, report


@torch.no_grad()
def evaluate(model, loader, criterion, device, label_names: List[str]):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask, pixel_values)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)

    report = classification_report(all_labels, all_preds, target_names=label_names, zero_division=0)
    macro_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, report, macro_precision, macro_recall, macro_f1


def save_label_mapping(label_to_id: Dict[str, int], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: int(v) for k, v in label_to_id.items()}, f, ensure_ascii=False)



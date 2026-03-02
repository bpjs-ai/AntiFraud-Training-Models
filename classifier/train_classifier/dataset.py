import os
from typing import Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer


class HybridOCRDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: BertTokenizer,
        transform: Optional[transforms.Compose] = None,
        max_length: int = 512,
        image_path_col: str = "image_path",
        text_col: str = "text",
        label_id_col: str = "label_id",
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        self.image_path_col = image_path_col
        self.text_col = text_col
        self.label_id_col = label_id_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = row[self.image_path_col]
        text_value = str(row[self.text_col]) if row[self.text_col] is not None else ""
        label_value = int(row[self.label_id_col])

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        encoding = self.tokenizer(
            text_value,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "pixel_values": image,
            "labels": torch.tensor(label_value, dtype=torch.long),
        }


def build_transforms(image_size: Tuple[int, int]) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_tokenizer(model_name: str) -> BertTokenizer:
    return BertTokenizer.from_pretrained(model_name)


def load_and_prepare_dataframe(
    data_csv_path: str,
    image_root_dir: str,
    filename_col: str = "filename",
    label_col: str = "label",
    text_col: str = "text",
    image_path_col: str = "image_path",
    drop_empty_text: bool = True,
):
    df = pd.read_csv(data_csv_path)

    if drop_empty_text and text_col in df.columns:
        df[text_col] = df[text_col].fillna("")
        df = df[df[text_col].astype(str).str.strip() != ""]

    # Build image path from root/label/filename
    df[image_path_col] = df.apply(
        lambda row: os.path.join(image_root_dir, str(row[label_col]), str(row[filename_col])),
        axis=1,
    )

    # Encode labels
    unique_labels = df[label_col].astype(str).unique().tolist()
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    df["label_id"] = df[label_col].astype(str).map(label_to_id)

    return df, label_to_id


def split_dataframe(
    df: pd.DataFrame,
    stratify_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
):
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[stratify_col] if stratify_col in df.columns else None,
        random_state=random_state,
    )
    return train_df, val_df


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer: BertTokenizer,
    transform: transforms.Compose,
    max_length: int,
    batch_size: int,
    num_workers: int,
    shuffle_train: bool = True,
):
    train_dataset = HybridOCRDataset(
        train_df, tokenizer, transform=transform, max_length=max_length
    )
    val_dataset = HybridOCRDataset(
        val_df, tokenizer, transform=transform, max_length=max_length
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader



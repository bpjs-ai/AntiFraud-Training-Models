from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime


@dataclass
class TrainConfig:
    # Data
    data_csv_path: str = "dataset/data.csv"
    image_root_dir: str = "dataset"
    filename_col: str = "filename"
    label_col: str = "label"
    text_col: str = "text"
    image_size: Tuple[int, int] = (224, 224)
    max_length: int = 512
    drop_empty_text: bool = True
    val_size: float = 0.2
    seed: int = 42

    # Model
    bert_model_name: str = "indobenchmark/indobert-base-p2"

    # Optimization
    epochs: int = 50
    batch_size: int = 16
    num_workers: int = 8
    lr_bert: float = 2e-5
    lr_resnet: float = 1e-4
    lr_heads: float = 1e-4
    weight_decay: float = 0.01
    scheduler_patience: int = 5

    # Training tricks
    early_stop_patience: int = 15

    # Output
    output_dir: str = "outputs_training/full"
    best_state_dict_path: str = "state_dicts/best_state_dict.pt"
    label_map_path: str = "labelmaps/label2id.json"
    best_onnx_model_path: str = "models/full_model.onnx"
    result_path: str = "results/training_report.json"


@dataclass
class TransferConfig:
    """Default settings for the transfer-learning pipeline."""
    
    # Data
    data_csv_path: str = "dataset/dataset.csv"
    image_root_dir: str = "dataset"
    val_size: float = 0.2
    max_length: int = 512
    image_size: Tuple[int, int] = (224, 224)

    bert_model_name: str = "indobenchmark/indobert-base-p2"

    state_dict_path: Optional[str] = "outputs_training/full/state_dicts/state_dict.pt"
    
    previous_label_map_path: Optional[str] = "outputs_training/full/state_dicts/label2id.json"
    
    artifact_dir: str = "outputs_training/full/state_dicts"
    export_onnx: bool = True

    # Outputs
    output_dir: str = "outputs_training/transfer"
    label_map_s3_key: str = "outputs_training/label2id.json"
    state_dict_s3_key: str = "outputs_training/state_dict.pt"
    label_map_path: str = "labelmaps/label2id_transfer.json"
    best_state_dict_path: str = "state_dicts/state_dict_transfer.pt"
    best_onnx_model_path: str = "models/hybrid_model_transfer.onnx"
    result_path: str = "results/training_report.json"

    # Optimization
    epochs: int = 50
    batch_size: int = 16
    num_workers: int = 8
    lr_bert: float = 2e-5
    lr_resnet: float = 1e-4
    lr_heads: float = 1e-4

    # Training tricks
    early_stop_patience: int = 15
    freeze_backbone: bool = True

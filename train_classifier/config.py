from dataclasses import dataclass
from typing import Tuple
from datetime import datetime


@dataclass
class TrainConfig:
    # Data
    data_csv_path: str = "ocr_results_paddle.csv"
    image_root_dir: str = "Augmented"
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
    batch_size: int = 32
    num_workers: int = 8
    lr_bert: float = 2e-5
    lr_resnet: float = 1e-4
    lr_heads: float = 1e-4
    weight_decay: float = 0.01
    scheduler_patience: int = 5

    # Training tricks
    early_stop_patience: int = 15

    # Output
    output_dir: str = "outputs"
    best_state_dict_path: str = "outputs/classifier/state_dict{timestamp}.pt"
    label_map_path: str = "outputs/classifier/label2id.json"
    best_onnx_model_path: str = "outputs/classifier/full_model_{timestamp}.onnx"



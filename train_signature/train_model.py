"""Training utilities for YOLOv11.

This module defines a function for training a YOLOv11 model using the
Ultralytics high‑level API.  Keeping the training logic in its own file
promotes separation of concerns and simplifies testing individual components.
"""

from pathlib import Path

try:  # pragma: no cover
    from ultralytics import YOLO
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Ultralytics package is not installed.\n"
        "Install it with 'pip install \"ultralytics<=8.3.40\"' before running this script."
    ) from e


def train_model(
    data_yaml: Path,
    model_checkpoint: str,
    epochs: int,
    imgsz: int,
    patience: int,
    out_dir: Path,
) -> YOLO:
    """Train a YOLOv11 model using the Ultralytics high‑level API.

    Parameters
    ----------
    data_yaml : pathlib.Path
        Path to the ``data.yaml`` file describing the dataset splits and class names.
    model_checkpoint : str
        Name of the base model checkpoint to fine‑tune.
    epochs : int
        Number of training epochs.
    imgsz : int
        Input image size used for training and evaluation.
    patience : int
        Early stopping patience (number of epochs without improvement).
    out_dir : pathlib.Path
        Directory where training results will be stored.  A subdirectory named
        ``train`` will be created inside ``out_dir``.

    Returns
    -------
    ultralytics.YOLO
        The trained YOLO model instance.  The best weights can be accessed via
        ``model.best``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_checkpoint)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        patience=patience,
        project=str(out_dir),
        name="train",
    )
    return model
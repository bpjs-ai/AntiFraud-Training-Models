"""Inference utilities for YOLOv11 models.

This module provides a helper to perform inference on a directory of images
with a trained YOLOv11 model and save annotated predictions to an output
folder.  Keeping this functionality separate makes it easier to reuse and
test in isolation.
"""

from pathlib import Path
from ultralytics import YOLO  # type: ignore[import]


def run_inference(model: YOLO, images_dir: Path, out_dir: Path, conf: float = 0.25) -> None:
    """Run inference on a folder of images using a trained YOLO model.

    Parameters
    ----------
    model : ultralytics.YOLO
        A trained YOLO model.
    images_dir : pathlib.Path
        Directory containing the images on which to perform inference.
    out_dir : pathlib.Path
        Directory where prediction results will be saved.
    conf : float, optional
        Confidence threshold for filtering predictions.  Defaults to 0.25.
    """
    model.predict(
        source=str(images_dir),
        conf=conf,
        project=str(out_dir),
        name="predict",
        save=True,
        imgsz=model.model.args.imgsz,
    )
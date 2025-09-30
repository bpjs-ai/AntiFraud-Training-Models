"""Validation utilities for YOLOv11 models.

This module defines a function to run validation on a trained YOLOv11 model.
Validation results (metrics, plots, etc.) will be stored in a dedicated
subdirectory under the specified output folder.
"""

from pathlib import Path
from ultralytics import YOLO  # type: ignore[import]


def validate_model(model: YOLO, data_yaml: Path, out_dir: Path) -> None:
    """Run validation on the trained model and save results.

    Parameters
    ----------
    model : ultralytics.YOLO
        A trained YOLO model.
    data_yaml : pathlib.Path
        Path to the dataset's ``data.yaml`` file.
    out_dir : pathlib.Path
        Directory where validation results will be written.
    """
    model.val(
        data=str(data_yaml),
        project=str(out_dir),
        name="val",
    )
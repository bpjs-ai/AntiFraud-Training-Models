"""Model export utilities for YOLOv11.

This module defines a function to export a trained YOLOv11 model to a
portable format such as ONNX.  By isolating the export logic here, you
can reuse it across different pipelines or integrate it into deployment
scripts without pulling in the entire training workflow.
"""

import os
from pathlib import Path
from ultralytics import YOLO  # type: ignore[import]


def export_model(model: YOLO, out_dir: Path, format: str = "onnx") -> None:
    """Export the trained model weights to a portable format.

    Parameters
    ----------
    model : ultralytics.YOLO
        A trained YOLO model.
    out_dir : pathlib.Path
        Directory where the exported file will be saved.
    format : str, optional
        The target model format (e.g. 'onnx', 'torchscript', 'coreml').  Defaults to
        'onnx'.
    """
    cwd = Path.cwd()
    try:
        os.chdir(out_dir)
        model.export(format=format)
    finally:
        os.chdir(cwd)
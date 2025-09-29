"""Command‑line argument parser for the YOLOv11 training pipeline.

This module defines a single function, :func:`parse_args`, which returns an
``argparse.Namespace`` populated with sensible default values for all required
arguments.  Separating the argument parsing into its own file makes the
command‑line interface easier to maintain and to reuse in other scripts.
"""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments for the training pipeline.

    Returns
    -------
    argparse.Namespace
        A namespace containing parsed arguments with sensible defaults.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Fine‑tune YOLOv11 on a custom dataset downloaded from Roboflow."
        )
    )
    parser.add_argument(
        "--workspace",
        default="ahmad-zaki-firdaus",
        help="Name of the Roboflow workspace containing your project",
    )
    parser.add_argument(
        "--project",
        default="signature-detection-hlx8j-mem1d",
        help="Roboflow project name (slug). The dataset will be downloaded from this project.",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=3,
        help="Version number of the Roboflow project to download.",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="YOLOv11 base model checkpoint to fine‑tune (e.g. yolo11n.pt, yolo11s.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=70,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size for training and validation.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of epochs with no improvement before early stopping.",
    )
    parser.add_argument(
        "--out-dir",
        default="qr-signature-yolo",
        help=(
            "Output directory where training runs, validation results and predictions will be saved."
        ),
    )
    parser.add_argument(
        "--data-format",
        default="yolov11",
        help="Export format to request from Roboflow (default: yolov11)."
    )
    return parser.parse_args()
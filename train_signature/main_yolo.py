"""
Main orchestration script for training, validating, and exporting a YOLOv11 model.

This script ties together the individual components of the training pipeline
into a single executable.  It leverages helper modules to parse
command‑line arguments, download datasets from Roboflow, train and validate
the model, run inference on test images, and export the resulting weights
to a portable format (ONNX by default).

To run this script, ensure you have installed the required dependencies
(``ultralytics`` and ``roboflow``) and have set the ``ROBOFLOW_API_KEY``
environment variable.  See the module‑level docstring in
``train_yolo11_pipeline.py`` for detailed usage instructions.
"""

import sys
from pathlib import Path

from .parse_args import parse_args
from .download_dataset import download_dataset
from .train_model import train_model
from .validate_model import validate_model
from .run_inference import run_inference
from .export_model import export_model


def main() -> None:
    args = parse_args()
    out_path = Path(args.out_dir).expanduser().resolve()

    # Summarise configuration
    print("\n=== YOLOv11 Training Pipeline ===")
    print(f"Workspace: {args.workspace}")
    print(f"Project:   {args.project}")
    print(f"Version:   {args.version}")
    print(f"Model:     {args.model}")
    print(f"Epochs:    {args.epochs}")
    print(f"Image size:{args.imgsz}")
    print(f"Patience:  {args.patience}")
    print(f"Output dir:{out_path}\n")

    # Ensure datasets directory exists
    datasets_dir = Path.cwd() / "datasets"
    datasets_dir.mkdir(exist_ok=True)

    # Download dataset
    print("Downloading dataset from Roboflow…")
    dataset_dir = download_dataset(
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        export_format=args.data_format,
    )
    data_yaml = dataset_dir / "data.yaml"

    # Train model
    print("\nStarting training…")
    model = train_model(
        data_yaml=data_yaml,
        model_checkpoint=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        patience=args.patience,
        out_dir=out_path,
    )

    # Validate model
    print("\nValidating model…")
    validate_model(model, data_yaml, out_path)

    # Inference on test images
    test_images = dataset_dir / "test" / "images"
    if test_images.exists() and any(test_images.iterdir()):
        print("\nRunning inference on test images…")
        run_inference(model, test_images, out_path, conf=0.25)
    else:
        print("\nNo test images found for inference.")

    # Export model
    print("\nExporting model weights to ONNX format…")
    export_model(model, out_path, format="onnx")

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
#!/usr/bin/env python3
"""Classifier or transfer-learning training pipeline orchestrator."""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.s3_utils import S3Handler
from train_classifier.main_classifier import main as classifier_full_main
from train_classifier.main_classifier_TL import main as classifier_transfer_main


def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure basic logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("pipeline")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Classifier Training Pipeline Orchestrator",
    )
    parser.add_argument(
        "--classifier_mode",
        choices=["full", "transfer"],
        default="full",
        help="Choose between full training and transfer learning (default: full).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset",
        help="Local dataset directory (default: dataset)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Base output directory for all pipeline results (default: outputs)",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip classifier dataset download from S3.",
    )
    parser.add_argument("--data_csv_path", type=str, default=None)
    parser.add_argument("--image_root_dir", type=str, default=None)
    parser.add_argument("--val_size", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lr_bert", type=float, default=None)
    parser.add_argument("--lr_resnet", type=float, default=None)
    parser.add_argument("--lr_heads", type=float, default=None)
    parser.add_argument("--bert_model_name", type=str, default=None)

    # Transfer learning specific overrides (optional)
    parser.add_argument("--state_dict_path", type=str, default=None)
    parser.add_argument("--state_dict_s3_key", type=str, default=None)
    parser.add_argument("--previous_label_map_path", type=str, default=None)
    parser.add_argument("--label_map_s3_key", type=str, default=None)
    parser.add_argument("--artifact_dir", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--image_size", type=str, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--export_onnx", action="store_true")

    return parser.parse_args()


def download_classifier_dataset(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Download the classifier dataset from S3 when configured."""
    if args.classifier_mode != "full":
        logger.info("Skipping dataset download because classifier_mode='%s' does not require it.", args.classifier_mode)
        return

    if args.skip_download:
        logger.info("Skipping classifier dataset download because --skip_download is set.")
        return

    prefix = os.getenv("S3_CLASSIFIER_PREFIX")
    if not prefix:
        logger.info("Environment variable S3_CLASSIFIER_PREFIX is not set; skipping dataset download.")
        return

    target_dir = Path(args.dataset_dir) / "classifier" / "namadocs"
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        handler = S3Handler()
        downloaded, skipped = handler.download_folder_recursive(
            s3_prefix=prefix,
            local_dir=str(target_dir),
            skip_existing=True,
        )
        if downloaded:
            logger.info("Downloaded %d classifier files from S3.", len(downloaded))
        if skipped:
            logger.info("Skipped %d existing classifier files.", len(skipped))
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to download classifier dataset from S3: %s", exc)
        logger.info("Continuing with local dataset only.")


def build_common_classifier_args(args: argparse.Namespace, run_output_dir: Path) -> List[str]:
    """Arguments shared by full and transfer training."""
    cli_args: List[str] = ["--output_dir", str(run_output_dir)]

    common_mapping = {
        "data_csv_path": "--data_csv_path",
        "image_root_dir": "--image_root_dir",
        "val_size": "--val_size",
        "epochs": "--epochs",
        "batch_size": "--batch_size",
        "num_workers": "--num_workers",
        "lr_bert": "--lr_bert",
        "lr_resnet": "--lr_resnet",
        "lr_heads": "--lr_heads",
        "bert_model_name": "--bert_model_name",
    }

    for attr, flag in common_mapping.items():
        value = getattr(args, attr, None)
        if value is not None:
            cli_args.extend([flag, str(value)])

    return cli_args


def build_transfer_args(args: argparse.Namespace) -> List[str]:
    """Arguments specific to the transfer learning entry point."""
    transfer_mapping = {
        "state_dict_path": "--state_dict_path",
        "state_dict_s3_key": "--state_dict_s3_key",
        "previous_label_map_path": "--previous_label_map_path",
        "label_map_s3_key": "--label_map_s3_key",
        "artifact_dir": "--artifact_dir",
        "max_length": "--max_length",
        "image_size": "--image_size",
        "early_stop_patience": "--early_stop_patience",
    }

    cli_args: List[str] = []
    for attr, flag in transfer_mapping.items():
        value = getattr(args, attr, None)
        if value is not None:
            cli_args.extend([flag, str(value)])

    if args.freeze_backbone:
        cli_args.append("--freeze_backbone")
    if args.export_onnx:
        cli_args.append("--export_onnx")

    return cli_args


def build_classifier_cli_args(args: argparse.Namespace, run_output_dir: Path) -> List[str]:
    """Build CLI arguments for the selected classifier training entry point."""
    cli_args = build_common_classifier_args(args, run_output_dir)
    if args.classifier_mode == "transfer":
        cli_args.extend(build_transfer_args(args))
    return cli_args


def run_classifier_pipeline(args: argparse.Namespace, logger: logging.Logger) -> bool:
    """Execute the classifier or transfer-learning training pipeline."""
    logger.info("Starting classifier pipeline in '%s' mode.", args.classifier_mode)

    mode_dir = "classifier" if args.classifier_mode == "full" else "classifier_transfer"
    run_output_dir = Path(args.output_dir) / mode_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir.mkdir(parents=True, exist_ok=True)

    classifier_args = build_classifier_cli_args(args, run_output_dir)
    logger.info("Invoking %s with args: %s", args.classifier_mode, classifier_args)

    entrypoint = classifier_full_main if args.classifier_mode == "full" else classifier_transfer_main

    original_argv = sys.argv[:]
    try:
        sys.argv = ["main_classifier.py"] + classifier_args
        entrypoint()
    except Exception as exc:  # noqa: BLE001
        logger.exception("%s training failed: %s", args.classifier_mode.capitalize(), exc)
        return False
    finally:
        sys.argv = original_argv

    logger.info("%s training completed successfully.", args.classifier_mode.capitalize())
    return True


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("=" * 60)
    logger.info("Classifier Training Pipeline Started (mode=%s)", args.classifier_mode)
    logger.info("Dataset directory: %s", Path(args.dataset_dir).resolve())
    logger.info("Output directory: %s", output_dir.resolve())

    download_classifier_dataset(args, logger)

    success = run_classifier_pipeline(args, logger)

    logger.info("=" * 60)
    if success:
        logger.info("Classifier pipeline finished successfully.")
    else:
        logger.error("Classifier pipeline failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()

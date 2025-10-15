from utils.s3_utils import S3Handler
from train_classifier.config import TrainConfig, TransferConfig
from pathlib import Path
import shutil
import logging
import subprocess
import argparse  # ✅ tambahkan ini

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_classifier_dataset(mode) -> None:
    """Download the classifier dataset from S3 when configured."""
    prefix = "dataset/classifier"
    if mode == "full":
        target_dir = Path(TrainConfig.dataset_dir)
    elif mode == "transfer":
        target_dir = Path(TransferConfig.dataset_dir)
    else:
        logger.warning("Unknown mode '%s', skipping dataset download.")
        return

    # --- 🔹 HAPUS SEMUA YANG ADA DI dataset/* ---
    dataset_root = Path("dataset")
    if dataset_root.exists():
        for item in dataset_root.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                logger.info("Removed %s", item)
            except Exception as exc:
                logger.warning("Failed to remove %s: %s", item, exc)

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
    except Exception as exc:
        logger.error("Failed to download classifier dataset from S3: %s", exc)
        logger.info("Continuing with local dataset only.")


def ocr_training(mode):
    if mode == "full":
        subprocess.run([
            "python",
            "train_classifier/ocr_paddle.py",
            "--input_folder", TrainConfig.image_root_dir,
            "--output_csv", TrainConfig.data_csv_path,
            "--num_workers", "50",
        ])
    elif mode == "transfer":
        subprocess.run([
            "python",
            "train_classifier/ocr_paddle.py",
            "--input_folder", TransferConfig.image_root_dir,
            "--output_csv", TransferConfig.data_csv_path,
            "--num_workers", "50",
        ])
    else:
        logger.info("Skipping OCR training because mode='%s' does not require it.", mode)


def main(mode):
    download_classifier_dataset(mode)  # ✅ tambahkan argumen mode ke sini
    ocr_training(mode)


if __name__ == "__main__":
    # ✅ Tambahkan argument parser
    parser = argparse.ArgumentParser(description="Run OCR training pipeline.")
    parser.add_argument(
        "mode",
        choices=["full", "transfer"],
        help="Mode training yang akan dijalankan ('full' atau 'transfer')."
    )
    args = parser.parse_args()

    main(args.mode)

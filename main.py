#!/usr/bin/env python3
"""
BPJS Training Pipeline Orchestrator

This script orchestrates the training pipelines for both classifier and signature detection.
It handles dataset download from S3 to local paths and pipeline execution.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.s3_utils import S3Handler
from train_classifier.main_classifier import main as classifier_main
from train_signature.main_yolo import main as signature_main
from train_signature.download_dataset import download_dataset



def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_file = output_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="BPJS Training Pipeline Orchestrator"
    )
    
    # Pipeline selection
    parser.add_argument(
        "--pipeline",
        choices=["classifier", "signature", "both"],
        default="both",
        help="Which pipeline to run (default: both)"
    )
    
    # Dataset directory configuration
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset",
        help="Local dataset directory"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Base output directory for all pipeline results"
    )
    
    # Skip dataset download
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip dataset download and use existing data"
    )
    
    parser.add_argument("--val_size", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lr_bert", type=float, default=None)
    parser.add_argument("--lr_resnet", type=float, default=None)
    parser.add_argument("--lr_heads", type=float, default=None)
    parser.add_argument("--bert_model_name", type=str, default=None)

    
    # Signature specific arguments  
    parser.add_argument(
        "--signature_workspace",
        type=str,
        default="signature-detection",
        help="Roboflow workspace for signature training (fallback if S3 fails)"
    )
    parser.add_argument(
        "--signature_project", 
        type=str,
        default="signature-detection-hlx8j-mem1d",
        help="Roboflow project for signature training (fallback if S3 fails)"
    )
    parser.add_argument(
        "--signature_version",
        type=int,
        default=3,
        help="Roboflow version for signature training (fallback if S3 fails)"
    )
    parser.add_argument(
        "--signature_epochs",
        type=int,
        default=70,
        help="Number of epochs for signature training"
    )
    
    return parser.parse_args()


def download_datasets(args: argparse.Namespace) -> bool:
    """Download datasets from S3 with smart skip functionality using environment variables."""
    import os
    
    try:
        s3_handler = S3Handler()
        
        # Get S3 prefixes from environment variables
        s3_classifier_prefix = os.getenv("S3_CLASSIFIER_PREFIX")
        s3_signature_prefix = os.getenv("S3_SIGNATURE_PREFIX")
        
        # Download classifier dataset
        if args.pipeline in ["classifier", "both"] and s3_classifier_prefix:
            logging.info("Checking for classifier dataset updates from S3...")
            classifier_dir = Path(args.dataset_dir) / "classifier" / "namadocs"
            classifier_dir.mkdir(parents=True, exist_ok=True)
            
            downloaded_files, skipped_files = s3_handler.download_folder_recursive(
                s3_prefix=s3_classifier_prefix,
                local_dir=str(classifier_dir),
                skip_existing=True
            )
            
            if downloaded_files:
                logging.info(f"Downloaded {len(downloaded_files)} new classifier files")
            if skipped_files:
                logging.info(f"Skipped {len(skipped_files)} existing classifier files")
        
        # Download signature dataset
        if args.pipeline in ["signature", "both"] and s3_signature_prefix:
            logging.info("Checking for signature dataset updates from S3...")
            signature_dir = Path(args.dataset_dir) / "signature"
            signature_dir.mkdir(parents=True, exist_ok=True)
            
            downloaded_files, skipped_files = s3_handler.download_folder_recursive(
                s3_prefix=s3_signature_prefix,
                local_dir=str(signature_dir),
                skip_existing=True
            )
            
            if downloaded_files:
                logging.info(f"Downloaded {len(downloaded_files)} new signature files")
            if skipped_files:
                logging.info(f"Skipped {len(skipped_files)} existing signature files")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to download datasets from S3: {str(e)}")
        logging.info("Continuing with local dataset...")
        return True  # Don't fail the pipeline, just use local data


def training_classifier(args: argparse.Namespace, logger: logging.Logger) -> bool:
    """Execute classifier training pipeline."""
    logger.info("🚀 Starting Classifier Training Pipeline")
    print("\n=== Downloading Classifier Dataset ===")
    # try:
    #     classifier_path = download_dataset(
    #         dataset_type="classifier",
    #         s3_prefix="dataset/classifier/",  # Replace with actual S3 prefix
    #         local_base_dir=""  # Current directory
    #     )
    #     print(f"Classifier dataset downloaded to: {classifier_path}")
    # except Exception as e:
    #     print(f"Error downloading classifier dataset: {e}")
    try:
        classifier_args = []

        # output_dir khusus tiap run
        classifier_output_dir = Path(args.output_dir) / "classifier" / datetime.now().strftime('%Y%m%d_%H%M%S')
        classifier_args.extend(["--output_dir", str(classifier_output_dir)])

        mapping = {
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
            # "config": "--config",  # hapus kalau tidak ada di parse_args
        }

        for attr, flag in mapping.items():
            val = getattr(args, attr, None)
            if val is not None:
                classifier_args.extend([flag, str(val)])

        original_argv = sys.argv[:]
        try:
            sys.argv = ["main_classifier.py"] + classifier_args
            logger.info(f"🔧 Running classifier with args: {classifier_args}")
            classifier_main()
        finally:
            sys.argv = original_argv

        logger.info("✅ Classifier training completed successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Classifier training failed: {str(e)}")
        return False

def training_signature(args: argparse.Namespace, logger: logging.Logger) -> bool:
    """Execute signature training pipeline."""
    logger.info("🚀 Starting Signature Training Pipeline")

    print("\n=== Downloading Signature Dataset ===")
    # jangan dihapus
    # try:
    #     signature_path = download_dataset(
    #         dataset_type="signature",
    #         s3_prefix="dataset/signature/",  # Replace with actual S3 prefix
    #         local_base_dir="."  # Current directory
    #     )
    #     print(f"Signature dataset downloaded to: {signature_path}")
    # except Exception as e:
    #     print(f"Error downloading signature dataset: {e}")
    
    try:
        
        # Prepare signature arguments
        signature_args = []
        
        # Add output directory
        signature_output_dir = Path(args.output_dir) / "signature" / datetime.now().strftime('%Y%m%d_%H%M%S')
        signature_args.extend(["--out-dir", str(signature_output_dir)])
        
        # Add signature specific arguments
        signature_args.extend([
            "--workspace", args.signature_workspace,
            "--project", args.signature_project,
            "--version", str(args.signature_version),
            "--epochs", str(args.signature_epochs)
        ])
        
        # Override sys.argv for signature main
        original_argv = sys.argv.copy()
        sys.argv = ["main_yolo.py"] + signature_args
        
        logger.info(f"🔧 Running signature training with args: {signature_args}")
        
        # Run signature training
        signature_main()
        
        # Restore original argv
        sys.argv = original_argv
        
        logger.info("✅ Signature training completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Signature training failed: {str(e)}")
        return False


def main():
    """Main orchestration function."""
    args = parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    logger.info("=" * 60)
    logger.info("🏥 BPJS Training Pipeline Started")
    logger.info("=" * 60)
    logger.info(f"Pipeline mode: {args.pipeline}")
    logger.info(f"Dataset directory: {args.dataset_dir}")
    logger.info(f"Output directory: {output_dir.absolute()}")
    
    # Always download/update datasets from S3 (with skip existing)
    logger.info("🔄 Updating datasets from S3...")
    download_datasets(args)  # Continue even if S3 fails, use local data
    
    success_count = 0
    total_pipelines = 0
    
    try:
        if args.pipeline in ["classifier", "both"]:
            total_pipelines += 1
            if training_classifier(args, logger):
                success_count += 1
        
        if args.pipeline in ["signature", "both"]:
            total_pipelines += 1
            if training_signature(args, logger):
                success_count += 1
        
        # Summary
        logger.info("=" * 60)
        if success_count == total_pipelines:
            logger.info("🎉 All pipelines completed successfully!")
            logger.info(f"✅ {success_count}/{total_pipelines} pipelines succeeded")
        else:
            logger.warning(f"⚠️  {success_count}/{total_pipelines} pipelines succeeded")
            if success_count == 0:
                logger.error("❌ All pipelines failed")
                sys.exit(1)
        
        logger.info(f"📁 Results saved to: {output_dir.absolute()}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("🛑 Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
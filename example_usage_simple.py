#!/usr/bin/env python3
"""
Example usage of the BPJS Training Pipeline

This example shows how to use the pipeline with S3 dataset updates.
S3 configuration is handled via environment variables.
"""

import subprocess
import sys
from pathlib import Path

def run_classifier_only():
    """Run classifier training with S3 update via environment variables."""
    print("=== Running Classifier Training ===")
    
    cmd = [
        sys.executable, "main.py",
        "--pipeline", "classifier",
        "--dataset-dir", "dataset",
        "--output-dir", "outputs"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("Note: S3 configuration is handled via environment variables")
    # subprocess.run(cmd)  # Uncomment when ready to run

def run_signature_only():
    """Run signature training with S3 update via environment variables."""
    print("=== Running Signature Training ===")
    
    cmd = [
        sys.executable, "main.py",
        "--pipeline", "signature",
        "--dataset-dir", "dataset",
        "--output-dir", "outputs",
        "--model", "yolo11n.pt",
        "--epochs", "50"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("Note: S3 configuration is handled via environment variables")
    # subprocess.run(cmd)  # Uncomment when ready to run

def run_both_pipelines():
    """Run both pipelines with S3 updates via environment variables."""
    print("=== Running Both Pipelines with S3 Updates ===")
    
    cmd = [
        sys.executable, "main.py",
        "--pipeline", "both",
        "--dataset-dir", "dataset",
        "--output-dir", "outputs",
        "--model", "yolo11n.pt",
        "--epochs", "50"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("Note: S3 configuration is handled via environment variables")
    # subprocess.run(cmd)  # Uncomment when ready to run

def run_with_custom_config():
    """Run classifier with custom config and S3 update via environment variables."""
    print("=== Running Classifier with Custom Config ===")
    
    cmd = [
        sys.executable, "main.py",
        "--pipeline", "classifier",
        "--dataset-dir", "dataset",
        "--output-dir", "outputs",
        "--classifier-config", "custom_config.py"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("Note: S3 configuration is handled via environment variables")
    # subprocess.run(cmd)  # Uncomment when ready to run

if __name__ == "__main__":
    print("BPJS Training Pipeline - S3 Environment Variable Examples")
    print("=" * 65)
    print("🔧 S3 configuration is handled via environment variables")
    print("🔄 Pipeline automatically checks S3 for dataset updates")
    print("⏭️  Existing files are skipped, only new/changed files downloaded")
    print("📝 Set S3_BUCKET_NAME, S3_CLASSIFIER_PREFIX, S3_SIGNATURE_PREFIX in environment")
    print()
    
    # Check if dataset directory exists
    dataset_dir = Path("dataset")
    if not dataset_dir.exists():
        print("Warning: dataset directory not found. Creating structure...")
        (dataset_dir / "classifier" / "namadocs").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "signature" / "train" / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "signature" / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "signature" / "test").mkdir(parents=True, exist_ok=True)
        print("Dataset structure created.")
    
    # Show examples (uncomment the one you want to run)
    
    # Example 1: Run classifier with S3 via environment variables
    # run_classifier_only()
    
    # Example 2: Run signature with S3 via environment variables
    # run_signature_only()
    
    # Example 3: Run both pipelines with S3 via environment variables
    # run_both_pipelines()
    
    # Example 4: Run with custom config and S3 via environment variables
    # run_with_custom_config()
    
    print("\nTo run any example, uncomment the corresponding function call in this script.")
    print("\nDirect command examples:")
    print("# Both pipelines (default)")
    print("python main.py")
    print()
    print("# Classifier only")
    print("python main.py --pipeline classifier")
    print()
    print("# Signature only")
    print("python main.py --pipeline signature")
    print()
    print("⚠️  IMPORTANT: Set environment variables for S3 configuration:")
    print("   S3_BUCKET_NAME=your-bucket-name")
    print("   S3_CLASSIFIER_PREFIX=datasets/classifier/")
    print("   S3_SIGNATURE_PREFIX=datasets/signature/")
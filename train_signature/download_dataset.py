"""Dataset download utilities for the training pipeline.

This module provides helper functions to download datasets from S3 storage.
It supports both classifier and signature dataset structures and organizes
them according to each pipeline's requirements. The download logic uses
recursive S3 download to ensure complete dataset contents are retrieved.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List

# Add utils to path for S3Handler import
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from s3_utils import S3Handler


def download_classifier_dataset(s3_prefix: str, local_base_dir: Optional[str] = None) -> Path:
    """Download classifier dataset from S3.
    
    Downloads data to: dataset/classifier/namadocs/
    
    Parameters
    ----------
    s3_prefix : str
        The S3 prefix (folder path) containing the classifier dataset.
    local_base_dir : str, optional
        Base directory for downloads. Defaults to current working directory.
        
    Returns
    -------
    pathlib.Path
        Path to the classifier dataset directory.
        
    Raises
    ------
    ValueError
        If S3 configuration is missing or download fails.
    """
    if local_base_dir is None:
        local_base_dir = os.getcwd()
    
    # Ensure the directory structure exists
    classifier_dir = Path(local_base_dir) / "dataset" / "classifier"
    classifier_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize S3 handler
    s3_handler = S3Handler()
    
    print(f"📥 Downloading classifier dataset from S3 prefix: {s3_prefix}")
    print(f"📁 Target directory: {classifier_dir}")
    
    # Download recursively from S3
    downloaded_files, skipped_files = s3_handler.download_folder_recursive(
        s3_prefix=s3_prefix,
        local_dir=str(classifier_dir),
        skip_existing=False
    )
    
    print(f"✅ Downloaded {len(downloaded_files)} files for classifier dataset")
    if skipped_files:
        print(f"⏭️  Skipped {len(skipped_files)} existing files")
    
    return classifier_dir


def download_signature_dataset(s3_prefix: str, local_base_dir: Optional[str] = None) -> Path:
    """Download signature dataset from S3.
    
    Downloads data to: dataset/signature/train/images/, dataset/signature/train/labels/, dataset/signature/test/
    
    Parameters
    ----------
    s3_prefix : str
        The S3 prefix (folder path) containing the signature dataset.
    local_base_dir : str, optional
        Base directory for downloads. Defaults to current working directory.
        
    Returns
    -------
    pathlib.Path
        Path to the signature dataset directory.
        
    Raises
    ------
    ValueError
        If S3 configuration is missing or download fails.
    """
    if local_base_dir is None:
        local_base_dir = os.getcwd()
    
    # Ensure the directory structure exists
    signature_dir = Path(local_base_dir) / "dataset" / "signature"
    train_images_dir = signature_dir / "train" / "images"
    train_labels_dir = signature_dir / "train" / "labels"
    test_dir = signature_dir / "test"
    
    for dir_path in [train_images_dir, train_labels_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize S3 handler
    s3_handler = S3Handler()
    
    print(f"📥 Downloading signature dataset from S3 prefix: {s3_prefix}")
    print(f"📁 Target directory: {signature_dir}")
    
    # Download recursively from S3
    downloaded_files, skipped_files = s3_handler.download_folder_recursive(
        s3_prefix=s3_prefix,
        local_dir=str(signature_dir),
        skip_existing=False
    )
    
    print(f"✅ Downloaded {len(downloaded_files)} files for signature dataset")
    if skipped_files:
        print(f"⏭️  Skipped {len(skipped_files)} existing files")
    
    return signature_dir


def download_dataset(dataset_type: str, s3_prefix: str, local_base_dir: Optional[str] = None) -> Path:
    """Download dataset from S3 based on type.
    
    Parameters
    ----------
    dataset_type : str
        Type of dataset to download. Either 'classifier' or 'signature'.
    s3_prefix : str
        The S3 prefix (folder path) containing the dataset.
    local_base_dir : str, optional
        Base directory for downloads. Defaults to current working directory.
        
    Returns
    -------
    pathlib.Path
        Path to the downloaded dataset directory.
        
    Raises
    ------
    ValueError
        If dataset_type is not supported or S3 configuration is missing.
    """
    if dataset_type.lower() == 'classifier':
        return download_classifier_dataset(s3_prefix, local_base_dir)
    elif dataset_type.lower() == 'signature':
        return download_signature_dataset(s3_prefix, local_base_dir)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. Use 'classifier' or 'signature'.")


def list_s3_datasets(s3_prefix: str = "") -> List[str]:
    """List available datasets in S3.
    
    Parameters
    ----------
    s3_prefix : str, optional
        S3 prefix to search within. Defaults to root.
        
    Returns
    -------
    List[str]
        List of available dataset prefixes in S3.
    """
    s3_handler = S3Handler()
    
    print(f"🔍 Listing datasets in S3 prefix: {s3_prefix or 'root'}")
    
    objects = s3_handler.list_objects(s3_prefix)
    
    # Extract unique folder prefixes
    folders = set()
    for obj in objects:
        if '/' in obj:
            folder = '/'.join(obj.split('/')[:-1]) + '/'
            folders.add(folder)
    
    folder_list = sorted(list(folders))
    
    print(f"📂 Found {len(folder_list)} dataset folders:")
    for folder in folder_list:
        print(f"  - {folder}")
    
    return folder_list
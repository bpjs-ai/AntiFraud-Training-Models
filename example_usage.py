"""Example usage of the dataset download utilities.

This script demonstrates how to use the updated download_dataset.py
to download datasets from S3 with the proper directory structure.
"""

from train_signature.download_dataset import download_dataset, list_s3_datasets

def main():
    """Example usage of dataset download functions."""
    
    # List available datasets in S3
    print("=== Available Datasets ===")
    datasets = list_s3_datasets()
    
    # Example: Download classifier dataset
    print("\n=== Downloading Classifier Dataset ===")
    try:
        classifier_path = download_dataset(
            dataset_type="classifier",
            s3_prefix="datasets/classifier/",  # Replace with actual S3 prefix
            local_base_dir="."  # Current directory
        )
        print(f"Classifier dataset downloaded to: {classifier_path}")
    except Exception as e:
        print(f"Error downloading classifier dataset: {e}")
    
    # Example: Download signature dataset
    print("\n=== Downloading Signature Dataset ===")
    try:
        signature_path = download_dataset(
            dataset_type="signature",
            s3_prefix="datasets/signature/",  # Replace with actual S3 prefix
            local_base_dir="."  # Current directory
        )
        print(f"Signature dataset downloaded to: {signature_path}")
    except Exception as e:
        print(f"Error downloading signature dataset: {e}")

if __name__ == "__main__":
    main()
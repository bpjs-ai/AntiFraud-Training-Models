
import boto3
import os
from botocore.exceptions import ClientError
from typing import List
import logging
from tqdm import tqdm
from dotenv import load_dotenv


load_dotenv()

class S3Handler:
    def __init__(self, region_name='ap-southeast-1'):
        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        self.s3_client = boto3.client(
            's3',
            endpoint_url=f"http://{os.getenv('S3_ENDPOINT')}",
            aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("S3_SECRET_KEY"), 
            region_name=region_name
        )

    def list_objects(self, prefix):
        """List all objects in a given prefix (folder)"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
        except Exception as e:
            logging.error(f"Error listing objects with prefix {prefix}: {str(e)}")
            return []

    def file_exists(self, s3_key):
        """Check if a file exists in S3"""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                # Re-raise other errors
                raise
        except Exception as e:
            logging.error(f"Error checking if file exists {s3_key}: {str(e)}")
            return False

        
    def upload_folder(self, dir_path, s3_key="Data", file_extension=""):
        # Get list of files to upload
        files_to_upload = []
        
        for filename in os.listdir(dir_path):
            if not file_extension or filename.lower().endswith(file_extension.lower()):
                local_path = os.path.join(dir_path, filename)  
                s3_file_key = f"{s3_key}/{filename}"  
                
                files_to_upload.append((filename, local_path, s3_file_key))
        
        if not files_to_upload:
            print("No new files to upload")
            return
        
        # Upload files with progress bar
        with tqdm(total=len(files_to_upload), desc="Uploading files", unit="file") as pbar:
            for filename, local_path, s3_file_key in files_to_upload:
                try:
                    pbar.set_description(f"Uploading {filename}")
                    self.s3_client.upload_file(local_path, self.bucket_name, s3_file_key)
                    pbar.set_postfix({"status": "✅"})
                except ClientError as e:
                    pbar.set_postfix({"status": "❌"})
                    print(f"\n❌ Failed to upload {filename}: {e}")
                except Exception as e:
                    pbar.set_postfix({"status": "⚠️"})
                    print(f"\n⚠️ Other error while uploading {filename}: {e}")
                finally:
                    pbar.update(1)

    def upload_folder_recursive(self, dir_path, s3_key="Data", file_extension=""):
        # Get list of files to upload recursively
        files_to_upload = []
        skipped_files = []
        
        # Use os.walk to traverse all subdirectories recursively
        for root, dirs, files in os.walk(dir_path):
            for filename in files:
                if not file_extension or filename.lower().endswith(file_extension.lower()):
                    local_path = os.path.join(root, filename)
                    
                    # Calculate relative path from dir_path to preserve folder structure
                    relative_path = os.path.relpath(local_path, dir_path)
                    s3_file_key = f"{s3_key}/{relative_path}".replace('\\', '/')  # Normalize path separators

                    # Skip upload only for existing PDF files, overwrite others
                    if filename.lower().endswith('.pdf'):
                        try:
                            # Check if the PDF file already exists in S3
                            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_file_key)
                            # If exists, skip upload
                            skipped_files.append(relative_path)
                            continue
                        except ClientError as e:
                            # If not found, proceed to upload
                            if e.response['Error']['Code'] != '404':
                                # If other error, raise
                                raise

                    files_to_upload.append((relative_path, local_path, s3_file_key))
        
        if skipped_files:
            print(f"⏭️  Skipped {len(skipped_files)} existing PDF files: {', '.join(skipped_files)}")
        
        if not files_to_upload:
            print("No new files to upload")
            return
        
        # Upload files with progress bar
        with tqdm(total=len(files_to_upload), desc="Uploading files", unit="file") as pbar:
            for relative_path, local_path, s3_file_key in files_to_upload:
                try:
                    pbar.set_description(f"Uploading {relative_path}")
                    self.s3_client.upload_file(local_path, self.bucket_name, s3_file_key)
                    pbar.set_postfix({"status": "✅"})
                except ClientError as e:
                    pbar.set_postfix({"status": "❌"})
                    print(f"\n❌ Failed to upload {relative_path}: {e}")
                except Exception as e:
                    pbar.set_postfix({"status": "⚠️"})
                    print(f"\n⚠️ Other error while uploading {relative_path}: {e}")
                finally:
                    pbar.update(1)


    def upload_file(self, file_path, s3_key):
        """Upload a file to S3 bucket"""
        try:
            filename = os.path.basename(file_path)
            with tqdm(total=1, desc=f"Uploading {filename}", unit="file") as pbar:
                self.s3_client.upload_file(file_path, self.bucket_name, s3_key)
                pbar.set_postfix({"status": f"{s3_key}"})
                pbar.update(1)
            return True
        except ClientError as e:
            print(f"Error uploading file: {e}")
            return False
        

    def upload_fileobj(self, file_content, s3_key):
        """Upload file content (bytes) to S3 bucket"""
        try:
            import io
            filename = os.path.basename(s3_key)
            with tqdm(total=1, desc=f"Uploading {filename}", unit="file") as pbar:
                # Create BytesIO object from file content
                file_obj = io.BytesIO(file_content)
                self.s3_client.upload_fileobj(file_obj, self.bucket_name, s3_key)
                pbar.set_postfix({"status": f"{s3_key}"})
                pbar.update(1)
            return True
        except ClientError as e:
            print(f"Error uploading file: {e}")
            return False
        except Exception as e:
            print(f"Error uploading file: {e}")
            return False


    def download_file(self, s3_key, local_path):
        """Download a file from S3"""
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            return True
        except Exception as e:
            return False


    def download_folder(self, s3_prefix: str, local_dir: str) -> List[str]:
        """
        Download all objects under the given S3 prefix into local_dir, preserving folder structure.
        
        :param s3_prefix: the "folder" in S3 (e.g. 'path/to/folder/')
        :param local_dir: local directory to write files into
        :return: list of local file paths downloaded
        """
        paginator = self.s3_client.get_paginator('list_objects_v2')
        downloaded_files: List[str] = []
        skipped_files: List[str] = []

        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=s3_prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('/'):
                    continue

                relative_path = key[len(s3_prefix):].lstrip('/')
                local_path = os.path.join(local_dir, relative_path)

                # Tambahkan skip file jika file pdf sudah ada
                if local_path.lower().endswith('.pdf') and os.path.isfile(local_path):
                    skipped_files.append(relative_path)
                    continue

                if self.download_file(key, local_path):
                    downloaded_files.append(local_path)

        if skipped_files:
            print(f"⏭️  Skipped {len(skipped_files)} existing PDF files: {', '.join(skipped_files[:5])}{'...' if len(skipped_files) > 5 else ''}")

        return downloaded_files

    def download_folder_recursive(self, s3_prefix: str, local_dir: str, file_extension: str = "", skip_existing: bool = True):
        """
        Download all objects under the given S3 prefix recursively into local_dir, 
        preserving folder structure.
        
        :param s3_prefix: the "folder" in S3 (e.g. 'path/to/folder/')
        :param local_dir: local directory to write files into
        :param file_extension: optional file extension filter (e.g. '.pdf', '.txt')
        :param skip_existing: whether to skip files that already exist locally
        :return: tuple of (downloaded_files, skipped_files)
        """
        # Get list of files to download recursively
        files_to_download = []
        skipped_files = []
        
        # Use paginator to handle large numbers of objects
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=s3_prefix):
            for obj in page.get('Contents', []):
                s3_key = obj['Key']
                
                # Skip directories (keys ending with '/')
                if s3_key.endswith('/'):
                    continue

                if s3_key.endswith('.md') or s3_key.endswith('.xlsx'):
                    continue
                
                # Apply file extension filter if specified
                if file_extension and not s3_key.lower().endswith(file_extension.lower()):
                    continue
                
                # Calculate relative path from s3_prefix to preserve folder structure
                relative_path = s3_key[len(s3_prefix):].lstrip('/')
                local_path = os.path.join(local_dir, relative_path)
                
                # Check if file already exists locally (skip based on file existence, not folder)
                if skip_existing and os.path.isfile(local_path):
                    skipped_files.append(relative_path)
                    continue
                
                files_to_download.append((relative_path, s3_key, local_path))
        
        if skipped_files:
            print(f"⏭️  Skipped {len(skipped_files)} existing files: {', '.join(skipped_files[:5])}{'...' if len(skipped_files) > 5 else ''}")
        
        if not files_to_download:
            print("No new files to download")
            return [], skipped_files
        
        # Download files with progress bar
        downloaded_files = []
        with tqdm(total=len(files_to_download), desc="Downloading files", unit="file") as pbar:
            for relative_path, s3_key, local_path in files_to_download:
                try:
                    pbar.set_description(f"Downloading {relative_path}")
                    
                    # Create directory structure if it doesn't exist
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    
                    # Download the file
                    self.s3_client.download_file(self.bucket_name, s3_key, local_path)
                    downloaded_files.append(local_path)
                    pbar.set_postfix({"status": "✅"})
                    
                except ClientError as e:
                    pbar.set_postfix({"status": "❌"})
                    print(f"\n❌ Failed to download {relative_path}: {e}")
                except Exception as e:
                    pbar.set_postfix({"status": "⚠️"})
                    print(f"\n⚠️ Other error while downloading {relative_path}: {e}")
                finally:
                    pbar.update(1)
        
        return downloaded_files, skipped_files

    def file_exists(self, s3_key):
        """Check if a file exists in S3 bucket"""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise e 
        
    def dir_exists(self, s3_prefix: str) -> bool:
        """
        Check if any object exists under the given prefix (i.e. treat prefix as a "directory").
        Returns True if at least one object is found, False otherwise.
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=s3_prefix,
                MaxKeys=1
            )
            # If the call succeeded and returned at least one object, the "directory" exists
            return response.get('KeyCount', 0) > 0
        except ClientError as e:
            # You could log e here if you want—in most cases list_objects_v2
            # will not 404, so any ClientError probably should be raised
            raise


    def delete_folder_recursive(self, folder_path: str) -> int:
        """Delete folder and all its contents recursively"""
        deleted_count = 0
        
        try:
            # List all objects in folder
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=folder_path):
                if 'Contents' in page:
                    # Delete all objects in folder
                    objects_to_delete = [{'Key': obj['Key']} for obj in page['Contents']]
                    
                    if objects_to_delete:
                        try:
                            response = self.s3_client.delete_objects(
                                Bucket=self.bucket_name,
                                Delete={'Objects': objects_to_delete}
                            )
                            deleted_count += len(response.get('Deleted', []))
                            
                            # Print deleted objects for debugging
                            for deleted_obj in response.get('Deleted', []):
                                print(f"✅ Berhasil menghapus: {deleted_obj['Key']}")
                                
                        except ClientError as e:
                            print(f"❌ Gagal menghapus batch objects: {e}")
                            # Try deleting one by one if batch delete fails
                            for obj in objects_to_delete:
                                try:
                                    self.s3_client.delete_object(Bucket=self.bucket_name, Key=obj['Key'])
                                    deleted_count += 1
                                    print(f"✅ Berhasil menghapus: {obj['Key']}")
                                except ClientError as e:
                                    print(f"❌ Gagal menghapus {obj['Key']}: {e}")
                                except Exception as e:
                                    print(f"⚠️ Error lain saat menghapus {obj['Key']}: {e}")
            
            return deleted_count
            
        except Exception as e:
            print(f"⚠️ Error saat listing objek: {e}")
            return deleted_count
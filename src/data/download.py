"""
Dataset download utilities for Sapling ML
Handles downloading and organizing the Mendeley Plant Leaf Diseases dataset
Because apparently we need to get this fucking data somehow
"""

import os
import requests
import zipfile
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Handles downloading and organizing datasets for crop disease detection - the data fetching bitch"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset URLs and metadata
        self.datasets = {
            "mendeley_original": {
                "url": "https://data.mendeley.com/datasets/tywbtsjrjv/1/files/",
                "filename": "Plant_leaf_diseases_dataset_without_augmentation.zip",
                "description": "Original Mendeley dataset without augmentation",
                "expected_size": "~2.5GB"
            },
            "mendeley_augmented": {
                "url": "https://data.mendeley.com/datasets/tywbtsjrjv/1/files/",
                "filename": "Plant_leaf_diseases_dataset_with_augmentation.zip", 
                "description": "Mendeley dataset with augmentation",
                "expected_size": "~15GB"
            },
            "plantdoc": {
                "url": "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip",
                "filename": "PlantDoc-Dataset-master.zip",
                "description": "PlantDoc field images for cross-dataset evaluation",
                "expected_size": "~200MB"
            }
        }
    
    def download_file(self, url: str, filepath: Path, chunk_size: int = 8192) -> bool:
        """
        Download a file from URL with progress tracking
        
        Args:
            url: URL to download from
            filepath: Local path to save the file
            chunk_size: Size of chunks to download at a time
            
        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            logger.info(f"Downloading {url} to {filepath}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rDownload progress: {progress:.1f}%", end="", flush=True)
            
            print()  # New line after progress
            logger.info(f"Successfully downloaded {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {str(e)}")
            return False
    
    def extract_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """
        Extract a zip archive
        
        Args:
            archive_path: Path to the zip file
            extract_to: Directory to extract to
            
        Returns:
            bool: True if extraction successful, False otherwise
        """
        try:
            logger.info(f"Extracting {archive_path} to {extract_to}")
            
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            
            logger.info(f"Successfully extracted to {extract_to}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract {archive_path}: {str(e)}")
            return False
    
    def verify_download(self, filepath: Path, expected_size: Optional[int] = None) -> bool:
        """
        Verify downloaded file integrity
        
        Args:
            filepath: Path to the downloaded file
            expected_size: Expected file size in bytes
            
        Returns:
            bool: True if file is valid, False otherwise
        """
        if not filepath.exists():
            logger.error(f"File {filepath} does not exist")
            return False
        
        actual_size = filepath.stat().st_size
        
        if expected_size and actual_size != expected_size:
            logger.warning(f"File size mismatch: expected {expected_size}, got {actual_size}")
            return False
        
        logger.info(f"File verification passed: {filepath} ({actual_size} bytes)")
        return True
    
    def download_dataset(self, dataset_name: str, force_download: bool = False) -> bool:
        """
        Download a specific dataset
        
        Args:
            dataset_name: Name of the dataset to download
            force_download: Whether to re-download if file exists
            
        Returns:
            bool: True if download successful, False otherwise
        """
        if dataset_name not in self.datasets:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
        
        dataset_info = self.datasets[dataset_name]
        dataset_dir = self.data_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        # Check if already downloaded
        zip_path = dataset_dir / dataset_info["filename"]
        if zip_path.exists() and not force_download:
            logger.info(f"Dataset {dataset_name} already exists. Use force_download=True to re-download")
            return True
        
        # Download the file
        if not self.download_file(dataset_info["url"], zip_path):
            return False
        
        # Verify download
        if not self.verify_download(zip_path):
            return False
        
        # Extract the archive
        if not self.extract_archive(zip_path, dataset_dir):
            return False
        
        logger.info(f"Successfully downloaded and extracted {dataset_name}")
        return True
    
    def download_all_datasets(self, force_download: bool = False) -> Dict[str, bool]:
        """
        Download all available datasets
        
        Args:
            force_download: Whether to re-download existing files
            
        Returns:
            Dict mapping dataset names to download success status
        """
        results = {}
        
        for dataset_name in self.datasets:
            logger.info(f"Downloading dataset: {dataset_name}")
            results[dataset_name] = self.download_dataset(dataset_name, force_download)
        
        return results
    
    def list_available_datasets(self) -> None:
        """Print information about available datasets"""
        print("Available datasets:")
        print("-" * 50)
        
        for name, info in self.datasets.items():
            print(f"Name: {name}")
            print(f"Description: {info['description']}")
            print(f"Expected size: {info['expected_size']}")
            print(f"URL: {info['url']}")
            print("-" * 50)


def main():
    """CLI interface for dataset download"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download datasets for Sapling ML")
    parser.add_argument("--dataset", type=str, help="Specific dataset to download")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader()
    
    if args.list:
        downloader.list_available_datasets()
    elif args.all:
        results = downloader.download_all_datasets(args.force)
        print("\nDownload results:")
        for dataset, success in results.items():
            status = "✓" if success else "✗"
            print(f"{status} {dataset}")
    elif args.dataset:
        success = downloader.download_dataset(args.dataset, args.force)
        status = "✓" if success else "✗"
        print(f"{status} {args.dataset}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

"""
Image deduplication utilities for Sapling ML
Handles perceptual hashing and deduplication of augmented datasets
Because apparently we have too many fucking duplicate images
"""

import os
import hashlib
import imagehash
from pathlib import Path
from typing import Dict, List, Tuple, Set
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import logging
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class ImageDeduplicator:
    """Handles deduplication of images using perceptual hashing - the duplicate killer"""
    
    def __init__(self, hash_size: int = 8, hash_threshold: int = 5):
        """
        Initialize the deduplicator - because we need to kill these fucking duplicates
        
        Args:
            hash_size: Size of the perceptual hash (8x8 = 64 bits)
            hash_threshold: Maximum Hamming distance for considering images as duplicates
        """
        self.hash_size = hash_size
        self.hash_threshold = hash_threshold
        self.image_hashes = {}
        self.duplicate_groups = defaultdict(list)
        self.original_to_augmented = defaultdict(list)
    
    def compute_perceptual_hash(self, image_path: Path) -> str:
        """
        Compute perceptual hash for an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            str: Hexadecimal representation of the hash
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Compute perceptual hash
                phash = imagehash.phash(img, hash_size=self.hash_size)
                return str(phash)
                
        except Exception as e:
            logger.error(f"Failed to compute hash for {image_path}: {str(e)}")
            return None
    
    def compute_sha256(self, image_path: Path) -> str:
        """
        Compute SHA256 hash for file integrity
        
        Args:
            image_path: Path to the image file
            
        Returns:
            str: SHA256 hash
        """
        sha256_hash = hashlib.sha256()
        try:
            with open(image_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to compute SHA256 for {image_path}: {str(e)}")
            return None
    
    def get_image_metadata(self, image_path: Path) -> Dict:
        """
        Extract metadata from an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict: Image metadata
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                
                return {
                    "filename": image_path.name,
                    "width": width,
                    "height": height,
                    "mode": img.mode,
                    "format": img.format
                }
        except Exception as e:
            logger.error(f"Failed to get metadata for {image_path}: {str(e)}")
            return None
    
    def find_duplicates(self, image_paths: List[Path]) -> Dict[str, List[Path]]:
        """
        Find duplicate images using perceptual hashing
        
        Args:
            image_paths: List of image paths to check
            
        Returns:
            Dict mapping hash to list of duplicate image paths
        """
        logger.info(f"Computing hashes for {len(image_paths)} images")
        
        # Compute hashes for all images
        hash_to_paths = defaultdict(list)
        
        for i, image_path in enumerate(image_paths):
            if i % 1000 == 0:
                logger.info(f"Processing image {i+1}/{len(image_paths)}")
            
            phash = self.compute_perceptual_hash(image_path)
            if phash:
                hash_to_paths[phash].append(image_path)
        
        # Find groups of similar images
        duplicate_groups = {}
        processed_hashes = set()
        
        for phash, paths in hash_to_paths.items():
            if len(paths) > 1 and phash not in processed_hashes:
                duplicate_groups[phash] = paths
                processed_hashes.add(phash)
        
        logger.info(f"Found {len(duplicate_groups)} groups of duplicate images")
        return duplicate_groups
    
    def group_original_and_augmented(self, image_paths: List[Path], 
                                   original_dir: Path, 
                                   augmented_dir: Path) -> Dict[str, List[Path]]:
        """
        Group original images with their augmented variants
        
        Args:
            image_paths: List of all image paths
            original_dir: Directory containing original images
            augmented_dir: Directory containing augmented images
            
        Returns:
            Dict mapping original image ID to list of all variants
        """
        logger.info("Grouping original and augmented images")
        
        # Compute hashes for all images
        all_hashes = {}
        for image_path in image_paths:
            phash = self.compute_perceptual_hash(image_path)
            if phash:
                all_hashes[image_path] = phash
        
        # Group by hash
        hash_to_paths = defaultdict(list)
        for path, phash in all_hashes.items():
            hash_to_paths[phash].append(path)
        
        # Identify original images and their augmented variants
        original_groups = {}
        
        for phash, paths in hash_to_paths.items():
            if len(paths) > 1:  # Multiple images with same hash
                # Find the original (usually in original directory)
                original_paths = [p for p in paths if original_dir in p.parents]
                augmented_paths = [p for p in paths if augmented_dir in p.parents]
                
                if original_paths:
                    # Use the first original as the group identifier
                    original_id = original_paths[0].stem
                    original_groups[original_id] = paths
                else:
                    # No original found, use first image as identifier
                    original_id = paths[0].stem
                    original_groups[original_id] = paths
        
        logger.info(f"Grouped {len(original_groups)} original images with their variants")
        return original_groups
    
    def create_manifest(self, image_paths: List[Path], 
                       class_mapping: Dict[str, int],
                       output_path: Path) -> pd.DataFrame:
        """
        Create a manifest file with image metadata and deduplication info
        
        Args:
            image_paths: List of image paths
            class_mapping: Mapping from class name to class ID
            output_path: Path to save the manifest CSV
            
        Returns:
            pd.DataFrame: Manifest dataframe
        """
        logger.info("Creating manifest file")
        
        manifest_data = []
        
        for i, image_path in enumerate(image_paths):
            if i % 1000 == 0:
                logger.info(f"Processing image {i+1}/{len(image_paths)}")
            
            # Extract class from path
            class_name = image_path.parent.name
            class_id = class_mapping.get(class_name, -1)
            
            # Compute hashes
            phash = self.compute_perceptual_hash(image_path)
            sha256 = self.compute_sha256(image_path)
            
            # Get metadata
            metadata = self.get_image_metadata(image_path)
            if not metadata:
                continue
            
            # Determine source
            if "original" in str(image_path):
                source = "original"
            elif "augmented" in str(image_path):
                source = "augmented"
            else:
                source = "unknown"
            
            # Generate original ID (for grouping)
            orig_id = f"{class_name}_{phash[:8]}" if phash else f"{class_name}_{i}"
            
            manifest_data.append({
                "source": source,
                "orig_id": orig_id,
                "filename": image_path.name,
                "filepath": str(image_path),
                "class": class_name,
                "class_id": class_id,
                "width": metadata["width"],
                "height": metadata["height"],
                "phash": phash,
                "sha256": sha256,
                "license": "CC0 1.0",
                "notes": ""
            })
        
        # Create DataFrame
        manifest_df = pd.DataFrame(manifest_data)
        
        # Save to CSV
        manifest_df.to_csv(output_path, index=False)
        logger.info(f"Manifest saved to {output_path}")
        
        return manifest_df
    
    def deduplicate_dataset(self, original_dir: Path, 
                          augmented_dir: Path,
                          output_dir: Path,
                          class_mapping: Dict[str, int]) -> pd.DataFrame:
        """
        Main deduplication pipeline
        
        Args:
            original_dir: Directory containing original images
            augmented_dir: Directory containing augmented images
            output_dir: Directory to save processed data
            class_mapping: Mapping from class name to class ID
            
        Returns:
            pd.DataFrame: Deduplicated manifest
        """
        logger.info("Starting dataset deduplication")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        all_images = []
        
        for ext in image_extensions:
            all_images.extend(original_dir.rglob(f"*{ext}"))
            all_images.extend(augmented_dir.rglob(f"*{ext}"))
        
        logger.info(f"Found {len(all_images)} total images")
        
        # Group original and augmented images
        grouped_images = self.group_original_and_augmented(
            all_images, original_dir, augmented_dir
        )
        
        # Create manifest
        manifest_path = output_dir / "manifest.csv"
        manifest_df = self.create_manifest(all_images, class_mapping, manifest_path)
        
        # Save deduplication info
        dedup_info = {
            "total_images": len(all_images),
            "unique_groups": len(grouped_images),
            "deduplication_stats": {
                "original_images": len([p for p in all_images if "original" in str(p)]),
                "augmented_images": len([p for p in all_images if "augmented" in str(p)]),
                "unique_originals": len([g for g in grouped_images.values() 
                                       if any("original" in str(p) for p in g)])
            }
        }
        
        with open(output_dir / "deduplication_info.json", "w") as f:
            json.dump(dedup_info, f, indent=2)
        
        logger.info("Deduplication completed")
        return manifest_df


def main():
    """CLI interface for deduplication"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deduplicate image datasets")
    parser.add_argument("--original", type=str, required=True, help="Original images directory")
    parser.add_argument("--augmented", type=str, required=True, help="Augmented images directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--classes", type=str, help="Path to class mapping JSON file")
    
    args = parser.parse_args()
    
    # Load class mapping
    if args.classes:
        with open(args.classes, "r") as f:
            class_mapping = json.load(f)
    else:
        # Default class mapping (you might want to load from config)
        class_mapping = {f"class_{i}": i for i in range(39)}
    
    # Run deduplication
    deduplicator = ImageDeduplicator()
    manifest_df = deduplicator.deduplicate_dataset(
        Path(args.original),
        Path(args.augmented), 
        Path(args.output),
        class_mapping
    )
    
    print(f"Deduplication completed. Processed {len(manifest_df)} images.")
    print(f"Manifest saved to {args.output}/manifest.csv")


if __name__ == "__main__":
    main()

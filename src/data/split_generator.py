"""
Dataset splitting utilities for Sapling ML
Handles stratified splitting while preventing data leakage from augmented images
Because apparently we need to split this shit properly
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging
import json
from collections import defaultdict

logger = logging.getLogger(__name__)


class DatasetSplitter:
    """Handles dataset splitting with proper stratification and leakage prevention - the data splitting bitch"""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the dataset splitter - because we need to split this shit properly
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def load_manifest(self, manifest_path: Path) -> pd.DataFrame:
        """
        Load the dataset manifest
        
        Args:
            manifest_path: Path to the manifest CSV file
            
        Returns:
            pd.DataFrame: Loaded manifest
        """
        logger.info(f"Loading manifest from {manifest_path}")
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        
        manifest_df = pd.read_csv(manifest_path)
        logger.info(f"Loaded {len(manifest_df)} images from manifest")
        
        return manifest_df
    
    def validate_manifest(self, manifest_df: pd.DataFrame) -> bool:
        """
        Validate the manifest dataframe
        
        Args:
            manifest_df: Manifest dataframe to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        required_columns = ['source', 'orig_id', 'filename', 'class', 'class_id']
        
        for col in required_columns:
            if col not in manifest_df.columns:
                logger.error(f"Missing required column: {col}")
                return False
        
        # Check for missing values in critical columns
        critical_cols = ['orig_id', 'class', 'class_id']
        for col in critical_cols:
            if manifest_df[col].isnull().any():
                logger.error(f"Missing values found in column: {col}")
                return False
        
        logger.info("Manifest validation passed")
        return True
    
    def get_unique_originals(self, manifest_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get unique original images (one per orig_id)
        
        Args:
            manifest_df: Full manifest dataframe
            
        Returns:
            pd.DataFrame: Unique original images
        """
        # Filter for original images only
        original_df = manifest_df[manifest_df['source'] == 'original'].copy()
        
        if len(original_df) == 0:
            logger.warning("No original images found, using all images")
            # If no original images, use the first image from each orig_id group
            original_df = manifest_df.groupby('orig_id').first().reset_index()
        else:
            # Remove duplicates by orig_id (keep first occurrence)
            original_df = original_df.drop_duplicates(subset=['orig_id'], keep='first')
        
        logger.info(f"Found {len(original_df)} unique original images")
        return original_df
    
    def stratified_split(self, df: pd.DataFrame, 
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15,
                        test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform stratified split based on class distribution
        
        Args:
            df: Dataframe to split
            train_ratio: Training set ratio
            val_ratio: Validation set ratio  
            test_ratio: Test set ratio
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        logger.info(f"Performing stratified split: {train_ratio:.1%} train, {val_ratio:.1%} val, {test_ratio:.1%} test")
        
        # First split: separate train from (val + test)
        train_df, temp_df = train_test_split(
            df, 
            test_size=(val_ratio + test_ratio),
            stratify=df['class'],
            random_state=self.random_seed
        )
        
        # Second split: separate val from test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            stratify=temp_df['class'],
            random_state=self.random_seed
        )
        
        logger.info(f"Split completed: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        return train_df, val_df, test_df
    
    def expand_splits_with_augmented(self, original_splits: Dict[str, pd.DataFrame], 
                                   full_manifest: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Expand splits to include augmented images based on original image assignments
        
        Args:
            original_splits: Dictionary with 'train', 'val', 'test' dataframes
            full_manifest: Full manifest with all images
            
        Returns:
            Dict with expanded splits including augmented images
        """
        logger.info("Expanding splits with augmented images")
        
        expanded_splits = {}
        
        for split_name, split_df in original_splits.items():
            # Get orig_ids for this split
            split_orig_ids = set(split_df['orig_id'].unique())
            
            # Find all images (original + augmented) with these orig_ids
            expanded_df = full_manifest[
                full_manifest['orig_id'].isin(split_orig_ids)
            ].copy()
            
            expanded_splits[split_name] = expanded_df
            
            logger.info(f"{split_name}: {len(split_df)} originals -> {len(expanded_df)} total images")
        
        return expanded_splits
    
    def save_splits(self, splits: Dict[str, pd.DataFrame], 
                   output_dir: Path) -> None:
        """
        Save split dataframes to CSV files
        
        Args:
            splits: Dictionary with split dataframes
            output_dir: Directory to save the splits
        """
        splits_dir = output_dir / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_df in splits.items():
            split_path = splits_dir / f"{split_name}.csv"
            split_df.to_csv(split_path, index=False)
            logger.info(f"Saved {split_name} split to {split_path}")
    
    def generate_split_report(self, splits: Dict[str, pd.DataFrame], 
                            output_dir: Path) -> None:
        """
        Generate a detailed report of the dataset splits
        
        Args:
            splits: Dictionary with split dataframes
            output_dir: Directory to save the report
        """
        report = {
            "split_statistics": {},
            "class_distribution": {},
            "source_distribution": {}
        }
        
        for split_name, split_df in splits.items():
            # Basic statistics
            report["split_statistics"][split_name] = {
                "total_images": len(split_df),
                "unique_originals": len(split_df[split_df['source'] == 'original']),
                "augmented_images": len(split_df[split_df['source'] == 'augmented']),
                "unique_orig_ids": split_df['orig_id'].nunique()
            }
            
            # Class distribution
            class_counts = split_df['class'].value_counts().to_dict()
            report["class_distribution"][split_name] = class_counts
            
            # Source distribution
            source_counts = split_df['source'].value_counts().to_dict()
            report["source_distribution"][split_name] = source_counts
        
        # Save report
        report_path = output_dir / "split_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Split report saved to {report_path}")
        
        # Print summary
        print("\nDataset Split Summary:")
        print("=" * 50)
        for split_name, stats in report["split_statistics"].items():
            print(f"\n{split_name.upper()}:")
            print(f"  Total images: {stats['total_images']}")
            print(f"  Original images: {stats['unique_originals']}")
            print(f"  Augmented images: {stats['augmented_images']}")
            print(f"  Unique original IDs: {stats['unique_orig_ids']}")
    
    def split_dataset(self, manifest_path: Path,
                     output_dir: Path,
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15,
                     include_augmented: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Main dataset splitting pipeline
        
        Args:
            manifest_path: Path to the manifest CSV file
            output_dir: Directory to save the splits
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            include_augmented: Whether to include augmented images in splits
            
        Returns:
            Dict with split dataframes
        """
        logger.info("Starting dataset splitting pipeline")
        
        # Load and validate manifest
        manifest_df = self.load_manifest(manifest_path)
        if not self.validate_manifest(manifest_df):
            raise ValueError("Manifest validation failed")
        
        # Get unique original images for splitting
        original_df = self.get_unique_originals(manifest_df)
        
        # Perform stratified split on original images
        train_df, val_df, test_df = self.stratified_split(
            original_df, train_ratio, val_ratio, test_ratio
        )
        
        original_splits = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
        
        if include_augmented:
            # Expand splits to include augmented images
            splits = self.expand_splits_with_augmented(original_splits, manifest_df)
        else:
            splits = original_splits
        
        # Save splits
        self.save_splits(splits, output_dir)
        
        # Generate report
        self.generate_split_report(splits, output_dir)
        
        logger.info("Dataset splitting completed")
        return splits


def main():
    """CLI interface for dataset splitting"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest CSV")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test set ratio")
    parser.add_argument("--no-augmented", action="store_true", help="Exclude augmented images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        print("Error: Split ratios must sum to 1.0")
        return
    
    # Run splitting
    splitter = DatasetSplitter(random_seed=args.seed)
    splits = splitter.split_dataset(
        Path(args.manifest),
        Path(args.output),
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        include_augmented=not args.no_augmented
    )
    
    print(f"\nDataset splitting completed successfully!")
    print(f"Splits saved to {args.output}/splits/")


if __name__ == "__main__":
    main()

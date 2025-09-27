"""
Dataset loading utilities for Sapling ML
Handles loading and preprocessing of plant disease images with fucking augmentation
Because apparently we need to make our data more interesting
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
import yaml
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class PlantDiseaseDataset(Dataset):
    """PyTorch dataset for plant disease classification - the sexy data loader"""
    
    def __init__(self, 
                 manifest_df: pd.DataFrame,
                 image_dir: Path,
                 class_mapping: Dict[str, int],
                 transform: Optional[Callable] = None,
                 is_training: bool = True):
        """
        Initialize the dataset - because we need to load this shit somehow
        
        Args:
            manifest_df: DataFrame with image metadata
            image_dir: Base directory containing images
            class_mapping: Mapping from class name to class ID
            transform: Albumentations transform pipeline
            is_training: Whether this is training data (affects augmentation)
        """
        self.manifest_df = manifest_df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.class_mapping = class_mapping
        self.transform = transform
        self.is_training = is_training
        
        # Validate that all images exist
        self._validate_images()
        
        logger.info(f"Initialized dataset with {len(self.manifest_df)} images")
    
    def _validate_images(self):
        """Validate that all images in manifest exist - because missing files are a bitch"""
        missing_images = []
        
        for idx, row in self.manifest_df.iterrows():
            image_path = self.image_dir / row['filepath']
            if not image_path.exists():
                missing_images.append(str(image_path))
        
        if missing_images:
            logger.warning(f"Found {len(missing_images)} missing images")
            # Remove missing images from manifest - fuck those missing files
            self.manifest_df = self.manifest_df[
                ~self.manifest_df['filepath'].isin(missing_images)
            ].reset_index(drop=True)
            logger.info(f"Removed missing images, {len(self.manifest_df)} images remaining")
    
    def __len__(self) -> int:
        return len(self.manifest_df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Get a single item from the dataset
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (image_tensor, class_id, metadata)
        """
        row = self.manifest_df.iloc[idx]
        
        # Load image
        image_path = self.image_dir / row['filepath']
        try:
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {str(e)}")
            # Return a black image as fallback
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Get class ID
        class_id = int(row['class_id'])
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Convert to tensor if no transform
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Prepare metadata
        metadata = {
            'filename': row['filename'],
            'class_name': row['class'],
            'orig_id': row['orig_id'],
            'source': row['source'],
            'width': row['width'],
            'height': row['height']
        }
        
        return image, class_id, metadata


class AugmentationFactory:
    """Factory for creating augmentation pipelines"""
    
    @staticmethod
    def get_training_transforms(image_size: Tuple[int, int] = (224, 224),
                              augmentation_config: Optional[Dict] = None) -> A.Compose:
        """
        Get training augmentation pipeline
        
        Args:
            image_size: Target image size (height, width)
            augmentation_config: Configuration for augmentation parameters
            
        Returns:
            Albumentations compose object
        """
        if augmentation_config is None:
            augmentation_config = {
                'horizontal_flip_prob': 0.5,
                'vertical_flip_prob': 0.2,
                'rotation_limit': 15,
                'brightness_limit': 0.2,
                'contrast_limit': 0.2,
                'saturation_limit': 0.2,
                'hue_limit': 0.1,
                'blur_limit': 3,
                'noise_variance': 0.01
            }
        
        transforms = [
            # Geometric transforms
            A.HorizontalFlip(p=augmentation_config['horizontal_flip_prob']),
            A.VerticalFlip(p=augmentation_config['vertical_flip_prob']),
            A.Rotate(limit=augmentation_config['rotation_limit'], p=0.5),
            A.RandomResizedCrop(height=image_size[0], width=image_size[1], scale=(0.8, 1.0), p=0.5),
            
            # Color transforms
            A.RandomBrightnessContrast(
                brightness_limit=augmentation_config['brightness_limit'],
                contrast_limit=augmentation_config['contrast_limit'],
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=int(augmentation_config['hue_limit'] * 180),
                sat_shift_limit=int(augmentation_config['saturation_limit'] * 255),
                val_shift_limit=int(augmentation_config['brightness_limit'] * 255),
                p=0.5
            ),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(augmentation_config['noise_variance'] * 255, 
                                      augmentation_config['noise_variance'] * 255), p=0.3),
                A.GaussianBlur(blur_limit=augmentation_config['blur_limit'], p=0.3),
                A.MotionBlur(blur_limit=augmentation_config['blur_limit'], p=0.3),
            ], p=0.3),
            
            # Occlusion and shadows
            A.OneOf([
                A.RandomShadow(p=0.3),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.3),
                A.RandomRain(p=0.3),
            ], p=0.2),
            
            # JPEG compression artifacts
            A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
            
            # Normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225],   # ImageNet std
            ),
            
            # Convert to tensor
            ToTensorV2()
        ]
        
        return A.Compose(transforms)
    
    @staticmethod
    def get_validation_transforms(image_size: Tuple[int, int] = (224, 224)) -> A.Compose:
        """
        Get validation/test augmentation pipeline (minimal augmentation)
        
        Args:
            image_size: Target image size (height, width)
            
        Returns:
            Albumentations compose object
        """
        transforms = [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225],   # ImageNet std
            ),
            ToTensorV2()
        ]
        
        return A.Compose(transforms)


class DataLoaderFactory:
    """Factory for creating data loaders"""
    
    @staticmethod
    def create_data_loaders(config: Dict,
                           train_df: pd.DataFrame,
                           val_df: pd.DataFrame,
                           test_df: pd.DataFrame,
                           image_dir: Path,
                           class_mapping: Dict[str, int]) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test data loaders
        
        Args:
            config: Configuration dictionary
            train_df: Training data manifest
            val_df: Validation data manifest
            test_df: Test data manifest
            image_dir: Base directory containing images
            class_mapping: Mapping from class name to class ID
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Get augmentation config
        aug_config = config.get('dataset', {}).get('augmentation', {})
        image_size = tuple(config.get('dataset', {}).get('image_size', [224, 224]))
        
        # Create transforms
        train_transform = AugmentationFactory.get_training_transforms(image_size, aug_config)
        val_transform = AugmentationFactory.get_validation_transforms(image_size)
        
        # Create datasets
        train_dataset = PlantDiseaseDataset(
            train_df, image_dir, class_mapping, train_transform, is_training=True
        )
        val_dataset = PlantDiseaseDataset(
            val_df, image_dir, class_mapping, val_transform, is_training=False
        )
        test_dataset = PlantDiseaseDataset(
            test_df, image_dir, class_mapping, val_transform, is_training=False
        )
        
        # Get data loader config
        batch_size = config.get('training', {}).get('batch_size', 32)
        num_workers = config.get('hardware', {}).get('num_workers', 4)
        pin_memory = config.get('hardware', {}).get('pin_memory', True)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        logger.info(f"Created data loaders: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
        
        return train_loader, val_loader, test_loader


def load_class_mapping(config_path: Path) -> Dict[str, int]:
    """
    Load class mapping from config file
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Dictionary mapping class names to class IDs
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    class_names = config.get('class_names', {})
    # Convert string keys to int and reverse the mapping
    class_mapping = {v: int(k) for k, v in class_names.items()}
    
    return class_mapping


def main():
    """CLI interface for testing dataset loading"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test dataset loading")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--train-split", type=str, required=True, help="Path to train split CSV")
    parser.add_argument("--val-split", type=str, required=True, help="Path to val split CSV")
    parser.add_argument("--test-split", type=str, required=True, help="Path to test split CSV")
    parser.add_argument("--image-dir", type=str, required=True, help="Base image directory")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load class mapping
    class_mapping = load_class_mapping(Path(args.config))
    
    # Load split dataframes
    train_df = pd.read_csv(args.train_split)
    val_df = pd.read_csv(args.val_split)
    test_df = pd.read_csv(args.test_split)
    
    # Create data loaders
    train_loader, val_loader, test_loader = DataLoaderFactory.create_data_loaders(
        config, train_df, val_df, test_df, Path(args.image_dir), class_mapping
    )
    
    # Test loading a batch
    print("Testing data loaders...")
    
    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        print(f"\n{name.upper()} loader:")
        print(f"  Number of batches: {len(loader)}")
        
        # Load one batch
        batch = next(iter(loader))
        images, labels, metadata = batch
        
        print(f"  Batch shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Sample class: {metadata['class_name'][0]}")
        print(f"  Sample filename: {metadata['filename'][0]}")


if __name__ == "__main__":
    main()

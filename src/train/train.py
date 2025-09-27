"""
Training pipeline for Sapling ML
Config-driven training with comprehensive logging and monitoring
Because apparently we need to train this bitch properly
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
from datetime import datetime
import json
from tqdm import tqdm
import wandb

from ..data.dataset_loader import DataLoaderFactory, load_class_mapping
from ..models.mobilenetv3 import create_mobilenetv3_large
from ..models.efficientnet import create_efficientnet_b0, create_efficientnet_b3
from ..models.resnet import create_resnet101
from .callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, MetricsLogger

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance - because some classes are bitches"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy Loss - because hard labels are too fucking harsh"""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_preds = F.log_softmax(inputs, dim=1)
        targets = targets * (1 - self.smoothing) + self.smoothing / inputs.size(1)
        loss = (-targets * log_preds).sum(dim=1).mean()
        return loss


class ModelFactory:
    """Factory for creating models based on configuration - the model creation bitch"""
    
    @staticmethod
    def create_model(config: Dict) -> nn.Module:
        """
        Create model based on configuration - because we need to build this shit somehow
        
        Args:
            config: Configuration dictionary
            
        Returns:
            PyTorch model
        """
        model_config = config.get('model', {})
        architecture = model_config.get('architecture', 'efficientnet_b0')
        num_classes = model_config.get('num_classes', 39)
        dropout_rate = model_config.get('dropout_rate', 0.2)
        pretrained = model_config.get('pretrained', True)
        
        if architecture == 'mobilenetv3_large':
            model = create_mobilenetv3_large(num_classes, dropout_rate, pretrained)
        elif architecture == 'efficientnet_b0':
            model = create_efficientnet_b0(num_classes, dropout_rate, pretrained)
        elif architecture == 'efficientnet_b3':
            model = create_efficientnet_b3(num_classes, dropout_rate, pretrained)
        elif architecture == 'resnet101':
            model = create_resnet101(num_classes, dropout_rate, pretrained)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        logger.info(f"Created model: {architecture} with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model


class OptimizerFactory:
    """Factory for creating optimizers and schedulers"""
    
    @staticmethod
    def create_optimizer(model: nn.Module, config: Dict) -> optim.Optimizer:
        """Create optimizer based on configuration"""
        training_config = config.get('training', {})
        optimizer_name = training_config.get('optimizer', 'adamw')
        learning_rate = training_config.get('learning_rate', 1e-4)
        weight_decay = training_config.get('weight_decay', 1e-4)
        
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = training_config.get('momentum', 0.9)
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        logger.info(f"Created optimizer: {optimizer_name} with lr={learning_rate}")
        return optimizer
    
    @staticmethod
    def create_scheduler(optimizer: optim.Optimizer, config: Dict) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration"""
        training_config = config.get('training', {})
        scheduler_name = training_config.get('scheduler', 'cosine_annealing')
        num_epochs = training_config.get('num_epochs', 100)
        
        if scheduler_name == 'cosine_annealing':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        elif scheduler_name == 'step':
            step_size = training_config.get('step_size', 30)
            gamma = training_config.get('gamma', 0.1)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        else:
            scheduler = None
        
        if scheduler:
            logger.info(f"Created scheduler: {scheduler_name}")
        
        return scheduler


class LossFactory:
    """Factory for creating loss functions"""
    
    @staticmethod
    def create_loss(config: Dict) -> nn.Module:
        """Create loss function based on configuration"""
        training_config = config.get('training', {})
        loss_name = training_config.get('loss', 'cross_entropy')
        label_smoothing = config.get('model', {}).get('label_smoothing', 0.0)
        
        if loss_name == 'cross_entropy':
            if label_smoothing > 0:
                loss_fn = LabelSmoothingCrossEntropy(label_smoothing)
            else:
                loss_fn = nn.CrossEntropyLoss()
        elif loss_name == 'focal_loss':
            alpha = training_config.get('focal_loss_alpha', 0.25)
            gamma = training_config.get('focal_loss_gamma', 2.0)
            loss_fn = FocalLoss(alpha, gamma)
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")
        
        logger.info(f"Created loss function: {loss_name}")
        return loss_fn


class Trainer:
    """Main training class for Sapling ML"""
    
    def __init__(self, config: Dict, device: torch.device):
        """
        Initialize trainer
        
        Args:
            config: Configuration dictionary
            device: Device to run training on
        """
        self.config = config
        self.device = device
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.callbacks = []
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.training_history = []
        
        # Setup logging
        self._setup_logging()
        
        # Setup callbacks
        self._setup_callbacks()
    
    def _setup_logging(self):
        """Setup logging and experiment tracking"""
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = Path("experiments") / f"exp_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup TensorBoard
        self.writer = SummaryWriter(self.experiment_dir / "tensorboard")
        
        # Setup Weights & Biases
        wandb_config = self.config.get('logging', {}).get('wandb', {})
        if wandb_config.get('enabled', False):
            wandb.init(
                project=wandb_config.get('project', 'sapling-ml'),
                entity=wandb_config.get('entity', None),
                config=self.config,
                dir=str(self.experiment_dir)
            )
        
        # Save config
        with open(self.experiment_dir / "config.yaml", 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def _setup_callbacks(self):
        """Setup training callbacks"""
        # Early stopping
        early_stopping_config = self.config.get('training', {}).get('early_stopping', {})
        if early_stopping_config.get('enabled', True):
            self.callbacks.append(EarlyStopping(
                patience=early_stopping_config.get('patience', 10),
                monitor=early_stopping_config.get('monitor', 'val_macro_f1'),
                mode=early_stopping_config.get('mode', 'max')
            ))
        
        # Model checkpointing
        checkpoint_config = self.config.get('training', {}).get('checkpoint_dir', 'experiments/checkpoints')
        self.callbacks.append(ModelCheckpoint(
            save_dir=Path(checkpoint_config),
            monitor='val_macro_f1',
            mode='max',
            save_best=True,
            save_last=True
        ))
        
        # Learning rate scheduler
        self.callbacks.append(LearningRateScheduler())
        
        # Metrics logger
        self.callbacks.append(MetricsLogger(self.writer))
    
    def setup_model(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
        """Setup model, optimizer, and loss function"""
        # Create model
        self.model = ModelFactory.create_model(self.config)
        self.model = self.model.to(self.device)
        
        # Create optimizer
        self.optimizer = OptimizerFactory.create_optimizer(self.model, self.config)
        
        # Create scheduler
        self.scheduler = OptimizerFactory.create_scheduler(self.optimizer, self.config)
        
        # Create loss function
        self.loss_fn = LossFactory.create_loss(self.config)
        
        # Setup mixed precision training
        self.use_amp = self.config.get('hardware', {}).get('mixed_precision', True)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        logger.info("Model setup completed")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (images, labels, metadata) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return {
            'train_loss': epoch_loss,
            'train_accuracy': epoch_acc
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, metadata in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.loss_fn(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)
                
                total_loss += loss.item()
                
                # Store predictions and labels for metrics calculation
                probabilities = torch.softmax(outputs, dim=1)
                all_predictions.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Basic metrics
        predicted_classes = np.argmax(all_predictions, axis=1)
        accuracy = np.mean(predicted_classes == all_labels)
        
        # Macro F1 score
        from sklearn.metrics import f1_score
        macro_f1 = f1_score(all_labels, predicted_classes, average='macro')
        
        # Per-class metrics
        from sklearn.metrics import classification_report
        class_report = classification_report(all_labels, predicted_classes, output_dict=True)
        
        return {
            'val_loss': total_loss / len(val_loader),
            'val_accuracy': accuracy * 100,
            'val_macro_f1': macro_f1,
            'val_class_report': class_report
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
        """Main training loop"""
        logger.info("Starting training")
        
        num_epochs = self.config.get('training', {}).get('num_epochs', 100)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics['epoch'] = epoch
            epoch_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_macro_f1'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            self.training_history.append(epoch_metrics)
            
            # Call callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, epoch_metrics, self.model, self.optimizer)
            
            # Check for early stopping
            if any(isinstance(cb, EarlyStopping) and cb.should_stop for cb in self.callbacks):
                logger.info("Early stopping triggered")
                break
            
            logger.info(f"Epoch {epoch}: Train Loss={train_metrics['train_loss']:.4f}, "
                       f"Val Loss={val_metrics['val_loss']:.4f}, "
                       f"Val Acc={val_metrics['val_accuracy']:.2f}%, "
                       f"Val F1={val_metrics['val_macro_f1']:.4f}")
        
        # Final evaluation on test set
        logger.info("Evaluating on test set")
        test_metrics = self.validate_epoch(test_loader)
        
        # Save final results
        self._save_results(test_metrics)
        
        logger.info("Training completed")
    
    def _save_results(self, test_metrics: Dict[str, float]):
        """Save final training results"""
        results = {
            'training_history': self.training_history,
            'test_metrics': test_metrics,
            'best_metric': self.best_metric,
            'total_epochs': self.current_epoch
        }
        
        with open(self.experiment_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save training history as CSV
        history_df = pd.DataFrame(self.training_history)
        history_df.to_csv(self.experiment_dir / "training_history.csv", index=False)
        
        logger.info(f"Results saved to {self.experiment_dir}")


def main():
    """CLI interface for training"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Sapling ML model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--train-split", type=str, required=True, help="Path to train split CSV")
    parser.add_argument("--val-split", type=str, required=True, help="Path to val split CSV")
    parser.add_argument("--test-split", type=str, required=True, help="Path to test split CSV")
    parser.add_argument("--image-dir", type=str, required=True, help="Base image directory")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
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
    
    # Create trainer
    trainer = Trainer(config, device)
    trainer.setup_model(train_loader, val_loader, test_loader)
    
    # Start training
    trainer.train(train_loader, val_loader, test_loader)


if __name__ == "__main__":
    main()

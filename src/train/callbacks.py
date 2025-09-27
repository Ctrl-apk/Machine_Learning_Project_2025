"""
Training callbacks for Sapling ML
Early stopping, model checkpointing, and metrics logging
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class Callback:
    """Base callback class"""
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]):
        """Called at the beginning of each epoch"""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        """Called at the end of each epoch"""
        pass
    
    def on_training_begin(self, logs: Dict[str, Any]):
        """Called at the beginning of training"""
        pass
    
    def on_training_end(self, logs: Dict[str, Any]):
        """Called at the end of training"""
        pass


class EarlyStopping(Callback):
    """Early stopping callback to prevent overfitting"""
    
    def __init__(self, 
                 patience: int = 10,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 min_delta: float = 0.0,
                 restore_best_weights: bool = True):
        """
        Initialize early stopping callback
        
        Args:
            patience: Number of epochs to wait before stopping
            monitor: Metric to monitor
            mode: 'min' or 'max' - whether to minimize or maximize the metric
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_metric = None
        self.should_stop = False
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        """Check if training should stop"""
        current_metric = logs.get(self.monitor)
        
        if current_metric is None:
            logger.warning(f"Early stopping conditioned on metric '{self.monitor}' which is not available")
            return
        
        if self.best_metric is None:
            self.best_metric = current_metric
            self.best_weights = model.state_dict().copy()
        elif self.monitor_op(current_metric, self.best_metric + self.min_delta):
            self.best_metric = current_metric
            self.wait = 0
            self.best_weights = model.state_dict().copy()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.should_stop = True
                
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    logger.info(f"Restored best weights from epoch {epoch - self.wait}")
                
                logger.info(f"Early stopping triggered at epoch {epoch}")
    
    def on_training_end(self, logs: Dict[str, Any]):
        """Called at the end of training"""
        if self.stopped_epoch > 0:
            logger.info(f"Training stopped early at epoch {self.stopped_epoch}")


class ModelCheckpoint(Callback):
    """Model checkpointing callback"""
    
    def __init__(self, 
                 save_dir: Path,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 save_best: bool = True,
                 save_last: bool = True,
                 save_freq: int = 1,
                 filename_template: str = 'checkpoint_epoch_{epoch:03d}.pth'):
        """
        Initialize model checkpoint callback
        
        Args:
            save_dir: Directory to save checkpoints
            monitor: Metric to monitor for best model
            mode: 'min' or 'max' - whether to minimize or maximize the metric
            save_best: Whether to save the best model
            save_last: Whether to save the last model
            save_freq: Frequency of saving (every N epochs)
            filename_template: Template for checkpoint filenames
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last
        self.save_freq = save_freq
        self.filename_template = filename_template
        
        self.best_metric = None
        self.best_epoch = None
        
        if mode == 'min':
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        """Save model checkpoint if needed"""
        current_metric = logs.get(self.monitor)
        
        if current_metric is None:
            logger.warning(f"Model checkpoint conditioned on metric '{self.monitor}' which is not available")
            return
        
        # Check if this is the best model so far
        is_best = False
        if self.best_metric is None or self.monitor_op(current_metric, self.best_metric):
            self.best_metric = current_metric
            self.best_epoch = epoch
            is_best = True
        
        # Save best model
        if self.save_best and is_best:
            self._save_checkpoint(model, optimizer, epoch, logs, 'best')
            logger.info(f"Saved best model at epoch {epoch} with {self.monitor}={current_metric:.4f}")
        
        # Save last model
        if self.save_last and (epoch + 1) % self.save_freq == 0:
            self._save_checkpoint(model, optimizer, epoch, logs, 'last')
        
        # Save periodic checkpoint
        if (epoch + 1) % self.save_freq == 0:
            self._save_checkpoint(model, optimizer, epoch, logs, f'epoch_{epoch:03d}')
    
    def _save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                        epoch: int, logs: Dict[str, Any], suffix: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'logs': logs,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add scheduler state if available
        if hasattr(optimizer, 'scheduler') and optimizer.scheduler is not None:
            checkpoint['scheduler_state_dict'] = optimizer.scheduler.state_dict()
        
        filename = self.filename_template.format(epoch=epoch).replace('epoch_{epoch:03d}', suffix)
        filepath = self.save_dir / filename
        
        torch.save(checkpoint, filepath)
        logger.debug(f"Saved checkpoint to {filepath}")


class LearningRateScheduler(Callback):
    """Learning rate scheduler callback"""
    
    def __init__(self, log_lr: bool = True):
        """
        Initialize learning rate scheduler callback
        
        Args:
            log_lr: Whether to log learning rate changes
        """
        self.log_lr = log_lr
        self.lr_history = []
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]):
        """Log current learning rate"""
        if self.log_lr and 'learning_rate' in logs:
            current_lr = logs['learning_rate']
            self.lr_history.append(current_lr)
            logger.debug(f"Epoch {epoch}: Learning rate = {current_lr:.2e}")
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        """Update learning rate if scheduler is available"""
        if hasattr(optimizer, 'scheduler') and optimizer.scheduler is not None:
            # Log learning rate after scheduler step
            current_lr = optimizer.param_groups[0]['lr']
            logs['learning_rate'] = current_lr


class MetricsLogger(Callback):
    """Metrics logging callback for TensorBoard and other loggers"""
    
    def __init__(self, writer=None, log_freq: int = 1):
        """
        Initialize metrics logger callback
        
        Args:
            writer: TensorBoard writer
            log_freq: Frequency of logging (every N epochs)
        """
        self.writer = writer
        self.log_freq = log_freq
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        """Log metrics to TensorBoard"""
        if self.writer and (epoch + 1) % self.log_freq == 0:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, epoch)
            
            # Log learning rate
            if 'learning_rate' in logs:
                self.writer.add_scalar('Learning_Rate', logs['learning_rate'], epoch)
            
            # Log model parameters (histograms)
            if (epoch + 1) % (self.log_freq * 5) == 0:  # Less frequent
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        self.writer.add_histogram(f'Parameters/{name}', param, epoch)
                        self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
            
            self.writer.flush()


class ValidationCallback(Callback):
    """Validation callback for custom validation logic"""
    
    def __init__(self, validation_fn, frequency: int = 1):
        """
        Initialize validation callback
        
        Args:
            validation_fn: Function to call for validation
            frequency: How often to run validation (every N epochs)
        """
        self.validation_fn = validation_fn
        self.frequency = frequency
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        """Run validation if needed"""
        if (epoch + 1) % self.frequency == 0:
            validation_results = self.validation_fn(model, epoch)
            logs.update(validation_results)


class ModelSummaryCallback(Callback):
    """Model summary callback for logging model information"""
    
    def __init__(self, log_freq: int = 10):
        """
        Initialize model summary callback
        
        Args:
            log_freq: How often to log model summary (every N epochs)
        """
        self.log_freq = log_freq
    
    def on_training_begin(self, logs: Dict[str, Any]):
        """Log model summary at the beginning of training"""
        logger.info("Model Summary:")
        logger.info(f"  Total parameters: {sum(p.numel() for p in logs.get('model', {}).parameters()):,}")
        logger.info(f"  Trainable parameters: {sum(p.numel() for p in logs.get('model', {}).parameters() if p.requires_grad):,}")
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        """Log model summary periodically"""
        if (epoch + 1) % self.log_freq == 0:
            # Log gradient norms
            total_norm = 0
            param_count = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                logs['gradient_norm'] = total_norm
                logger.info(f"Epoch {epoch}: Gradient norm = {total_norm:.4f}")


class ProgressCallback(Callback):
    """Progress callback for training progress visualization"""
    
    def __init__(self, total_epochs: int):
        """
        Initialize progress callback
        
        Args:
            total_epochs: Total number of epochs
        """
        self.total_epochs = total_epochs
        self.start_time = None
    
    def on_training_begin(self, logs: Dict[str, Any]):
        """Initialize progress tracking"""
        self.start_time = datetime.now()
        logger.info(f"Starting training for {self.total_epochs} epochs")
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        """Log progress information"""
        if self.start_time:
            elapsed_time = datetime.now() - self.start_time
            progress = (epoch + 1) / self.total_epochs
            eta = elapsed_time / progress - elapsed_time
            
            logger.info(f"Progress: {progress:.1%} ({epoch + 1}/{self.total_epochs}) - "
                       f"Elapsed: {elapsed_time} - ETA: {eta}")


def create_default_callbacks(config: Dict) -> list:
    """Create default set of callbacks based on configuration"""
    callbacks = []
    
    # Early stopping
    early_stopping_config = config.get('training', {}).get('early_stopping', {})
    if early_stopping_config.get('enabled', True):
        callbacks.append(EarlyStopping(
            patience=early_stopping_config.get('patience', 10),
            monitor=early_stopping_config.get('monitor', 'val_macro_f1'),
            mode=early_stopping_config.get('mode', 'max')
        ))
    
    # Model checkpointing
    checkpoint_config = config.get('training', {}).get('checkpoint_dir', 'experiments/checkpoints')
    callbacks.append(ModelCheckpoint(
        save_dir=Path(checkpoint_config),
        monitor='val_macro_f1',
        mode='max',
        save_best=True,
        save_last=True
    ))
    
    # Learning rate scheduler
    callbacks.append(LearningRateScheduler())
    
    # Progress tracking
    total_epochs = config.get('training', {}).get('num_epochs', 100)
    callbacks.append(ProgressCallback(total_epochs))
    
    # Model summary
    callbacks.append(ModelSummaryCallback())
    
    return callbacks

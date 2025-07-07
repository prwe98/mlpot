"""
Training interface and utilities for molecular potential models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable, List, Tuple
import time
import numpy as np
from abc import ABC, abstractmethod

from .base_model import BasePotential


class TrainingInterface(ABC):
    """
    Abstract interface for training molecular potential models.
    """
    
    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step."""
        pass
    
    @abstractmethod
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single validation step."""
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str, **kwargs):
        """Save model checkpoint."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        pass


class PotentialTrainer(TrainingInterface):
    """
    Comprehensive trainer for molecular potential models.
    """
    
    def __init__(
        self,
        model: BasePotential,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        loss_config: Dict[str, Any] = None,
        device: str = 'cuda',
        gradient_clip: Optional[float] = None,
        **kwargs
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_clip = gradient_clip
        
        # Loss configuration
        self.loss_config = loss_config or {
            'energy_weight': 1.0,
            'force_weight': 50.0,
            'loss_type': 'l1'
        }
        
        # Initialize loss function
        self.criterion = self._init_loss_function()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metrics = {}
        self.training_history = []
        
    def _init_loss_function(self) -> nn.Module:
        """Initialize loss function based on configuration."""
        loss_type = self.loss_config.get('loss_type', 'l1')
        if loss_type == 'l1':
            return nn.L1Loss()
        elif loss_type == 'l2':
            return nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def compute_loss(
        self, 
        pred_energy: torch.Tensor,
        pred_forces: torch.Tensor,
        true_energy: torch.Tensor,
        true_forces: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss and individual components.
        """
        energy_loss = self.criterion(pred_energy, true_energy)
        force_loss = self.criterion(pred_forces, true_forces)
        
        total_loss = (
            self.loss_config['energy_weight'] * energy_loss +
            self.loss_config['force_weight'] * force_loss
        )
        
        loss_dict = {
            'total_loss': float(total_loss.item()),
            'energy_loss': float(energy_loss.item()),
            'force_loss': float(force_loss.item())
        }
        
        return total_loss, loss_dict
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step."""
        self.model.train()
        
        try:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(batch)
            if isinstance(outputs, tuple) and len(outputs) == 2:
                pred_energy, pred_forces = outputs
            else:
                raise ValueError(f"Model output should be a tuple of (energy, forces), got {type(outputs)}")
            
            true_energy = batch['energy']
            true_forces = batch['forces']
            
            # Compute loss
            loss, loss_dict = self.compute_loss(
                pred_energy, pred_forces, true_energy, true_forces
            )
            
            # Check for numerical issues
            if not torch.isfinite(loss):
                raise ValueError("Loss is not finite (NaN or Inf detected)")
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
                loss_dict['grad_norm'] = float(grad_norm.item())
            
            self.optimizer.step()
            self.global_step += 1
            
            # Ensure all values are Python floats, not numpy types
            loss_dict = {k: float(v) for k, v in loss_dict.items()}
            
            return loss_dict
            
        except Exception as e:
            print(f"Error in train_step: {e}")
            # Return a safe fallback dict
            return {
                'total_loss': float('inf'),
                'energy_loss': float('inf'),
                'force_loss': float('inf')
            }
    
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single validation step."""
        self.model.eval()
        
        try:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            with torch.no_grad():
                outputs = self.model(batch)
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    pred_energy, pred_forces = outputs
                else:
                    raise ValueError(f"Model output should be a tuple of (energy, forces), got {type(outputs)}")
                
                true_energy = batch['energy']
                true_forces = batch['forces']
                
                # Compute loss
                loss, loss_dict = self.compute_loss(
                    pred_energy, pred_forces, true_energy, true_forces
                )
                
                # Check for numerical issues
                if not torch.isfinite(loss):
                    raise ValueError("Validation loss is not finite (NaN or Inf detected)")
                
                # Ensure all values are Python floats
                loss_dict = {k: float(v) for k, v in loss_dict.items()}
            
            return loss_dict
            
        except Exception as e:
            print(f"Error in validate_step: {e}")
            # Return a safe fallback dict
            return {
                'total_loss': float('inf'),
                'energy_loss': float('inf'),
                'force_loss': float('inf')
            }
    
    def train_epoch(
        self, 
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        log_interval: int = 100
    ) -> Dict[str, float]:
        """Train for one epoch."""
        start_time = time.time()
        train_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            loss_dict = self.train_step(batch)
            if isinstance(loss_dict, dict):
                train_losses.append(loss_dict)
            else:
                print(f"Warning: train_step returned {type(loss_dict)}, expected dict")
                continue
            
            if batch_idx % log_interval == 0 and train_losses:
                recent_losses = train_losses[-min(log_interval, len(train_losses)):]
                avg_loss = sum(l['total_loss'] for l in recent_losses) / len(recent_losses)
                print(f"Epoch {self.epoch}, Step {batch_idx}, Loss: {avg_loss:.6f}")
        
        # Check if we have valid training losses
        if not train_losses:
            raise ValueError("No valid training losses collected")
        
        # Compute epoch averages using native Python operations
        epoch_metrics = {}
        for key in train_losses[0].keys():
            values = [float(l[key]) for l in train_losses if key in l and not np.isnan(l[key])]
            if values:
                epoch_metrics[f'train_{key}'] = sum(values) / len(values)
            else:
                epoch_metrics[f'train_{key}'] = float('inf')
        
        # Validation
        if val_loader is not None:
            val_losses = []
            for batch in val_loader:
                val_loss_dict = self.validate_step(batch)
                if isinstance(val_loss_dict, dict):
                    val_losses.append(val_loss_dict)
            
            if val_losses:
                for key in val_losses[0].keys():
                    values = [float(l[key]) for l in val_losses if key in l and not np.isnan(l[key])]
                    if values:
                        epoch_metrics[f'val_{key}'] = sum(values) / len(values)
                    else:
                        epoch_metrics[f'val_{key}'] = float('inf')
        
        # Learning rate scheduling
        if self.scheduler is not None:
            if hasattr(self.scheduler, 'step'):
                if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                    val_loss = epoch_metrics.get('val_total_loss', epoch_metrics.get('train_total_loss', float('inf')))
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
        
        epoch_metrics['epoch_time'] = float(time.time() - start_time)
        epoch_metrics['learning_rate'] = float(self.optimizer.param_groups[0]['lr'])
        
        self.epoch += 1
        self.training_history.append(epoch_metrics)
        
        return epoch_metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        log_interval: int = 100,
        save_best: bool = True,
        checkpoint_dir: str = './checkpoints'
    ) -> List[Dict[str, float]]:
        """
        Train the model for specified number of epochs.
        """
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            epoch_metrics = self.train_epoch(train_loader, val_loader, log_interval)
            
            # Print epoch summary
            print(f"Train Loss: {epoch_metrics['train_total_loss']:.6f}")
            if val_loader is not None:
                print(f"Val Loss: {epoch_metrics['val_total_loss']:.6f}")
                
                # Save best model
                if save_best and epoch_metrics['val_total_loss'] < best_val_loss:
                    best_val_loss = epoch_metrics['val_total_loss']
                    self.save_checkpoint(
                        os.path.join(checkpoint_dir, 'best_model.pt'),
                        epoch=epoch,
                        metrics=epoch_metrics
                    )
                    print(f"New best model saved! Val Loss: {best_val_loss:.6f}")
        
        return self.training_history
    
    def save_checkpoint(self, path: str, **kwargs):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'training_history': self.training_history,
            'loss_config': self.loss_config,
            **kwargs
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.training_history = checkpoint.get('training_history', [])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint

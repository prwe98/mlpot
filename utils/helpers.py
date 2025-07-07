"""
General utilities for the MLPot framework.
This module contains helper functions and utilities used throughout the framework.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
import pickle
import time
from typing import Dict, Any, Optional, Union, Callable
import logging
from pathlib import Path


class AverageTracker:
    """
    Track running averages during training.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the tracker."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """Update the tracker with new value(s)."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def get_average(self) -> float:
        """Get current average."""
        return self.avg


class BestTracker:
    """
    Track best values during training.
    """
    
    def __init__(self, mode: str = "min"):
        assert mode in ["min", "max"], "mode must be 'min' or 'max'"
        self.mode = mode
        self.reset()
    
    def reset(self):
        """Reset the tracker."""
        self.best = float('inf') if self.mode == 'min' else float('-inf')
        self.count_since_best = 0
    
    def update(self, val: float) -> bool:
        """
        Update with new value.
        Returns True if this is a new best value.
        """
        is_best = False
        
        if self.mode == 'min':
            if val < self.best:
                self.best = val
                self.count_since_best = 0
                is_best = True
            else:
                self.count_since_best += 1
        else:  # mode == 'max'
            if val > self.best:
                self.best = val
                self.count_since_best = 0
                is_best = True
            else:
                self.count_since_best += 1
        
        return is_best
    
    def get_best(self) -> float:
        """Get best value."""
        return self.best
    
    def steps_since_best(self) -> int:
        """Get number of steps since best value."""
        return self.count_since_best


class ExponentialMovingAverage:
    """
    Exponential Moving Average for model parameters.
    Helps stabilize training and improve final performance.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow_params = {}
        self.backup_params = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone()
    
    def update(self, model: Optional[nn.Module] = None):
        """Update EMA parameters."""
        if model is None:
            model = self.model
        
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.shadow_params[name] -= (1 - self.decay) * (
                    self.shadow_params[name] - param.data
                )
    
    def apply_shadow(self):
        """Apply EMA parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.backup_params[name] = param.data.clone()
                param.data.copy_(self.shadow_params[name])
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup_params:
                param.data.copy_(self.backup_params[name])
        self.backup_params.clear()
    
    def state_dict(self) -> Dict[str, Any]:
        """Get EMA state dict."""
        return {
            'decay': self.decay,
            'shadow_params': self.shadow_params
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load EMA state dict."""
        self.decay = state_dict['decay']
        self.shadow_params = state_dict['shadow_params']


class Timer:
    """
    Simple timing utility.
    """
    
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        if self.start_time is None:
            return 0
        
        self.elapsed = time.time() - self.start_time
        self.start_time = None
        return self.elapsed
    
    def get_elapsed(self) -> float:
        """Get elapsed time without stopping."""
        if self.start_time is None:
            return self.elapsed
        return time.time() - self.start_time


class ConfigManager:
    """
    Configuration management utility.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    def load_from_file(self, file_path: str):
        """Load configuration from file."""
        file_path = Path(file_path)
        
        if file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                self.config = json.load(f)
        elif file_path.suffix == '.yaml' or file_path.suffix == '.yml':
            import yaml
            with open(file_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {file_path.suffix}")
    
    def save_to_file(self, file_path: str):
        """Save configuration to file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_path.suffix == '.json':
            with open(file_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        elif file_path.suffix == '.yaml' or file_path.suffix == '.yml':
            import yaml
            with open(file_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config file format: {file_path.suffix}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, other_config: Dict[str, Any]):
        """Update configuration with another config."""
        self._deep_update(self.config, other_config)
    
    def _deep_update(self, base: Dict, update: Dict):
        """Recursively update nested dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value


class Logger:
    """
    Simple logging utility.
    """
    
    def __init__(self, name: str = 'mlpot', log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)


def set_random_seed(seed: int):
    """
    Set random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # For deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    **kwargs
):
    """
    Save training checkpoint.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        **kwargs
    }
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def compute_gradient_norm(model: nn.Module) -> float:
    """
    Compute gradient norm of model parameters.
    """
    total_norm = 0.0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** (1. / 2)
    return total_norm


def get_device(prefer_gpu: bool = True) -> str:
    """
    Get appropriate device for computation.
    """
    if prefer_gpu and torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'  # Apple Silicon
    else:
        return 'cpu'


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def memory_usage() -> Dict[str, float]:
    """
    Get current memory usage.
    """
    import psutil
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    usage = {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': memory_info.vms / 1024 / 1024   # Virtual Memory Size
    }
    
    if torch.cuda.is_available():
        usage['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        usage['gpu_cached_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
    
    return usage

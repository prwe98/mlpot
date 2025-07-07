"""
Enhanced trainer with multi-GPU support for molecular potential models.
Uses DistributedDataParallel (DDP) for better compatibility with graph neural networks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Dict, Any, Optional, List, Union
import time
import numpy as np
import os
import socket
from contextlib import contextmanager

from .trainer import PotentialTrainer
from .base_model import BasePotential


class MultiGPUPotentialTrainer(PotentialTrainer):
    """
    Enhanced trainer with multi-GPU support for molecular potential models.
    Uses DistributedDataParallel (DDP) for better compatibility with graph neural networks.
    """
    
    def __init__(
        self,
        model: BasePotential,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        loss_config: Dict[str, Any] = None,
        device: Union[str, List[str], List[int]] = 'auto',
        gradient_clip: Optional[float] = None,
        use_distributed: bool = True,
        backend: str = 'nccl',
        **kwargs
    ):
        """
        Initialize multi-GPU trainer with DDP.
        
        Args:
            model: The molecular potential model
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler (optional)
            loss_config: Loss function configuration
            device: Device specification:
                - 'auto': automatically detect and use all available GPUs
                - 'cuda': use primary GPU only
                - 'cpu': use CPU only
                - List[str]: specific devices like ['cuda:0', 'cuda:1']
                - List[int]: GPU indices like [0, 1, 2]
            gradient_clip: Gradient clipping threshold
            use_distributed: Whether to use DDP for multi-GPU
            backend: DDP backend ('nccl' for GPU, 'gloo' for CPU)
        """
        # Store DDP settings
        self.use_distributed = use_distributed
        self.backend = backend
        self.rank = None
        self.world_size = None
        self.local_rank = None
        self.is_distributed_initialized = False
        
        # Process device configuration
        self.device_config = self._process_device_config(device)
        self.primary_device = self.device_config['primary_device']
        
        # Initialize base trainer with primary device
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_config=loss_config,
            device=self.primary_device,
            gradient_clip=gradient_clip,
            **kwargs
        )
        
        # Note: DDP setup will be done in setup_distributed_training()
        self.is_multi_gpu = False
        self.ddp_model = None
        
    def _process_device_config(self, device: Union[str, List[str], List[int]]) -> Dict[str, Any]:
        """Process device configuration and return standardized format."""
        if not torch.cuda.is_available():
            return {
                'primary_device': 'cpu',
                'gpu_ids': [],
                'physical_gpu_ids': [],
                'device_count': 0,
                'devices': ['cpu'],
                'cuda_visible_devices': None
            }
        
        # 获取CUDA_VISIBLE_DEVICES信息
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        physical_gpu_ids = []
        
        if device == 'auto':
            # Auto-detect based on CUDA_VISIBLE_DEVICES or all available GPUs
            if visible_devices:
                # 解析物理GPU ID用于日志显示
                physical_gpu_ids = [int(x.strip()) for x in visible_devices.split(',') if x.strip()]
            
            # 当设置CUDA_VISIBLE_DEVICES时，PyTorch会自动重新编号GPU为0,1,2...
            # 我们只需要使用这些重新编号的索引
            gpu_ids = list(range(torch.cuda.device_count()))
                
        elif device == 'cuda':
            gpu_ids = [0] if torch.cuda.is_available() else []
            if visible_devices:
                physical_gpu_ids = [int(x.strip()) for x in visible_devices.split(',') if x.strip()][:1]
            
        elif device == 'cpu':
            gpu_ids = []
            
        elif isinstance(device, list):
            if all(isinstance(d, int) for d in device):
                gpu_ids = device
                # 如果有CUDA_VISIBLE_DEVICES，映射到物理GPU
                if visible_devices:
                    visible_list = [int(x.strip()) for x in visible_devices.split(',') if x.strip()]
                    physical_gpu_ids = [visible_list[i] for i in gpu_ids if i < len(visible_list)]
            elif all(isinstance(d, str) and d.startswith('cuda:') for d in device):
                gpu_ids = [int(d.split(':')[1]) for d in device]
                if visible_devices:
                    visible_list = [int(x.strip()) for x in visible_devices.split(',') if x.strip()]
                    physical_gpu_ids = [visible_list[i] for i in gpu_ids if i < len(visible_list)]
            else:
                raise ValueError(f"Invalid device list format: {device}")
        else:
            raise ValueError(f"Unsupported device specification: {device}")
        
        # Filter valid GPU IDs
        if gpu_ids:
            available_gpus = torch.cuda.device_count()
            gpu_ids = [gid for gid in gpu_ids if 0 <= gid < available_gpus]
            # 同步过滤物理GPU ID
            if physical_gpu_ids and len(physical_gpu_ids) > len(gpu_ids):
                physical_gpu_ids = physical_gpu_ids[:len(gpu_ids)]
        
        primary_device = f'cuda:{gpu_ids[0]}' if gpu_ids else 'cpu'
        devices = [f'cuda:{gid}' for gid in gpu_ids] if gpu_ids else ['cpu']
        
        return {
            'primary_device': primary_device,
            'gpu_ids': gpu_ids,
            'physical_gpu_ids': physical_gpu_ids,
            'device_count': len(gpu_ids),
            'devices': devices,
            'cuda_visible_devices': visible_devices
        }
    
    
    def setup_distributed_training(self, rank: int, world_size: int, master_addr: str = 'localhost', master_port: str = '12355'):
        """
        Setup distributed training for DDP.
        
        Args:
            rank: Process rank
            world_size: Total number of processes
            master_addr: Master node address
            master_port: Master node port
        """
        self.rank = rank
        self.world_size = world_size
        self.local_rank = rank  # Assuming single-node setup
        
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        
        # Initialize process group
        dist.init_process_group(
            backend=self.backend,
            rank=rank,
            world_size=world_size
        )
        
        # Set device for this process
        if torch.cuda.is_available() and self.backend == 'nccl':
            torch.cuda.set_device(rank)
            device = f'cuda:{rank}'
        else:
            device = 'cpu'
        
        # Move model to device and wrap with DDP
        self.model = self.model.to(device)
        self.ddp_model = DDP(
            self.model,
            device_ids=[rank] if torch.cuda.is_available() and self.backend == 'nccl' else None
        )
        
        # Update device settings
        self.device = device
        self.primary_device = device
        self.is_multi_gpu = world_size > 1
        self.is_distributed_initialized = True
        
        print(f"Rank {rank}: DDP initialized with device {device}")
    
    def cleanup_distributed_training(self):
        """Cleanup distributed training."""
        if self.is_distributed_initialized:
            dist.destroy_process_group()
            self.is_distributed_initialized = False
    
    @contextmanager
    def distributed_context(self, rank: int, world_size: int, master_addr: str = 'localhost', master_port: str = '12355'):
        """Context manager for distributed training."""
        try:
            self.setup_distributed_training(rank, world_size, master_addr, master_port)
            yield
        finally:
            self.cleanup_distributed_training()
    
    def create_distributed_dataloader(self, dataset, batch_size: int, shuffle: bool = True, **kwargs) -> DataLoader:
        """
        Create a distributed dataloader for DDP training.
        
        Args:
            dataset: The dataset
            batch_size: Batch size per process
            shuffle: Whether to shuffle data
            **kwargs: Additional DataLoader arguments
        
        Returns:
            DataLoader with DistributedSampler
        """
        if self.is_distributed_initialized:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle
            )
            # Don't shuffle when using DistributedSampler
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=False,  # Handled by DistributedSampler
                **kwargs
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                **kwargs
            )
    
    def get_model_for_training(self):
        """Get the model for training (DDP wrapped if distributed)."""
        if self.is_distributed_initialized and self.ddp_model is not None:
            return self.ddp_model
        return self.model
    
    def _setup_multi_gpu(self):
        """
        Legacy method - now DDP setup is handled by setup_distributed_training.
        This method is kept for backward compatibility but prints a warning.
        """
        print("Warning: _setup_multi_gpu is deprecated. Use setup_distributed_training() for DDP.")
        print("For graph neural networks like MLPot, DDP is recommended over DataParallel.")
        
        if not self.use_distributed:
            return
            
        print(f"Multi-GPU setup deferred to setup_distributed_training()")
        print(f"Available devices: {self.device_config['devices']}")
    
    def get_model_for_inference(self) -> BasePotential:
        """Get the underlying model for inference (unwrapped from DDP)."""
        if self.is_distributed_initialized and self.ddp_model is not None:
            return self.ddp_model.module
        return self.model
    
    
    def save_checkpoint(self, path: str, **kwargs):
        """Save model checkpoint (handles DDP wrapper)."""
        # Get the actual model state dict (unwrapped if needed)
        if self.is_distributed_initialized and self.ddp_model is not None:
            model_state_dict = self.ddp_model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
            
        checkpoint = {
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'training_history': self.training_history,
            'loss_config': self.loss_config,
            'device_config': self.device_config,
            'is_multi_gpu': self.is_multi_gpu,
            'is_distributed': self.is_distributed_initialized,
            'rank': self.rank,
            'world_size': self.world_size,
            **kwargs
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Only save from rank 0 in distributed training
        if not self.is_distributed_initialized or self.rank == 0:
            torch.save(checkpoint, path)
            if self.is_distributed_initialized:
                print(f"Checkpoint saved from rank 0 to {path}")
        
    def load_checkpoint(self, path: str, strict: bool = True) -> Dict[str, Any]:
        """Load model checkpoint (handles DDP wrapper)."""
        # Map to current device
        map_location = self.device if hasattr(self, 'device') else self.primary_device
        checkpoint = torch.load(path, map_location=map_location)
        
        # Load model state dict (handle DDP wrapper)
        if self.is_distributed_initialized and self.ddp_model is not None:
            self.ddp_model.module.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.training_history = checkpoint.get('training_history', [])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint
    
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information."""
        info = {
            'primary_device': self.primary_device,
            'is_multi_gpu': self.is_multi_gpu,
            'is_distributed': self.is_distributed_initialized,
            'device_count': self.device_config['device_count'],
            'gpu_ids': self.device_config['gpu_ids'],
            'physical_gpu_ids': self.device_config.get('physical_gpu_ids', []),
            'devices': self.device_config['devices'],
            'cuda_visible_devices': self.device_config.get('cuda_visible_devices', None),
            'rank': self.rank,
            'world_size': self.world_size,
            'local_rank': self.local_rank,
            'backend': self.backend if self.is_distributed_initialized else None
        }
        
        if torch.cuda.is_available():
            info['cuda_device_count'] = torch.cuda.device_count()
            info['current_device'] = torch.cuda.current_device()
            
            # Memory info for each GPU
            gpu_memory = {}
            gpu_list = self.device_config['gpu_ids'] if not self.is_distributed_initialized else [self.local_rank]
            
            for gpu_id in gpu_list:
                try:
                    if gpu_id < torch.cuda.device_count():
                        gpu_memory[gpu_id] = {
                            'allocated': torch.cuda.memory_allocated(gpu_id) / 1024**3,  # GB
                            'reserved': torch.cuda.memory_reserved(gpu_id) / 1024**3,    # GB
                            'max_memory': torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3  # GB
                        }
                    else:
                        gpu_memory[gpu_id] = {
                            'allocated': 0.0,
                            'reserved': 0.0,
                            'max_memory': 0.0,
                            'error': f'GPU {gpu_id} not available'
                        }
                except RuntimeError as e:
                    # 如果无法访问某个GPU，记录警告
                    gpu_memory[gpu_id] = {
                        'allocated': 0.0,
                        'reserved': 0.0,
                        'max_memory': 0.0,
                        'error': str(e)
                    }
            info['gpu_memory'] = gpu_memory
            
        return info
    
    def print_device_info(self):
        """Print comprehensive device information."""
        info = self.get_device_info()
        
        print("=== Device Configuration ===")
        print(f"Primary device: {info['primary_device']}")
        print(f"Multi-GPU training: {info['is_multi_gpu']}")
        print(f"Distributed training (DDP): {info['is_distributed']}")
        
        if info['is_distributed']:
            print(f"Rank: {info['rank']}/{info['world_size']-1}")
            print(f"Backend: {info['backend']}")
        
        print(f"Number of GPUs: {info['device_count']}")
        
        # 显示CUDA_VISIBLE_DEVICES信息
        if info.get('cuda_visible_devices'):
            print(f"CUDA_VISIBLE_DEVICES: {info['cuda_visible_devices']}")
        
        if info['gpu_ids']:
            if not info['is_distributed']:
                print(f"Available GPU IDs: {info['gpu_ids']}")
            else:
                print(f"Current GPU ID: {info['local_rank']}")
            
            # 显示物理GPU映射
            if info.get('physical_gpu_ids') and not info['is_distributed']:
                mapping = []
                for i, pytorch_id in enumerate(info['gpu_ids']):
                    if i < len(info['physical_gpu_ids']):
                        physical_id = info['physical_gpu_ids'][i]
                        mapping.append(f"PyTorch-{pytorch_id}→Physical-{physical_id}")
                    else:
                        mapping.append(f"PyTorch-{pytorch_id}→Physical-?")
                print(f"GPU Mapping: {', '.join(mapping)}")
            
            if not info['is_distributed']:
                print(f"Device list: {info['devices']}")
            
            if 'gpu_memory' in info:
                print("\n=== GPU Memory Status ===")
                for gpu_id, mem_info in info['gpu_memory'].items():
                    if 'error' in mem_info:
                        print(f"GPU {gpu_id}: 无法访问 ({mem_info['error']})")
                    else:
                        physical_info = ""
                        if (info.get('physical_gpu_ids') and 
                            not info['is_distributed'] and 
                            gpu_id < len(info['physical_gpu_ids'])):
                            physical_info = f" (Physical GPU {info['physical_gpu_ids'][gpu_id]})"
                        print(f"GPU {gpu_id}{physical_info}: {mem_info['allocated']:.2f}/{mem_info['max_memory']:.2f} GB "
                              f"(Reserved: {mem_info['reserved']:.2f} GB)")
        
        print("=" * 30)


def run_distributed_training(
    trainer_fn,
    model,
    optimizer,
    dataset,
    batch_size: int = 32,
    epochs: int = 100,
    world_size: int = None,
    master_addr: str = 'localhost',
    master_port: str = '12355',
    backend: str = 'nccl',
    **trainer_kwargs
):
    """
    Convenience function to run distributed training with DDP.
    
    Args:
        trainer_fn: Function that creates and returns a trainer instance
        model: The model to train
        optimizer: Optimizer instance
        dataset: Training dataset
        batch_size: Batch size per process
        epochs: Number of training epochs
        world_size: Number of processes (default: number of available GPUs)
        master_addr: Master node address
        master_port: Master node port
        backend: DDP backend
        **trainer_kwargs: Additional arguments for trainer
    
    Example:
        def create_trainer(model, optimizer, **kwargs):
            return MultiGPUPotentialTrainer(
                model=model,
                optimizer=optimizer,
                **kwargs
            )
        
        run_distributed_training(
            trainer_fn=create_trainer,
            model=my_model,
            optimizer=my_optimizer,
            dataset=my_dataset,
            batch_size=32,
            epochs=100
        )
    """
    if world_size is None:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    if world_size <= 1:
        print("Using single GPU/CPU training")
        trainer = trainer_fn(model, optimizer, **trainer_kwargs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        trainer.train(dataloader, epochs=epochs)
        return trainer
    
    print(f"Starting distributed training with {world_size} processes")
    
    def worker(rank):
        # Create trainer
        trainer = trainer_fn(
            model, 
            optimizer, 
            use_distributed=True,
            backend=backend,
            **trainer_kwargs
        )
        
        # Setup distributed training
        trainer.setup_distributed_training(
            rank=rank,
            world_size=world_size,
            master_addr=master_addr,
            master_port=master_port
        )
        
        # Create distributed dataloader
        dataloader = trainer.create_distributed_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        try:
            # Train
            trainer.train(dataloader, epochs=epochs)
        finally:
            # Cleanup
            trainer.cleanup_distributed_training()
        
        return trainer
    
    # Spawn processes
    mp.spawn(worker, nprocs=world_size, join=True)


class DDPTrainingExample:
    """
    Example class showing how to use the DDP trainer.
    """
    
    @staticmethod
    def create_simple_trainer():
        """Example of creating a trainer for DDP training."""
        from mlpot.models.equivariant_net import EquivariantNet
        import torch.optim as optim
        
        # Create model
        model = EquivariantNet(
            hidden_dim=128,
            num_layers=3,
            cutoff_radius=6.0,
            max_neighbors=20
        )
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Create trainer
        trainer = MultiGPUPotentialTrainer(
            model=model,
            optimizer=optimizer,
            use_distributed=True,
            backend='nccl'
        )
        
        return trainer
    
    @staticmethod
    def manual_ddp_training():
        """Example of manual DDP training setup."""
        # This would typically be called from a script like train_ddp.py
        
        def worker(rank, world_size):
            # Create trainer
            trainer = DDPTrainingExample.create_simple_trainer()
            
            # Setup distributed training
            trainer.setup_distributed_training(rank, world_size)
            
            # Create dummy dataset (replace with real dataset)
            from torch.utils.data import TensorDataset
            dummy_data = {
                'pos': torch.randn(1000, 10, 3),
                'atomic_numbers': torch.randint(1, 19, (1000, 10)),
                'batch': torch.repeat_interleave(torch.arange(1000), 10),
                'cell': torch.randn(1000, 3, 3),
                'energy': torch.randn(1000),
                'forces': torch.randn(1000, 10, 3)
            }
            dataset = TensorDataset(*dummy_data.values())
            
            # Create distributed dataloader
            dataloader = trainer.create_distributed_dataloader(
                dataset, batch_size=16, shuffle=True
            )
            
            try:
                # Training loop would go here
                print(f"Rank {rank}: Training started")
                # trainer.train(dataloader, epochs=10)
            finally:
                trainer.cleanup_distributed_training()
        
        # Launch distributed training
        world_size = torch.cuda.device_count()
        if world_size > 1:
            mp.spawn(worker, args=(world_size,), nprocs=world_size)
        else:
            print("Single GPU training")
            worker(0, 1)

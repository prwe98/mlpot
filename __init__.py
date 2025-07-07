"""
MLPot - Machine Learning Potential Framework
A modular framework for building equivariant neural network potentials.
"""

from .core.base_model import BasePotential
from .core.trainer import PotentialTrainer
from .core.multi_gpu_trainer import MultiGPUPotentialTrainer
from .models.equivariant_net import EquivariantNet
from .data.dataset import MolecularDataset, OUTCARDataset, create_outcar_dataset
from .utils.metrics import EnergyForceMetrics
from .utils.chemistry import get_atomic_number, get_atomic_symbol, element_properties
from .device_config import get_device_manager, auto_device, auto_config

__version__ = "0.1.0"
__author__ = "MLPot Development Team"

__all__ = [
    "BasePotential",
    "PotentialTrainer", 
    "MultiGPUPotentialTrainer",
    "EquivariantNet",
    "MolecularDataset",
    "OUTCARDataset",
    "create_outcar_dataset",
    "EnergyForceMetrics",
    "get_atomic_number",
    "get_atomic_symbol", 
    "element_properties",
    "get_device_manager",
    "auto_device",
    "auto_config"
]

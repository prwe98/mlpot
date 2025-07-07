"""
Core module for MLPot framework.
Contains base classes and training interfaces.
"""

from .base_model import (
    BasePotential,
    MessagePassingInterface,
    EquivarianceInterface
)

from .trainer import (
    TrainingInterface,
    PotentialTrainer
)

from .multi_gpu_trainer import (
    MultiGPUPotentialTrainer
)

__all__ = [
    'BasePotential',
    'MessagePassingInterface', 
    'EquivarianceInterface',
    'TrainingInterface',
    'PotentialTrainer',
    'MultiGPUPotentialTrainer'
]
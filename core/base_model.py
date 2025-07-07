"""
Core base classes and interfaces for MLPot framework.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional


class BasePotential(nn.Module, ABC):
    """
    Abstract base class for all molecular potential models.
    Defines the core interface that all potential models must implement.
    """
    
    def __init__(self, **kwargs):
        super(BasePotential, self).__init__()
        self.config = kwargs
        
    @abstractmethod
    def forward(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            data: Dictionary containing molecular data
                - pos: atomic positions [N, 3]
                - atomic_numbers: atomic numbers [N]
                - batch: batch indices [N]
                - cell: unit cell (optional) [batch_size, 3, 3]
                
        Returns:
            Tuple of (energy, forces)
            - energy: predicted energy [batch_size]
            - forces: predicted forces [N, 3]
        """
        pass
    
    @abstractmethod
    def get_energy_and_forces(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute energy and forces for given molecular configuration.
        """
        pass
    
    def predict(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        High-level prediction interface.
        """
        energy, forces = self.get_energy_and_forces(data)
        return {
            'energy': energy,
            'forces': forces
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return model configuration and metadata.
        """
        return {
            'model_type': self.__class__.__name__,
            'config': self.config,
            'parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class MessagePassingInterface(ABC):
    """
    Interface for message passing operations.
    """
    
    @abstractmethod
    def message(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Compute messages between nodes.
        """
        pass
    
    @abstractmethod
    def aggregate(self, messages: torch.Tensor, edge_index: torch.Tensor,
                  num_nodes: int) -> torch.Tensor:
        """
        Aggregate messages for each node.
        """
        pass
    
    @abstractmethod
    def update(self, x: torch.Tensor, aggregated: torch.Tensor) -> torch.Tensor:
        """
        Update node features.
        """
        pass


class EquivarianceInterface(ABC):
    """
    Interface for equivariant operations.
    """
    
    @abstractmethod
    def apply_rotation(self, features: torch.Tensor, rotation_matrix: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation to equivariant features.
        """
        pass
    
    @abstractmethod
    def check_equivariance(self, data: Dict[str, torch.Tensor], 
                          rotation_matrix: torch.Tensor, 
                          translation: torch.Tensor) -> bool:
        """
        Verify equivariance property.
        """
        pass

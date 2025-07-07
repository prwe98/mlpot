"""
Equivariant neural network implementation based on E2GNN architecture.
This module implements the core equivariant graph neural network model.
"""

import math
import torch
import torch.nn as nn
from torch_scatter import scatter
from typing import Dict, Tuple, Optional

try:
    from ..core.base_model import BasePotential, MessagePassingInterface, EquivarianceInterface
    from ..layers.geometric_layers import RadialBasisFunction, AtomicEmbedding, ScaledActivation
    from ..layers.graph_ops import construct_radius_graph_pbc
except ImportError:
    # 如果相对导入失败，尝试直接导入
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'layers'))
    
    from base_model import BasePotential, MessagePassingInterface, EquivarianceInterface
    from geometric_layers import RadialBasisFunction, AtomicEmbedding, ScaledActivation
    from graph_ops import construct_radius_graph_pbc

# Try to import global_mean_pool, but provide fallback if not available
try:
    from torch_geometric.nn import global_mean_pool
except ImportError:
    def global_mean_pool(x, batch):
        """Fallback implementation of global_mean_pool without torch_geometric"""
        batch_size = batch.max().item() + 1
        result = torch.zeros(batch_size, x.size(1), device=x.device, dtype=x.dtype)
        for i in range(batch_size):
            mask = batch == i
            if mask.any():
                result[i] = x[mask].mean(dim=0)
        return result


class GlobalScalarProcessor(nn.Module):
    """
    Global scalar feature processing module.
    Handles aggregation and update of scalar features across the molecular system.
    """
    
    def __init__(self, hidden_dim: int, use_residual: bool = False):
        super(GlobalScalarProcessor, self).__init__()
        self.use_residual = use_residual
        
        self.node_mixer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            ScaledActivation(),
            nn.Linear(hidden_dim, hidden_dim),
            ScaledActivation()
        )
        
        self.global_mixer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            ScaledActivation(),
            nn.Linear(hidden_dim, hidden_dim),
            ScaledActivation()
        )
    
    def update_local_features(self, node_features: torch.Tensor, 
                            batch_idx: torch.Tensor, 
                            global_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update local node features using global context."""
        enhanced_features = self.node_mixer(
            torch.cat([node_features, global_features[batch_idx]], dim=1)
        ) + node_features
        
        return enhanced_features, global_features
    
    def update_global_features(self, node_features: torch.Tensor,
                             batch_idx: torch.Tensor,
                             global_features: torch.Tensor) -> torch.Tensor:
        """Update global features using aggregated node features."""
        pooled_features = global_mean_pool(node_features, batch_idx)
        updated_global = self.global_mixer(
            torch.cat([pooled_features, global_features], dim=-1)
        )
        
        if self.use_residual:
            global_features = global_features + updated_global
        else:
            global_features = updated_global
            
        return global_features


class GlobalVectorProcessor(nn.Module):
    """
    Global vector feature processing module.
    Handles aggregation and update of vector features while maintaining equivariance.
    """
    
    def __init__(self, hidden_dim: int, use_residual: bool = False):
        super(GlobalVectorProcessor, self).__init__()
        self.use_residual = use_residual
        
        self.node_mixer = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.global_mixer = nn.Linear(hidden_dim, hidden_dim, bias=False)
    
    def update_local_vectors(self, node_vectors: torch.Tensor,
                           batch_idx: torch.Tensor,
                           global_vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update local node vectors using global context."""
        enhanced_vectors = self.node_mixer(
            node_vectors + global_vectors[batch_idx]
        ) + node_vectors
        
        return enhanced_vectors, global_vectors
    
    def update_global_vectors(self, node_vectors: torch.Tensor,
                            batch_idx: torch.Tensor,
                            global_vectors: torch.Tensor) -> torch.Tensor:
        """Update global vectors using aggregated node vectors."""
        pooled_vectors = scatter(node_vectors, batch_idx, dim=0, reduce='mean', 
                               dim_size=global_vectors.size(0))
        updated_global = self.global_mixer(pooled_vectors + global_vectors)
        
        if self.use_residual:
            global_vectors = global_vectors + updated_global
        else:
            global_vectors = updated_global
            
        return global_vectors


class EquivariantMessageLayer(nn.Module, MessagePassingInterface):
    """
    Equivariant message passing layer.
    Implements message computation between neighboring atoms while preserving equivariance.
    """
    
    def __init__(self, hidden_dim: int, num_radial_basis: int):
        super(EquivariantMessageLayer, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Feature projection layers - exactly like E2GNNMessage
        self.feature_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            ScaledActivation(),
            nn.Linear(hidden_dim // 2, hidden_dim * 3)
        )
        
        self.radial_projector = nn.Linear(num_radial_basis, hidden_dim * 3)
        
        # Normalization constants - exactly like E2GNN
        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_hidden = 1 / math.sqrt(hidden_dim)
    
    def message(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        实现MessagePassingInterface的message方法
        """
        # 这是接口要求的简化版本，实际使用forward方法
        return torch.zeros_like(x)
    
    def aggregate(self, messages: torch.Tensor, edge_index: torch.Tensor,
                  num_nodes: int) -> torch.Tensor:
        """
        实现MessagePassingInterface的aggregate方法
        """
        target_idx = edge_index[1]
        from torch_scatter import scatter
        return scatter(messages, target_idx, dim=0, dim_size=num_nodes, reduce='sum')
    
    def update(self, x: torch.Tensor, aggregated: torch.Tensor) -> torch.Tensor:
        """
        实现MessagePassingInterface的update方法
        """
        return x + aggregated
    
    def compute_messages(self, node_features: torch.Tensor, vector_features: torch.Tensor,
                edge_index: torch.Tensor, edge_radial: torch.Tensor, 
                edge_vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute equivariant messages between connected nodes.
        Implementation exactly follows E2GNNMessage.forward()
        """
        source_idx, target_idx = edge_index
        
        # Project radial basis functions
        radial_weights = self.radial_projector(edge_radial)
        
        # Project node features
        feature_weights = self.feature_projector(node_features)
        
        # Combine source features with radial weights - exactly like E2GNN
        combined_weights = feature_weights[source_idx] * radial_weights * self.inv_sqrt_3
        w1, w2, w3 = torch.split(combined_weights, self.hidden_dim, dim=-1)
        
        # Compute vector messages - exactly like E2GNN
        source_vectors = vector_features[source_idx]
        vector_messages = (
            w1.unsqueeze(1) * source_vectors + 
            w2.unsqueeze(1) * edge_vectors.unsqueeze(2)
        ) * self.inv_sqrt_hidden
        
        return w3, vector_messages
    
    def aggregate_messages(self, scalar_messages: torch.Tensor, vector_messages: torch.Tensor,
                  edge_index: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate messages for each target node.
        """
        target_idx = edge_index[1]
        
        aggregated_scalars = scatter(scalar_messages, target_idx, dim=0, 
                                   dim_size=num_nodes, reduce='sum')
        aggregated_vectors = scatter(vector_messages, target_idx, dim=0,
                                   dim_size=num_nodes, reduce='sum')
        
        return aggregated_scalars, aggregated_vectors
    
    def forward(self, node_features: torch.Tensor, vector_features: torch.Tensor,
                edge_index: torch.Tensor, edge_radial: torch.Tensor,
                edge_vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass exactly following E2GNNMessage.forward()"""
        source_idx, target_idx = edge_index
        
        # Project radial basis functions
        radial_weights = self.radial_projector(edge_radial)
        
        # Project node features  
        feature_weights = self.feature_projector(node_features)
        w1, w2, w3 = torch.split(
            feature_weights[source_idx] * radial_weights * self.inv_sqrt_3, 
            self.hidden_dim, dim=-1
        )
        
        # Compute vector messages
        vector_messages = (
            w1.unsqueeze(1) * vector_features[source_idx] + 
            w2.unsqueeze(1) * edge_vectors.unsqueeze(2)
        ) * self.inv_sqrt_hidden
        
        # Aggregate messages directly
        aggregated_vectors = scatter(vector_messages, target_idx, dim=0, 
                                   dim_size=node_features.size(0), reduce='sum')
        aggregated_scalars = scatter(w3, target_idx, dim=0, 
                                   dim_size=node_features.size(0), reduce='sum')
        
        return aggregated_scalars, aggregated_vectors


class EquivariantUpdateLayer(nn.Module):
    """
    Equivariant feature update layer.
    Updates both scalar and vector features while maintaining equivariance.
    Implementation exactly follows E2GNNUpdate.
    """
    
    def __init__(self, hidden_dim: int):
        super(EquivariantUpdateLayer, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Exactly like E2GNNUpdate
        self.vector_projector = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        
        self.mixed_projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            ScaledActivation(),
            nn.Linear(hidden_dim, hidden_dim * 3)
        )
        
        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
    
    def forward(self, scalar_features: torch.Tensor, 
                vector_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update scalar and vector features exactly like E2GNNUpdate.forward()
        """
        # Split vector features - exactly like E2GNN
        v1, v2 = torch.split(
            self.vector_projector(vector_features), self.hidden_dim, dim=-1
        )
        
        # Compute vector norms for scalar-vector coupling - exactly like E2GNN
        vector_norms = torch.sqrt(torch.sum(v2**2, dim=-2) + 1e-8)
        
        # Mixed projection combining scalar features and vector norms
        mixed_features = self.mixed_projector(
            torch.cat([scalar_features, vector_norms], dim=-1)
        )
        
        # Split mixed features - exactly like E2GNN
        x1, x2, x3 = torch.split(mixed_features, self.hidden_dim, dim=-1)
        
        # Apply gating and updates - exactly like E2GNN
        gate = torch.tanh(x3)
        scalar_update = x2 * self.inv_sqrt_2 + scalar_features * gate
        vector_update = x1.unsqueeze(1) * v1
        
        return scalar_update, vector_update


class EquivariantNet(BasePotential, EquivarianceInterface):
    """
    Main equivariant neural network model for molecular potential learning.
    Based on the E2GNN architecture but with modular, interface-driven design.
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_radial_basis: int = 128,
        cutoff_radius: float = 6.0,
        max_neighbors: int = 20,
        radial_basis_config: Dict = None,
        envelope_config: Dict = None,
        predict_forces: bool = True,
        direct_force_prediction: bool = True,
        use_periodic_boundary: bool = False,
        online_graph_construction: bool = True,
        num_elements: int = 83,
        **kwargs
    ):
        super(EquivariantNet, self).__init__(**kwargs)
        
        # Model configuration
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_radial_basis = num_radial_basis
        self.cutoff_radius = cutoff_radius
        self.max_neighbors = max_neighbors
        self.predict_forces = predict_forces
        self.direct_force_prediction = direct_force_prediction
        self.online_graph_construction = online_graph_construction
        self.use_periodic_boundary = use_periodic_boundary
        
        # Initialize embedding layers
        self.atomic_embedding = AtomicEmbedding(hidden_dim, num_elements)
        self.global_embedding = nn.Embedding(1, hidden_dim)
        
        # Initialize radial basis function
        radial_config = radial_basis_config or {"name": "gaussian"}
        envelope_config = envelope_config or {"name": "polynomial", "exponent": 5}
        
        self.radial_basis = RadialBasisFunction(
            num_radial=num_radial_basis,
            cutoff=self.cutoff_radius,
            rbf_config=radial_config,
            envelope_config=envelope_config
        )
        
        # Initialize message passing layers
        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()
        self.global_scalar_layers = nn.ModuleList()
        self.global_vector_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.message_layers.append(
                EquivariantMessageLayer(hidden_dim, num_radial_basis)
            )
            self.update_layers.append(
                EquivariantUpdateLayer(hidden_dim)
            )
            self.global_scalar_layers.append(
                GlobalScalarProcessor(hidden_dim, use_residual=True)
            )
            self.global_vector_layers.append(
                GlobalVectorProcessor(hidden_dim, use_residual=True)
            )
        
        # Output layers
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            ScaledActivation(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.force_head = nn.Linear(hidden_dim, 1, bias=False)
        
        # Normalization constant
        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
    
    def _construct_graph(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct molecular graph from atomic positions.
        """
        positions = data['pos']
        batch_idx = data['batch']
        
        if self.online_graph_construction:
            # Always use our internal graph construction to avoid torch-cluster dependency
            edge_index, cell_offsets, _ = construct_radius_graph_pbc(
                data, self.cutoff_radius, self.max_neighbors
            )
            
            if 'cell' in data and cell_offsets is not None and len(cell_offsets) > 0:
                # Handle periodic boundary conditions
                source_idx, target_idx = edge_index
                
                # Get cells for each edge based on target atom's batch
                edge_batch = batch_idx[target_idx]  # Which molecule each edge belongs to
                edge_cells = data['cell'][edge_batch]  # Cell for each edge
                
                # Apply periodic offsets
                cell_shifts = (cell_offsets.float().unsqueeze(1) @ edge_cells).squeeze(1)
                edge_vectors = (
                    positions[source_idx] + cell_shifts
                ) - positions[target_idx]
            else:
                # Non-periodic case
                source_idx, target_idx = edge_index
                edge_vectors = positions[source_idx] - positions[target_idx]
        else:
            # Use precomputed graph
            edge_index = data['edge_index']
            source_idx, target_idx = edge_index
            
            if self.use_periodic_boundary and 'cell_offsets' in data:
                # Handle precomputed periodic graph
                cell_offsets = data['cell_offsets']
                cell = data['cell']
                num_neighbors = data['neighbors']
                
                cell_offsets_expanded = cell_offsets.unsqueeze(1).float()
                cell_expanded = cell.repeat_interleave(num_neighbors, dim=0)
                
                edge_vectors = (
                    positions[source_idx] + 
                    (cell_offsets_expanded @ cell_expanded).squeeze(1)
                ) - positions[target_idx]
            else:
                edge_vectors = positions[source_idx] - positions[target_idx]
        
        # Compute edge distances and unit vectors
        edge_distances = edge_vectors.norm(dim=-1)
        edge_unit_vectors = -edge_vectors / edge_distances.unsqueeze(-1)
        
        return edge_index, edge_distances, edge_unit_vectors
    
    def forward(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the equivariant network.
        """
        # Extract input data
        positions = data['pos']
        batch_idx = data['batch']
        atomic_numbers = data['atomic_numbers'].long()
        
        assert atomic_numbers.dim() == 1 and atomic_numbers.dtype == torch.long
        
        # Construct molecular graph
        edge_index, edge_distances, edge_unit_vectors = self._construct_graph(data)
        
        # Encode edge distances with radial basis functions
        edge_radial_features = self.radial_basis(edge_distances)
        
        # Initialize node features
        node_features = self.atomic_embedding(atomic_numbers)
        vector_features = torch.zeros(
            node_features.size(0), 3, node_features.size(1), 
            device=node_features.device
        )
        
        # Initialize global features
        num_molecules = batch_idx[-1].item() + 1
        global_scalar_features = self.global_embedding(
            torch.zeros(num_molecules, dtype=torch.long, device=node_features.device)
        )
        global_vector_features = torch.zeros(
            num_molecules, 3, node_features.size(1),
            dtype=node_features.dtype, device=node_features.device
        )
        
        # Message passing and feature updates
        for layer_idx in range(self.num_layers):
            # Update local features with global context
            node_features, global_scalar_features = self.global_scalar_layers[layer_idx].update_local_features(
                node_features, batch_idx, global_scalar_features
            )
            vector_features, global_vector_features = self.global_vector_layers[layer_idx].update_local_vectors(
                vector_features, batch_idx, global_vector_features
            )
            
            # Message passing
            scalar_messages, vector_messages = self.message_layers[layer_idx](
                node_features, vector_features, edge_index, 
                edge_radial_features, edge_unit_vectors
            )
            
            # Apply messages
            node_features = node_features + scalar_messages
            vector_features = vector_features + vector_messages
            node_features = node_features * self.inv_sqrt_2
            
            # Feature updates
            scalar_updates, vector_updates = self.update_layers[layer_idx](
                node_features, vector_features
            )
            
            node_features = node_features + scalar_updates
            vector_features = vector_features + vector_updates
            
            # Update global features
            global_scalar_features = self.global_scalar_layers[layer_idx].update_global_features(
                node_features, batch_idx, global_scalar_features
            )
            global_vector_features = self.global_vector_layers[layer_idx].update_global_vectors(
                vector_features, batch_idx, global_vector_features
            )
        
        # Predict energy
        per_atom_energy = self.energy_head(node_features).squeeze(1)
        molecular_energy = scatter(per_atom_energy, batch_idx, dim=0, reduce='sum')
        
        # Predict forces
        atomic_forces = self.force_head(vector_features).squeeze(-1)
        
        return molecular_energy, atomic_forces
    
    def get_energy_and_forces(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute energy and forces for given molecular configuration.
        """
        return self.forward(data)
    
    def apply_rotation(self, features: torch.Tensor, rotation_matrix: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation to equivariant features.
        """
        if features.dim() == 3:  # Vector features
            return torch.einsum('ij,nkj->nki', rotation_matrix, features)
        else:  # Scalar features
            return features
    
    def check_equivariance(self, data: Dict[str, torch.Tensor], 
                          rotation_matrix: torch.Tensor, 
                          translation: torch.Tensor) -> bool:
        """
        Verify equivariance property of the model.
        """
        # Original prediction
        original_energy, original_forces = self.forward(data)
        
        # Transform input
        transformed_data = data.copy()
        transformed_data['pos'] = torch.einsum('ij,nj->ni', rotation_matrix, data['pos']) + translation
        
        # Prediction on transformed input
        transformed_energy, transformed_forces = self.forward(transformed_data)
        
        # Transform original forces
        rotated_forces = torch.einsum('ij,nj->ni', rotation_matrix, original_forces)
        
        # Check invariance of energy and equivariance of forces
        energy_invariant = torch.allclose(original_energy, transformed_energy, atol=1e-5)
        force_equivariant = torch.allclose(rotated_forces, transformed_forces, atol=1e-5)
        
        return energy_invariant and force_equivariant

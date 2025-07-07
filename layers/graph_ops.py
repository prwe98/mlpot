"""
Graph construction and manipulation operations.
This module provides utilities for building molecular graphs and handling periodic boundary conditions.
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
from torch_geometric.nn import radius_graph
from scipy.spatial.distance import cdist


# Offset list for periodic boundary conditions
PERIODIC_OFFSETS = [
    [-1, -1, -1], [-1, -1, 0], [-1, -1, 1],
    [-1, 0, -1], [-1, 0, 0], [-1, 0, 1],
    [-1, 1, -1], [-1, 1, 0], [-1, 1, 1],
    [0, -1, -1], [0, -1, 0], [0, -1, 1],
    [0, 0, -1], [0, 0, 0], [0, 0, 1],
    [0, 1, -1], [0, 1, 0], [0, 1, 1],
    [1, -1, -1], [1, -1, 0], [1, -1, 1],
    [1, 0, -1], [1, 0, 0], [1, 0, 1],
    [1, 1, -1], [1, 1, 0], [1, 1, 1],
]


def construct_radius_graph_pbc(
    data: Dict[str, torch.Tensor],
    cutoff: float,
    max_neighbors: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct radius graph with periodic boundary conditions.
    
    Args:
        data: Dictionary containing molecular data with 'pos', 'cell', 'batch'
        cutoff: Cutoff radius for neighbor search
        max_neighbors: Maximum number of neighbors per atom
        
    Returns:
        Tuple of (edge_index, cell_offsets, num_neighbors)
    """
    positions = data['pos']
    cell = data['cell']
    batch = data['batch']
    
    device = positions.device
    num_atoms = positions.size(0)
    
    # Convert to numpy for efficient computation
    pos_np = positions.cpu().numpy()
    cell_np = cell.cpu().numpy()
    batch_np = batch.cpu().numpy()
    
    edge_indices = []
    cell_offsets = []
    
    # Process each molecule in the batch
    unique_batches = np.unique(batch_np)
    
    for batch_idx in unique_batches:
        # Get atoms in this molecule
        mask = batch_np == batch_idx
        mol_positions = pos_np[mask]
        mol_cell = cell_np[batch_idx]
        atom_indices = np.where(mask)[0]
        
        # Generate all periodic images within cutoff
        mol_edges, mol_offsets = _find_neighbors_pbc(
            mol_positions, mol_cell, cutoff, max_neighbors, atom_indices
        )
        
        if len(mol_edges) > 0:
            edge_indices.append(mol_edges)
            cell_offsets.append(mol_offsets)
    
    if edge_indices:
        edge_index = np.concatenate(edge_indices, axis=1)
        all_offsets = np.concatenate(cell_offsets, axis=0)
        
        # Convert back to tensors
        edge_index = torch.from_numpy(edge_index).long().to(device)
        cell_offsets = torch.from_numpy(all_offsets).to(device)
        
        # Count neighbors for each atom
        num_neighbors = torch.bincount(edge_index[1], minlength=num_atoms)
    else:
        # No edges found
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        cell_offsets = torch.empty((0, 3), device=device)
        num_neighbors = torch.zeros(num_atoms, dtype=torch.long, device=device)
    
    return edge_index, cell_offsets, num_neighbors


def _find_neighbors_pbc(
    positions: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
    max_neighbors: int,
    atom_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find neighbors for a single molecule with periodic boundary conditions.
    """
    num_atoms = len(positions)
    
    # Generate all possible periodic images
    all_positions = []
    all_atom_indices = []
    all_offsets = []
    
    for offset in PERIODIC_OFFSETS:
        offset_vec = np.array(offset)
        image_positions = positions + offset_vec @ cell
        
        all_positions.append(image_positions)
        # Use range(num_atoms) for each image to get proper local indices
        all_atom_indices.extend(range(num_atoms))
        all_offsets.extend([offset_vec] * num_atoms)
    
    all_positions = np.concatenate(all_positions, axis=0)
    all_atom_indices = np.array(all_atom_indices)
    all_offsets = np.array(all_offsets)
    
    # Compute pairwise distances
    distances = cdist(positions, all_positions)
    
    edge_list = []
    offset_list = []
    
    for i in range(num_atoms):
        # Find neighbors within cutoff
        neighbor_mask = (distances[i] < cutoff) & (distances[i] > 1e-8)
        
        if neighbor_mask.sum() == 0:
            continue
            
        neighbor_distances = distances[i][neighbor_mask]
        neighbor_indices = all_atom_indices[neighbor_mask]
        neighbor_offsets = all_offsets[neighbor_mask]
        
        # Sort by distance and limit to max_neighbors
        sort_indices = np.argsort(neighbor_distances)
        if len(sort_indices) > max_neighbors:
            sort_indices = sort_indices[:max_neighbors]
        
        for j_idx in sort_indices:
            j_local = neighbor_indices[j_idx]  # Local index in this molecule
            j_global = atom_indices[j_local]   # Global index in batch
            offset = neighbor_offsets[j_idx]
            
            # Add edge from global j to global i
            edge_list.append([j_global, atom_indices[i]])
            offset_list.append(offset)
    
    if edge_list:
        edges = np.array(edge_list).T  # Shape: (2, num_edges)
        offsets = np.array(offset_list)  # Shape: (num_edges, 3)
    else:
        edges = np.empty((2, 0), dtype=np.int64)
        offsets = np.empty((0, 3))
    
    return edges, offsets


class GraphConstructor:
    """
    Graph construction utilities for molecular systems.
    """
    
    def __init__(
        self,
        cutoff: float,
        max_neighbors: int = 50,
        use_pbc: bool = False
    ):
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.use_pbc = use_pbc
    
    def construct_graph(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Construct molecular graph from atomic positions.
        """
        if self.use_pbc and 'cell' in data:
            edge_index, cell_offsets, num_neighbors = construct_radius_graph_pbc(
                data, self.cutoff, self.max_neighbors
            )
            
            # Add graph information to data
            data['edge_index'] = edge_index
            data['cell_offsets'] = cell_offsets
            data['neighbors'] = num_neighbors
        else:
            edge_index = radius_graph(
                data['pos'], self.cutoff, data['batch'],
                max_num_neighbors=self.max_neighbors
            )
            data['edge_index'] = edge_index
        
        return data
    
    def add_edge_features(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Add edge features to the graph.
        """
        positions = data['pos']
        edge_index = data['edge_index']
        
        if self.use_pbc and 'cell_offsets' in data:
            # Handle periodic boundary conditions
            source_idx, target_idx = edge_index
            cell = data['cell']
            cell_offsets = data['cell_offsets']
            num_neighbors = data['neighbors']
            
            # Expand cell for each edge
            cell_expanded = cell.repeat_interleave(num_neighbors, dim=0)
            cell_offsets_expanded = cell_offsets.unsqueeze(1).float()
            
            # Compute edge vectors with PBC
            edge_vectors = (
                positions[source_idx] + 
                (cell_offsets_expanded @ cell_expanded).squeeze(1)
            ) - positions[target_idx]
        else:
            # Standard edge vectors
            source_idx, target_idx = edge_index
            edge_vectors = positions[source_idx] - positions[target_idx]
        
        # Compute edge distances and unit vectors
        edge_distances = edge_vectors.norm(dim=-1)
        edge_unit_vectors = edge_vectors / (edge_distances.unsqueeze(-1) + 1e-8)
        
        data['edge_vectors'] = edge_vectors
        data['edge_distances'] = edge_distances
        data['edge_unit_vectors'] = edge_unit_vectors
        
        return data


class NeighborList:
    """
    Efficient neighbor list for molecular dynamics simulations.
    """
    
    def __init__(
        self,
        cutoff: float,
        skin: float = 1.0,
        max_neighbors: int = 50
    ):
        self.cutoff = cutoff
        self.skin = skin
        self.max_neighbors = max_neighbors
        self.extended_cutoff = cutoff + skin
        
        # Cache for neighbor list
        self._cached_positions = None
        self._cached_neighbors = None
        self._cached_distances = None
    
    def update_if_needed(self, positions: torch.Tensor, cell: Optional[torch.Tensor] = None) -> bool:
        """
        Update neighbor list if atoms have moved significantly.
        """
        if self._cached_positions is None:
            self._update_neighbors(positions, cell)
            return True
        
        # Check if any atom has moved more than skin/2
        max_displacement = torch.max(
            torch.norm(positions - self._cached_positions, dim=-1)
        )
        
        if max_displacement > self.skin / 2:
            self._update_neighbors(positions, cell)
            return True
        
        return False
    
    def _update_neighbors(self, positions: torch.Tensor, cell: Optional[torch.Tensor] = None):
        """
        Update the neighbor list.
        """
        self._cached_positions = positions.clone()
        
        # Build neighbor list with extended cutoff
        if cell is not None:
            # Use PBC neighbor search
            data = {'pos': positions, 'cell': cell, 'batch': torch.zeros(len(positions))}
            edge_index, _, _ = construct_radius_graph_pbc(
                data, self.extended_cutoff, self.max_neighbors
            )
        else:
            # Standard neighbor search
            batch = torch.zeros(len(positions), dtype=torch.long, device=positions.device)
            edge_index = radius_graph(
                positions, self.extended_cutoff, batch,
                max_num_neighbors=self.max_neighbors
            )
        
        self._cached_neighbors = edge_index
        
        # Compute distances
        source_idx, target_idx = edge_index
        if cell is not None:
            # TODO: Handle PBC distances
            edge_vectors = positions[source_idx] - positions[target_idx]
        else:
            edge_vectors = positions[source_idx] - positions[target_idx]
        
        self._cached_distances = edge_vectors.norm(dim=-1)
    
    def get_neighbors(self, positions: torch.Tensor, 
                     cell: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get current neighbors within cutoff.
        """
        self.update_if_needed(positions, cell)
        
        # Filter cached neighbors by actual cutoff
        mask = self._cached_distances <= self.cutoff
        filtered_edges = self._cached_neighbors[:, mask]
        filtered_distances = self._cached_distances[mask]
        
        return filtered_edges, filtered_distances


def compute_graph_properties(edge_index: torch.Tensor, num_nodes: int) -> Dict[str, torch.Tensor]:
    """
    Compute various graph properties.
    """
    source_idx, target_idx = edge_index
    
    # Node degrees
    degrees = torch.bincount(target_idx, minlength=num_nodes)
    
    # Edge count
    num_edges = edge_index.size(1)
    
    # Graph connectivity (simplified check)
    is_connected = num_edges >= num_nodes - 1
    
    return {
        'degrees': degrees,
        'num_edges': torch.tensor(num_edges),
        'avg_degree': degrees.float().mean(),
        'max_degree': degrees.max(),
        'is_connected': torch.tensor(is_connected)
    }

"""
Dataset classes and data processing utilities for molecular systems.
This module provides interfaces for loading and processing various molecular datasets.
"""

import os
import lmdb
import pickle
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod
import warnings


class MolecularDataInterface(ABC):
    """
    Abstract interface for molecular datasets.
    """
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset."""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, float]:
        """Get dataset statistics for normalization."""
        pass


class MolecularDataset(Dataset, MolecularDataInterface):
    """
    General molecular dataset class supporting multiple data formats.
    """
    
    def __init__(
        self,
        data_path: str,
        format_type: str = 'lmdb',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        precompute_edges: bool = False,
        cutoff: float = 6.0,
        max_neighbors: int = 50,
        use_pbc: bool = False
    ):
        self.data_path = data_path
        self.format_type = format_type
        self.transform = transform
        self.target_transform = target_transform
        self.precompute_edges = precompute_edges
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.use_pbc = use_pbc
        
        # Initialize data loader based on format
        self._init_data_loader()
        
        # Cache for statistics
        self._statistics = None
    
    def _init_data_loader(self):
        """Initialize the appropriate data loader."""
        if self.format_type == 'lmdb':
            self._init_lmdb()
        elif self.format_type == 'pickle':
            self._init_pickle()
        elif self.format_type == 'npz':
            self._init_npz()
        elif self.format_type == 'h5' or self.format_type == 'hdf5':
            self._init_h5()
        else:
            raise ValueError(f"Unsupported format type: {self.format_type}")
    
    def _init_lmdb(self):
        """Initialize LMDB database connection."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"LMDB database not found: {self.data_path}")
        
        self.env = lmdb.open(
            self.data_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        
        with self.env.begin() as txn:
            self.length = int(txn.get('length'.encode()).decode())
    
    def _init_pickle(self):
        """Initialize pickle file loader."""
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.length = len(self.data)
    
    def _init_npz(self):
        """Initialize NPZ file loader."""
        self.data = np.load(self.data_path)
        # Assume the first array determines the length
        first_key = list(self.data.keys())[0]
        self.length = len(self.data[first_key])
    
    def _init_h5(self):
        """Initialize HDF5 file loader."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 format support")
        
        # Check if this is the format from OUTCARDatasetBuilder
        with h5py.File(self.data_path, 'r') as f:
            if 'metadata' in f:
                # Standard format from OUTCARDatasetBuilder
                self.length = f['metadata'].attrs.get('total_frames', len(f['energies']))
            else:
                # Check if it's in numbered group format (from OUTCARDataset)
                groups = [key for key in f.keys() if key.isdigit()]
                if groups:
                    self.length = len(groups)
                else:
                    # Try to infer from arrays
                    for key in ['energies', 'energy']:
                        if key in f:
                            self.length = len(f[key])
                            break
                    else:
                        raise ValueError("Cannot determine dataset length from H5 file")
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single molecular sample."""
        if self.format_type == 'lmdb':
            data = self._get_lmdb_item(idx)
        elif self.format_type == 'pickle':
            data = self._get_pickle_item(idx)
        elif self.format_type == 'npz':
            data = self._get_npz_item(idx)
        elif self.format_type == 'h5' or self.format_type == 'hdf5':
            data = self._get_h5_item(idx)
        else:
            raise ValueError(f"Unsupported format type: {self.format_type}")
        
        # Apply transforms
        if self.transform:
            data = self.transform(data)
        
        return data
    
    def _get_lmdb_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from LMDB database."""
        with self.env.begin() as txn:
            data_bytes = txn.get(str(idx).encode())
            if data_bytes is None:
                raise KeyError(f"Key {idx} not found in LMDB database")
            
            data = pickle.loads(data_bytes)
        
        return self._convert_to_tensors(data)
    
    def _get_pickle_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from pickle data."""
        data = self.data[idx]
        return self._convert_to_tensors(data)
    
    def _get_npz_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from NPZ data."""
        data = {}
        for key in self.data.keys():
            data[key] = self.data[key][idx]
        
        return self._convert_to_tensors(data)
    
    def _get_h5_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from HDF5 data."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 format support")
        
        with h5py.File(self.data_path, 'r') as f:
            # Check format type
            if str(idx) in f:
                # Numbered group format (from OUTCARDataset)
                group = f[str(idx)]
                data = {}
                for key in group.keys():
                    data[key] = group[key][:]
            else:
                # Array format (structured as (n_structures, n_atoms, 3) or (n_structures,))
                data = {}
                
                # Map from standard OUTCAR format to mlpot format
                key_mapping = {
                    'positions': 'pos',
                    'energies': 'energy',
                    'forces': 'forces',
                    'atomic_numbers': 'atomic_numbers',
                    'cells': 'cell',
                    'num_atoms': 'natoms'
                }
                
                # Extract data for structure at index idx
                for h5_key, mlpot_key in key_mapping.items():
                    if h5_key in f:
                        dataset = f[h5_key]
                        if h5_key in ['positions', 'forces']:
                            # Shape: (n_structures, n_atoms, 3) -> (n_atoms, 3)
                            data[mlpot_key] = dataset[idx]
                        elif h5_key == 'atomic_numbers':
                            # Shape: (n_structures, n_atoms) -> (n_atoms,)
                            data[mlpot_key] = dataset[idx]
                        elif h5_key == 'cells':
                            # Shape: (n_structures, 3, 3) -> (3, 3)
                            data[mlpot_key] = dataset[idx]
                        elif h5_key in ['energies', 'num_atoms']:
                            # Shape: (n_structures,) -> scalar
                            data[mlpot_key] = dataset[idx]
        
        return self._convert_to_tensors(data)

    def _convert_to_tensors(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert data arrays to PyTorch tensors."""
        tensor_data = {}
        
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                tensor_data[key] = torch.from_numpy(value).float()
            elif isinstance(value, torch.Tensor):
                tensor_data[key] = value.float()
            elif isinstance(value, (int, float)):
                tensor_data[key] = torch.tensor(value, dtype=torch.float32)
            else:
                # Keep non-numeric data as is
                tensor_data[key] = value
        
        # Ensure required fields are present
        self._validate_data(tensor_data)
        
        return tensor_data
    
    def _validate_data(self, data: Dict[str, torch.Tensor]):
        """Validate that required fields are present."""
        required_fields = ['pos', 'atomic_numbers']
        
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Required field '{field}' not found in data")
        
        # Add batch index if not present (for single molecule)
        if 'batch' not in data:
            num_atoms = data['pos'].size(0)
            data['batch'] = torch.zeros(num_atoms, dtype=torch.long)
    
    def get_statistics(self) -> Dict[str, float]:
        """Compute and cache dataset statistics."""
        if self._statistics is not None:
            return self._statistics
        
        print("Computing dataset statistics...")
        
        energies = []
        forces = []
        num_atoms_list = []
        
        for i in range(len(self)):
            try:
                data = self[i]
                
                if 'energy' in data:
                    energies.append(data['energy'].item())
                
                if 'forces' in data:
                    forces.extend(data['forces'].flatten().tolist())
                
                num_atoms_list.append(data['pos'].size(0))
                
                if i % 1000 == 0:
                    print(f"Processed {i}/{len(self)} samples")
                    
            except Exception as e:
                warnings.warn(f"Error processing sample {i}: {e}")
                continue
        
        self._statistics = {}
        
        if energies:
            energies = np.array(energies)
            self._statistics.update({
                'energy_mean': float(np.mean(energies)),
                'energy_std': float(np.std(energies)),
                'energy_min': float(np.min(energies)),
                'energy_max': float(np.max(energies))
            })
        
        if forces:
            forces = np.array(forces)
            self._statistics.update({
                'force_mean': float(np.mean(forces)),
                'force_std': float(np.std(forces)),
                'force_min': float(np.min(forces)),
                'force_max': float(np.max(forces)),
                'force_rmse': float(np.sqrt(np.mean(forces**2)))
            })
        
        if num_atoms_list:
            num_atoms_array = np.array(num_atoms_list)
            self._statistics.update({
                'avg_num_atoms': float(np.mean(num_atoms_array)),
                'min_num_atoms': int(np.min(num_atoms_array)),
                'max_num_atoms': int(np.max(num_atoms_array))
            })
        
        print("Dataset statistics computed successfully")
        return self._statistics


class TrajectoryDataset(MolecularDataset):
    """
    Dataset for molecular dynamics trajectories.
    Specialized for handling time-series molecular data.
    """
    
    def __init__(
        self,
        data_path: str,
        sequence_length: int = 1,
        stride: int = 1,
        **kwargs
    ):
        self.sequence_length = sequence_length
        self.stride = stride
        
        super(TrajectoryDataset, self).__init__(data_path, **kwargs)
        
        # Adjust length for sequence sampling
        if self.sequence_length > 1:
            self.length = max(0, (self.length - self.sequence_length) // self.stride + 1)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence of molecular configurations."""
        if self.sequence_length == 1:
            return super(TrajectoryDataset, self).__getitem__(idx)
        
        # Get sequence of frames
        start_idx = idx * self.stride
        sequence_data = []
        
        for i in range(self.sequence_length):
            frame_idx = start_idx + i
            frame_data = super(TrajectoryDataset, self).__getitem__(frame_idx)
            sequence_data.append(frame_data)
        
        # Stack sequences
        batched_data = {}
        for key in sequence_data[0].keys():
            if isinstance(sequence_data[0][key], torch.Tensor):
                batched_data[key] = torch.stack([frame[key] for frame in sequence_data])
            else:
                batched_data[key] = [frame[key] for frame in sequence_data]
        
        return batched_data


def collate_molecular_data(
    batch: List[Dict[str, torch.Tensor]],
    online_graph_construction: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Collate function for molecular data.
    Handles batching of variable-size molecular systems.
    """
    if len(batch) == 0:
        return {}
    
    # Initialize batch data
    batch_data = {}
    
    # Handle different data types
    for key in batch[0].keys():
        if key in ['pos', 'forces']:
            # Concatenate atomic coordinates and forces
            batch_data[key] = torch.cat([sample[key] for sample in batch], dim=0)
        
        elif key in ['atomic_numbers', 'atomic_masses']:
            # Concatenate atomic properties
            batch_data[key] = torch.cat([sample[key] for sample in batch], dim=0)
        
        elif key in ['energy']:
            # Stack energies
            batch_data[key] = torch.stack([sample[key] for sample in batch], dim=0)
        
        elif key == 'cell':
            # Stack unit cells
            batch_data[key] = torch.stack([sample[key] for sample in batch], dim=0)
        
        elif key == 'edge_index' and not online_graph_construction:
            # Handle precomputed edges
            edge_indices = []
            atom_offset = 0
            
            for sample in batch:
                edge_index = sample[key] + atom_offset
                edge_indices.append(edge_index)
                atom_offset += sample['pos'].size(0)
            
            batch_data[key] = torch.cat(edge_indices, dim=1)
        
        else:
            # Handle other data types
            values = [sample[key] for sample in batch]
            if isinstance(values[0], torch.Tensor):
                if values[0].dim() == 0:  # Scalar tensors
                    batch_data[key] = torch.stack(values, dim=0)
                else:
                    batch_data[key] = torch.cat(values, dim=0)
            else:
                batch_data[key] = values
    
    # Create batch index
    batch_idx = []
    for i, sample in enumerate(batch):
        num_atoms = sample['pos'].size(0)
        batch_idx.extend([i] * num_atoms)
    
    batch_data['batch'] = torch.tensor(batch_idx, dtype=torch.long)
    
    # Add number of atoms per molecule
    natoms = torch.tensor([sample['pos'].size(0) for sample in batch], dtype=torch.long)
    batch_data['natoms'] = natoms
    
    return batch_data


class DataNormalizer:
    """
    Data normalization utilities for molecular datasets.
    """
    
    def __init__(self, method: str = 'zscore'):
        self.method = method
        self.statistics = {}
        self.fitted = False
    
    def fit(self, dataset: MolecularDataset):
        """Fit normalizer to dataset statistics."""
        self.statistics = dataset.get_statistics()
        self.fitted = True
    
    def normalize_energy(self, energy: torch.Tensor) -> torch.Tensor:
        """Normalize energy values."""
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before use")
        
        if self.method == 'zscore':
            mean = self.statistics.get('energy_mean', 0.0)
            std = self.statistics.get('energy_std', 1.0)
            return (energy - mean) / std
        elif self.method == 'minmax':
            min_val = self.statistics.get('energy_min', 0.0)
            max_val = self.statistics.get('energy_max', 1.0)
            return (energy - min_val) / (max_val - min_val)
        else:
            return energy
    
    def denormalize_energy(self, energy: torch.Tensor) -> torch.Tensor:
        """Denormalize energy values."""
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before use")
        
        if self.method == 'zscore':
            mean = self.statistics.get('energy_mean', 0.0)
            std = self.statistics.get('energy_std', 1.0)
            return energy * std + mean
        elif self.method == 'minmax':
            min_val = self.statistics.get('energy_min', 0.0)
            max_val = self.statistics.get('energy_max', 1.0)
            return energy * (max_val - min_val) + min_val
        else:
            return energy
    
    def normalize_forces(self, forces: torch.Tensor) -> torch.Tensor:
        """Normalize force values."""
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before use")
        
        if self.method == 'zscore':
            std = self.statistics.get('force_std', 1.0)
            return forces / std
        else:
            return forces
    
    def denormalize_forces(self, forces: torch.Tensor) -> torch.Tensor:
        """Denormalize force values."""
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before use")
        
        if self.method == 'zscore':
            std = self.statistics.get('force_std', 1.0)
            return forces * std
        else:
            return forces


def create_dataloader(
    dataset: MolecularDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    online_graph_construction: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for molecular datasets.
    """
    from functools import partial
    
    collate_fn = partial(
        collate_molecular_data,
        online_graph_construction=online_graph_construction
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs
    )


# Add OUTCAR support
from .outcar_processor import OUTCARDatasetBuilder, OUTCARParser, find_outcar_files


class OUTCARDataset(MolecularDataset):
    """
    Dataset specialized for VASP OUTCAR files.
    Handles multiple OUTCAR files and converts them to mlpot format.
    """
    
    def __init__(
        self,
        outcar_paths: Union[str, List[str]],
        cache_dir: Optional[str] = None,
        rebuild_cache: bool = False,
        **kwargs
    ):
        """
        Initialize OUTCAR dataset.
        
        Args:
            outcar_paths: Path to directory containing OUTCAR files, or list of OUTCAR file paths
            cache_dir: Directory to cache processed data
            rebuild_cache: Whether to rebuild cache even if it exists
        """
        self.outcar_paths = outcar_paths
        self.cache_dir = cache_dir
        self.rebuild_cache = rebuild_cache
        
        # Process OUTCAR files
        processed_data = self._process_outcar_files()
        
        # Create temporary file for parent class
        import tempfile
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        self.temp_file.close()
        
        # Save processed data
        self._save_processed_data(processed_data, self.temp_file.name)
        
        # Initialize parent class
        super(OUTCARDataset, self).__init__(
            data_path=self.temp_file.name,
            format_type='h5',
            **kwargs
        )
    
    def _process_outcar_files(self) -> Dict[str, Any]:
        """Process OUTCAR files and return structured data."""
        # Determine OUTCAR files to process
        if isinstance(self.outcar_paths, str):
            if os.path.isdir(self.outcar_paths):
                outcar_files = find_outcar_files(self.outcar_paths, recursive=True)
            else:
                outcar_files = [self.outcar_paths]
        else:
            outcar_files = self.outcar_paths
        
        if not outcar_files:
            raise ValueError("No OUTCAR files found")
        
        print(f"🔄 Processing {len(outcar_files)} OUTCAR files...")
        
        # Check cache
        if self.cache_dir and not self.rebuild_cache:
            cache_file = os.path.join(self.cache_dir, 'outcar_cache.h5')
            if os.path.exists(cache_file):
                print(f"📁 Loading from cache: {cache_file}")
                return self._load_cached_data(cache_file)
        
        # Process OUTCAR files
        builder = OUTCARDatasetBuilder(output_format='h5', verbose=True)
        
        # Create temporary output for builder
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_output = tmp.name
        
        try:
            stats = builder.build_dataset(outcar_files, tmp_output)
            processed_data = self._load_processed_data(tmp_output)
            
            # Cache if requested
            if self.cache_dir:
                os.makedirs(self.cache_dir, exist_ok=True)
                cache_file = os.path.join(self.cache_dir, 'outcar_cache.h5')
                import shutil
                shutil.copy2(tmp_output, cache_file)
                print(f"💾 Cached data to: {cache_file}")
            
            return processed_data
            
        finally:
            if os.path.exists(tmp_output):
                os.unlink(tmp_output)
    
    def _load_cached_data(self, cache_file: str) -> Dict[str, Any]:
        """Load data from cache file."""
        return self._load_processed_data(cache_file)
    
    def _load_processed_data(self, file_path: str) -> Dict[str, Any]:
        """Load processed data from HDF5 file."""
        import h5py
        
        data = {}
        with h5py.File(file_path, 'r') as f:
            for key in ['positions', 'atomic_numbers', 'forces', 'energies', 'cells', 'num_atoms']:
                if key in f:
                    data[key] = f[key][:]
        
        return data
    
    def _save_processed_data(self, data: Dict[str, Any], output_path: str):
        """Save processed data to HDF5 file."""
        import h5py
        
        # Convert to mlpot format
        mlpot_data = []
        
        num_structures = len(data['energies'])
        atom_offset = 0
        
        for i in range(num_structures):
            num_atoms = data['num_atoms'][i]
            
            # Extract data for this structure
            pos = data['positions'][atom_offset:atom_offset + num_atoms]
            atomic_numbers = data['atomic_numbers'][atom_offset:atom_offset + num_atoms]
            forces = data['forces'][atom_offset:atom_offset + num_atoms]
            energy = data['energies'][i]
            cell = data['cells'][i]
            
            structure_data = {
                'pos': pos,
                'atomic_numbers': atomic_numbers,
                'forces': forces,
                'energy': energy,
                'cell': cell
            }
            
            mlpot_data.append(structure_data)
            atom_offset += num_atoms
        
        # Save in format compatible with parent class
        with h5py.File(output_path, 'w') as f:
            # Save each structure as a group
            for i, structure in enumerate(mlpot_data):
                group = f.create_group(str(i))
                for key, value in structure.items():
                    group.create_dataset(key, data=value)
        
        self.length = len(mlpot_data)
    
    def _get_h5_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from HDF5 data."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 format support")
        
        with h5py.File(self.data_path, 'r') as f:
            # Check format type
            if str(idx) in f:
                # Numbered group format (from OUTCARDataset)
                group = f[str(idx)]
                data = {}
                for key in group.keys():
                    data[key] = group[key][:]
            else:
                # Array format (from OUTCARDatasetBuilder)
                data = {}
                
                # Map from standard OUTCAR format to mlpot format
                key_mapping = {
                    'positions': 'pos',
                    'energies': 'energy',
                    'forces': 'forces',
                    'atomic_numbers': 'atomic_numbers',
                    'cells': 'cell',
                    'num_atoms': 'natoms'
                }
                
                # Get number of atoms for this structure
                if 'num_atoms' in f:
                    num_atoms_array = f['num_atoms'][:]
                    start_idx = sum(num_atoms_array[:idx])
                    end_idx = start_idx + num_atoms_array[idx]
                    
                    # Extract atomic-level data
                    for h5_key, mlpot_key in key_mapping.items():
                        if h5_key in f:
                            if h5_key in ['positions', 'forces', 'atomic_numbers']:
                                data[mlpot_key] = f[h5_key][start_idx:end_idx]
                            elif h5_key in ['energies']:
                                data[mlpot_key] = f[h5_key][idx]
                            elif h5_key in ['cells']:
                                data[mlpot_key] = f[h5_key][idx]
                else:
                    # Fallback: assume single structure or try direct access
                    for h5_key, mlpot_key in key_mapping.items():
                        if h5_key in f:
                            data[mlpot_key] = f[h5_key][idx]
        
        return self._convert_to_tensors(data)
    
    def __del__(self):
        """Clean up temporary file."""
        if hasattr(self, 'temp_file') and os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)


def create_outcar_dataset(
    outcar_paths: Union[str, List[str]],
    output_path: str,
    format_type: str = 'h5',
    cache_dir: Optional[str] = None
) -> str:
    """
    Convenient function to create a dataset from OUTCAR files.
    
    Args:
        outcar_paths: Path to directory or list of OUTCAR files
        output_path: Output dataset path
        format_type: Output format ('h5', 'pickle', 'npz')
        cache_dir: Cache directory
        
    Returns:
        str: Path to created dataset
    """
    # Find OUTCAR files
    if isinstance(outcar_paths, str) and os.path.isdir(outcar_paths):
        outcar_files = find_outcar_files(outcar_paths, recursive=True)
    elif isinstance(outcar_paths, str):
        outcar_files = [outcar_paths]
    else:
        outcar_files = outcar_paths
    
    if not outcar_files:
        raise ValueError("No OUTCAR files found")
    
    print(f"🔄 Creating dataset from {len(outcar_files)} OUTCAR files...")
    
    # Build dataset
    builder = OUTCARDatasetBuilder(output_format=format_type, verbose=True)
    stats = builder.build_dataset(outcar_files, output_path)
    
    print(f"✅ Dataset created: {output_path}")
    return output_path


# 更新__all__列表
__all__ = [
    'MolecularDataInterface',
    'MolecularDataset',
    'OUTCARDataset',
    'TrajectoryDataset', 
    'DataNormalizer',
    'collate_molecular_data',
    'create_dataloader',
    'create_outcar_dataset'
]

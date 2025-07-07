"""
Crystal structure processing utilities for MLPot framework.
Provides integration with ASE and pymatgen for loading and processing crystal structures.
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
import warnings

from .chemistry import get_atomic_number, get_atomic_symbol, get_atomic_mass, HAS_ASE, HAS_PYMATGEN

# Try to import crystal structure libraries
if HAS_ASE:
    from ase import Atoms
    from ase.io import read, write

if HAS_PYMATGEN:
    from pymatgen.core import Structure, Lattice
    from pymatgen.io.ase import AseAtomsAdaptor


class CrystalLoader:
    """
    Universal crystal structure loader supporting multiple formats.
    
    Supports:
    - Material Project JSON files
    - ASE-compatible formats (CIF, POSCAR, XYZ, etc.)
    - pymatgen Structure objects
    """
    
    def __init__(self, use_ase: bool = True, use_pymatgen: bool = True):
        """
        Initialize crystal loader.
        
        Args:
            use_ase: Whether to use ASE if available
            use_pymatgen: Whether to use pymatgen if available
        """
        self.use_ase = use_ase and HAS_ASE
        self.use_pymatgen = use_pymatgen and HAS_PYMATGEN
        
        if not (self.use_ase or self.use_pymatgen):
            warnings.warn(
                "Neither ASE nor pymatgen is available or enabled. "
                "Only Material Project JSON format will be supported."
            )
    
    def load_structure(self, file_path: Union[str, Path]) -> Dict:
        """
        Load crystal structure from file.
        
        Args:
            file_path: Path to structure file
            
        Returns:
            Dictionary with structure information
        """
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.json':
            return self._load_mp_json(file_path)
        
        elif file_path.suffix.lower() in ['.cif', '.vasp', '.poscar', '.xyz', '.pdb']:
            if self.use_ase:
                return self._load_ase_format(file_path)
            elif self.use_pymatgen:
                return self._load_pymatgen_format(file_path)
            else:
                raise ValueError(f"No library available to read {file_path.suffix} format")
        
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _load_mp_json(self, file_path: Path) -> Dict:
        """Load Material Project JSON format."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle different possible field names for formula
        formula = data.get('pretty_formula', 
                          data.get('formula', 
                          data.get('reduced_formula', 'Unknown')))
        
        # Extract lattice and sites
        lattice = np.array(data['lattice']['matrix'])
        sites = data['sites']
        
        # Process atomic sites
        atomic_numbers = []
        positions = []
        
        for site in sites:
            # Get the element symbol (handle multiple species)
            if isinstance(site['species'], list):
                # Take the first (most abundant) species
                element = list(site['species'][0].keys())[0]
            else:
                element = list(site['species'].keys())[0]
            
            atomic_number = get_atomic_number(element)
            atomic_numbers.append(atomic_number)
            
            # Get fractional coordinates and convert to Cartesian
            frac_coords = np.array(site['xyz'])
            cart_coords = frac_coords @ lattice
            positions.append(cart_coords)
        
        return {
            'lattice': lattice,
            'positions': np.array(positions),
            'atomic_numbers': np.array(atomic_numbers),
            'formula': formula,
            'source': 'mp_json'
        }
    
    def _load_ase_format(self, file_path: Path) -> Dict:
        """Load structure using ASE."""
        atoms = read(str(file_path))
        
        return {
            'lattice': atoms.cell.array,
            'positions': atoms.positions,
            'atomic_numbers': atoms.numbers,
            'formula': atoms.get_chemical_formula(),
            'source': 'ase'
        }
    
    def _load_pymatgen_format(self, file_path: Path) -> Dict:
        """Load structure using pymatgen."""
        structure = Structure.from_file(str(file_path))
        
        return {
            'lattice': structure.lattice.matrix,
            'positions': structure.cart_coords,
            'atomic_numbers': np.array([site.specie.Z for site in structure]),
            'formula': structure.formula,
            'source': 'pymatgen'
        }


class CrystalBatcher:
    """
    Batch multiple crystal structures for model input.
    """
    
    def __init__(self, loader: Optional[CrystalLoader] = None):
        """
        Initialize crystal batcher.
        
        Args:
            loader: CrystalLoader instance (creates default if None)
        """
        self.loader = loader or CrystalLoader()
    
    def prepare_batch(self, 
                     file_paths: List[Union[str, Path]], 
                     max_structures: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Prepare batch data from multiple crystal structure files.
        
        Args:
            file_paths: List of structure file paths
            max_structures: Maximum number of structures to load
            
        Returns:
            Dictionary with batched data ready for model input
        """
        if max_structures is not None:
            file_paths = file_paths[:max_structures]
        
        all_atomic_numbers = []
        all_positions = []
        all_cells = []
        all_batch_idx = []
        structure_info = []
        
        for batch_id, file_path in enumerate(file_paths):
            try:
                structure = self.loader.load_structure(file_path)
                
                # Add to batch
                num_atoms = len(structure['atomic_numbers'])
                all_atomic_numbers.extend(structure['atomic_numbers'])
                all_positions.extend(structure['positions'])
                all_cells.append(structure['lattice'])
                all_batch_idx.extend([batch_id] * num_atoms)
                
                structure_info.append({
                    'file_path': str(file_path),
                    'formula': structure['formula'],
                    'num_atoms': num_atoms,
                    'source': structure['source']
                })
                
                print(f"Loaded {structure['formula']}: {num_atoms} atoms from {structure['source']}")
                
            except Exception as e:
                warnings.warn(f"Failed to load {file_path}: {e}")
                continue
        
        if not all_atomic_numbers:
            raise ValueError("No structures were successfully loaded")
        
        return {
            'atomic_numbers': torch.tensor(all_atomic_numbers, dtype=torch.long),
            'positions': torch.tensor(all_positions, dtype=torch.float32),
            'cells': torch.stack([torch.tensor(cell, dtype=torch.float32) for cell in all_cells]),
            'batch': torch.tensor(all_batch_idx, dtype=torch.long),
            'num_graphs': len(structure_info),
            'structure_info': structure_info
        }


def create_simple_crystal(element: str = 'Fe', 
                         lattice_parameter: float = 2.8,
                         crystal_system: str = 'cubic') -> Dict:
    """
    Create a simple crystal structure for testing.
    
    Args:
        element: Element symbol
        lattice_parameter: Lattice parameter in Angstroms
        crystal_system: Crystal system ('cubic', 'hexagonal', etc.)
        
    Returns:
        Structure dictionary
    """
    atomic_number = get_atomic_number(element)
    
    if crystal_system == 'cubic':
        # Simple cubic
        lattice = np.eye(3) * lattice_parameter
        positions = np.array([[0.0, 0.0, 0.0]])
        atomic_numbers = np.array([atomic_number])
    
    elif crystal_system == 'fcc':
        # Face-centered cubic
        lattice = np.eye(3) * lattice_parameter
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5]
        ]) * lattice_parameter
        atomic_numbers = np.array([atomic_number] * 4)
    
    elif crystal_system == 'bcc':
        # Body-centered cubic
        lattice = np.eye(3) * lattice_parameter
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5]
        ]) * lattice_parameter
        atomic_numbers = np.array([atomic_number] * 2)
    
    else:
        raise ValueError(f"Unsupported crystal system: {crystal_system}")
    
    return {
        'lattice': lattice,
        'positions': positions,
        'atomic_numbers': atomic_numbers,
        'formula': element,
        'source': 'generated'
    }


def analyze_crystal_structure(structure: Dict) -> Dict:
    """
    Analyze crystal structure properties.
    
    Args:
        structure: Structure dictionary
        
    Returns:
        Analysis results
    """
    lattice = structure['lattice']
    positions = structure['positions']
    atomic_numbers = structure['atomic_numbers']
    
    # Basic properties
    num_atoms = len(atomic_numbers)
    volume = np.abs(np.linalg.det(lattice))
    density = np.sum([get_atomic_mass(z) for z in atomic_numbers]) / volume * 1.66054  # g/cm³
    
    # Lattice parameters
    a, b, c = np.linalg.norm(lattice, axis=1)
    alpha = np.arccos(np.dot(lattice[1], lattice[2]) / (b * c)) * 180 / np.pi
    beta = np.arccos(np.dot(lattice[0], lattice[2]) / (a * c)) * 180 / np.pi
    gamma = np.arccos(np.dot(lattice[0], lattice[1]) / (a * b)) * 180 / np.pi
    
    # Element composition
    unique_elements, counts = np.unique(atomic_numbers, return_counts=True)
    composition = {get_atomic_symbol(z): count for z, count in zip(unique_elements, counts)}
    
    return {
        'num_atoms': num_atoms,
        'volume': volume,
        'density': density,
        'lattice_parameters': {'a': a, 'b': b, 'c': c},
        'lattice_angles': {'alpha': alpha, 'beta': beta, 'gamma': gamma},
        'composition': composition,
        'formula': structure.get('formula', 'Unknown')
    }


def print_structure_summary(structure: Dict):
    """Print a summary of the crystal structure."""
    analysis = analyze_crystal_structure(structure)
    
    print(f"Crystal Structure Summary:")
    print(f"  Formula: {analysis['formula']}")
    print(f"  Number of atoms: {analysis['num_atoms']}")
    print(f"  Volume: {analysis['volume']:.3f} Ų")
    print(f"  Density: {analysis['density']:.3f} g/cm³")
    print(f"  Lattice parameters: a={analysis['lattice_parameters']['a']:.3f}, "
          f"b={analysis['lattice_parameters']['b']:.3f}, "
          f"c={analysis['lattice_parameters']['c']:.3f} Å")
    print(f"  Lattice angles: α={analysis['lattice_angles']['alpha']:.1f}°, "
          f"β={analysis['lattice_angles']['beta']:.1f}°, "
          f"γ={analysis['lattice_angles']['gamma']:.1f}°")
    print(f"  Composition: {analysis['composition']}")


# Example usage
if __name__ == "__main__":
    from .chemistry import print_library_status
    
    print_library_status()
    
    # Test crystal loader
    loader = CrystalLoader()
    
    # Create test structure
    test_structure = create_simple_crystal('Fe', crystal_system='bcc')
    print_structure_summary(test_structure)
    
    # Test batcher with generated structure
    print("\nTesting crystal batcher...")
    batcher = CrystalBatcher(loader)
    
    # This would work with real files:
    # batch_data = batcher.prepare_batch(['structure1.json', 'structure2.cif'])
    
    print("Crystal processing utilities ready!")

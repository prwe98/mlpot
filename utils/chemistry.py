"""
Chemical utilities for the MLPot framework.
This module provides chemical constants and helper functions for molecular systems.
"""

import numpy as np
from typing import Dict, List, Union, Optional
import warnings

# Try to import from ASE or pymatgen if available
try:
    from ase.data import atomic_numbers, chemical_symbols, atomic_masses
    from ase import Atoms
    HAS_ASE = True
except ImportError:
    HAS_ASE = False

try:
    from pymatgen.core import Element
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False

# Fallback atomic data (up to element 118)
ATOMIC_NUMBERS = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
    'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
    'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
    'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
    'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
    'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
    'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
    'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
    'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
    'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
    'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
    'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92,
    'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99,
    'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106,
    'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113,
    'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
}

# Reverse mapping for atomic numbers to symbols
ATOMIC_SYMBOLS = {v: k for k, v in ATOMIC_NUMBERS.items()}

# Atomic masses (in amu)
ATOMIC_MASSES = {
    1: 1.008, 2: 4.003, 3: 6.941, 4: 9.012, 5: 10.811, 6: 12.011, 7: 14.007, 8: 15.999,
    9: 18.998, 10: 20.180, 11: 22.990, 12: 24.305, 13: 26.982, 14: 28.086, 15: 30.974,
    16: 32.065, 17: 35.453, 18: 39.948, 19: 39.098, 20: 40.078, 21: 44.956, 22: 47.867,
    23: 50.942, 24: 51.996, 25: 54.938, 26: 55.845, 27: 58.933, 28: 58.693, 29: 63.546,
    30: 65.38, 31: 69.723, 32: 72.630, 33: 74.922, 34: 78.971, 35: 79.904, 36: 83.798
    # Add more as needed...
}


def get_atomic_number(element: Union[str, int]) -> int:
    """
    Get atomic number from element symbol or verify atomic number.
    
    Uses ASE or pymatgen if available, falls back to internal data.
    
    Args:
        element: Element symbol (str) or atomic number (int)
        
    Returns:
        Atomic number (int)
        
    Raises:
        ValueError: If element is not found
    """
    if isinstance(element, int):
        if element < 1 or element > 118:
            raise ValueError(f"Invalid atomic number: {element}")
        return element
    
    if isinstance(element, str):
        # Try ASE first
        if HAS_ASE:
            try:
                return atomic_numbers[element]
            except KeyError:
                pass
        
        # Try pymatgen
        if HAS_PYMATGEN:
            try:
                return Element(element).Z
            except (ValueError, KeyError):
                pass
        
        # Fallback to internal data
        element = element.capitalize()  # Handle case variations
        if element in ATOMIC_NUMBERS:
            return ATOMIC_NUMBERS[element]
        
        raise ValueError(f"Unknown element: {element}")
    
    raise TypeError(f"Element must be str or int, got {type(element)}")


def get_atomic_symbol(atomic_number: int) -> str:
    """
    Get element symbol from atomic number.
    
    Args:
        atomic_number: Atomic number
        
    Returns:
        Element symbol
    """
    if HAS_ASE:
        try:
            return chemical_symbols[atomic_number]
        except (IndexError, KeyError):
            pass
    
    if HAS_PYMATGEN:
        try:
            return Element.from_Z(atomic_number).symbol
        except (ValueError, KeyError):
            pass
    
    if atomic_number in ATOMIC_SYMBOLS:
        return ATOMIC_SYMBOLS[atomic_number]
    
    raise ValueError(f"Unknown atomic number: {atomic_number}")


def get_atomic_mass(element: Union[str, int]) -> float:
    """
    Get atomic mass from element symbol or atomic number.
    
    Args:
        element: Element symbol or atomic number
        
    Returns:
        Atomic mass in amu
    """
    if isinstance(element, str):
        atomic_number = get_atomic_number(element)
    else:
        atomic_number = element
    
    if HAS_ASE:
        try:
            return atomic_masses[atomic_number]
        except (IndexError, KeyError):
            pass
    
    if HAS_PYMATGEN:
        try:
            return Element.from_Z(atomic_number).atomic_mass
        except (ValueError, KeyError):
            pass
    
    if atomic_number in ATOMIC_MASSES:
        return ATOMIC_MASSES[atomic_number]
    
    # Rough approximation for missing elements
    return atomic_number * 2.0


def validate_atomic_numbers(atomic_numbers: Union[List[int], np.ndarray]) -> bool:
    """
    Validate that all atomic numbers are valid.
    
    Args:
        atomic_numbers: List or array of atomic numbers
        
    Returns:
        True if all valid
        
    Raises:
        ValueError: If any atomic number is invalid
    """
    atomic_numbers = np.asarray(atomic_numbers)
    
    if np.any(atomic_numbers < 1) or np.any(atomic_numbers > 118):
        invalid = atomic_numbers[(atomic_numbers < 1) | (atomic_numbers > 118)]
        raise ValueError(f"Invalid atomic numbers found: {invalid}")
    
    return True


def parse_chemical_formula(formula: str) -> Dict[str, int]:
    """
    Parse a chemical formula into element counts.
    
    Args:
        formula: Chemical formula (e.g., "H2O", "CaCO3")
        
    Returns:
        Dictionary of element counts
    """
    import re
    
    # Simple regex for parsing formulas like H2O, CaCO3, etc.
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)
    
    composition = {}
    for element, count in matches:
        count = int(count) if count else 1
        composition[element] = composition.get(element, 0) + count
    
    return composition


def get_formula_weight(formula: str) -> float:
    """
    Calculate molecular weight from chemical formula.
    
    Args:
        formula: Chemical formula
        
    Returns:
        Molecular weight in amu
    """
    composition = parse_chemical_formula(formula)
    total_mass = 0.0
    
    for element, count in composition.items():
        atomic_mass = get_atomic_mass(element)
        total_mass += atomic_mass * count
    
    return total_mass


def element_properties(element: Union[str, int]) -> Dict[str, Union[str, int, float]]:
    """
    Get comprehensive element properties.
    
    Args:
        element: Element symbol or atomic number
        
    Returns:
        Dictionary with element properties
    """
    atomic_number = get_atomic_number(element)
    symbol = get_atomic_symbol(atomic_number)
    mass = get_atomic_mass(atomic_number)
    
    properties = {
        'symbol': symbol,
        'atomic_number': atomic_number,
        'atomic_mass': mass,
        'period': get_period(atomic_number),
        'group': get_group(atomic_number)
    }
    
    # Add more properties if ASE or pymatgen available
    if HAS_PYMATGEN:
        try:
            elem = Element.from_Z(atomic_number)
            properties.update({
                'name': elem.name,
                'atomic_radius': elem.atomic_radius,
                'electronegativity': elem.X,
                'electron_configuration': elem.electronic_structure
            })
        except:
            pass
    
    return properties


def get_period(atomic_number: int) -> int:
    """Get period (row) in periodic table."""
    if atomic_number <= 2:
        return 1
    elif atomic_number <= 10:
        return 2
    elif atomic_number <= 18:
        return 3
    elif atomic_number <= 36:
        return 4
    elif atomic_number <= 54:
        return 5
    elif atomic_number <= 86:
        return 6
    else:
        return 7


def get_group(atomic_number: int) -> Optional[int]:
    """Get group (column) in periodic table."""
    # Simplified group assignment
    noble_gases = [2, 10, 18, 36, 54, 86, 118]
    if atomic_number in noble_gases:
        return 18
    
    # This is a simplified version - real implementation would be more complex
    return None


def print_library_status():
    """Print status of available chemical libraries."""
    print("Chemical Library Status:")
    print(f"  ASE available: {HAS_ASE}")
    print(f"  pymatgen available: {HAS_PYMATGEN}")
    
    if not HAS_ASE and not HAS_PYMATGEN:
        warnings.warn(
            "Neither ASE nor pymatgen is available. "
            "Install one of them for better chemical data support:\n"
            "  pip install ase\n"
            "  pip install pymatgen"
        )


# Example usage and testing
if __name__ == "__main__":
    print_library_status()
    
    # Test basic functionality
    print(f"\nElement tests:")
    elements = ['H', 'He', 'Li', 'C', 'N', 'O', 'Fe', 'Au']
    
    for elem in elements:
        try:
            atomic_num = get_atomic_number(elem)
            symbol = get_atomic_symbol(atomic_num)
            mass = get_atomic_mass(elem)
            print(f"  {elem}: Z={atomic_num}, symbol={symbol}, mass={mass:.3f} amu")
        except Exception as e:
            print(f"  {elem}: Error - {e}")
    
    # Test formula parsing
    print(f"\nFormula tests:")
    formulas = ['H2O', 'CO2', 'CaCO3', 'C6H12O6']
    for formula in formulas:
        try:
            composition = parse_chemical_formula(formula)
            weight = get_formula_weight(formula)
            print(f"  {formula}: {composition}, MW={weight:.3f} amu")
        except Exception as e:
            print(f"  {formula}: Error - {e}")

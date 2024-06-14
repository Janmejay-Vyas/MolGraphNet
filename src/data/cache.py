"""
TODO: Add docstring
"""

from rdkit import Chem
from typing import Dict

# Initialize the caches for graphs and RDKit molecule objects
CACHE_GRAPH = True
SMILES_TO_GRAPH = {}  # Cache to store pre-computed graph objects from SMILES


def cache_graph():
    """Check if graph caching is enabled."""
    return CACHE_GRAPH


def set_cache_graph(cache_graph: bool):
    """Enable or disable graph caching."""
    global CACHE_GRAPH
    CACHE_GRAPH = cache_graph


CACHE_MOL = True
SMILES_TO_MOL: Dict[str, Chem.Mol] = {}  # Cache to store RDKit molecule objects from SMILES


def cache_mol() -> bool:
    """Check if molecule caching is enabled."""
    return CACHE_MOL


def set_cache_mol(cache_mol: bool):
    """Enable or disable molecule caching."""
    global CACHE_MOL
    CACHE_MOL = cache_mol

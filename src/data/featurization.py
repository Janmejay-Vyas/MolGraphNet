"""
TODO: Add docstring
"""

# Importing the necessary libraries
from rdkit import Chem
from typing import List, Dict

# Define constant features for atoms and bonds in molecules
MAX_ATOMIC_NUM = 100  # Maximum number of atomic numbers to consider for encoding
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],  # Degrees of connectivity for an atom
    'formal_charge': [-1, -2, 1, 2, 0],  # Possible formal charges on an atom
    'chiral_tag': [0, 1, 2, 3],  # Chirality tags
    'num_Hs': [0, 1, 2, 3, 4],  # Number of hydrogen atoms attached
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],  # Hybridization states
}

PATH_DISTANCE_BINS = list(range(10))  # Binning for path distances in the molecular graph
THREE_D_DISTANCE_MAX = 20  # Maximum distance for 3D spatial features
THREE_D_DISTANCE_STEP = 1  # Step size for incrementing distance
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
EXTRA_ATOM_FDIM = 0
BOND_FDIM = 14


def get_atom_fdim() -> int:
    """Calculate the dimensionality of atom features."""
    return sum(
        len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2  # Additional features for aromaticity and mass


def get_bond_fdim(atom_messages=False) -> int:
    """Calculate the dimensionality of bond features."""
    return BOND_FDIM + (not atom_messages) * get_atom_fdim()


def onek_encoding_unk(value: int, choices: List[int]):
    """One-hot encode a value with a provision for unknown values."""
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None):
    """Build a feature vector for an atom based on its chemical properties."""
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
               onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
               onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
               onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
               onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
               onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
               [1 if atom.GetIsAromatic() else 0] + \
               [atom.GetMass() * 0.01]  # Mass is scaled for feature consistency
    if functional_groups is not None:
        features += functional_groups
    return features


def bond_features(bond: Chem.rdchem.Bond):
    """Build a feature vector for a bond based on its chemical properties."""
    if bond is None:
        return [1] + [0] * (BOND_FDIM - 1)  # Handle the case of non-existent bonds
    else:
        bt = bond.GetBondType()
        fbond = [
                    0,  # Bond is not None
                    bt == Chem.rdchem.BondType.SINGLE,
                    bt == Chem.rdchem.BondType.DOUBLE,
                    bt == Chem.rdchem.BondType.TRIPLE,
                    bt == Chem.rdchem.BondType.AROMATIC,
                    bond.GetIsConjugated() if bt is not None else 0,
                    bond.IsInRing() if bt is not None else 0
                ] + onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
        return fbond

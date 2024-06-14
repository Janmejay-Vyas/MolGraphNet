"""
TODO: Add docstring
"""

# Importing the necessary libraries
import numpy as np
import torch
from torch.utils.data import Dataset


class MoleculeDatapoint:
    """
    Represents a single molecule's data including features, targets, and its SMILES string.
    """

    def __init__(self, smiles: str, targets: List[Optional[float]] = None, row: OrderedDict = None):
        """
        Initializes a MoleculeDatapoint instance.
        Args:
            smiles (str): The SMILES string representing the molecule.
            targets (List[Optional[float]]): The target properties to predict.
            row (OrderedDict): Row from the dataset containing additional data.
        """
        self.smiles = smiles
        self.targets = targets
        self.features = []
        self.row = row

    @property
    def mol(self) -> Chem.Mol:
        """
        Lazy loads and returns the RDKit molecule object corresponding to the SMILES string.
        Caches the molecule if caching is enabled.
        """
        mol = SMILES_TO_MOL.get(self.smiles, Chem.MolFromSmiles(self.smiles))
        if cache_mol():
            SMILES_TO_MOL[self.smiles] = mol
        return mol

    def set_features(self, features: np.ndarray) -> None:
        """Sets the molecule's features."""
        self.features = features

    def extend_features(self, features: np.ndarray) -> None:
        """Appends additional features to the molecule's feature array."""
        self.features = np.append(self.features, features) if self.features is not None else features

    def num_tasks(self) -> int:
        """Returns the number of prediction tasks based on the targets list."""
        return len(self.targets)

    def set_targets(self, targets: List[Optional[float]]):
        """Sets the prediction targets for the molecule."""
        self.targets = targets

    def reset_features_and_targets(self) -> None:
        """Resets the features and targets to their original values."""
        self.features, self.targets = self.raw_features, self.raw_targets


class MoleculeDataset(Dataset):
    """
    PyTorch dataset class for handling collections of MoleculeDatapoints.
    """

    def __init__(self, data: List[MoleculeDatapoint]):
        """
        Initializes a MoleculeDataset instance.
        Args:
            data (List[MoleculeDatapoint]): List of molecule data points.
        """
        self._data = data
        self._scaler = None
        self._batch_graph = None
        self._random = Random()

    def smiles(self) -> List[str]:
        """Returns a list of SMILES strings for all molecules in the dataset."""
        return [d.smiles for d in self._data]

    def mols(self) -> List[Chem.Mol]:
        """Returns a list of RDKit molecule objects for all molecules in the dataset."""
        return [d.mol for d in self._data]

    def targets(self) -> List[List[Optional[float]]]:
        """Returns a list of target properties for all molecules in the dataset."""
        return [d.targets for d in self._data]

    def num_tasks(self) -> int:
        """Returns the number of tasks based on the first data point, assuming uniformity."""
        return self._data[0].num_tasks() if len(self._data) > 0 else None

    def set_targets(self, targets: List[List[Optional[float]]]) -> None:
        """
        Sets the target properties for each molecule in the dataset.
        Args:
            targets (List[List[Optional[float]]]): Nested list of target values.
        """
        assert len(self._data) == len(targets)
        for i in range(len(self._data)):
            self._data[i].set_targets(targets[i])

    def reset_features_and_targets(self) -> None:
        """Resets features and targets for all molecules in the dataset to their original values."""
        for d in self._data:
            d.reset_features_and_targets()

    def __len__(self) -> int:
        """Returns the number of molecules in the dataset."""
        return len(self._data)

    def __getitem__(self, item) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        """Returns the molecule data point or a list of data points indexed by item."""
        return self._data[item]

    def batch_graph(self):
        """
        Prepares and caches the graph representation for a batch of molecules, if not already cached.
        """
        if self._batch_graph is None:
            self._batch_graph = []
            mol_graphs = []
            for d in self._data:
                mol_graphs_list = []
                if d.smiles in SMILES_TO_GRAPH:
                    mol_graph = SMILES_TO_GRAPH[d.smiles]
                else:
                    mol_graph = MolGraph(d.mol)
                    if cache_graph():
                        SMILES_TO_GRAPH[d.smiles] = mol_graph
                mol_graphs.append([mol_graph])
            self._batch_graph = [BatchMolGraph([g[i] for g in mol_graphs]) for i in range(len(mol_graphs[0]))]
        return self._batch_graph

    def features(self) -> List[np.ndarray]:
        """
        Returns the features associated with each molecule (if they exist).

        :return: A list of 1D numpy arrays containing the features for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].features is None:
            return None

        return [d.features for d in self._data]

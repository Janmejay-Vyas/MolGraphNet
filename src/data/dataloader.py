"""
TODO: Add docstring
"""

# Importing the necessary libraries
import torch
from torch.utils.data import DataLoader, Sampler
from data.dataset import MoleculeDataset


def construct_molecule_batch(data):
    """
    Prepare and process a batch of molecular data for use in a DataLoader.

    This function takes a list of data points, wraps them into a MoleculeDataset object,
    and ensures that all necessary graph computations for the batch are performed in advance.

    Args:
        data (list): A list of molecular data points.

    Returns:
        MoleculeDataset: A dataset object with precomputed graph representations.
    """
    data = MoleculeDataset(data)
    data.batch_graph()  # Ensures all molecular graphs are processed and ready for model input
    return data


class MoleculeSampler(Sampler):
    """
    Custom sampler for molecular datasets, supporting optional shuffling.

    Attributes:
        dataset (Dataset): The dataset from which to sample.
        shuffle (bool): Whether to shuffle the data every epoch.
        _random (Random): Random number generator for shuffling.
        length (int): Number of items in the dataset.

    Methods:
        __iter__: Returns an iterator over the dataset indices.
        __len__: Returns the number of items in the dataset.
    """

    def __init__(self, dataset, shuffle=False, seed=0):
        super(Sampler, self).__init__()
        self.dataset = dataset
        self.shuffle = shuffle
        self._random = Random(seed)  # Random generator for shuffling
        self.length = len(self.dataset)  # Store dataset length for easy access

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            self._random.shuffle(indices)  # Shuffle indices if required
        return iter(indices)  # Return iterator over indices

    def __len__(self):
        return self.length  # Return the length of the dataset


class MoleculeDataLoader(DataLoader):
    """
    DataLoader for handling molecular data with specific needs like batch processing and optional shuffling.

    Extends PyTorch DataLoader, configuring it specifically for molecular datasets using a custom sampler
    and data preparation function.

    Attributes:
        _dataset (MoleculeDataset): The dataset to load.
        _batch_size (int): Number of items per batch.
        _num_workers (int): Number of subprocesses to use for data loading.
        _shuffle (bool): Whether to shuffle the data at every epoch.
        _seed (int): Random seed for shuffling.
        _context (str, optional): Multiprocessing context, set to 'forkserver' if not the main thread to avoid hangs.
        _timeout (int): Timeout for collecting a batch from workers.
        _sampler (Sampler): Custom sampler to manage the sampling of data indices based on shuffling and seeding.

    Methods:
        targets: Property that attempts to return targets from the dataset unless shuffling or class balance is active.
        iter_size: Property returning the size of an iterator over the dataset.
        __iter__: Provides an iterator over batches of data.
    """

    def __init__(self, dataset, batch_size=50, num_workers=8, shuffle=False, seed=0):
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._shuffle = shuffle
        self._seed = seed
        is_main_thread = threading.current_thread() is threading.main_thread()

        if not is_main_thread and self._num_workers > 0:
            self._context = 'forkserver'  # Use forkserver to avoid issues in non-main threads
            self._timeout = 3600  # Set a long timeout to handle possible delays in data loading

        self._sampler = MoleculeSampler(dataset=self._dataset, shuffle=self._shuffle, seed=self._seed)

        super(MoleculeDataLoader, self).__init__(
            dataset=self._dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=construct_molecule_batch,
            multiprocessing_context=self._context,
            timeout=self._timeout
        )

    @property
    def targets(self):
        """
        Retrieves targets from the dataset unless shuffle is enabled, in which case raising an error.

        Returns:
            List[List[Optional[float]]]: List of target values from the dataset.

        Raises:
            ValueError: If data shuffling is enabled, as target retrieval might not be safe.
        """
        if self._class_balance or self._shuffle:
            raise ValueError('Cannot safely extract targets when class balance or shuffle are enabled.')
        return [self._dataset[index].targets for index in self._sampler]

    @property
    def iter_size(self):
        """Returns the number of items that can be iterated over."""
        return len(self._sampler)

    def __iter__(self):
        """Provides an iterator over the batches of data."""
        return super(MoleculeDataLoader, self).__iter__()
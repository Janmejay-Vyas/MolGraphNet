"""
TODO: Add docstring
"""

import pandas as pd
from random import Random
from .dataset import MoleculeDataset, MoleculeDatapoint


def load_and_prepare_data(filepath):
    # Load the dataset
    df = pd.read_csv(filepath)

    # Prepare dataset
    data = MoleculeDataset([
        MoleculeDatapoint(
            smiles=row['SMILES'],
            targets=[row['ClogP']]
        ) for index, row in df.iterrows()
    ])

    # Split data into training, validation, and testing sets
    random = Random(42)  # Fixed seed for reproducibility
    indices = list(range(len(data)))
    random.shuffle(indices)

    train_size = int(0.8 * len(data))
    train_val_size = int(0.9 * len(data))

    train = [data[i] for i in indices[:train_size]]
    val = [data[i] for i in indices[train_size:train_val_size]]
    test = [data[i] for i in indices[train_val_size:]]

    # Wrap in MoleculeDataset for DataLoader compatibility
    train_data = MoleculeDataset(train)
    val_data = MoleculeDataset(val)
    test_data = MoleculeDataset(test)

    return train_data, val_data, test_data


def main():
    train_data, val_data, test_data = load_and_prepare_data('../data/enamine_discovery_diversity_set_10240.csv')


if __name__ == '__main__':
    main()

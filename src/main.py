"""
TODO: Add docstring
"""

import argparse
from config import TrainArgs
from training.train import train_model


def main():
    args = TrainArgs()
    args.data_path = '../data/enamine_discovery_diversity_set_10240.csv'
    args.target_column = 'ClogP'
    args.smiles_column = 'SMILES'
    args.dataset_type = 'regression'
    args.task_names = [args.target_column]
    args.num_tasks = 1

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'evaluate'], help='Mode to run the script in')
    args = parser.parse_args()

    train_model(args)


if __name__ == '__main__':
    main()

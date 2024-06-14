"""
TODO: Add docstring
"""

import torch


# Define a class to store training arguments and settings
class TrainArgs:
    """Container for training parameters and model configuration."""
    smiles_column = None  # Column name for SMILES strings, to be set later
    num_workers = 8  # Number of worker threads for loading data
    batch_size = 50  # Batch size for training
    no_cache_mol = False  # Toggle whether to cache molecule objects
    dataset_type = 'regression'  # Type of dataset/task, regression in this case
    task_names = []  # Names of the tasks, to be populated based on the dataset
    seed = 0  # Random seed for reproducibility
    hidden_size = 300  # Hidden layer size for the neural network
    bias = False  # Use bias in neural network layers
    depth = 3  # Depth of the network (number of layers)
    dropout = 0.0  # Dropout rate for training
    undirected = False  # Whether the molecular graph is undirected
    aggregation = 'mean'  # Method for aggregating node information
    aggregation_norm = 100  # Normalization factor for aggregation
    ffn_num_layers = 2  # Number of layers in the feed-forward network
    ffn_hidden_size = 300  # Hidden layer size of the feed-forward network
    init_lr = 1e-4  # Initial learning rate
    max_lr = 1e-3  # Maximum learning rate
    final_lr = 1e-4  # Final learning rate
    num_lrs = 1  # Number of different learning rates to use
    warmup_epochs = 2.0  # Number of epochs to linearly increase the learning rate
    epochs = 30  # Total number of epochs to train
    device = torch.device('cpu')  # Device to run the model on, default to CPU

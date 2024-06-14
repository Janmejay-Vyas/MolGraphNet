"""
TODO: Add docstring
"""
# Importing the necessary libraries
import torch
from torch import nn
from mpn import MPNEncoder


class MolGraphNet(nn.Module):
    """
    The primary model for molecular property prediction, integrating the MPN encoder and a fully connected feed-forward network (FFN) for final prediction.

    Attributes:
        classification (bool): Indicates if the model is used for classification tasks.
        featurizer (bool): Flag to determine if the model should act as a featurizer, i.e., outputting features rather than predictions.
        output_size (int): The number of output targets or classes.
        sigmoid (nn.Module, optional): Sigmoid activation, used in the output layer for classification tasks.
        encoder (MPN): An instance of the MPN class, which encodes molecular graphs.
        ffn (nn.Sequential): The feed-forward network that processes encoded molecular features into final predictions.

    Methods:
        create_encoder(args): Initializes the MPN encoder.
        create_ffn(args): Constructs the feed-forward layers based on the provided arguments.
        featurize(batch, features_batch=None, atom_descriptors_batch=None): Generates features from the input batch using the model.
        forward(batch): Processes input through the encoder and FFN to produce predictions or features.
    """

    def __init__(self, args, featurizer=False):
        super(MoleculeModel, self).__init__()
        self.classification = args.dataset_type == 'classification'
        self.featurizer = featurizer
        self.output_size = args.num_tasks

        if self.classification:
            self.sigmoid = nn.Sigmoid()  # Activation function for binary classification output

        # Initialize the components of the model
        self.create_encoder(args)
        self.create_ffn(args)
        initialize_weights(self)  # Custom function to initialize model weights

    def create_encoder(self, args):
        """
        Initializes the encoder component of the model using the MPN architecture.
        """
        self.encoder = MPN(args)

    def create_ffn(self, args):
        """
        Constructs the feed-forward network using the specifications provided in args.

        Args:
            args: Configuration arguments containing network settings.
        """
        first_linear_dim = args.hidden_size
        dropout = nn.Dropout(args.dropout)
        activation = nn.ReLU()

        # Build the FFN architecture dynamically based on the number of layers specified
        if args.ffn_num_layers == 1:
            ffn = [dropout, nn.Linear(first_linear_dim, self.output_size)]
        else:
            ffn = [dropout, nn.Linear(first_linear_dim, args.ffn_hidden_size)]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([activation, dropout, nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size)])
            ffn.extend([activation, dropout, nn.Linear(args.ffn_hidden_size, self.output_size)])

        self.ffn = nn.Sequential(*ffn)  # Wrap the list of layers into a Sequential container

    def featurize(self, batch, features_batch=None, atom_descriptors_batch=None):
        """
        Generates feature vectors from the input batch, omitting the final prediction layer.

        Args:
            batch: The batch of data to process.
            features_batch: Optional additional features to be included.
            atom_descriptors_batch: Optional atomic descriptors to be included.

        Returns:
            Tensor: The feature vectors for the batch.
        """
        return self.ffn[:-1](self.encoder(batch))  # Exclude the last layer (prediction layer)

    def forward(self, batch):
        """
        Defines the forward pass through the model.

        Args:
            batch: The batch of molecular graphs to be processed.

        Returns:
            Tensor: The output predictions or class probabilities.
        """
        output = self.ffn(self.encoder(batch))
        if self.classification and not self.training:
            output = self.sigmoid(output)  # Apply sigmoid for classification to get probabilities
        return output

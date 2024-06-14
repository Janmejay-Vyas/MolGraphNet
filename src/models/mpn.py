"""
TODO: Add doc-string
"""
# Importing the necessary libraries
import torch
from torch import nn
from utils.training_utils import index_select_ND


class MPNEncoder(nn.Module):
    """
    A PyTorch module for encoding molecular graphs using a Message Passing Neural Network (MPN).
    This encoder uses a combination of linear transformations and non-linear activations to process
    the graph structure of molecules for downstream prediction tasks.

    Attributes:
        atom_fdim (int): The feature dimension of each atom.
        bond_fdim (int): The feature dimension of each bond.
        hidden_size (int): The size of the hidden layers.
        bias (bool): Whether to add a bias term in the linear transformations.
        depth (int): The number of message passing iterations.
        dropout (float): Dropout rate for regularization during training.
        layers_per_message (int): Number of layers per message passing iteration (currently set statically to 1).
        undirected (bool): Whether the molecular graph is undirected (currently not used).
        atom_messages (bool): Whether messages are being passed at the atom level (currently not used).
        device (torch.device): The device (CPU or GPU) on which computations will be performed.
        aggregation (str): The method of aggregating messages ('mean', 'sum', 'norm').
        aggregation_norm (int): Normalization constant used when aggregation is 'norm'.

    Methods:
        forward(mol_graph): Processes a batch of molecular graphs and returns their vector representations.
    """

    def __init__(self, args, atom_fdim, bond_fdim):
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = False
        self.atom_messages = False
        self.device = args.device
        self.aggregation = args.aggregation
        self.aggregation_norm = args.aggregation_norm

        # Layers
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.act_func = nn.ReLU()
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Weight matrices for input and hidden layers
        self.W_i = nn.Linear(self.bond_fdim, self.hidden_size, bias=self.bias)
        self.W_h = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

    def forward(self, mol_graph):
        """
        Forward pass of the MPNEncoder.

        Args:
            mol_graph (BatchMolGraph): A batch of molecular graphs to encode.

        Returns:
            torch.Tensor: A tensor of molecular vectors, each representing a molecule in the batch.
        """
        # Extract graph components
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()
        f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.to(self.device), f_bonds.to(self.device), a2b.to(
            self.device), b2a.to(self.device), b2revb.to(self.device)

        # Initial message passing
        input = self.W_i(f_bonds)
        message = self.act_func(input)

        # Iterative message passing
        for depth in range(self.depth - 1):
            nei_a_message = index_select_ND(message, a2b)
            a_message = nei_a_message.sum(dim=1)
            rev_message = message[b2revb]
            message = self.W_h(a_message[b2a] - rev_message)
            message = self.act_func(input + message)
            message = self.dropout_layer(message)

        # Aggregating messages to atom vectors
        nei_a_message = index_select_ND(message, a2b)
        a_message = nei_a_message.sum(dim=1)
        a_input = torch.cat([f_atoms, a_message], dim=1)
        atom_hiddens = self.act_func(self.W_o(a_input))
        atom_hiddens = self.dropout_layer(atom_hiddens)

        # Readout phase to combine atom vectors into molecule vectors
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens
                if self.aggregation == 'mean':
                    mol_vec = mol_vec.sum(dim=0) / a_size
                elif self.aggregation == 'sum':
                    mol_vec = mol_vec.sum(dim=0)
                elif self.aggregation == 'norm':
                    mol_vec = mol_vec.sum(dim=0) / self.aggregation_norm
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)

        return mol_vecs


class MPN(nn.Module):
    """
    Message Passing Network (MPN) for encoding molecular graphs into a vector representation.

    This class wraps the MPNEncoder and provides an interface to process batches of molecular graphs, converting them
    into fixed-size embeddings that can be used for downstream tasks like property prediction.

    Attributes:
        atom_fdim (int): The feature dimension for atoms, automatically determined if not provided.
        bond_fdim (int): The feature dimension for bonds, automatically determined if not provided.
        device (torch.device): The device on which to perform calculations (CPU or GPU).
        encoder (MPNEncoder): The encoder that performs the message passing and encoding logic.

    Methods:
        forward(batch): Processes a batch of molecules through the encoder and aggregates the results.
    """

    def __init__(self, args, atom_fdim=None, bond_fdim=None):
        """
        Initializes the MPN model with the necessary settings and sub-models.

        Args:
            args: Configuration arguments which contain settings like the device and dimensions.
            atom_fdim (int, optional): The feature dimension for atoms. If None, uses a default function.
            bond_fdim (int, optional): The feature dimension for bonds. If None, uses a default function.
        """
        super(MPN, self).__init__()
        self.atom_fdim = atom_fdim or get_atom_fdim()  # Set atom feature dimension
        self.bond_fdim = bond_fdim or get_bond_fdim()  # Set bond feature dimension
        self.device = args.device  # Device configuration
        self.encoder = MPNEncoder(args, self.atom_fdim, self.bond_fdim)  # Initialize the encoder

    def forward(self, batch):
        """
        Forward pass through the network which processes a batch of molecular data.

        Args:
            batch (list): A batch of data that needs to be processed. Each element can be a molecule graph or data structure that the encoder can process.

        Returns:
            torch.Tensor: A tensor containing the encoded representations of the batch.
        """
        # Ensure that each item in the batch is a MolGraph; if not, convert it
        if type(batch[0]) != BatchMolGraph:
            batch = [mol2graph(b) for b in batch]  # Convert data to MolGraph if necessary

        # Encode the batch using the MPNEncoder
        encodings = [self.encoder(batch[0])]
        # Combine encodings if there are multiple (this example assumes a single encoding for simplicity)
        output = reduce(lambda x, y: torch.cat((x, y), dim=1), encodings)
        return output

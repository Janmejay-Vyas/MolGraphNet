"""
TODO: Add docstring
"""

import torch
import torch.nn as nn


def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Selects the message features from source corresponding to the atom or bond indices in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)
    return target


def initialize_weights(model: nn.Module) -> None:
    """Initializes the weights of a model in place.
    """
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
"""
TODO: Add docstring
"""
# Importing the necessary libraries
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler


class NoamLR(_LRScheduler):
    """
    Implements a learning rate scheduler that adjusts the learning rate according to the Noam scheme.
    It starts with a linear warm-up phase, followed by an exponential decay.

    Attributes:
        optimizer (Optimizer): The optimizer for which to adjust the learning rate.
        warmup_epochs (list of floats or ints): Number of epochs during which the learning rate increases.
        total_epochs (list of ints): Total number of epochs for training.
        steps_per_epoch (int): Number of batches (steps) per epoch.
        init_lr (list of floats): Initial learning rates for each parameter group.
        max_lr (list of floats): Maximum learning rates during the warm-up.
        final_lr (list of floats): Final learning rates after decay.
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs, steps_per_epoch, init_lr, max_lr, final_lr):
        """
        Initializes the NoamLR scheduler.

        Args:
            optimizer (Optimizer): Bound optimizer.
            warmup_epochs (list of float|int): Epochs to linearly increase the learning rate.
            total_epochs (list of int): Total duration of training to adjust learning rate.
            steps_per_epoch (int): Number of optimizer updates per epoch.
            init_lr (list of float): Initial learning rates for each parameter group.
            max_lr (list of float): Peak learning rates to reach after warmup.
            final_lr (list of float): Learning rates to decay towards by the end of training.
        """
        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == \
               len(init_lr) == len(max_lr) == len(final_lr), "Length of constructor arguments must match."

        self.num_lrs = len(optimizer.param_groups)
        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps
        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self):
        """
        Computes and returns the current learning rate for each parameter group.
        """
        return list(self.lr)

    def step(self, current_step=None):
        """
        Update the learning rate after each batch iteration.

        Args:
            current_step (int, optional): Optionally specify the current training step. If not provided,
            the internal step counter is used and incremented.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]

"""
TODO: add docstrings
"""

import torch
from torch.optim import Adam
from torch.nn import MSELoss
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.dataloader import MoleculeDataLoader
from models.mol_graph_net import MolGraphNet
from utils.scheduler import NoamLR
from utils.training_utils import initialize_weights


def setup_training(args, train_data, val_data, test_data):
    # Data loaders
    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True,
        seed=args.seed
    )

    val_data_loader = MoleculeDataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False
    )

    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False
    )

    # Model
    model = MolGraphNet(args)
    initialize_weights(model)

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.init_lr, weight_decay=0)
    scheduler = NoamLR(
        optimizer=optimizer,
        warmup_epochs=[args.warmup_epochs],
        total_epochs=[args.epochs] * args.num_lrs,
        steps_per_epoch=len(train_data) // args.batch_size,
        init_lr=[args.init_lr],
        max_lr=[args.max_lr],
        final_lr=[args.final_lr]
    )

    # Loss function
    loss_func = MSELoss(reduction='none')

    return train_data_loader, val_data_loader, test_data_loader, model, optimizer, scheduler, loss_func


def train_model(args, model, train_data_loader, val_data_loader, optimizer, scheduler, loss_func):
    for epoch in tqdm(range(args.epochs), desc='Training Epochs'):
        model.train()
        loss_sum = iter_count = 0

        for batch in tqdm(train_data_loader, total=len(train_data_loader), leave=False, desc='Training Batches'):
            mol_batch, target_batch = batch.batch_graph(), batch.targets()
            mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
            targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])

            model.zero_grad()
            preds = model(mol_batch)

            mask = mask.to(preds.device)
            targets = targets.to(preds.device)
            class_weights = torch.ones(targets.shape, device=preds.device)

            loss = loss_func(preds, targets) * class_weights * mask
            loss = loss.sum() / mask.sum()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_sum += loss.item()
            iter_count += 1

        model.eval()
        preds_list = []
        targets_list = []

        for batch in tqdm(val_data_loader, total=len(val_data_loader), leave=False, desc='Validation Batches'):
            mol_batch = batch.batch_graph()
            with torch.no_grad():
                batch_preds = model(mol_batch).data.cpu().numpy()

            batch_targets = [batch.targets[i] for i in range(len(batch_preds)) if batch.targets[i] is not None]
            preds_list.extend(batch_preds)
            targets_list.extend(batch_targets)

        valid_preds, valid_targets = zip(
            *[(pred, target) for pred, target in zip(preds_list, targets_list) if target is not None])
        mse = mean_squared_error(valid_targets, valid_preds)
        print(f'Epoch {epoch}: MSE = {mse}')

        # Plotting results
        plt.figure(figsize=(4, 4))
        plt.scatter(valid_targets, valid_preds, alpha=0.6)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Validation Predictions vs True Values')
        plt.grid(True)
        plt.savefig("../../plots/")


def main(args):
    train_data, val_data, test_data = setup_data() # TODO: setup_data is a placeholder, change this
    train_data_loader, val_data_loader, test_data_loader, model, optimizer, scheduler, loss_func = setup_training(args,
                                                                                                                  train_data,
                                                                                                                  val_data,
                                                                                                                  test_data)
    train_model(args, model, train_data_loader, val_data_loader, optimizer, scheduler, loss_func)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Training and evaluation script for molecular property prediction.")
    # Add other command-line arguments as needed
    args = parser.parse_args()
    main(args)

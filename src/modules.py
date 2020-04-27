"""Base PytorchLightningModules for extending"""
from typing import List
import torch
import pytorch_lightning as pl


class CollectOnEpochEnd(pl.LightningModule):
    """
    Here, I want to have all reporting logic in callback classes.
    However, the methods on the callback classes need to know targets and predictions
    for validation and training rounds in order to calculate metrics.
    Therefore, I'm overriding the `*_epoch_end` hooks to collect targets and predictions
    for training and validation after each epochs.
    The resulting containers are then used by the callbacks.
    
    In order for this to work the training and validation steps of the final module
    need to return `loss`, `preds`, and `targets`.
    
    :Example:

        def training_step(self, batch, batch_idx: int) -> dict:
            loss, preds = self._forward_pass(*batch)
            return {'loss': loss, 'preds': preds, 'targets': batch[1]}
    """

    def __init__(self):
        super(CollectOnEpochEnd, self).__init__()
        self.val_targets = torch.tensor([])
        self.val_predictions = torch.tensor([])
        self.val_loss = torch.tensor([])
        self.train_targets = torch.tensor([])
        self.train_predictions = torch.tensor([])
        self.train_loss = torch.tensor([])

    def validation_epoch_end(self, val_steps: List[dict]) -> dict:
        self.val_targets = torch.cat([d['targets'] for d in val_steps], dim=0)
        self.val_predictions = torch.cat([d['preds'] for d in val_steps], dim=0)
        self.val_loss = torch.stack([d['loss'] for d in val_steps]).sum() / len(self.val_targets)
        return {'val_loss': self.val_loss}

    def training_epoch_end(self, train_steps: List[dict]) -> dict:
        self.train_targets = torch.cat([d['targets'] for d in train_steps], dim=0)
        self.train_predictions = torch.cat([d['preds'] for d in train_steps], dim=0)
        self.train_loss = torch.stack([d['loss'] for d in train_steps]).sum() / len(self.train_targets)
        return {'train_loss': self.train_loss}

"""Base PytorchLightningModules for extending"""
from __future__ import annotations
from enum import Enum
from typing import List, Dict
import torch
import pytorch_lightning as pl


class Partition(Enum):
    """Since I'm always using those"""
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


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


class MetricsOnEpochEnd(pl.LightningModule):
    """
    Here, I decided to give the metrics map to the module directly.
    Again, the metrics need train/val targets and predictions.
    But in contrast to `CollectOnEpochEnd`, I can calculate them in the `*_epoch_end` hooks directly.
    Thus, I don't need to collect targets and prediction in separate attributes anymore.

    Again the training and validation steps of the final module
    need to return `loss`, `preds`, and `targets`.
    
    :Example:

        def training_step(self, batch, batch_idx: int) -> dict:
            loss, preds = self._forward_pass(*batch)
            return {'loss': loss, 'preds': preds, 'targets': batch[1]}
    """

    def validation_epoch_end(self, val_steps: List[dict]) -> dict:
        return self._get_epoch_results(val_steps, 'val')

    def training_epoch_end(self, train_steps: List[dict]) -> dict:
        return self._get_epoch_results(train_steps, 'train')

    def _get_epoch_results(self, steps: List[dict], partition: str) -> dict:
        targets = torch.cat([d['targets'] for d in steps], dim=0)
        predictions = torch.cat([d['preds'] for d in steps], dim=0)
        loss = torch.stack([d['loss'] for d in steps]).sum() / len(targets)

        log = {f'loss/{partition}': loss}
        for name, metric in self.metrics.items():
            log[f'{name}/{partition}'] = metric(targets, predictions)

        return {f'{partition}_loss': loss, 'log': log}


class EpochSummary:
    """Container for collecting epoch-wise results"""

    class Results:
        def __init__(self, targets: torch.Tensor, predictions: torch.Tensor, loss: float):
            self.targets = targets
            self.predictions = predictions
            self.loss = loss

    def __init__(self):
        self.summaries: Dict[str, EpochSummary.Results] = {}

    def set_results(self, name: Partition, steps: List[dict]) -> float:
        EpochSummary._validate_steps(steps)
        targets = torch.cat([d['targets'] for d in steps], dim=0)
        predictions = torch.cat([d['predictions'] for d in steps], dim=0)
        loss = float((torch.stack([d['loss'] for d in steps]).sum() / len(targets)).cpu().numpy())
        self.summaries[name] = EpochSummary.Results(targets, predictions, loss)
        return loss
    
    def get_results(self, name: Partition) -> EpochSummary.Results:
        return self.summaries[name]

    @staticmethod
    def _validate_steps(steps: dict):
        for step in steps:
            assert len(step['targets']) == len(step['predictions'])
            assert step['loss'].dim() == 0


class SummarizeEpochs(pl.LightningModule):
    """
    Basically the same as CollectOnEpochEnd.
    I want to collect everything needed for calculating the metrics, so that I can do all
    the metrics reporting in callbacks.
    Here, I cleaned up the module a bit by using a container class.
    
    In order for this to work the training and validation steps of the final module
    need to return `loss`, `predictions`, and `targets`.
    
    :Example:

        def training_step(self, batch, batch_idx: int) -> dict:
            loss, preds = self._forward_pass(*batch)
            return {'loss': loss, 'predictions': preds, 'targets': batch[1]}
    """
    # TODO: test_epoch_end ?

    def __init__(self):
        super(SummarizeEpochs, self).__init__()
        self.epoch_summary = EpochSummary()

    def validation_epoch_end(self, val_steps: List[dict]) -> dict:
        loss = self.epoch_summary.set_results(Partition.VAL, val_steps)
        return {f'{Partition.VAL.value}_loss': loss}

    def training_epoch_end(self, train_steps: List[dict]) -> dict:
        loss = self.epoch_summary.set_results(Partition.TRAIN, train_steps)
        return {f'{Partition.TRAIN.value}_loss': loss}

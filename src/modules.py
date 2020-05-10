"""Base PytorchLightningModules for extending"""
from __future__ import annotations
from typing import List, Dict
import torch
import pytorch_lightning as pl
from ray import tune
from src.dataloading import Partition


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

    def __init__(self):
        super(MetricsOnEpochEnd, self).__init__()
        self.best_epoch = {'best/val-loss': 999.9, 'best/epoch': -1}

    def on_train_end(self):
        #self.logger.log_hyperparams_metrics(self.hparams, self.best_epoch)
        self.logger.close()

    def validation_epoch_end(self, val_steps: List[dict]) -> dict:
        out = self._get_epoch_results(val_steps, 'val')
        self._update_best_epoch(out['val_loss'])
        return out

    def training_epoch_end(self, train_steps: List[dict]) -> dict:
        return self._get_epoch_results(train_steps, 'train')

    def _get_epoch_results(self, steps: List[dict], partition: str) -> dict:
        targets = torch.cat([d['targets'] for d in steps], dim=0)
        predictions = torch.cat([d['preds'] for d in steps], dim=0)
        loss = float((torch.stack([d['loss'] for d in steps]).sum() / len(targets)).cpu().numpy())

        log = {f'loss/{partition}': loss}
        for name, metric in self.metrics.items():
            log[f'{name}/{partition}'] = metric(targets, predictions)

        return {f'{partition}_loss': loss, 'log': log}

    def _update_best_epoch(self, loss: float):
        if loss < self.best_epoch['best/val-loss']:
            self.best_epoch = {'best/val-loss': loss, 'best/epoch': self.current_epoch}
            self.logger.log_hyperparams_metrics(self.hparams, self.best_epoch)
            print(f'New best epoch: {self.best_epoch}')


class MetricsAndBestLossOnEpochEnd(pl.LightningModule):
    """
    This is similar to MetricsOnEpochEnd above.
    I'm giving the metrics map to the module directly
    and use the validation/training_epoch_end hooks to calculate and log metrics.

    In addition, I can log a metric to my hyperparameters.
    For that I write a summary with a `best/val-loss` metric at the beginning of the training.
    This metric is then updated by normal `add_scalar` logs from the `log` key.
    For this to work, I need to use a logger which doesn't log hparams without a metric
    at the beginning of the training.
    I implemented this in the `HyperparamsSummaryTensorBoardLogger` logger.
    Idea from https://github.com/PyTorchLightning/pytorch-lightning/issues/1228#issuecomment-620558981

    Again the training and validation steps of the final module
    need to return `loss`, `preds`, and `targets`.
    
    :Example:

        def training_step(self, batch, batch_idx: int) -> dict:
            loss, preds = self._forward_pass(*batch)
            return {'loss': loss, 'preds': preds, 'targets': batch[1]}
    """

    def __init__(self):
        super(MetricsAndBestLossOnEpochEnd, self).__init__()
        self.best_val_loss = 999.9

    def on_train_start(self):
        self.logger.log_hyperparams_metrics(self.hparams, {'best/val-loss': self.best_val_loss})

    def validation_epoch_end(self, val_steps: List[dict]) -> dict:
        return self._get_epoch_results(val_steps, 'val')

    def training_epoch_end(self, train_steps: List[dict]) -> dict:
        return self._get_epoch_results(train_steps, 'train')

    def _get_epoch_results(self, steps: List[dict], partition: str) -> dict:
        targets = torch.cat([d['targets'] for d in steps], dim=0)
        predictions = torch.cat([d['preds'] for d in steps], dim=0)
        loss = float((torch.stack([d['loss'] for d in steps]).sum() / len(targets)).cpu().numpy())

        log = {f'loss/{partition}': loss}
        if partition == 'val':
            log['best/val-loss'] = self._update_best_loss(loss)
        for name, metric in self.metrics.items():
            log[f'{name}/{partition}'] = metric(targets, predictions)

        return {f'{partition}_loss': loss, 'log': log}

    def _update_best_loss(self, loss: float):
        if loss < self.best_val_loss:
            self.best_val_loss = loss
        return self.best_val_loss


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
    If no test step was run `test_epoch_end` is not called.
    
    :Example:

        def training_step(self, batch, batch_idx: int) -> dict:
            loss, preds = self._forward_pass(*batch)
            return {'loss': loss, 'predictions': preds, 'targets': batch[1]}
    """

    def __init__(self):
        super(SummarizeEpochs, self).__init__()
        self.epoch_summary = EpochSummary()

    def validation_epoch_end(self, val_steps: List[dict]) -> dict:
        loss = self.epoch_summary.set_results(Partition.VAL, val_steps)
        return {f'{Partition.VAL.value}_loss': loss}

    def training_epoch_end(self, train_steps: List[dict]) -> dict:
        loss = self.epoch_summary.set_results(Partition.TRAIN, train_steps)
        return {f'{Partition.TRAIN.value}_loss': loss}

    def test_epoch_end(self, test_steps: List[dict]) -> dict:
        loss = self.epoch_summary.set_results(Partition.TEST, test_steps)
        return {f'{Partition.TEST.value}_loss': loss}


class Tune(pl.LightningModule):
    """
    This is similar to MetricsOnEpochEnd above.
    I'm giving the metrics map to the module directly
    and use the validation/training_epoch_end hooks to calculate and log metrics.

    In addition, I can log a metric to my hyperparameters.
    For that I write a summary with a `best/val-loss` metric at the beginning of the training.
    This metric is then updated by normal `add_scalar` logs from the `log` key.
    For this to work, I need to use a logger which doesn't log hparams without a metric
    at the beginning of the training.
    I implemented this in the `HyperparamsSummaryTensorBoardLogger` logger.
    Idea from https://github.com/PyTorchLightning/pytorch-lightning/issues/1228#issuecomment-620558981

    Again the training and validation steps of the final module
    need to return `loss`, `preds`, and `targets`.
    
    :Example:

        def training_step(self, batch, batch_idx: int) -> dict:
            loss, preds = self._forward_pass(*batch)
            return {'loss': loss, 'preds': preds, 'targets': batch[1]}
    """

    def __init__(self):
        super(Tune, self).__init__()
        self.best_val_loss = 999.9
        self.logs = {}

    def training_epoch_end(self, steps: List[dict]) -> dict:
        targets = torch.cat([d['targets'] for d in steps], dim=0)
        predictions = torch.cat([d['preds'] for d in steps], dim=0)
        loss = float((torch.stack([d['loss'] for d in steps]).sum() / len(targets)).cpu().numpy())

        partition = 'train'
        self.logs[f'{partition}_loss'] = loss
        return {f'{partition}_loss': loss}

    def validation_epoch_end(self, steps: List[dict]) -> dict:
        targets = torch.cat([d['targets'] for d in steps], dim=0)
        predictions = torch.cat([d['preds'] for d in steps], dim=0)
        loss = float((torch.stack([d['loss'] for d in steps]).sum() / len(targets)).cpu().numpy())

        partition = 'val'
        self.logs[f'{partition}_loss'] = loss
        self.logs['best_val_loss'] = self._update_best_loss(loss)
        self.logs['epoch'] = self.current_epoch
        tune.track.log(**self.logs)
        return {f'{partition}_loss': loss}

    def _get_epoch_results(self, steps: List[dict], partition: str) -> dict:
        # TODO: currently not using this method
        # TODO: not implemented metrics
        #targets = torch.cat([d['targets'] for d in steps], dim=0)
        #predictions = torch.cat([d['preds'] for d in steps], dim=0)
        #loss = float((torch.stack([d['loss'] for d in steps]).sum() / len(targets)).cpu().numpy())
        #
        #log = {f'loss/{partition}': loss}
        #if partition == 'val':
        #    log['best/val-loss'] = self._update_best_loss(loss)
        #for name, metric in self.metrics.items():
        #    log[f'{name}/{partition}'] = metric(targets, predictions)
        #tune.track.log(**log)
        pass

    def _update_best_loss(self, loss: float):
        if loss < self.best_val_loss:
            self.best_val_loss = loss
        return self.best_val_loss
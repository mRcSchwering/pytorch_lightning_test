from typing import Dict
import torch
import pytorch_lightning as pl
from src.metrics import Metric


class TrainingProgress(pl.Callback):
    """
    Callback for logging epoch-wise metrics at the end of each epoch.
    
    Train/val loss will always be logged.
    Provide additional metrics in a map from name to class.
    E.g. if the name is `auc`, `auc/train` and `auc/val` will be logged.
    """

    def __init__(self, metrics: Dict[str, Metric]):
        """
        :param metrics: map metric name to Metric class (name will be prepended with train/val)
        """
        super(TrainingProgress, self).__init__()
        self.metrics = {}
        self._init_metrics(metrics)

    def on_epoch_end(self, trainer, module):
        """Gets called by trainer after epoch end"""
        self._log_metrics(module)

    def _init_metrics(self, metrics: Dict[str, Metric]):
        for name, metric in metrics.items():
            self.metrics[name + '/train'] = metric
            self.metrics[name + '/val'] = metric

    def _log_metrics(self, module):
        row = {'loss/train': module.train_loss, 'loss/val': module.val_loss}
        for name, metric in self.metrics.items():
            if '/val' in name:
                row[name] = metric(module.val_targets, module.val_predictions)
            elif '/train' in name:
                row[name] = metric(module.train_targets, module.train_predictions)
        module.logger.log_metrics(row, module.current_epoch)
    

class BestLoss(pl.Callback):

    def __init__(self):
        super(BestLoss, self).__init__()
        self.best_loss = float('inf')

    def on_epoch_end(self, trainer, module):
        """Gets called by trainer after epoch end"""
        current_loss = float(module.val_loss.cpu().numpy())
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            metrics = {'hparam/loss': self.best_loss}
            module.logger.log_hyperparams_metrics(params=module.hparams, metrics=metrics)

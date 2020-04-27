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
    
    #def on_train_end(self, trainer, module):
    #    """Gets called when whole training round (all epochs) is over"""
    #    # TODO: torch.tensorboard doesnt want to show the metric?! just empty...
    #    module.logger.log_hyperparams_metrics(module.hparams, {'hparams/val_loss': self.best_val_loss})

    #def _log_best_val_loss(self, module):
    #    module.logger.log_metrics({'best_loss/val': self.best_val_loss}, module.current_epoch)

    #def _log_loss(self, module) -> float:
    #    """[train/val]_losses must be collected in module.metrics"""
    #    train = torch.stack([d for d in module.metrics['train_losses']]).mean()
    #    val = torch.stack([d for d in module.metrics['val_losses']]).mean()
    #    module.logger.log_metrics({'loss/train': train, 'loss/val': val}, module.current_epoch)
    #    return val

    #def _log_acc(self, module) -> float:
    #    """[train/val]_preds must be collected in module.metrics"""
    #    train = torch.stack([d for d in module.metrics['train_preds']]).mean()
    #    val = torch.stack([d for d in module.metrics['val_preds']]).mean()
    #    module.logger.log_metrics({'acc/train': train, 'acc/val': val}, module.current_epoch)
    #    return val

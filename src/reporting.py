from typing import Dict
import torch
import pytorch_lightning as pl
from src.metrics import Metric


class TrainingProgress(pl.Callback):

    def __init__(self, metrics: Dict[str, Metric]):
        super(Reporter, self).__init__()
        self.metrics = metrics
        self.best_val_loss = None

    def on_epoch_end(self, trainer, module):
        """Gets called by trainer after epoch end"""
        val_loss = self._log_loss(module)
        if self.best_val_loss is None or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss.item()
            self._log_best_val_loss(module)
        if self.report_acc:
            self._log_acc(module)
    
    def on_train_end(self, trainer, module):
        """Gets called when whole training round (all epochs) is over"""
        # TODO: torch.tensorboard doesnt want to show the metric?! just empty...
        module.logger.log_hyperparams_metrics(module.hparams, {'hparams/val_loss': self.best_val_loss})

    def _log_best_val_loss(self, module):
        module.logger.log_metrics({'best_loss/val': self.best_val_loss}, module.current_epoch)

    def _log_loss(self, module) -> float:
        """[train/val]_losses must be collected in module.metrics"""
        train = torch.stack([d for d in module.metrics['train_losses']]).mean()
        val = torch.stack([d for d in module.metrics['val_losses']]).mean()
        module.logger.log_metrics({'loss/train': train, 'loss/val': val}, module.current_epoch)
        return val

    def _log_acc(self, module) -> float:
        """[train/val]_preds must be collected in module.metrics"""
        train = torch.stack([d for d in module.metrics['train_preds']]).mean()
        val = torch.stack([d for d in module.metrics['val_preds']]).mean()
        module.logger.log_metrics({'acc/train': train, 'acc/val': val}, module.current_epoch)
        return val

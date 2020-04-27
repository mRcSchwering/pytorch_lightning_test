"""
Module definition

Basically experimental setup w/ dataloading, training and validation steps.
"""
from typing import List, Dict
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from src.metrics import Metric
from src.dataloading import RandomClassData
from src.modeling import TanhMlp

# TODO ich muss eigentlich nur predictions und 


class MyModule(pl.LightningModule):

    def __init__(self, hparams: dict, metrics: Dict[str, Metric]):
        """Note: when hparams are logged their values cannot be None!"""
        super(MyModule, self).__init__()
        self.hparams = hparams
        self.metrics = {}
        self.net = TanhMlp(10, 16, 2)
        self._init_metrics(metrics)

    def forward(self, x):
        return self.net(x)

    def train_dataloader(self):
        return DataLoader(dataset=RandomClassData(200, 10), batch_size=self.hparams['batch-size'])
    
    def val_dataloader(self):
        return DataLoader(dataset=RandomClassData(100, 10), batch_size=self.hparams['batch-size'])

    def configure_optimizers(self):
        optimizers = [optim.Adam(self.parameters(), lr=self.hparams['start-lr'])]
        schedulers = [optim.lr_scheduler.ReduceLROnPlateau(optimizers[0], factor=.5, patience=20)]
        return optimizers, schedulers

    def training_step(self, batch, batch_idx: int) -> dict:
        loss = self._forward_pass(*batch, partition='train')
        return {'loss': loss}

    def validation_step(self, batch, batch_idx: int) -> dict:
        loss = self._forward_pass(*batch, partition='val')
        return {'loss': loss}
    
    def validation_epoch_end(self, val_steps: List[dict]) -> dict:
        self._log_metrics()
        self._reset_metrics()
        return {'val_loss': torch.stack([d['loss'] for d in val_steps]).mean()}

    def _forward_pass(self, x, y, partition: str) -> float:
        tanhs = self.forward(x)
        loss = F.cross_entropy(tanhs, y)
        self._add_metrics(y, tanhs, partition)
        return loss
    
    def _init_metrics(self, metrics: Dict[str, Metric]):
        for name in metrics:
            self.metrics[name + '/train'] = metrics[name]()
            self.metrics[name + '/val'] = metrics[name]()
    
    def _add_metrics(self, targets: torch.Tensor, predictions: torch.Tensor, partition: str):
        for name, metric in self.metrics.items():
            if '/' + partition in name:
                metric.add(targets, predictions)

    def _log_metrics(self):
        row = {k: d.get() for k, d in self.metrics.items()}
        self.logger.log_metrics(row, self.current_epoch)

    def _reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()

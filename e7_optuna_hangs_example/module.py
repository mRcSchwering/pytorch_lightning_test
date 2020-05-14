import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from src.dataloading import RandomClassData
from src.modeling import TanhMlp


class MyModule(pl.LightningModule):

    def __init__(self, hparams: dict):
        super(MyModule, self).__init__()
        self.net = TanhMlp(10, hparams['hidden-size'], 2)
        self.best_val_loss = 999.9
        self.hparams = hparams

    def forward(self, x):
        return self.net(x)

    def train_dataloader(self):
        return DataLoader(dataset=RandomClassData(200, 10), batch_size=self.hparams['batch-size'])
    
    def val_dataloader(self):
        return DataLoader(dataset=RandomClassData(100, 10), batch_size=self.hparams['batch-size'])

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx: int) -> dict:
        return self._forward_pass(*batch)

    def validation_step(self, batch, batch_idx: int) -> dict:
        return self._forward_pass(*batch)

    def validation_epoch_end(self, steps):
        targets = torch.cat([d['targets'] for d in steps], dim=0)
        loss = float((torch.stack([d['loss'] for d in steps]).sum() / len(targets)).cpu().numpy())
        if loss < self.best_val_loss:
            self.best_val_loss = loss
        log = {'best_val_loss': self.best_val_loss}
        return {'loss': loss, 'log': log}

    def _forward_pass(self, x, y) -> float:
        preds = self.forward(x)
        loss = F.cross_entropy(preds, y)
        return {'loss': loss, 'preds': preds, 'targets': y}
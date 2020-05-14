import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from src.config import N_CPUS
from src.modules import MetricsAndBestLossOnEpochEnd
from src.dataloading import RandomClassData
from src.modeling import TanhMlp


class MyModule(MetricsAndBestLossOnEpochEnd):

    def __init__(self, hparams: dict, metrics: dict):
        super(MyModule, self).__init__()
        self.net = TanhMlp(10, hparams['hidden-size'], 2)
        self.best_val_loss = 999.9
        self.hparams = hparams
        self.metrics = metrics

    def forward(self, x):
        return self.net(x)

    def train_dataloader(self):
        return DataLoader(dataset=RandomClassData(200, 10), batch_size=self.hparams['batch-size'], num_workers=N_CPUS)
    
    def val_dataloader(self):
        return DataLoader(dataset=RandomClassData(100, 10), batch_size=self.hparams['batch-size'], num_workers=N_CPUS)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx: int) -> dict:
        return self._forward_pass(*batch)

    def validation_step(self, batch, batch_idx: int) -> dict:
        return self._forward_pass(*batch)

    def _forward_pass(self, x, y) -> float:
        preds = self.forward(x)
        loss = F.cross_entropy(preds, y)
        return {'loss': loss, 'preds': preds, 'targets': y}
"""Simple binary classification experiment"""
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from src.modules import MetricsOnEpochEnd
from src.dataloading import RandomClassData
from src.modeling import TanhMlp
from src.config import WORKERS


class MyModule(MetricsOnEpochEnd):
    """
    Using a module that collects all train/val targets and predictions
    and also calculates the metrics on train/val epoch end.
    """

    def __init__(self, hparams: dict, metrics: dict):
        super(MyModule, self).__init__()
        self.hparams = hparams
        self.metrics = metrics
        self.net = TanhMlp(10, 16, 2)
        self.criterion = F.cross_entropy

    def forward(self, x):
        return self.net(x)

    def train_dataloader(self):
        return DataLoader(dataset=RandomClassData(200, 10), batch_size=self.hparams['batch-size'], num_workers=int(WORKERS))
    
    def val_dataloader(self):
        return DataLoader(dataset=RandomClassData(100, 10), batch_size=self.hparams['batch-size'], num_workers=int(WORKERS))

    def configure_optimizers(self):
        optimizer1 = optim.Adam(self.parameters(), lr=self.hparams['start-lr'])
        scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, factor=.5, patience=20, verbose=True)
        return [optimizer1], [scheduler1]

    def training_step(self, batch, batch_idx: int) -> dict:
        return self._forward_pass(*batch)

    def validation_step(self, batch, batch_idx: int) -> dict:
        return self._forward_pass(*batch)

    def _forward_pass(self, x, y) -> float:
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        return {'loss': loss, 'preds': preds, 'targets': y}

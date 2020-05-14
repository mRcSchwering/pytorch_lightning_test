"""
Trying to find out why e6 is hanging on the last job.
"""
import os
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import optuna

N_GPUS = os.environ.get('CUDA_VISIBLE_DEVICES', 0)


class RandomClassData(Dataset):
    """Standard normal distributed features and uniformly sampled discrete targets"""

    def __init__(self, n_samples: int, n_dim: int, n_classes: int = 2):
        super(RandomClassData, self).__init__()
        self.features = torch.rand((n_samples, n_dim))
        self.targets = torch.randint(0, n_classes, size=(n_samples,))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.features[i], self.targets[i]


class TanhMlp(nn.Module):
    """Multi layer perceptron with tanh activation function"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout_rate: float = .0):
        super(TanhMlp, self).__init__()
        self.hidden_fc = nn.Linear(input_size, hidden_size)
        self.out_fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        hidden = F.relu(self.hidden_fc(x))
        out = torch.tanh(self.out_fc(self.dropout(hidden)))
        return out


class MyModule(pl.LightningModule):

    def __init__(self):
        super(MyModule, self).__init__()
        self.net = TanhMlp(10, 128, 2)
        self.best_val_loss = 999.9

    def forward(self, x):
        return self.net(x)

    def train_dataloader(self):
        return DataLoader(dataset=RandomClassData(200, 10), batch_size=32)
    
    def val_dataloader(self):
        return DataLoader(dataset=RandomClassData(100, 10), batch_size=32)

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
        return {'loss': loss}

    def _forward_pass(self, x, y) -> float:
        preds = self.forward(x)
        loss = F.cross_entropy(preds, y)
        return {'loss': loss, 'preds': preds, 'targets': y}


def objective(trial):
    print(f'Starting trial {trial.number} pid:{os.getpid()} tid:{threading.get_ident()}')
    trainer = pl.Trainer(
        logger=False,
        max_epochs=10,
        gpus=0 if N_GPUS > 0 else None,
        weights_summary=None,
        num_sanity_val_steps=0,
        progress_bar_refresh_rate=0)

    model = MyModule()
    trainer.fit(model)
    print(f'Finished trial {trial.number} pid:{os.getpid()} tid:{threading.get_ident()}')
    return model.best_val_loss


if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(objective, n_trials=10, n_jobs=max(1, N_GPUS))


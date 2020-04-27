"""Training process for a module"""
import random
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from src.metrics import BinRocAuc
#from src.reporting import Reporter
from e1.module import MyModule

THIS_DIR = Path(__file__).parent.absolute()

module = MyModule({'asd': 'a'}, metrics)

[d for d in dir(module) if d[0] != '_']


def main(hparams: dict, run_i: int):
    print(f'starting {hparams}')
    metrics = {'auc': BinRocAuc}
    module = MyModule(hparams, metrics)

    #reporter = Reporter(metrics)
    trainer = Trainer(
        logger=TensorBoardLogger(str(THIS_DIR / 'logs'), f'run{run_i}'),
        max_epochs=hparams['max-epochs'],
        early_stop_callback=pl.callbacks.EarlyStopping(patience=50),
        callbacks=[])
    trainer.fit(module)


if __name__ == '__main__':
    n_rounds = 3
    max_epochs = 100
    folds = ('fold1', 'fold2')
    batch_sizes = (8, 16)
    lrs = (1e-3, 1e-4)

    for round_i in range(n_rounds):
        for fold in folds:
            args = {
                'batch-size': random.choice(batch_sizes),
                'start-lr': random.choice(lrs),
                'fold': fold,
                'max-epochs': max_epochs}
            main(args, round_i)

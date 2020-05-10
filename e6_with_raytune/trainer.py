import random
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from ray import tune
from ray.tune.logger import Logger
from src.metrics import BinRocAuc
#from src.loggers import HyperparamsSummaryTensorBoardLogger
from e6_with_raytune.module import MyModule

THIS_DIR = Path(__file__).parent.absolute()


def trainable(hparams: dict):
    metrics = {'auc': BinRocAuc()}
    module = MyModule(hparams, metrics=metrics)
    trainer = Trainer(
        logger=False,
        max_epochs=hparams['max-epochs'],
        progress_bar_refresh_rate=0,  # 20 or so, only if non-parallel
        early_stop_callback=pl.callbacks.EarlyStopping(patience=2 * 50))  # seems to be 2 https://github.com/PyTorchLightning/pytorch-lightning/issues/1751
    trainer.fit(module)


if __name__ == '__main__':
    SPACE = {
        'batch-size': tune.grid_search([32, 64]),
        'start-lr': tune.grid_search([1e-3, 1e-4]),
        'fold': tune.grid_search(['fold1', 'fold2']),
        'max-epochs': 100}
    tune.run(trainable, config=SPACE, name='__logs__', local_dir=str(THIS_DIR))

"""
Trainer using callback logging.
Training 2 random hyperparameter sampling rounds with 2 train/val folds.
I am injecting epoch-wise metrics as callbacks into the trainer.
I got reporting of the "best" epoch together with the hparam set working.
It uses a hacked tensorboard logger... :/

:Example:

   python e4_custom_logger/trainer.py
"""
import random
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from src.loggers import HyperparamsMetricsTensorBoardLogger
from src.metrics import BinRocAuc
from src.reporting import TrainingProgressFromSummary, BestEpochFromSummary
from e4_custom_logger.module import MyModule

THIS_DIR = Path(__file__).parent.absolute()


def main(hparams: dict, trial: str or int):
    print(f'starting {hparams}')
    
    module = MyModule(hparams)
    metrics = {'auc': BinRocAuc()}
    logger = HyperparamsMetricsTensorBoardLogger(str(THIS_DIR / 'logs'), name=str(trial))
    training_progress = TrainingProgressFromSummary(metrics)
    training_results = BestEpochFromSummary(metrics)
    
    trainer = Trainer(
        logger=logger,
        max_epochs=hparams['max-epochs'],
        early_stop_callback=pl.callbacks.EarlyStopping(patience=50),
        callbacks=[training_progress, training_results])
    trainer.fit(module)


if __name__ == '__main__':
    n_trials = 3
    max_epochs = 100
    folds = ('fold1', 'fold2')
    batch_sizes = (32, 64)
    lrs = (1e-3, 1e-4)

    for trial_i in range(n_trials):
        for fold in folds:
            args = {
                'trial': trial_i,
                'batch-size': random.choice(batch_sizes),
                'start-lr': random.choice(lrs),
                'fold': fold,
                'max-epochs': max_epochs}
            main(hparams=args, trial=trial_i)

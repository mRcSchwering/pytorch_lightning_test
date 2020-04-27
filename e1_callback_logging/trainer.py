"""
Trainer using callback logging.
Training 2 random hyperparameter sampling rounds with 2 train/val folds.

I am injecting epoch-wise metrics as callbacks into the trainer.
I tried to log the best validation loss to hparams in tensorboard, but it's still not working.

:Example:

   python e1_callback_logging/trainer.py
"""
import random
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from src.loggers import MyTensorBoardLogger
from src.metrics import BinRocAuc
from src.reporting import TrainingProgress, BestLoss
from e1_callback_logging.module import MyModule

THIS_DIR = Path(__file__).parent.absolute()


def main(hparams: dict, run_i: int):
    print(f'starting {hparams}')
    
    module = MyModule(hparams)
    logger = MyTensorBoardLogger(str(THIS_DIR / 'logs'), name=str(run_i))
    training_progress = TrainingProgress({'auc': BinRocAuc()})
    training_summary = BestLoss()
    
    trainer = Trainer(
        logger=logger,
        max_epochs=hparams['max-epochs'],
        early_stop_callback=pl.callbacks.EarlyStopping(patience=50),
        callbacks=[training_progress, training_summary])
    trainer.fit(module)


if __name__ == '__main__':
    n_rounds = 3
    max_epochs = 100
    folds = ('fold1', 'fold2')
    batch_sizes = (32, 64)
    lrs = (1e-3, 1e-4)

    for round_i in range(n_rounds):
        for fold in folds:
            args = {
                'round': round_i,
                'batch-size': random.choice(batch_sizes),
                'start-lr': random.choice(lrs),
                'fold': fold,
                'max-epochs': max_epochs}
            main(args, round_i)

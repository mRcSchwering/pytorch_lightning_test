"""
Debugging concurrency issues in e6.

Example:
    CUDA_VISIBLE_DEVICES=11,12 python e7_optuna_hangs_example.trainer.py
"""
import os
from pathlib import Path
import pytorch_lightning as pl
from multiprocessing import Process, Value
from optuna import Trial, create_study
from src.config import N_GPUS
from src.loggers import HyperparamsSummaryTensorBoardLogger
from src.multiproc import GpuQueue, subprocess
from e7_optuna_hangs_example.module import MyModule

THIS_DIR = Path(__file__).parent.absolute()


def train_with_params(hparams: dict, trial_i: int, gpus: list) -> dict:
    print(f'pid:{os.getpid()}, ppid:{os.getppid()} : starting trial {trial_i} on GPUs {gpus}')
    logger = HyperparamsSummaryTensorBoardLogger(
        save_dir=str(THIS_DIR / '__logs__'),
        name=f'{trial_i}_trial')

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=hparams['max-epochs'],
        gpus=None,  #gpus,
        weights_summary=None,
        num_sanity_val_steps=0,
        progress_bar_refresh_rate=0)

    model = MyModule(hparams, {})
    trainer.fit(model)
    return {'loss': model.best_val_loss}


def objective(trial: Trial, queue: GpuQueue) -> float:
    hparams = {
        'batch-size': 16 * 2**trial.suggest_int('batch_size_exp', 0, 4),
        'hidden-size': 16 * 2**trial.suggest_int('hidden_size_exp', 0, 10),
        'start-lr': trial.suggest_loguniform('start_lr', 1e-5, 1e-3),
        'fold': 'fold1',
        'max-epochs': 10}
    with queue.n_gpus() as gpus:
        res = subprocess(
            target=train_with_params,
            kwargs={'hparams': hparams, 'trial_i': trial.number, 'gpus': gpus})
        return res['loss']


def run_sampling_rounds(n: int):
    print(f'\nStarting {n} rounds TPE sampling over hparam space.\n')
    queue = GpuQueue()
    study = create_study()
    study.optimize(lambda d: objective(d, queue), n_trials=n, n_jobs=max(1, N_GPUS))
    return sorted(study.trials, key=lambda d: d.value)


if __name__ == "__main__":
    TRIALS = run_sampling_rounds(10)
    print('\ndone')


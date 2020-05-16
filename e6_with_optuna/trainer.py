"""
Using the pytorch_lightning pattern from e5 together with optuna.

Some specifics about parallizing:
- Trials during hparam search should run in parallel.
- I wanted to be able to define how many (and which) GPUs each trial should run on.
- While sampling a hparam space, I also want to just run specific sets of hparams.

Example:

    CUDA_VISIBLE_DEVICES=11,12 python e6_with_optuna/trainer.py
"""
import os
from pathlib import Path
import pytorch_lightning as pl
from optuna import Trial, create_study
from src.config import N_GPUS
from src.loggers import HyperparamsSummaryTensorBoardLogger
from src.parallel import GpuQueue, NonDaemonPool, subprocess_with_gpus
from e6_with_optuna.module import MyModule

THIS_DIR = Path(__file__).parent.absolute()
HPARAMS_INIT = {
    'batch-size': 32,
    'hidden-size': 64,
    'start-lr': 1e-3,
    'fold': 'fold1',
    'max-epochs': 10}


def train_with_params(hparams: dict, name: str, gpus: list) -> dict:
    print(f'pid:{os.getpid()}, ppid:{os.getppid()} : starting {name} on GPUs {gpus}')
    logger = HyperparamsSummaryTensorBoardLogger(
        save_dir=str(THIS_DIR / '__logs__'),
        name=name)

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=hparams['max-epochs'],
        gpus=gpus,
        weights_summary=None,
        num_sanity_val_steps=0,
        progress_bar_refresh_rate=0)

    model = MyModule(hparams, {})
    trainer.fit(model)
    return {'loss': model.best_val_loss}


def sampling_objective(trial: Trial, queue: GpuQueue) -> float:
    hparams = HPARAMS_INIT.copy()
    hparams['batch-size'] = 16 * 2**trial.suggest_int('batch_size_exp', 0, 4)
    hparams['hidden-size'] = 16 * 2**trial.suggest_int('hidden_size_exp', 0, 10)
    hparams['start-lr'] = trial.suggest_loguniform('start_lr', 1e-5, 1e-3)
    return subprocess_with_gpus(
        target=train_with_params,
        kwargs={'hparams': hparams, 'name': f'sampling_{trial.number}'},
        queue=queue)


def validation_objective(trial: dict, queue: GpuQueue):
    hparams = HPARAMS_INIT.copy()
    hparams['batch-size'] = 16 * 2**trial.params['batch_size_exp']
    hparams['hidden-size'] = 16 * 2**trial.params['hidden_size_exp']
    hparams['start-lr'] = trial.params['start_lr']
    hparams['fold'] = 'fold2'
    return subprocess_with_gpus(
        target=train_with_params,
        kwargs={'hparams': hparams, 'name': f'validation_{trial.number}'},
        queue=queue)


def run_sampling_rounds(n: int):
    print(f'\nStarting {n} rounds TPE sampling over hparam space.\n')
    queue = GpuQueue()
    study = create_study()
    study.optimize(lambda d: sampling_objective(d, queue), n_trials=n, n_jobs=max(1, N_GPUS))
    return sorted(study.trials, key=lambda d: d.value)


def run_validaton_rounds(n: int, trials: list):
    print(f'\nStarting {n} validation rounds on fold2.\n')
    queue = GpuQueue()
    with NonDaemonPool(processes=max(1, N_GPUS)) as pool:
        return pool.starmap(validation_objective, [(d, queue) for d in trials[:n]])


if __name__ == "__main__":
    TRIALS = run_sampling_rounds(10)
    RES = run_validaton_rounds(3, TRIALS)
    print(RES)
    print('\ndone')


"""
Trying to use pytorch_lightning with optuna.

Here, is a version where first, I do a sampling round with optuna,
and then I validate the 3 best hparam sets on another fold of the data.
The main issue here was that I want to control that each training trial
should be placed on 1 GPU, e.g. for a model that would more or less fill
1 GPU during training.

I achieved that adding a multiprocessing queue which holds the ids of
available GPUs.
It works mostly, sometimes I get weird multiprocessing error,
but I gave up fixing them.

:Example:
    CUDA_VISIBLE_DEVICE=13,14 python e6_with_optuna/trainer.py
"""
import os
import threading
from pathlib import Path
from typing import List
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from optuna import create_study
from optuna.trial import Trial
from src.config import N_GPUS
from src.multiproc import NonDaemonPool, GpuQueue
from src.metrics import BinRocAuc
from src.loggers import HyperparamsSummaryTensorBoardLogger
from e6_with_optuna.module import MyModule

THIS_DIR = Path(__file__).parent.absolute()


def train_with_params(trial_config: dict, gpu_i: int = None):
    """
    Actual training procedure.
    
    Takes trial config, which is not exactly the same as hparams.
    For more specific sampling I need to take an extra step and apply a function.
    Each training should run on 1 GPU, `gpu_i`.
    """
    print(f'\nStarting pid:{os.getpid()} tid:{threading.get_ident()} on GPU {gpu_i}')
    hparams = {
        'batch-size': 16 * 2**trial_config['batch_size_exp'],
        'hidden-size': 16 * 2**trial_config['hidden_size_exp'],
        'start-lr': trial_config['start_lr'],
        'fold': trial_config['fold'],
        'max-epochs': trial_config['max_epochs']}

    metrics = {'auc': BinRocAuc()}
    module = MyModule(hparams, metrics=metrics)
    logger = HyperparamsSummaryTensorBoardLogger(
        save_dir=str(THIS_DIR / '__logs__'),
        name=f'pid{os.getpid()}_tid{threading.get_ident()}')

    trainer = Trainer(
        logger=logger,
        max_epochs=hparams['max-epochs'],
        gpus=None if gpu_i is None else [gpu_i],
        weights_summary=None,  # disable summary print 
        num_sanity_val_steps=0,  # no sanity check
        progress_bar_refresh_rate=0,  # 20 or so, only if non-parallel
        early_stop_callback=pl.callbacks.EarlyStopping(patience=2 * 50))  # pl issue 1751
    trainer.fit(module)
    return module.best_val_loss


class Objective:
    """
    Optuna Objective class.
    
    Adapter for `study.optimize`.
    Adds the logic for aquiring a GPU within the `study.optimize` multithread loop.
    """
    
    def __init__(self, gpu_queue: GpuQueue):
        self.gpu_queue = gpu_queue

    def __call__(self, trial: Trial):
        with self.gpu_queue.one_gpu_per_process() as gpu_i:
            config = {
                'batch_size_exp': trial.suggest_int('batch_size_exp', 0, 4),
                'hidden_size_exp': trial.suggest_int('hidden_size_exp', 0, 10),
                'start_lr': trial.suggest_loguniform('start_lr', 1e-5, 1e-3),
                'fold': 'fold1',
                'max_epochs': 10}
            return train_with_params(trial_config=config, gpu_i=gpu_i)


def process(gpu_queue: GpuQueue, config: dict):
    """Adapter for aquiring GPU within a `multiprocess.Pool` loop."""
    with gpu_queue.one_gpu_per_process() as gpu_i:
        return train_with_params(trial_config=config, gpu_i=gpu_i)


def run_sampling_rounds(n: int) -> List[Trial]:
    print(f'\nStarting {n} round TPE sampling over hparam space.\n')
    study = create_study()
    study.optimize(Objective(gpu_queue=GpuQueue()), n_jobs=max(1, N_GPUS), n_trials=n)
    return sorted(study.trials, key=lambda d: d.value)


def repeat_on_fold2(trials: List[Trial]):
    print(f'\nRepeating best {len(trials)} trials with fold2.\n')
    configs = []
    for trial in trials:
        trial.params.update({'fold': 'fold2', 'max_epochs': 10})
        configs.append(trial.params)
    queue = GpuQueue()
    pool = NonDaemonPool()
    with pool as popen:
        return popen.starmap(process, [(queue, d) for d in configs])


if __name__ == '__main__':
    TRIALS = run_sampling_rounds(n=10)
    _ = repeat_on_fold2(TRIALS[:3])

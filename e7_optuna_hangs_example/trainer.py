"""
Trying to find out why e6 is hanging on the last job.
"""
import os
import threading
from pathlib import Path
import pytorch_lightning as pl
from src.loggers import HyperparamsSummaryTensorBoardLogger
import optuna
from src.multiproc import GpuQueue
from e7_optuna_hangs_example.module import MyModule

THIS_DIR = Path(__file__).parent.absolute()
CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES')
N_GPUS = 0 if CUDA_VISIBLE_DEVICES is None else len(CUDA_VISIBLE_DEVICES.split(','))


def train_with_params(trial_config, trial_i, gpu_i):
    print(f'Starting trial {trial_i} pid:{os.getpid()} tid:{threading.get_ident()}')
    hparams = {
        'batch-size': 16 * 2**trial_config['batch_size_exp'],
        'hidden-size': 16 * 2**trial_config['hidden_size_exp'],
        'start-lr': trial_config['start_lr'],
        'fold': trial_config['fold'],
        'max-epochs': trial_config['max_epochs']}

    logger = HyperparamsSummaryTensorBoardLogger(
        save_dir=str(THIS_DIR / '__logs__'),
        name=f'{trial_i}_trial')

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=hparams['max-epochs'],
        gpus=None if gpu_i is None else [gpu_i],
        weights_summary=None,
        num_sanity_val_steps=0,
        progress_bar_refresh_rate=0)

    model = MyModule(hparams)
    trainer.fit(model)
    print(f'Finished trial {trial_i} pid:{os.getpid()} tid:{threading.get_ident()}')
    return model.best_val_loss


class Objective:
    """
    Optuna Objective class.
    
    Adapter for `study.optimize`.
    Adds the logic for aquiring a GPU within the `study.optimize` multithread loop.
    """

    def __init__(self, gpu_queue: GpuQueue):
        self.gpu_queue = gpu_queue

    def __call__(self, trial):
        with self.gpu_queue.one_gpu_per_process() as gpu_i:
            config = {
                'batch_size_exp': trial.suggest_int('batch_size_exp', 0, 4),
                'hidden_size_exp': trial.suggest_int('hidden_size_exp', 0, 10),
                'start_lr': trial.suggest_loguniform('start_lr', 1e-5, 1e-3),
                'fold': 'fold1',
                'max_epochs': 10}
            return train_with_params(config, trial.number, gpu_i)


def run_sampling_rounds(n: int):
    print(f'\nStarting {n} round TPE sampling over hparam space.\n')
    study = optuna.create_study()
    study.optimize(Objective(GpuQueue()), n_trials=n, n_jobs=max(1, N_GPUS))
    return sorted(study.trials, key=lambda d: d.value)


if __name__ == "__main__":
    TRIALS = run_sampling_rounds(10)
    print('done')


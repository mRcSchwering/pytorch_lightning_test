"""
Trying to find out why e6 is hanging on the last job.
"""
import os
from pathlib import Path
import pytorch_lightning as pl
from multiprocessing import Process
import optuna
from src.loggers import HyperparamsSummaryTensorBoardLogger
from src.multiproc import GpuQueue
from e7_optuna_hangs_example.module import MyModule

THIS_DIR = Path(__file__).parent.absolute()


def train_with_params(hparams, trial_i, gpu_i):
    print(f'Training in pid:{os.getpid()}, ppid:{os.getppid()}')
    logger = HyperparamsSummaryTensorBoardLogger(
        save_dir=str(THIS_DIR / '__logs__'),
        name=f'{trial_i}_trial')

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=hparams['max-epochs'],
        gpus=gpu_i,
        weights_summary=None,
        num_sanity_val_steps=0,
        progress_bar_refresh_rate=0)

    model = MyModule(hparams, {})
    trainer.fit(model)
    return model.best_val_loss


def objective(trial, queue):
    print(f'Starting trial {trial.number} in pid:{os.getpid()}')
    hparams = {
        'batch-size': 16 * 2**trial.suggest_int('batch_size_exp', 0, 4),
        'hidden-size': 16 * 2**trial.suggest_int('hidden_size_exp', 0, 10),
        'start-lr': trial.suggest_loguniform('start_lr', 1e-5, 1e-3),
        'fold': 'fold1',
        'max-epochs': 10}
    with queue.one_gpu_per_process() as gpu_i:
        print(f'In trial {trial.number}, pid:{os.getpid()} starting new process with gpu {gpu_i}')
        p = Process(target=train_with_params, args=(hparams, trial.number, gpu_i))
        p.start()
        p.join()
    print(f'Finished trial {trial.number}')
    return 1


def run_sampling_rounds(n: int):
    print(f'\nStarting {n} round TPE sampling over hparam space.\n')
    queue = GpuQueue(2)
    study = optuna.create_study()
    study.optimize(lambda d: objective(d, queue), n_trials=n, n_jobs=2)
    return sorted(study.trials, key=lambda d: d.value)


if __name__ == "__main__":
    TRIALS = run_sampling_rounds(10)
    print('done')


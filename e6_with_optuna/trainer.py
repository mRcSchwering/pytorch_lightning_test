"""
python e6_with_optuna/trainer.py
"""
import os
import threading
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from optuna import create_study
from optuna.trial import Trial
from src.config import N_GPUS
from src.multiproc import NonDaemonPool
from src.metrics import BinRocAuc
from src.loggers import HyperparamsSummaryTensorBoardLogger
from e6_with_optuna.module import MyModule

THIS_DIR = Path(__file__).parent.absolute()


class Objective:
    
    def __init__(self, process_idx: list):
        self.process_idx = process_idx

    def __call__(self, trial: Trial):
        return self.train_with_params(
            batch_size_exp=trial.suggest_int('batch_size_exp', 0, 4),
            hidden_size_exp=trial.suggest_int('hidden_size_exp', 0, 10),
            start_lr=trial.suggest_loguniform('start_lr', 1e-5, 1e-3),
            fold='fold1')

    def train_with_params(
            self,
            batch_size_exp: int,
            hidden_size_exp: int,
            start_lr: float,
            fold: str,
            max_epochs: int = 100):
        print(f'\nTraining in pid:{os.getpid()} tid:{threading.get_ident()} starting trial...\n')
        # TODO: optuna uses threading, process id can be the same
        #       in contrast to the Pool, I cannot control which GPU each thread gets
        #       so far I would have to use the custom Pool for that
        #       started: https://optuna.readthedocs.io/en/stable/tutorial/first.html

        name = f'batch_size_exp={batch_size_exp}_=hidden_size_exp{hidden_size_exp}_start_lr={start_lr}'

        hparams = {
            'batch-size': 16 * 2**batch_size_exp,
            'hidden-size': 16 * 2**hidden_size_exp,
            'start-lr': start_lr,
            'fold': fold,
            'max-epochs': max_epochs}
        
        metrics = {'auc': BinRocAuc()}
        module = MyModule(hparams, metrics=metrics)
        logger = HyperparamsSummaryTensorBoardLogger(
            save_dir=str(THIS_DIR / '__logs__'),
            name=name)  # name=f'pid{os.getpid()}')

        trainer = Trainer(
            logger=logger,
            max_epochs=hparams['max-epochs'],
            gpus=[self.process_idx] if N_GPUS > 0 else None,
            weights_summary=None,  # disable summary print 
            num_sanity_val_steps=0,  # no sanity check
            progress_bar_refresh_rate=0,  # 20 or so, only if non-parallel
            early_stop_callback=pl.callbacks.EarlyStopping(patience=2 * 50))  # seems to be 2 https://github.com/PyTorchLightning/pytorch-lightning/issues/1751
        trainer.fit(module)
        return module.best_val_loss


def hparam_sampling(idx: int):
    print(f'\nProcess {os.getpid()} starting study...\n')
    study = create_study()
    study.optimize(Objective(process_idx=idx), n_trials=10)
    print(f'\nProcess {os.getpid()} finished study\n')
    return study


if __name__ == '__main__':
    study = create_study()
    study.optimize(Objective(process_idx=0), n_trials=10, n_jobs=2)

    """
    with NonDaemonPool() as pool:
        studies = pool.map(hparam_sampling, range(N_GPUS) if N_GPUS > 0 else [0])
    
    print('\nHparam sampling finished, consolidating results\n')
    trials = []
    for study in studies:
        trials.extend(study.trials)
    best_trials = sorted(trials, key=lambda d: d.value)[:3]

    print('\nRunning fold2 for 3 best trials\n')
    for best_trial in best_trials:
        best_trial.params['fold'] = 'fold2'
        obj = Objective(process_idx=0)
        obj.train_with_params(**best_trial.params)

    print('\nDone\n')
    """
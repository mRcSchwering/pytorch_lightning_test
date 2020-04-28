from typing import Dict
import pytorch_lightning as pl
from src.metrics import Metric
from src.modules import CollectOnEpochEnd, SummarizeEpochs
from src.dataloading import Partition


class TrainingProgress(pl.Callback):
    """
    Callback for logging epoch-wise metrics at the end of each epoch.
    
    Train/val loss will always be logged.
    Provide additional metrics in a map from name to class.
    E.g. if the name is `auc`, `auc/train` and `auc/val` will be logged.

    For this callback to work, there need to be targets, predictions, and losses for
    training and validation (e.g. `train_loss`).
    It's a pretty ugly solution because it clutters the module namespace.
    A fitting base module would be `CollectOnEpochEnd`.
    """

    def __init__(self, metrics: Dict[str, Metric]):
        """
        :param metrics: map metric name to Metric class (name will be prepended with train/val)
        """
        super(TrainingProgress, self).__init__()
        self.metrics = {}
        self._init_metrics(metrics)

    def on_epoch_end(self, trainer: pl.Trainer, module: CollectOnEpochEnd):
        """Gets called by trainer after epoch end"""
        self._log_metrics(module)

    def _init_metrics(self, metrics: Dict[str, Metric]):
        for name, metric in metrics.items():
            self.metrics[name + '/train'] = metric
            self.metrics[name + '/val'] = metric

    def _log_metrics(self, module: CollectOnEpochEnd):
        row = {'loss/train': module.train_loss, 'loss/val': module.val_loss}
        for name, metric in self.metrics.items():
            if '/val' in name:
                row[name] = metric(module.val_targets, module.val_predictions)
            elif '/train' in name:
                row[name] = metric(module.train_targets, module.train_predictions)
        module.logger.log_metrics(row, module.current_epoch)


class BestLoss(pl.Callback):
    """
    Reporting the best/lowest (val) loss.
    This actually never worked as intended...
    """

    def __init__(self):
        super(BestLoss, self).__init__()
        self.best_loss = float('inf')

    def on_epoch_end(self, trainer, module):
        """Gets called by trainer after epoch end"""
        current_loss = float(module.val_loss.cpu().numpy())
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            metrics = {'hparam/loss': 0.75}
            module.logger.log_hyperparams(params=module.hparams, metrics=metrics)


class TrainingProgressFromSummary(pl.Callback):
    """
    Callback for logging epoch-wise metrics at the end of each epoch.
    Basically the same as TrainingProgress, but much cleaner.
    A EpochSummary class has collected the epoch results.
    The fitting base module would be `SummarizeEpochs`.
    """

    def __init__(self, metrics: Dict[str, Metric]):
        """
        :param metrics: map metric name to Metric class (name will be appended with partition)
        """
        super(TrainingProgressFromSummary, self).__init__()
        self.metrics = {}
        self.metrics = metrics

    def on_epoch_end(self, trainer: pl.Trainer, module: SummarizeEpochs):
        """Gets called by trainer after epoch end"""
        self._log_metrics(module)

    def _log_metrics(self, module: SummarizeEpochs):
        log = {}
        for part in module.epoch_summary.summaries.keys():
            results = module.epoch_summary.get_results(part)
            log[f'loss/{part.value}'] = results.loss
            for name, metric in self.metrics.items():
                log[f'{name}/{part.value}'] = metric(results.targets, results.predictions)
        module.logger.log_metrics(log, module.current_epoch)


class BestEpochFromSummary(pl.Callback):
    """
    Reporting and updating metrics of the best epoch.
    Best epoch := lowest validation loss.

    Like the one above, this one needs the EpochSummary to have collected results.
    Thus, it works together with the `SummarizeEpochs` base module.
    
    Additionally, I need to use a hacked tensorboard logger: `HyperparamsMetricsTensorBoardLogger`.
    This logger works but it also has some issues (see its docstring).
    """

    def __init__(self, metrics: Dict[str, Metric]):
        super(BestEpochFromSummary, self).__init__()
        self.best_loss = float('inf')
        self.metrics = metrics

    def on_epoch_end(self, trainer: pl.Trainer, module: SummarizeEpochs):
        """Gets called by trainer after epoch end"""
        results = module.epoch_summary.get_results(Partition.VAL)
        if results.loss < self.best_loss:
            self.best_loss = results.loss
            self._log_hparams_metrics(module)

    def _log_hparams_metrics(self, module: SummarizeEpochs):
        metrics = {}
        for part in module.epoch_summary.summaries.keys():
            results = module.epoch_summary.get_results(part)
            metrics[f'best-loss/{part.value}'] = results.loss
            for name, metric in self.metrics.items():
                metrics[f'best-{name}/{part.value}'] = metric(results.targets, results.predictions)
        module.logger.log_hyperparams_metrics(params=module.hparams, metrics=metrics)

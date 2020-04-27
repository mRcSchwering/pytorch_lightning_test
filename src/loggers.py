"""this is a workaround, see issue: https://github.com/PyTorchLightning/pytorch-lightning/issues/1228"""
from typing import Union, Optional, Dict, Any
from argparse import Namespace
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.base import rank_zero_only
from torch.utils.tensorboard.summary import hparams


class MyTensorBoardLogger(TensorBoardLogger):
    """
    I am trying to log hyperparameters with metrics.
    For some reason the metrics dont end up in tensorboard.
    I had the suspicion that there is an issue with logging multiple times.
    That's why I have this class.

    Still not working though...
    """

    def __init__(self, *args, **kwargs):
        super(MyTensorBoardLogger, self).__init__(*args, **kwargs)

    def log_hyperparams(self, *args, **kwargs):
        pass

    @rank_zero_only
    def log_hyperparams_metrics(self, params: Union[Dict[str, Any], Namespace],
                        metrics: Optional[Dict[str, Any]] = None) -> None:
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        sanitized_params = self._sanitize_params(params)
            
        if metrics is None:
            metrics = {}
        exp, ssi, sei = hparams(sanitized_params, metrics)
        writer = self.experiment._get_file_writer()
        writer.add_summary(exp)
        writer.add_summary(ssi)
        writer.add_summary(sei)

        # some alternative should be added
        self.tags.update(sanitized_params)

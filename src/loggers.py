"""this is a workaround, see issue: https://github.com/PyTorchLightning/pytorch-lightning/issues/1228"""
from pytorch_lightning.loggers import TensorBoardLogger, LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_only
from torch.utils.tensorboard.summary import hparams


class MyTensorBoardLogger(TensorBoardLogger):

    def __init__(self, *args, **kwargs):
        super(MyTensorBoardLogger, self).__init__(*args, **kwargs)

    def log_hyperparams(self, *args, **kwargs):
        pass

    @rank_zero_only
    def log_hyperparams_metrics(self, params: dict, metrics: dict) -> None:
        params = self._convert_params(params)
        exp, ssi, sei = hparams(params, metrics)
        writer = self.experiment._get_file_writer()
        writer.add_summary(exp)
        writer.add_summary(ssi)
        writer.add_summary(sei)
        self.tags.update(params)

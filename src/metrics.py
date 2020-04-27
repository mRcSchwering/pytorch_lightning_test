"""Metrics reported during training"""
import torch
from sklearn.metrics import roc_auc_score


class Metric:
    """Base class for metrics used in training procedure"""

    def __call__(self, targets: torch.Tensor, predictions: torch.Tensor):
        """
        Called at the end of epoch. Calculate metric from collected values.
        
        :return: float or scalar tensor
        """
        raise NotImplementedError


class BinRocAuc(Metric):
    """Binary ROC AUC (class0, class1)"""
    decision_functions = ('class_index', 'cross_entropy')

    def __init__(self, decision_function: str = 'class_index'):
        """
        :param decision_function: on what to base the classification decision
        """
        assert decision_function in self.decision_functions
        self.decision_function = decision_function

    def __call__(self, targets: torch.Tensor, predictions: torch.Tensor):
        if len(predictions) == 0:
            return float('nan')
        
        if self.decision_function == 'class_index':
            scores = predictions[:, 1]
        elif self.decision_function == 'cross_entropy':
            scores = predictions[:, 1] - predictions.exp().sum(axis=1).log()
        else:
            raise ValueError(self.decision_function)
        
        try:
            return roc_auc_score(targets.cpu().detach().numpy(), scores.cpu().detach().numpy())
        except ValueError:
            return float('nan')


"""Metrics reported during training"""
import torch
from sklearn.metrics import roc_auc_score
DEVICE = 'cpu'  # get from config


class Metric:
    """
    Base class for metrics used in training procedure.
    start: worst value the metric can have
    """
    start = float('Inf')

    def __init__(self):
        """Initialized before training starts"""
        self.targets = torch.tensor([], dtype=torch.float).to(DEVICE)
        self.predictions = torch.tensor([], dtype=torch.float).to(DEVICE)

    def reset(self):
        """Called at beginning of new epoch. Reinitialize all containers"""
        self.__init__()

    def add(self, targets: torch.Tensor, predictions: torch.Tensor):
        """Called after each minibatch. Fill containers."""
        self.targets = torch.cat([self.targets, targets.float()], dim=0)
        self.predictions = torch.cat([self.predictions, predictions.float()], dim=0)

    def get(self):
        """Called at the end of epoch. Calculate metric from collected values."""
        raise NotImplementedError


class BinRocAuc(Metric):
    start = 0  # TODO: why?
    decision_functions = ('class_index', 'cross_entropy')

    def __init__(self, decision_function='class_index'):
        """Initialized before training starts"""
        assert decision_function in self.decision_functions
        self.decision_function = decision_function
        self.targets = torch.tensor([], dtype=torch.long).to(DEVICE)
        self.predictions = torch.tensor([], dtype=torch.float).to(DEVICE)
    
    def reset(self):
        """Called at beginning of new epoch. Reinitialize all containers"""
        self.__init__(decision_function=self.decision_function)

    def add(self, targets: torch.Tensor, predictions: torch.Tensor):
        self.targets = torch.cat([self.targets, targets.long()], dim=0)
        self.predictions = torch.cat([self.predictions, predictions.float()], dim=0)

    def get(self):
        if len(self.predictions) == 0:
            return 0
        if self.decision_function == 'class_index':
            scores = self.predictions[:, 1]
        elif self.decision_function == 'cross_entropy':
            scores = self.predictions[:, 1] - self.predictions.exp().sum(axis=1).log()
        else:
            raise ValueError(self.decision_function)
        
        try:
            return roc_auc_score(
                self.targets.cpu().detach().numpy(),
                scores.cpu().detach().numpy())
        except ValueError:
            return float('nan')

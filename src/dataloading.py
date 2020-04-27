import torch
from torch.utils.data import Dataset


class RandomClassData(Dataset):
    """Standard normal distributed features and uniformly sampled discrete targets"""

    def __init__(self, n_samples: int, n_dim: int, n_classes: int = 2):
        super(RandomClassData, self).__init__()
        self.features = torch.rand((n_samples, n_dim))
        self.targets = torch.randint(0, n_classes, size=(n_samples,))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.features[i], self.targets[i]

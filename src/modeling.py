import torch
import torch.nn as nn
import torch.nn.functional as F


class TanhMlp(nn.Module):
    """Multi layer perceptron with tanh activation function"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout_rate: float = .0):
        super(TanhMlp, self).__init__()
        self.hidden_fc = nn.Linear(input_size, hidden_size)
        self.out_fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        hidden = F.relu(self.hidden_fc(x))
        out = torch.tanh(self.out_fc(self.dropout(hidden)))
        return out

import pytest
import torch
import numpy as np
from utils.settings import DEVICE
import utils.metrics as metrics

regr_metrics = [metrics.Mse, metrics.Rsquared]
class_metrics = [metrics.Accuracy, metrics.BinRocAuc]


def test_not_implemented_raises():
    m = metrics.Metric()
    with pytest.raises(NotImplementedError):
        m.get()


@pytest.mark.parametrize('M', regr_metrics)
def test_smoke_regr_metrics(M):
    m = M()
    m.reset()
    m.add(torch.tensor([1.2, 0.3, 1.7]).to(DEVICE), torch.tensor([0.1, 1.2, 1.3]).to(DEVICE))
    assert isinstance(m.get(), float)


@pytest.mark.parametrize('M', class_metrics)
def test_smoke_class_metrics(M):
    m = M()
    m.reset()
    m.add(torch.tensor([1, 0, 1]).to(DEVICE), torch.tensor([[0.1, 1.2], [0.1, 0.2], [2.1, 1.2]]).to(DEVICE))
    assert isinstance(m.get(), float)


@pytest.mark.parametrize('M', class_metrics)
def test_class_metrics_extreme_cases(M):
    m = M()
    m.reset()
    m.add(torch.tensor([0]).to(DEVICE), torch.tensor([[0, 0]]).to(DEVICE))
    assert isinstance(m.get(), float)
    m.add(torch.tensor([0]).to(DEVICE), torch.tensor([[0, 0]]).to(DEVICE))
    assert isinstance(m.get(), float)


@pytest.mark.parametrize('M', regr_metrics)
def test_regr_metrics_extreme_cases(M):
    m = M()
    m.reset()
    m.add(torch.tensor([0]).to(DEVICE), torch.tensor([[0, 0]]).to(DEVICE))
    assert isinstance(m.get(), float)
    m.add(torch.tensor([0]).to(DEVICE), torch.tensor([[0, 0]]).to(DEVICE))
    assert isinstance(m.get(), float)


def test_accuracy():
    m = metrics.Accuracy()
    m.reset()
    m.add(torch.tensor([1]).to(DEVICE), torch.tensor([[.0, .1]]).to(DEVICE))
    assert m.get() == 1
    m.add(torch.tensor([0]).to(DEVICE), torch.tensor([[.0, .1]]).to(DEVICE))
    assert m.get() == 0.5
    m.add(torch.tensor([0]).to(DEVICE), torch.tensor([[.1, .0]]).to(DEVICE))
    m.add(torch.tensor([0]).to(DEVICE), torch.tensor([[.1, .0]]).to(DEVICE))
    assert m.get() == 0.75


def test_mse():
    m = metrics.Mse()
    m.reset()
    m.add(torch.tensor([.1]).to(DEVICE), torch.tensor([.1]).to(DEVICE))
    assert m.get() == .0
    m.add(torch.tensor([.1]).to(DEVICE), torch.tensor([1.1]).to(DEVICE))
    assert m.get() == 0.5
    

def test_r2():
    m = metrics.Rsquared()
    m.reset()
    m.add(torch.tensor([.1]).to(DEVICE), torch.tensor([.1]).to(DEVICE))
    assert np.isnan(m.get())
    m.add(torch.tensor([1.1]).to(DEVICE), torch.tensor([1.1]).to(DEVICE))
    assert m.get() == 1
    m.add(torch.tensor([0.5]).to(DEVICE), torch.tensor([1.1]).to(DEVICE))
    res = m.get()
    assert res < 1
    assert res > 0


def test_bin_roc_auc():
    m = metrics.BinRocAuc()
    m.reset()
    m.add(torch.tensor([1, 1, 1, 0, 0]).to(DEVICE),
          torch.tensor([[.0, .9], [.0, .7], [.0, .5], [.0, .3], [.0, .1]]).to(DEVICE))
    assert m.get() == 1
    m.add(torch.tensor([1, 1]).to(DEVICE), torch.tensor([[.0, .1], [.0, .1]]).to(DEVICE))
    assert m.get() == 0.7

    m = metrics.BinRocAuc(decision_function='cross_entropy')
    m.reset()
    m.add(torch.tensor([1, 1, 1, 0, 0]).to(DEVICE),
          torch.tensor([[.0, .9], [.0, .7], [.0, .5], [.0, .3], [.0, .1]]).to(DEVICE))
    assert m.get() == 1
    m.add(torch.tensor([1, 1]).to(DEVICE), torch.tensor([[.0, .1], [.0, .1]]).to(DEVICE))
    assert m.get() == 0.7

    m.reset()
    m.add(torch.tensor([1, 1, 1, 0, 0]).to(DEVICE),
          torch.tensor([[.0, .9], [.0, .7], [.9, .5], [.0, .3], [.0, .1]]).to(DEVICE))
    assert m.get() < 1

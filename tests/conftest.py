import pytest
import numpy as np
import torch


np.random.seed(6)

@pytest.fixture
def X():
    return np.random.randn(100, 3)

@pytest.fixture
def y_linear(X):
    theta = np.array([1, 2, 3])
    return X @ theta + np.random.randn(100)

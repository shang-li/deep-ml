import torch
import pytest
from deep_ml import solve_linear_regression

@pytest.mark.parametrize("mode", ["analytical", "gd", "torch_gd"])
def test_solve_linear_regression(X, y_linear, mode):
    if mode in ["analytical", "gd", "torch_gd"]:
        theta = solve_linear_regression(X, y_linear, mode=mode)
        assert theta.shape == (3,)
        torch.testing.assert_close(theta, torch.tensor([1., 2., 3.]), atol=10e-2, rtol=10e-1)
    else:
        with pytest.raises(ValueError, match=f"Invalid mode: {mode}"):
            solve_linear_regression(X, y_linear, mode=mode)
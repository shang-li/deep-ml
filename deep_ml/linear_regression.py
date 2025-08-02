import torch
import torch.nn as nn
import numpy as np

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(input_dim))

    def forward(self, X):
        return X @ self.theta

def solve_linear_regression(X: np.ndarray, y: np.ndarray, mode: str = "analytical"):  # "analytical", "gd", "torch_gd"
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    if mode == "analytical":
        return torch.linalg.inv(X_t.T @ X_t) @ X_t.T @ y_t

    elif mode == "gd":
        max_iters = 100
        lr = 1e-2
        theta = torch.randn(X_t.shape[1])
        grad = (X_t.T @ X_t @ theta - y_t @ X_t) / len(y_t)
        for i in range(max_iters):
            theta_new = theta - lr * grad
            if sum(abs(theta_new - theta)) < 1e-2:
                return theta_new
            theta = theta_new
        return theta

    elif mode == "torch_gd":
        linear_nn = LinearRegressionModel(input_dim=X_t.shape[1])
        optimizer = torch.optim.SGD(linear_nn.parameters(), lr=0.01)
        max_iters = 100
        for i in range(max_iters):
            pred = linear_nn(X_t)
            loss = torch.mean((pred - y_t)**2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return linear_nn.theta.data
    else:
        raise ValueError(f"Invalid mode: {mode}")


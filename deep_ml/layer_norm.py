import numpy as np
import torch


def layer_norm(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    # X = batch, seq_len, feature_dim
    X_t = torch.tensor(X)
    mu_t = X_t.mean(dim=-1, keepdim=True)
    variance_t = torch.mean((X_t - mu_t) ** 2, dim=-1, keepdim=True)

    gamma_t = torch.tensor(gamma).reshape(1, 1, -1)
    beta_t = torch.tensor(beta).reshape(1, 1, -1)
    return gamma_t * (X_t - mu_t) / torch.sqrt(variance_t + epsilon) + beta_t

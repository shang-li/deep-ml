import numpy as np

def noisy_topk_gating(
    X: np.ndarray,
    W_g: np.ndarray,
    W_noise: np.ndarray,
    N: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Args:
        X: Input data, shape (batch_size, features)
        W_g: Gating weight matrix, shape (features, num_experts)
        W_noise: Noise weight matrix, shape (features, num_experts)
        N: Noise samples, shape (batch_size, num_experts)
        k: Number of experts to keep per example
    Returns:
        Gating probabilities, shape (batch_size, num_experts)
    """
    H = X @ W_g + N * np.log(1 + np.exp(X @ W_noise))
    arg_topk = np.argpartition(-H, k-1, axis=1)[:, :k]
    mask = np.full_like(H, False)
    #import pdb; pdb.set_trace()
    row_idx = np.arange(X.shape[0])[:, None]
    mask[row_idx, arg_topk] = True
    H[np.logical_not(mask)] = -float('inf')
    H_n = H - H.max(axis=1).reshape(-1, 1)
    return np.exp(H_n) / np.exp(H_n).sum(axis=1)


import numpy as np 
X = np.array([[1.0, 2.0]]) 
W_g = np.array([[1.0, 0.0], [0.0, 1.0]]) 
W_noise = np.array([[0.5, 0.5], [0.5, 0.5]]) 
N = np.array([[1.0, -1.0]]) 
print(np.round(noisy_topk_gating(X, W_g, W_noise, N, k=2), 4))
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
    return np.exp(H_n) / np.exp(H_n).sum(axis=1, keepdims=True)



def moe(x: np.ndarray, We: np.ndarray, Wg: np.ndarray, n_experts: int, top_k: int) -> np.ndarray:
    """
    Args:
        x: Input tensor of shape (n_batch, l_seq, d_model)
        We: Expert weights of shape (n_experts, d_model, d_model)
        Wg: Gating weights of shape (d_model, n_experts)
        n_experts: Number of experts
        top_k: Number of experts to route each token to
    Returns:
        Output tensor of shape (n_batch, l_seq, d_model)
    """
    logit = x @ Wg # n_batch, l_seq, n_experts
    weights = logit - logit.max(axis=-1, keepdims=True)
    mask = np.zeros_like(weights)
    col_idx = np.argpartition(-weights, top_k - 1, axis=2)[:, :, :top_k]
    mask[np.arange(x.shape[0])[:, None, None], np.arange(x.shape[1])[None, :, None], col_idx] = True
    weights[np.logical_not(mask)] = -float('inf')
    weights = np.exp(weights) / np.exp(weights).sum(axis=-1, keepdims=True) # batch x l_seq, nexperts
    w = np.tensordot(weights, We, axes=(2, 0))
    return np.einsum('ijk, ijkm -> ijm', x, w)

"""
X = np.array([[1.0, 2.0]]) 
W_g = np.array([[1.0, 0.0], [0.0, 1.0]]) 
W_noise = np.array([[0.5, 0.5], [0.5, 0.5]]) 
N = np.array([[1.0, -1.0]]) 
print(np.round(noisy_topk_gating(X, W_g, W_noise, N, k=2), 4))
"""

np.random.seed(42)
d_model = 2
n_experts = 4
l_seq = 3
n_batch = 2
top_k = 2
x = np.random.rand(n_batch, l_seq, d_model)
We = np.random.rand(n_experts, d_model, d_model)
Wg = np.random.rand(d_model, n_experts)
output = moe(x, We, Wg, n_experts, top_k)
print(np.round(output, 4))
print(output.shape)
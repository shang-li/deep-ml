import torch
from deep_ml import layer_norm


def test_layer_norm():
    X_t = torch.tensor([[[1, 2, 3], [2, 6, 2]]], dtype=torch.float32)
    # Check mean ≈ 0 and std ≈ 1
    gamma = torch.tensor([1, 1, 1], dtype=torch.float32)
    beta = torch.tensor([0, 0, 0], dtype=torch.float32)
    X_normed = layer_norm(X_t.detach().numpy(), gamma.detach().numpy(), beta.detach().numpy())

    torch_norm = torch.nn.LayerNorm(normalized_shape=3, elementwise_affine=False)
    X_torch_normed = torch_norm(X_t)
    torch.testing.assert_close(X_normed, X_torch_normed, atol=1e-6, rtol=1e-2)

    # Check gamma = [1,2,3] and beta ≈ [-1, -2, -3]
    gamma = torch.tensor([1, 2, 3], dtype=torch.float32)
    beta = torch.tensor([-1, -2, -3], dtype=torch.float32)
    X_normed = layer_norm(X_t.detach().numpy(), gamma.detach().numpy(), beta.detach().numpy())
    torch.testing.assert_close(X_normed, X_torch_normed * gamma + beta, atol=1e-6, rtol=1e-2)

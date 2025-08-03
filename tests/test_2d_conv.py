import numpy as np
import torch
import torch.nn.functional as F
from deep_ml import simple_2d_conv


def test_simple_2d_conv():
    input_matrix = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ]
    )

    kernel = np.array([[1, 0],[-1, 1]])

    padding = 1
    stride = 2

    X_conv = simple_2d_conv(input_matrix, kernel, padding, stride)
    input_tensor = torch.tensor(input_matrix)
    kernel_tensor = torch.tensor(kernel)
    X_torch_conv = F.conv2d(input_tensor[None, None, :, :], weight=kernel_tensor[None, None, :, :], bias=None, padding=padding, stride=stride)
    X_torch_conv = X_torch_conv.squeeze().detach().numpy()
    np.testing.assert_allclose(X_conv, X_torch_conv, atol=1e-5, rtol=1e-2)
import numpy as np
import math

def simple_2d_conv(X: np.ndarray, kernel: np.ndarray, padding: int = 0, stride: int = 1):
    # padding add integer number of elements on each side of the matrix
    # output shape: height = floor((input_height_padded - kernel_height) / stride) + 1
    X_padded = np.pad(X, pad_width=padding, mode="constant", constant_values=0)
    in_height_padded, in_width_padded = X_padded.shape
    k_height, k_width = kernel.shape
    out_height = math.floor((in_height_padded - k_height) / stride) + 1
    out_width = math.floor((in_width_padded - k_width) / stride) + 1
    X_padded = np.pad(X, pad_width=padding, mode="constant", constant_values=0)

    X_conv = np.zeros((out_height, out_width))
    for i in range(out_height):
        for j in range(out_width):
            X_conv[i, j] = np.sum(X_padded[i * stride:i * stride + k_height, j * stride:j * stride + k_width] * kernel)
    return X_conv

input_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

kernel = np.array([
    [1, 0],
    [-1, 1]
])

padding = 1
stride = 2

print(simple_2d_conv(input_matrix, kernel, padding, stride))
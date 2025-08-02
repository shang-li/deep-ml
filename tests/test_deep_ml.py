import torch
from deep_ml import matrix_vector_dot


def test_matrix_vector_dot():
    # normal
    A = [[1, 2], [3, 4]]
    b = [5, 6]
    torch.testing.assert_close(matrix_vector_dot(A, b), torch.tensor([17., 39.]), rtol=1e-5, atol=1e-8)
    # A = 1x1
    A = [[1]]
    b = [5]
    torch.testing.assert_close(matrix_vector_dot(A, b), torch.tensor([5.0]), rtol=1e-5, atol=1e-8)
    # A has more columns than b has rows
    A = [[1, 2, 3], [4, 5, 6]]
    b = [7, 8]
    assert matrix_vector_dot(A, b) == -1
    # test 3: A has more rows than b has columns
    A = [[1, 2], [3, 4], [5, 6]]
    b = [7, 8, 9]
    assert matrix_vector_dot(A, b) == -1
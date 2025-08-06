import numpy as np
import pytest
from deep_ml import RowSparseMatrix

@pytest.mark.parametrize(
    "i, j, val",
    [
        (0, 0, 1),
        (0, 2, 0),
        (2, 2, 0),
        (1, 2, 6),
    ]
)
def test_row_sparse_matrix(i, j, val):
    X = np.array(
        [
            [1, 0, 0, 2],
            [0, 0, 6, 0],
            [0, 0, 0, 1],
        ]
    )
    X_sparse = RowSparseMatrix(X)
    assert X_sparse[i, j] == val
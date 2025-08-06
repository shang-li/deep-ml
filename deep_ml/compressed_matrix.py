import numpy as np


class RowSparseMatrix():
    def __init__(self, X: np.array):
        assert len(X.shape) == 2, "requires 2D matrix"
        self.values = []
        self.orig_col_idx = []
        self.row_indices = [0]
        self._compressed_row_matrix(X)

    def _compressed_row_matrix(self, X):
        for i in range(len(X)):
            n_nonzeros = 0
            for j in range(len(X[0])):
                if X[i, j] != 0:
                    n_nonzeros += 1
                    self.values.append(X[i, j])
                    self.orig_col_idx.append(j)
            
            self.row_indices.append(self.row_indices[-1] + n_nonzeros)
    
    def __getitem__(self, key):
        i, j = key
        for k in range(self.row_indices[i], self.row_indices[i + 1]):
            if j == self.orig_col_idx[k]:
                return self.values[k]
        return 0

    
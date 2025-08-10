import numpy as np
from typing import Tuple

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    def gini(y_subset: np.ndarray) -> float:  # O(n)
        if y_subset.size == 0:
            return 0.0
        p = y_subset.mean()
        return 1.0 - (p**2 + (1 - p)**2)

    n_samples, n_features = X.shape
    best_feature, best_threshold = -1, float('inf')
    best_gini = float('inf')
    # O(mn^2)
    for f in range(n_features):  # O(m)
        for threshold in np.unique(X[:, f]):  # O(n)
            left = y[X[:, f] <= threshold] # O(n)
            right = y[X[:, f] > threshold]   # O(n)
            g_left, g_right = gini(left), gini(right) #O(n)
            weighted = (len(left) * g_left + len(right) * g_right) / n_samples
            if weighted < best_gini:
                best_gini, best_feature, best_threshold = weighted, f, threshold

    return best_feature, best_threshold


def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    def gini(labels):
        p1 = np.mean(labels)
        return 1 - p1**2 - (1 - p1)**2
    
    best_gini = float('inf') # the smaller the purer
    cut_col = None
    cut_val = None
    for col in range(len(X[0])):
        for row in range(len(X)):
            threshold = X[row, col]
            left_gini = gini(y[X[:, col] <= threshold])
            right_gini = gini(y[X[:, col] > threshold])
            left_cnt = np.sum(X[:, col] <= threshold)
            gini_calc = (left_gini * left_cnt + right_gini * (len(y) - left_cnt)) / len(y)
            if gini_calc < best_gini:
                best_gini = gini_calc
                cut_col = col
                cut_val = threshold
    return cut_col, cut_val

X = np.array([[0, 1], [0, 2], [0, 3], [0, 4]])
y = np.array([0, 0, 1, 1])
print(find_best_split(X, y))
import torch

def matrix_vector_dot(A, b):
    """
    A: numpy array, torch.tensor, or list of lists
    b: numpy array, torch.tensor, or list of lists
    """
    A = torch.tensor(A, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32)
    if A.shape[1] != b.shape[0]:
        return -1
    return A @ b

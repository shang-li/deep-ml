import numpy as np

class DropoutLayer:
    def __init__(self, p: float):
        """Initialize the dropout layer."""
        self.p = p

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass of the dropout layer."""
        if training:
            return x
        else:
            self.masked = np.random.binomial(1, p=1 - self.p, size=x.shape)  # mask is independent across samples
            return x * self.masked / (1 - self.p)
            

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass of the dropout layer."""
        return grad * self.masked / (1 - self.p)
    
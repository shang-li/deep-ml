import numpy as np

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    """train a single neuron with sigmoid activation; manual grad calculation"""
    mse_values = []
    updated_weights = initial_weights
    updated_bias = initial_bias
    for _ in range(epochs):
        # forward pass
        #import pdb; pdb.set_trace()
        z = features @ updated_weights + updated_bias
        s = 1 / (1 + np.exp(-z))
        loss = np.mean((s - labels)**2).round(4)
        mse_values.append(loss)
        # backward pass
        grad_w = (2 * (s - labels) * s * (1 - s) @ features) / features.shape[0]
        grad_b = np.mean(2 * (s - labels) * s * (1 - s))
        updated_weights -= learning_rate * grad_w
        updated_bias -= learning_rate * grad_b
    return updated_weights.round(4), updated_bias.round(4), mse_values

print(train_neuron(np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]]), np.array([1, 0, 0]), np.array([0.1, -0.2]), 0.0, 0.1, 2))
import torch

def relu(x: torch.tensor) -> torch.tensor:
    return torch.max(0, x)

def leaky_relu(x: torch.tensor, negative_slope: float) -> torch.tensor:
    return torch.where(x >= 0, x, negative_slope * x)

def elu(x: torch.tensor, negative_slope: float) -> torch.tensor:
    return torch.where(x >= 0, x, negative_slope * (torch.exp(x) + 1))

def gelu(x: torch.tensor) -> torch.tensor:
    return x * 0.5 * (1 + torch.erf(x / torch.sqrt(2)))

def swish(x: torch.tensor) -> torch.tensor:
    return 1 / (1 + torch.exp(-x)) * x

def sigmoid(x: torch.tensor) -> torch.tensor:
    return 1 / (1 + torch.exp(-x))

def softmax(x: torch.tensor) -> torch.tensor:
    return torch.exp(x) / torch.sum(torch.exp(x))


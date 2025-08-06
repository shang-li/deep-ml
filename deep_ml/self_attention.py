import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_qkv(X: torch.tensor, W_q: torch.tensor, W_k: torch.tensor, W_v: torch.tensor):
    return X @ W_q, X @ W_k, X @ W_v

def self_attention(Q: torch.tensor, K: torch.tensor, V: torch.tensor):
    d_k = K.shape[1]
    return F.softmax(Q @ K.T / d_k**0.5, dim=-1) @ V

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.d_k = embed_dim // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Parameter(torch.randn(embed_dim, self.d_k * num_heads))
        self.W_k = nn.Parameter(torch.randn(embed_dim, self.d_k * num_heads))
        self.W_v = nn.Parameter(torch.randn(embed_dim, self.d_k * num_heads))
        self.W_o = nn.Parameter(torch.randn(self.d_k * num_heads, self.d_k * num_heads))

    def forward(self, X):
        batch_size, seq_len, _ = X.shape
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2) # batch_size, num_heads, seq_len, d_k
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2) # batch_size, num_heads, seq_len, d_k
        attn_weights = F.softmax(Q @ K.transpose(-2, -1) / self.d_k ** 0.5, dim=-1) # batch_size, num_heads, seq_len, seq_len
        out = attn_weights @ V  # batch_size, num_heads, seq_len, d_k
        return out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1) @ self.W_o



import torch
import torch.nn as nn
from deep_ml import MultiHeadSelfAttention


def test_multihead_self_attention():
    embed_dim = 4
    num_heads = 2
    batch_size = 2
    seq_len = 2

    custom_attn = MultiHeadSelfAttention(embed_dim, num_heads)
    torch_attn = nn.MultiheadAttention(embed_dim, num_heads, bias=False, batch_first=True)
    # Copy parameters to match
    with torch.no_grad():
        # PyTorch stores weights as [embed_dim, embed_dim] in q_proj, k_proj, v_proj slices of in_proj_weight
        in_proj_weight = torch.empty(3*embed_dim, embed_dim)
        #in_proj_bias = torch.zeros(3*embed_dim)

        in_proj_weight[0:embed_dim] = custom_attn.W_q.T
        in_proj_weight[embed_dim:2*embed_dim] = custom_attn.W_k.T
        in_proj_weight[2*embed_dim:3*embed_dim] = custom_attn.W_v.T

        torch_attn.in_proj_weight.copy_(in_proj_weight)
        #torch_attn.in_proj_bias.copy_(in_proj_bias)

        # out_proj is identity for now
        torch_attn.out_proj.weight.copy_(torch.eye(embed_dim))
        #torch_attn.out_proj.bias.zero_()
    
    X = torch.randn(batch_size, seq_len, embed_dim)

# Outputs
    out_custom = custom_attn(X)
    out_torch, _ = torch_attn(X, X, X)
    torch.testing.assert_close(out_custom, out_torch, atol=1e-3, rtol=1e-2)
    
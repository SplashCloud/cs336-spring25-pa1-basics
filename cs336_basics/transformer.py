from torch import nn
import torch
from cs336_basics.base_modules import MultiHeadSelfAttention, RMSNorm, SwiGLU, Embedding, Linear, softmax
from einops import repeat
import math

class TransformerBlock(nn.Module):

    '''
    1. Attention part: RMSNorm => Causal-Masking Multi-Head Self-Attention => Residual Add
    2. Feed-Forward part: RMSNorm => FFN => Residual Add
    '''

    def __init__(self, d_embedding: int, d_attn: int,
                 num_heads: int, d_ff: int,
                 theta: float = 0, max_seq_len: int = 0,
                 device = None, dtype = None):
        super().__init__()
        self.pre_attention_norm = RMSNorm(d_model=d_embedding, device=device, dtype=dtype)
        self.multihead_self_attention = MultiHeadSelfAttention(d_embedding=d_embedding, d_attn=d_attn, num_heads=num_heads,
                                                                theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype)
        self.pre_ffn_norm = RMSNorm(d_model=d_attn, device=device, dtype=dtype)
        self.position_wise_ffn = SwiGLU(d_model=d_attn, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''x.shape = (... seq_len d_embedding)'''
        hidden1 = self.pre_attention_norm.forward(x)
        token_positions = torch.arange(x.shape[-2])
        token_positions = repeat(token_positions, "seq_len -> batch_size seq_len", batch_size=math.prod(x.shape[:-2]))
        token_positions = token_positions.reshape(x.shape[:-1])
        hidden1 = self.multihead_self_attention.forward(hidden1, token_positions=token_positions)
        hidden1 += x

        hidden2 = self.pre_ffn_norm(hidden1)
        hidden2 = self.position_wise_ffn(hidden2)
        hidden2 += hidden1

        return hidden2
        

class Transformer(nn.Module):

    def __init__(self, vocab_size: int, d_embedding: int, num_heads: int, d_attn: int, d_ff: int,
                 num_layers: int, context_length: int, theta: float, device = None, dtype = None):
        super().__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_embedding, device=device, dtype=dtype)
        self.transformer_blocks = nn.Sequential(*[
                TransformerBlock(d_embedding=d_embedding, d_attn=d_attn, num_heads=num_heads, d_ff=d_ff,
                                 theta=theta, max_seq_len=context_length, device=device, dtype=dtype) for _ in range(num_layers)
                ])
        self.norm = RMSNorm(d_model=d_attn, device=device, dtype=dtype)
        self.linear = Linear(in_features=d_attn, out_features=vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed = self.embedding.forward(x)
        output = self.transformer_blocks.forward(embed)
        output = self.norm.forward(output)
        output = self.linear.forward(output)
        # output = softmax(output, dim=-1)
        return output
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
        token_positions = torch.arange(x.shape[-2], device=x.device)
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


def get_computation(vocab_size: int, batch_size: int, seq_len: int, d_model: int, num_layers: int, num_heads: int, d_ff: int):
    d_k = d_v = d_model // num_heads
    embedding_comp = 2 * vocab_size * batch_size * seq_len * d_model
    attn_wqwkwv = 3 * (2 * d_model * batch_size * seq_len * d_model)
    attn_rope = 2 * (2 * seq_len * batch_size * num_heads * seq_len * d_k * d_k + 2 * d_k * batch_size * num_heads * seq_len * d_k)
    attn_qk = 2 * d_k * batch_size * num_heads * seq_len * seq_len
    attn_qkv = 2 * seq_len * batch_size * num_heads * seq_len * d_v
    attn_o = 2 * d_model * batch_size * seq_len * d_model
    attention_comp = attn_wqwkwv + attn_rope + attn_qk + attn_qkv + attn_o
    ffn_comp = 2 * (2 * d_model * batch_size * seq_len * d_ff) + 2 * batch_size * seq_len * d_ff + 2 * d_ff * batch_size * seq_len * d_model
    final_linear_comp = 2 * d_model * batch_size * seq_len * vocab_size
    total =  embedding_comp + num_layers * (attention_comp + ffn_comp) + final_linear_comp
    print(f'computation ratio of each part: embed={embedding_comp/total}, attention(single layer)={attention_comp/total}, ffn(single layer)={ffn_comp/total}, final_linear={final_linear_comp/total}')
    return total

def get_paras_number(vocab_size: int, d_model: int, num_layers: int, d_ff: int):
    embedding_weight = vocab_size * d_model
    attention_weight = 4 * d_model * d_model
    ffn_weight = 3 * d_model * d_ff
    norm_weight = d_model
    final_linear_weight = vocab_size * d_model
    return embedding_weight + num_layers * (attention_weight + ffn_weight + 2 * norm_weight) + norm_weight + final_linear_weight

if __name__ == "__main__":
    vocab_size = 50257
    batch_size = 1
    context_length = 1024
    d_ff = 6400
    data_size = 4

    # GPT-2 XL
    num_layers = 48
    d_model = 1600
    num_heads = 25
    # context_length = 16384
    paras = get_paras_number(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers, d_ff=d_ff)
    computation = get_computation(vocab_size=vocab_size, batch_size=batch_size, seq_len=context_length, num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff)
    print(f"GPT-2 XL need {paras} parameters, and need {paras * data_size / (1024 ** 3)} GB to store parameters")
    print(f"a single forward need {computation} FLOPs")

    # GPT-2 small
    num_layers = 12
    d_model = 768
    num_heads = 12
    paras = get_paras_number(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers, d_ff=d_ff)
    computation = get_computation(vocab_size=vocab_size, batch_size=batch_size, seq_len=context_length, num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff)
    print(f"GPT-2 small need {paras} parameters, and need {paras * data_size / (1024 ** 3)} GB to store parameters")
    print(f"GPT-2 small run a single forward need {computation} FLOPs")

    # GPT-2 medium
    num_layers = 24
    d_model = 1024
    num_heads = 16
    paras = get_paras_number(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers, d_ff=d_ff)
    computation = get_computation(vocab_size=vocab_size, batch_size=batch_size, seq_len=context_length, num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff)
    print(f"GPT-2 medium need {paras} parameters, and need {paras * data_size / (1024 ** 3)} GB to store parameters")
    print(f"GPT-2 medium run a single forward need {computation} FLOPs")

    # GPT-2 large
    num_layers = 36
    d_model = 1280
    num_heads = 20
    paras = get_paras_number(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers, d_ff=d_ff)
    computation = get_computation(vocab_size=vocab_size, batch_size=batch_size, seq_len=context_length, num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff)
    print(f"GPT-2 large need {paras} parameters, and need {paras * data_size / (1024 ** 3)} GB to store parameters")
    print(f"GPT-2 large run a single forward need {computation} FLOPs")
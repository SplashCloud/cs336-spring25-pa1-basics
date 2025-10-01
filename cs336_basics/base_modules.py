from torch import nn
import torch
import math
from einops import einsum, reduce, rearrange


def init_linear_weight(d_out: int, d_in: int, device = None, dtype = None) -> nn.Parameter:
    weight = torch.empty((d_out, d_in), device=device, dtype=dtype)
    std = math.sqrt(2 / (d_in + d_out))
    weight_init = nn.init.trunc_normal_(weight, mean=0, std=std, a=-3*std, b=3*std)
    return nn.Parameter(data=weight_init, requires_grad=True)


class Linear(nn.Module):

    def __init__(self, in_features, out_features, device = None, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        
        self.W = init_linear_weight(d_out=self.out_features, d_in=self.in_features, device=self.device, dtype=self.dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[-1] == self.in_features
        return einsum(input, self.W, "... d_in, d_out d_in -> ... d_out")

    
class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, device = None, dtype = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        weight = torch.empty((self.num_embeddings, self.embedding_dim), device=self.device, dtype=self.dtype)
        self.vocab = nn.Parameter(data=nn.init.trunc_normal_(weight, mean=0, std=1, a=-3, b=3), requires_grad=True)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        shape = token_ids.shape
        if not token_ids.is_contiguous():
            token_ids = token_ids.contiguous()
        token_ids = nn.functional.one_hot(token_ids.view(-1), num_classes=self.num_embeddings).view(*shape, self.num_embeddings)
        token_ids_dtype = token_ids.to(dtype=self.vocab.dtype)
        return einsum(token_ids_dtype, self.vocab, "... seq_len vocab_size, vocab_size d_embedding -> ... seq_len d_embedding")


class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(data=torch.ones(self.d_model), requires_grad=True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_model
        origin_type = x.dtype
        x_float32 = x.to(dtype=torch.float32)
        x_float32_squared_sum = reduce((x_float32 ** 2), "... d_model -> ... 1", 'sum')
        x_dtype_norm = einsum(x_float32 / torch.sqrt(x_float32_squared_sum / self.d_model + self.eps), self.g, "... d_model, d_model -> ... d_model")
        return x_dtype_norm.to(dtype=origin_type)


class SwiGLU(nn.Module):

    def __init__(self, d_model: int, d_ff: int, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        self.W1 = init_linear_weight(d_out=self.d_ff, d_in=self.d_model, device=self.device, dtype=self.dtype)
        self.W2 = init_linear_weight(d_out=self.d_model, d_in=self.d_ff, device=self.device, dtype=self.dtype)
        self.W3 = init_linear_weight(d_out=self.d_ff, d_in=self.d_model, device=self.device, dtype=self.dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[-1] == self.d_model
        apply_W1 = einsum(input, self.W1, "... d_model, d_ff d_model -> ... d_ff")
        apply_swi = einsum(apply_W1, torch.sigmoid(apply_W1), "... d_ff, ... d_ff -> ... d_ff")
        apply_W3 = einsum(input, self.W3, "... d_model, d_ff d_model -> ... d_ff")
        ele_wise_multiply = einsum(apply_swi, apply_W3, "... d_ff, ... d_ff -> ... d_ff")
        apply_W2 = einsum(ele_wise_multiply, self.W2, "... d_ff, d_model d_ff -> ... d_model")
        return apply_W2


class RoPE(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self._register_rotary_matrices() # generate and save the rotary matrices (max_seq_len, d_k, d_k)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_k
        assert x.shape[-2] == token_positions.shape[-1] and x.shape[-2] <= self.max_seq_len
        if not token_positions.is_contiguous():
            token_positions = token_positions.contiguous()
        token_positions_indices = nn.functional.one_hot(token_positions.view(-1), num_classes=self.max_seq_len).view(*token_positions.shape, self.max_seq_len)
        token_positions_indices = token_positions_indices.to(dtype=torch.float32, device=self.device)
        rotary_matrix = einsum(token_positions_indices, self.rotary_matrices, "... seq_len max_seq_len, max_seq_len d_k1 d_k2 -> ... seq_len d_k1 d_k2")
        return einsum(rotary_matrix, x, "... seq_len d_k1 d_k2, ... seq_len d_k2 -> ... seq_len d_k1")

    def _register_rotary_matrices(self):
        rotary_matrices = []
        for i in range(self.max_seq_len):
            rotary_matrix_for_i = []
            for k in range(1, self.d_k//2 + 1):
                theta = i * self.theta ** (-2*(k-1)/self.d_k)
                rotary_matrix_for_i.append(torch.Tensor([[math.cos(theta), -math.sin(theta)],
                                            [math.sin(theta), math.cos(theta)]], device=self.device))
            rotary_matrices.append(torch.block_diag(*rotary_matrix_for_i))
        self.register_buffer("rotary_matrices", torch.stack(tensors=rotary_matrices, dim=0))


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_embedding: int, d_attn: int,num_heads: int,
                       theta: float = 0, max_seq_len: int = 0, device = None, dtype = None):
        super().__init__()
        self.d_embedding = d_embedding
        self.d_attn = d_attn
        self.num_heads = num_heads
        self.d_k = d_attn // num_heads
        self.d_v = d_attn // num_heads
        self.device = device
        self.dtype = dtype

        self.WQ = init_linear_weight(d_out=self.num_heads*self.d_k, d_in=self.d_embedding, device=self.device, dtype=self.dtype)
        self.WK = init_linear_weight(d_out=self.num_heads*self.d_k, d_in=self.d_embedding, device=self.device, dtype=self.dtype)
        self.WV = init_linear_weight(d_out=self.num_heads*self.d_v, d_in=self.d_embedding, device=self.device, dtype=self.dtype)
        self.WO = init_linear_weight(d_out=self.d_attn, d_in=self.num_heads*self.d_v, device=self.device, dtype=self.dtype)

        self.enable_rope = ((theta != 0) and (max_seq_len != 0))
        if self.enable_rope:
            self.rope = RoPE(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=self.device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        assert x.shape[-1] == self.d_embedding # x.shape = (batch_size, ..., seq_len, d_embedding)
        Q = einsum(x, self.WQ, "... seq_len d_embedding, h_d_k d_embedding -> ... seq_len h_d_k")
        K = einsum(x, self.WK, "... seq_len d_embedding, h_d_k d_embedding -> ... seq_len h_d_k")
        V = einsum(x, self.WV, "... seq_len d_embedding, h_d_v d_embedding -> ... seq_len h_d_v")
        Q = rearrange(Q, "... seq_len (head d_k) -> ... head seq_len d_k", head=self.num_heads, d_k=self.d_k)
        K = rearrange(K, "... seq_len (head d_k) -> ... head seq_len d_k", head=self.num_heads, d_k=self.d_k)
        V = rearrange(V, "... seq_len (head d_v) -> ... head seq_len d_v", head=self.num_heads, d_v=self.d_v)
        if self.enable_rope:
            assert token_positions is not None
            Q = self.rope.forward(x=Q, token_positions=token_positions)
            K = self.rope.forward(x=K, token_positions=token_positions)
        mask = torch.tril(torch.ones(x.shape[-2], x.shape[-2]))
        mask = mask.to(dtype=torch.bool)
        Attn = scaled_dot_product_attention(Q, K, V, mask=mask)
        Attn = rearrange(Attn, "... head seq_len d_v -> ... seq_len (head d_v)")
        return einsum(Attn, self.WO, "... seq_len h_d_v, d_attn h_d_v -> ... seq_len d_attn")

## Functions

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_transpose = x.transpose(dim, len(x.shape)-1)
    x_max = reduce(x, "... dim -> ... 1", "max")
    x_norm_exp = torch.exp(x_transpose - x_max)
    x_norm_exp_sum = reduce(x_norm_exp, "... dim -> ... 1", "sum")
    x_softmax = x_norm_exp / x_norm_exp_sum
    return x_softmax.transpose(dim, len(x.shape)-1)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    QK = einsum(Q, K, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k")
    scaled_QK = QK / math.sqrt(Q.shape[-1])
    if mask is not None:
        masked_matrix = torch.full(mask.shape, -torch.inf, dtype=Q.dtype)
        masked_matrix[mask] = 0
        scaled_QK += masked_matrix
    softmax_scaled_QK = softmax(scaled_QK, dim=-1)
    return einsum(softmax_scaled_QK, V, "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v")
from torch import nn
import torch
import math
from einops import einsum, reduce


def init_linear_weight(weight: torch.Tensor):
    d_out, d_in = weight.shape
    std = math.sqrt(2 / (d_in + d_out))
    return nn.init.trunc_normal_(weight, mean=0, std=std, a=-3*std, b=3*std)


class Linear(nn.Module):

    def __init__(self, in_features, out_features, device = None, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        
        weight = torch.empty((self.out_features, self.in_features), device=self.device, dtype=self.dtype)
        std = math.sqrt(2 / (self.in_features + self.out_features))
        self.W = nn.Parameter(data=nn.init.trunc_normal_(weight, mean=0, std=std, a=-3*std, b=3*std), requires_grad=True)

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

        weight1 = torch.empty(size=(self.d_ff, self.d_model), device=self.device, dtype=self.dtype)
        weight2 = torch.empty(size=(self.d_model, self.d_ff), device=self.device, dtype=self.dtype)
        weight3 = torch.empty(size=(self.d_ff, self.d_model), device=self.device, dtype=self.dtype)
        self.W1 = nn.Parameter(data=init_linear_weight(weight1), requires_grad=True)
        self.W2 = nn.Parameter(data=init_linear_weight(weight2), requires_grad=True)
        self.W3 = nn.Parameter(data=init_linear_weight(weight3), requires_grad=True)

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

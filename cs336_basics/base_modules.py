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


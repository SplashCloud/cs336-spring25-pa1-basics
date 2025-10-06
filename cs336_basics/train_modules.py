import torch
from torch import nn
from einops import reduce, einsum, repeat

def cross_entropy(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    vocab_size = logit.shape[-1]
    # minus max value to keep value stability
    logit_max = reduce(logit, "... vocab_size -> ... 1", "max")
    logit_max = repeat(logit_max, "... 1 -> ... vocab_size", vocab_size=vocab_size)
    logit_norm = logit - logit_max
    # calculate the cross entropy for all positions
    logit_exp = torch.exp(logit_norm)
    logit_exp_sum = reduce(logit_exp, "... vocab_size -> ... 1", "sum")
    logit_exp_sum = repeat(logit_exp_sum, "... 1 -> ... vocab_size", vocab_size=vocab_size)
    logit_exp_sum_log = torch.log(logit_exp_sum)
    all_cross_entropy = logit_exp_sum_log - logit_norm # (..., vocab_size)
    # use target to indexing
    assert target.dtype == torch.int64 or target.dtype == torch.long
    slice_indices = tuple(torch.arange(d) for d in all_cross_entropy.shape[:-1])
    slice_indices += (target,)
    return all_cross_entropy[slice_indices].mean() # need first k-1 dimensions to be [0,...d-1]

def calculate_perplexity(loss: torch.Tensor) -> torch.Tensor:
    ''' loss[..., i] is the loss of token `i` in the sequence: loss(x_i | x_{1...i-1}) '''
    return torch.exp(torch.sum(loss, dim=-1) / loss.shape[-1]).mean()
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
    target_one_hot = nn.functional.one_hot(target.view(-1), num_classes=vocab_size).view(*target.shape, vocab_size).to(dtype=torch.float32)
    target_ce = einsum(all_cross_entropy, target_one_hot, "... vocab_size, ... vocab_size -> ... vocab_size")
    target_ce = reduce(target_ce, "... vocab_size -> ... 1", "sum")
    return reduce(target_ce, "... 1 -> 1", "mean")


def calculate_perplexity(loss: torch.Tensor) -> torch.Tensor:
    ''' loss[..., i] is the loss of token `i` in the sequence: loss(x_i | x_{1...i-1}) '''
    return torch.exp(torch.sum(loss, dim=-1) / loss.shape[-1]).mean()
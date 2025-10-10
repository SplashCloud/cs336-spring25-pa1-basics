import torch
from torch import nn
from einops import reduce, einsum, repeat
from typing import Iterable
import math

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

def learning_rate_schedule(current_t: int, max_lr: float, min_lr: float, warmup_t: int, cosine_annealing_t: int) -> float:
    if current_t < warmup_t:
        return (current_t / warmup_t) * max_lr # increase lr from 0 to max_lr linearly
    elif current_t <= cosine_annealing_t:
        # lr decrease from max_lr to min_lr smoothly
        return min_lr + (1 + math.cos((current_t - warmup_t)/(cosine_annealing_t - warmup_t)*math.pi)) / 2 * (max_lr - min_lr)
    else:
        return min_lr # at the end of training, keep small lr for stability

def gradient_clipping(params: Iterable[nn.Parameter], max_norm: float, eps: float = 1e-6):
    all_grads = [p.grad.data for p in params if p.grad is not None]
    # why need to concat the gradients of all the parameters and then calculate the norm?
    # To keep the direction of the gradient vector!
    # In one backward and step, all the parameters can be seen one single large vector
    # and the gradient of each parameter consist the gradient vector
    # if calculate for each gradient, the composed gradient vector will have the different directions
    total_norm = torch.norm(torch.stack(tuple(g.view(-1) for g in all_grads))) # for l2-norm, flatten the gradient is the same
    if total_norm >= max_norm:
        factor = max_norm / (total_norm + eps)
        for p in params:
            if p.grad is not None:
                p.grad.data *= factor

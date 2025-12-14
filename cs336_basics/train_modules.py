import torch
import numpy.typing as npt
import numpy as np
from torch import nn, optim
from einops import reduce, einsum, repeat
from typing import Iterable, BinaryIO, IO
import math
import os

def cross_entropy(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Optimized version: use PyTorch native operations and gather instead of creating large index tensors
    # This avoids creating many torch.arange tensors during backward pass

    # Subtract max for numerical stability (using PyTorch native operations instead of einops)
    logit_max = logit.max(dim=-1, keepdim=True)[0]  # (..., 1)
    logit_norm = logit - logit_max  # (..., vocab_size)
    
    # Compute log-sum-exp: log(sum(exp(logit_norm)))
    log_sum_exp = torch.log(torch.exp(logit_norm).sum(dim=-1, keepdim=True))  # (..., 1)
    
    # Get logits for target positions only using gather (more efficient than creating index tuples)
    # logit has shape (..., vocab_size), target has shape (...)
    target_logits = logit_norm.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)  # (...)
    
    # Cross-entropy = log-sum-exp - target_logit
    loss = log_sum_exp.squeeze(-1) - target_logits  # (...)
    
    return loss.mean()

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

def data_loading(x: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    l = len(x)
    max_start_idx = l - context_length - 1
    # Generate random indices directly without creating a huge list
    selected_ele = np.random.randint(0, max_start_idx + 1, size=batch_size)
    # Use advanced indexing to get all slices at once, then convert to tensor
    # This avoids creating many intermediate tensors and reduces CPU-GPU transfers
    indices = np.arange(context_length)[None, :] + selected_ele[:, None]  # (batch_size, context_length)
    s = torch.from_numpy(x[indices]).to(dtype=torch.int64, device=device)
    t = torch.from_numpy(x[indices + 1]).to(dtype=torch.int64, device=device)
    return s, t

def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]):
    model_state = model.state_dict()
    opti_state = optimizer.state_dict()
    states = {
        "model_state": model_state,
        "opti_state": opti_state,
        "iteration": iteration
    }
    torch.save(states, out)

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: nn.Module, optimizer: optim.Optimizer = None) -> int:
    states = torch.load(src)
    model.load_state_dict(states["model_state"])
    if optimizer != None:
        optimizer.load_state_dict(states["opti_state"])
    return states["iteration"]
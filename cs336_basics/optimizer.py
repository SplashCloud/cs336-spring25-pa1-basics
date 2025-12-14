from typing import Callable, Optional
from torch import optim, nn
import torch
import math

from cs336_basics.logger import LoggerManager
from cs336_basics.train_modules import learning_rate_schedule
from cs336_basics.config import TRAINING_LOG_FILE

logger = LoggerManager("OptimizerLogger", log_to_file=True, log_file_path=TRAINING_LOG_FILE)

class SGD(optim.Optimizer):

    def __init__(self, parameters, lr: float=1e-3):
        defaults = {"lr": lr}
        super().__init__(params=parameters, defaults=defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups: # only one dict in the self.param_groups
            lr = group["lr"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # self.state store the state associated with tensor p
                t = state.get("t", 0)

                p.data -= lr / math.sqrt(t + 1) * p.grad.data # update in-place

                state["t"] = t + 1
                
        return loss


class AdamW(optim.Optimizer):

    def __init__(self, params, lr: float, weight_decay: float, betas: tuple[float, float], eps: float, 
                        enable_lr_schedule: bool = False, max_lr: float = None, warmup_t: int = None, cosine_annealing_t: int = None):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": eps
        }
        if enable_lr_schedule:
            defaults["max_lr"] = max_lr
            defaults["warmup_t"] = warmup_t
            defaults["cosine_annealing_t"] = cosine_annealing_t
        self.enable_lr_schedule = enable_lr_schedule
        logger.info(f'enable_lr_schedule = {self.enable_lr_schedule} in AdamW...')
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        logging = False
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                m = state.get("m", 0)
                v = state.get("v", 0)
                t = state.get("t", 1)

                if self.enable_lr_schedule:
                    lr = learning_rate_schedule(t, group["max_lr"], lr, group["warmup_t"], group["cosine_annealing_t"])
                if not logging:
                    logger.info(f"learning rate = {lr} at iteration#{t}...")
                    logging = True

                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * torch.pow(grad, 2)
                alpha_t = lr * math.sqrt(1 - math.pow(beta2, t)) / (1 - math.pow(beta1, t))
                p.data -= alpha_t * (m / (torch.sqrt(v) + eps))
                p.data *= (1 - lr * weight_decay)

                state["m"] = m
                state["v"] = v
                state["t"] = t + 1

        return loss

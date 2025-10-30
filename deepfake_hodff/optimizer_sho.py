# deepfake_hodff/optimizer_sho.py
import math, random
from typing import Iterable, List
import torch
from torch.optim.optimizer import Optimizer

class SpottedHyena(Optimizer):
    """
    Simplified Spotted Hyena Optimizer (population-based) adapted to PyTorch params.
    Practical variant: acts as a multiplicative step-size controller over gradients.
    Use small hyena_steps per .step() to keep compute bounded.
    """
    def __init__(self, params, lr=1e-5, hyena_steps=1, a0=2.0, b0=1.0, seed=1337):
        defaults = dict(lr=lr, hyena_steps=hyena_steps, a0=a0, b0=b0)
        super().__init__(params, defaults)
        self.rng = random.Random(seed)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']; hs = group['hyena_steps']; a0 = group['a0']; b0 = group['b0']
            for p in group['params']:
                if p.grad is None: continue
                g = p.grad
                # alpha/beta coefficients simulate hyena encircling factors
                for _ in range(hs):
                    r1 = torch.rand_like(g)
                    r2 = torch.rand_like(g)
                    A = 2*a0*r1 - a0           # [-a0, +a0]
                    C = 2*r2                   # [0,2]
                    # prey estimate ~ gradient direction
                    D = torch.abs(C * g)
                    # position update: X(t+1) = X(t) - A * D * lr * b(t)
                    b = b0 * torch.rand_like(g)  # stochastic attack factor
                    p.add_( - A * D * lr * b )
        return loss

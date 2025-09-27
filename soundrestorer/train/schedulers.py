from __future__ import annotations

import math

import torch

__all__ = ["ensure_initial_lr", "make_warmup_cosine"]


def ensure_initial_lr(optimizer: torch.optim.Optimizer, default_lr: float):
    """
    PyTorch LambdaLR expects 'initial_lr' in each param group when resuming.
    """
    for g in optimizer.param_groups:
        if "initial_lr" not in g:
            g["initial_lr"] = g.get("lr", default_lr)


def make_warmup_cosine(
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int,
        already_done_steps: int,
        base_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Warmup (linear) → Cosine decay to 0 over total_steps.
    `already_done_steps` seeds the scheduler for resume.
    """

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        p = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * p))

    ensure_initial_lr(optimizer, base_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=already_done_steps - 1)
    return scheduler

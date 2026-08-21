"""Build the shared pre-training learning-rate scheduler.

Input is a DeepSpeed basic optimizer with one or more ordered parameter
groups. Output is a checkpoint-compatible ``WarmupCosineLR`` whose groups
are all initialized at optimizer step zero before training begins.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

from deepspeed.runtime.lr_schedules import WarmupCosineLR


INITIAL_SCHEDULER_STEP = 0


def initialize_warmup_cosine_scheduler(
    optimizer: Any,
    *,
    total_num_steps: int,
    warmup_num_steps: int,
    warmup_min_lr_ratio: float,
    cosine_min_ratio: float,
) -> WarmupCosineLR:
    """Return a consistently initialized multi-group cosine scheduler.

    Parameters
    ----------
    optimizer : Any
        DeepSpeed basic optimizer containing ordered parameter groups.
    total_num_steps : int
        Total number of successful optimizer updates.
    warmup_num_steps : int
        Number of optimizer updates assigned to warmup.
    warmup_min_lr_ratio : float
        Initial learning-rate multiplier for every parameter group.
    cosine_min_ratio : float
        Final cosine-decay learning-rate multiplier.

    Returns
    -------
    deepspeed.runtime.lr_schedules.WarmupCosineLR
        Scheduler initialized at optimizer step zero for every group.
    """
    scheduler = WarmupCosineLR(
        optimizer=optimizer,
        total_num_steps=total_num_steps,
        warmup_num_steps=warmup_num_steps,
        warmup_min_ratio=warmup_min_lr_ratio,
        cos_min_ratio=cosine_min_ratio,
        last_batch_iteration=INITIAL_SCHEDULER_STEP,
    )
    scheduler.step(last_batch_iteration=INITIAL_SCHEDULER_STEP)
    return scheduler


def warmup_cosine_scheduler_factory(
    *,
    total_num_steps: int,
    warmup_ratio: float,
    warmup_min_lr_ratio: float,
    cosine_min_ratio: float,
) -> Callable[[Any], WarmupCosineLR]:
    """Return the DeepSpeed client-scheduler factory for one campaign.

    Parameters
    ----------
    total_num_steps : int
        Total number of successful optimizer updates.
    warmup_ratio : float
        Fraction of total optimizer updates assigned to warmup.
    warmup_min_lr_ratio : float
        Initial learning-rate multiplier for every parameter group.
    cosine_min_ratio : float
        Final cosine-decay learning-rate multiplier.

    Returns
    -------
    Callable[[Any], WarmupCosineLR]
        Callable accepted by ``deepspeed.initialize``.
    """
    if total_num_steps < 1:
        raise ValueError(
            "Expected at least one optimizer step, got "
            f"{total_num_steps}."
        )
    for name, ratio in (
        ("warmup_ratio", warmup_ratio),
        ("warmup_min_lr_ratio", warmup_min_lr_ratio),
        ("cosine_min_ratio", cosine_min_ratio),
    ):
        if not 0 <= ratio <= 1:
            raise ValueError(
                f"Expected {name} in [0, 1], got {ratio}."
            )
    return partial(
        initialize_warmup_cosine_scheduler,
        total_num_steps=total_num_steps,
        warmup_num_steps=int(total_num_steps * warmup_ratio),
        warmup_min_lr_ratio=warmup_min_lr_ratio,
        cosine_min_ratio=cosine_min_ratio,
    )

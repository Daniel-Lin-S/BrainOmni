"""Warning-emitting in-memory integrity checks for pre-training.

The helpers inspect transient batches, losses, and gradients. They do not
write metrics, diagnostics, or sample identifiers to campaign artifacts.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping

import torch

MAXIMUM_REPORTED_NAMES = 5


def nonfinite_tensor_names(
    values: Mapping[str, object],
) -> list[str]:
    """Return names of floating or complex tensors with non-finite values.

    Parameters
    ----------
    values : Mapping[str, object]
        Named transient values. Non-tensor and integral values are ignored.

    Returns
    -------
    list[str]
        Names of tensors containing NaN or infinity.
    """
    names = []
    for name, value in values.items():
        if not isinstance(value, torch.Tensor):
            continue
        if not (value.is_floating_point() or value.is_complex()):
            continue
        if not torch.isfinite(value).all().item():
            names.append(name)
    return names


def nonfinite_gradient_names(module: torch.nn.Module) -> list[str]:
    """Return trainable parameter names with non-finite gradients.

    Parameters
    ----------
    module : torch.nn.Module
        Model module after backward propagation.

    Returns
    -------
    list[str]
        Parameter names whose present gradients contain NaN or infinity.
    """
    names = []
    for name, parameter in module.named_parameters():
        gradient = parameter.grad
        if not parameter.requires_grad or gradient is None:
            continue
        if not torch.isfinite(gradient).all().item():
            names.append(name)
    return names


def warn_and_raise_distributed_failure(
    local_names: list[str],
    subject: str,
    rank: int,
    device: torch.device | int,
    reduce_max: Callable[[torch.Tensor], None],
) -> None:
    """Emit a warning and stop all ranks when an integrity check fails.

    Parameters
    ----------
    local_names : list[str]
        Local values that failed the check.
    subject : str
        Human-readable check subject.
    rank : int
        Distributed rank issuing a local warning when applicable.
    device : torch.device or int
        Device used for the collective failure flag.
    reduce_max : Callable[[torch.Tensor], None]
        In-place maximum reduction across all training ranks.

    Raises
    ------
    FloatingPointError
        If any rank reports non-finite values.
    """
    if local_names:
        reported = ", ".join(local_names[:MAXIMUM_REPORTED_NAMES])
        remaining = len(local_names) - MAXIMUM_REPORTED_NAMES
        suffix = "" if remaining <= 0 else f", and {remaining} more"
        warnings.warn(
            f"Non-finite {subject} detected on rank {rank}: "
            f"{reported}{suffix}. Training will stop before an optimizer "
            "update.",
            RuntimeWarning,
            stacklevel=2,
        )
    failure = torch.tensor(
        int(bool(local_names)),
        dtype=torch.int32,
        device=device,
    )
    reduce_max(failure)
    if failure.item() != 0:
        raise FloatingPointError(
            f"Pre-training stopped because at least one rank had non-finite "
            f"{subject}. See the emitted warning for local details."
        )

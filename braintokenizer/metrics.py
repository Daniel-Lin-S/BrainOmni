"""Aggregate reconstruction quality for normalized neural-signal windows.

Inputs are matching tensors shaped ``(batch, channels, windows, samples)``
and sensor categories shaped ``(batch, channels)``. Output mappings group
scalar waveform and spectral metrics by modality. Spectral amplitude and
circular phase errors use the same definitions as the training objective.
"""

from collections.abc import Callable
import warnings

import torch

from factory.brain_constant import SENSOR_TYPE_DICT
from factory.pretraining_monitors import TensorSums, ensure_finite

from model_utils.loss import get_frequency_domain_loss


class MetricsComputer:
    """Accumulate trace-weighted reconstruction metrics by sensor modality."""

    def __init__(self) -> None:
        self.evaluate_func = {
            "mae": compute_mae,
            "mse": compute_mse,
            "amp": compute_amp,
            "phase": compute_phase,
            "pcc": compute_pcc,
        }
        self.sums = TensorSums()

    def step(
        self,
        rec: torch.Tensor,
        raw: torch.Tensor,
        sensor_type: torch.Tensor,
    ) -> None:
        """Accumulate matching B,C,W,T waveforms with B,C sensor types."""
        if rec.shape != raw.shape or rec.ndim != 4:
            raise ValueError(
                "Expected equal (batch, channels, windows, samples) shapes, "
                f"got {tuple(rec.shape)} and {tuple(raw.shape)}."
            )
        if tuple(sensor_type.shape) != tuple(rec.shape[:2]):
            raise ValueError(
                f"Expected sensor shape {tuple(rec.shape[:2])}, "
                f"got {tuple(sensor_type.shape)}."
            )
        rec, raw = rec.detach().float(), raw.detach().float()
        ensure_finite(rec, "evaluation reconstruction")
        ensure_finite(raw, "evaluation target")
        masks = {
            "all": torch.ones_like(sensor_type, dtype=torch.bool),
            "eeg": sensor_type == SENSOR_TYPE_DICT["EEG"],
            "meg": sensor_type != SENSOR_TYPE_DICT["EEG"],
        }
        for group, mask in masks.items():
            count = mask.sum().double() * rec.shape[2]
            self.sums.add(f"{group}_count", count)
            for name, function in self.evaluate_func.items():
                value = count.new_zeros(())
                if count > 0:
                    value = function(
                        rec[mask].unsqueeze(0), raw[mask].unsqueeze(0),
                    ).double()
                    ensure_finite(value, f"evaluation {group} {name}")
                self.sums.add(f"{group}_{name}", value * count)

    def reduce_(self, reduce_sum: Callable[[torch.Tensor], None]) -> None:
        """Sum sufficient statistics across evaluation ranks."""
        self.sums.reduce_(reduce_sum)

    def get_metrics(self) -> dict[str, dict[str, float]]:
        """Return weighted finite metrics, omitting absent modalities."""
        if self.sums.require("all_count") <= 0:
            raise ValueError("No reconstruction traces were evaluated.")
        metrics = {}
        for group in ("all", "eeg", "meg"):
            count = self.sums.require(f"{group}_count")
            if count <= 0:
                warnings.warn(
                    f"No {group} evaluation traces; omitting this modality.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            metrics[group] = {}
            for name in self.evaluate_func:
                value = self.sums.require(f"{group}_{name}") / count
                ensure_finite(value, f"evaluation {group} {name}")
                metrics[group][name] = value.item()
        return metrics

    def reset(self) -> None:
        """Discard the accumulated reconstruction statistics."""
        self.sums = TensorSums()


def compute_mae(rec, raw):
    mae = torch.abs(rec - raw)
    return torch.mean(mae)


def compute_mse(rec, raw):
    mse = torch.square(rec - raw)
    return torch.mean(mse)


def compute_amp(rec: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    """Return the training spectral amplitude error.

    Parameters
    ----------
    rec, raw : torch.Tensor
        Matching waveforms shaped ``(batch, channels, windows, samples)``.

    Returns
    -------
    torch.Tensor
        Scalar mean absolute spectral amplitude difference.
    """
    amplitude, _ = get_frequency_domain_loss(rec, raw)
    return amplitude


def compute_phase(rec: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    """Return the training circular phase error.

    Parameters
    ----------
    rec, raw : torch.Tensor
        Matching waveforms shaped ``(batch, channels, windows, samples)``.

    Returns
    -------
    torch.Tensor
        Scalar mean shortest angular difference in radians.
    """
    _, phase = get_frequency_domain_loss(rec, raw)
    return phase


def compute_pcc(rec: torch.Tensor, raw: torch.Tensor):
    # B C W
    B, C, W, D = rec.shape
    x = rec.reshape(B * C * W, 1, D)
    y = raw.reshape(B * C * W, 1, D)
    c = (
        (x - x.mean(dim=-1, keepdim=True))
        @ ((y - y.mean(dim=-1, keepdim=True)).transpose(1, 2))
        * (1.0 / (D - 1))
    ).squeeze()
    sigma = (torch.std(x, dim=-1) * torch.std(y, dim=-1)).squeeze() + 1e-6
    return (c / sigma).mean()

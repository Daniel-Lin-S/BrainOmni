"""Waveform and spectral losses for aligned neural-signal windows.

Inputs are matching tensors of shape ``(batch, channels, windows, samples)``.
Outputs are scalar losses. Spectral comparison uses an orthonormal real FFT
with a Hamming window; phase error is the shortest angular distance in
radians. Frequency bins retain equal weight.
"""

import torch
from einops import rearrange


def get_pcc(rec: torch.Tensor, raw: torch.Tensor):
    # B C W
    B, C, W, D = rec.shape
    x = rearrange(rec, "B C W D->(B C W) 1 D")
    y = rearrange(raw, "B C W D -> (B C W) 1 D")
    c = (
        (x - x.mean(dim=-1, keepdim=True))
        @ ((y - y.mean(dim=-1, keepdim=True)).transpose(1, 2))
        * (1.0 / (D - 1))
    ).squeeze()
    sigma = (torch.std(x, dim=-1) * torch.std(y, dim=-1)).squeeze() + 1e-6
    return (c / sigma).mean()


def compute_l1_loss(rec, raw):
    """
    rec  B C W D
    raw  B C W D
    """
    l1_distance = torch.abs(rec - raw)
    return torch.mean(l1_distance)


def get_time_loss(predicted, target):
    """
    rec  B C W D
    raw  B C W D
    """
    return compute_l1_loss(predicted, target)


def get_frequency_domain_loss(
    predicted: torch.Tensor, target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return amplitude L1 and circular phase L1 for matching waveforms.

    Parameters
    ----------
    predicted, target : torch.Tensor
        Finite, non-empty signals with equal shape and a final sample axis.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Scalar amplitude error and mean angular error in ``[0, pi]``.
    """
    if predicted.shape != target.shape or predicted.ndim == 0:
        raise ValueError(
            "Expected matching waveform shapes with a sample axis, got "
            f"{tuple(predicted.shape)} and {tuple(target.shape)}."
        )
    if predicted.numel() == 0:
        raise ValueError("Spectral losses require non-empty waveforms.")
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise ValueError("Spectral losses require finite waveforms.")
    predicted, target = predicted.float(), target.float()
    window = torch.hamming_window(target.shape[-1], device=predicted.device)
    predicted = window * predicted
    target = window * target

    pred_fft = torch.fft.rfft(predicted, dim=-1, norm="ortho")
    target_fft = torch.fft.rfft(target, dim=-1, norm="ortho")

    pred_magnitude = torch.abs(pred_fft)
    target_magnitude = torch.abs(target_fft)

    pred_phase = torch.angle(pred_fft)
    target_phase = torch.angle(target_fft)

    magnitude_loss = compute_l1_loss(pred_magnitude, target_magnitude)
    phase_difference = pred_phase - target_phase
    phase_loss = torch.atan2(
        torch.sin(phase_difference), torch.cos(phase_difference)
    ).abs().mean()
    if not torch.isfinite(magnitude_loss) or not torch.isfinite(phase_loss):
        raise ValueError("Spectral losses produced non-finite values.")
    return magnitude_loss, phase_loss

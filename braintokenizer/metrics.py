"""Aggregate reconstruction quality for normalized neural-signal windows.

Inputs are matching tensors shaped ``(batch, channels, windows, samples)``
and sensor categories shaped ``(batch, channels)``. Output mappings group
scalar waveform and spectral metrics by modality. Spectral amplitude and
circular phase errors use the same definitions as the training objective.
"""

import torch

from model_utils.loss import get_frequency_domain_loss


class MetricsComputer:
    """
    Accumulate reconstruction metrics for one evaluation setting.
    """

    def __init__(self):
        self.evaluate_func = {
            "mae": compute_mae,
            "mse": compute_mse,
            "amp": compute_amp,
            "phase": compute_phase,
            "pcc": compute_pcc,
        }
        self.record = []

    def step(
        self,
        rec: torch.Tensor,
        raw: torch.Tensor,
        sensor_type: torch.Tensor,
    ):
        rec = rec.detach().float()
        raw = raw.detach().float()

        cur_metrics = {
            key: self.evaluate_func[key](rec, raw).item()
            for key in self.evaluate_func.keys()
        }
        cur_metrics["is_eeg"] = (sensor_type == 0).all()
        self.record.append(cur_metrics)

    def get_metrics(self):
        metrics = {"all": {}, "eeg": {}, "meg": {}}
        for key in self.evaluate_func.keys():
            metrics["all"][key] = (
                torch.tensor([i[key] for i in self.record]).mean().item()
            )
            metrics["eeg"][key] = (
                torch.tensor([i[key] for i in self.record if i["is_eeg"]])
                .mean().item()
            )
            metrics["meg"][key] = (
                torch.tensor([i[key] for i in self.record if not i["is_eeg"]])
                .mean()
                .item()
            )
        return metrics

    def reset(self):
        self.record = []


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

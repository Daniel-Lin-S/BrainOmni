"""Compute lightweight pre-training monitoring sufficient statistics.

The module accepts transient model outputs and returns scalar quantities for
TensorBoard. It never writes activations, token assignments, attention maps, or
parameter snapshots to disk. Stage-specific trainers own cadence and logging.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from factory.brain_constant import SENSOR_TYPE_DICT

EEG_SENSOR_TYPE = SENSOR_TYPE_DICT["EEG"]
MONITOR_EPSILON = 1.0e-12
TAG_PARTS = 4
VALID_CADENCES = {"epoch", "micro_step", "step"}
VALID_SPLITS = {"evaluation", "train", "validation"}


def canonical_tag(
    split: str,
    cadence: str,
    family: str,
    metric: str,
    dimension: str | None = None,
) -> str:
    """Return one validated canonical TensorBoard tag.

    Parameters
    ----------
    split : str
        One of ``train``, ``validation``, or ``evaluation``.
    cadence : str
        One of ``step``, ``epoch``, or ``micro_step``.
    family : str
        Non-empty metric family.
    metric : str
        Non-empty metric name.
    dimension : str, optional
        Optional final grouping component, by default ``None``.

    Returns
    -------
    str
        Slash-delimited canonical tag.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown monitor split {split!r}.")
    if cadence not in VALID_CADENCES:
        raise ValueError(f"Unknown monitor cadence {cadence!r}.")
    parts = [split, cadence, family, metric]
    if dimension is not None:
        parts.append(dimension)
    if any(not part or "/" in part for part in parts):
        raise ValueError(f"Invalid monitor tag components: {parts!r}.")
    return "/".join(parts)


def level_name(index: int) -> str:
    """Return a zero-padded RVQ level label."""
    if index < 0:
        raise ValueError(f"RVQ level index must be non-negative, got {index}.")
    return f"level_{index:02d}"


def source_name(index: int) -> str:
    """Return a zero-padded latent-source label."""
    if index < 0:
        raise ValueError(
            f"Latent-source index must be non-negative, got {index}."
        )
    return f"source_{index:02d}"


def modality_channel_groups(
    sensor_type: torch.Tensor,
) -> dict[str, list[tuple[torch.Tensor, torch.Tensor]]]:
    """Group samples with matching EEG or MEG channel selections.

    Parameters
    ----------
    sensor_type : torch.Tensor
        Sensor-category indices with shape ``(batch, channels)``. Category
        zero is EEG; all other categories are MEG.

    Returns
    -------
    dict[str, list[tuple[torch.Tensor, torch.Tensor]]]
        Per-modality sample indices and channel masks. Mixed EMEG samples
        appear in both modality collections.
    """
    if sensor_type.ndim != 2 or sensor_type.numel() == 0:
        raise ValueError(
            "Expected non-empty sensor types with shape (batch, channels), "
            f"got {tuple(sensor_type.shape)}."
        )
    result: dict[
        str,
        list[tuple[torch.Tensor, torch.Tensor]],
    ] = {"eeg": [], "meg": []}
    for name, channel_selector in (
        ("eeg", sensor_type == EEG_SENSOR_TYPE),
        ("meg", sensor_type != EEG_SENSOR_TYPE),
    ):
        grouped_samples: dict[tuple[bool, ...], list[int]] = {}
        for sample in range(sensor_type.shape[0]):
            key = tuple(channel_selector[sample].tolist())
            if any(key):
                grouped_samples.setdefault(key, []).append(sample)
        for key, samples in grouped_samples.items():
            sample_indices = torch.tensor(
                samples,
                dtype=torch.long,
                device=sensor_type.device,
            )
            channel_mask = torch.tensor(
                key,
                dtype=torch.bool,
                device=sensor_type.device,
            )
            result[name].append((sample_indices, channel_mask))
    return result


def ensure_finite(value: torch.Tensor, name: str) -> None:
    """Raise when a monitor tensor contains NaN or infinity."""
    if not torch.isfinite(value).all():
        raise ValueError(f"Monitor {name} contains non-finite values.")


def checked_ratio(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """Return an elementwise ratio with a strictly positive denominator."""
    ensure_finite(numerator, f"{name} numerator")
    ensure_finite(denominator, f"{name} denominator")
    if torch.any(denominator <= 0):
        raise ValueError(
            f"Monitor {name} requires positive denominators, got "
            f"{denominator.detach().cpu().tolist()}."
        )
    result = numerator / denominator
    ensure_finite(result, name)
    return result


def assignment_metrics(counts: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute RVQ utilisation and perplexity from assignment counts.

    Parameters
    ----------
    counts : torch.Tensor
        Non-negative assignment counts of shape ``(levels, codebook_size)``.

    Returns
    -------
    dict[str, torch.Tensor]
        Per-level utilisation, perplexity, normalized perplexity, and
        normalized entropy tensors of shape ``(levels,)``.
    """
    if counts.ndim != 2 or counts.shape[1] < 2:
        raise ValueError(
            "Expected RVQ assignment counts with shape "
            f"(levels, codebook_size>=2), got {tuple(counts.shape)}."
        )
    if torch.any(counts < 0):
        raise ValueError("RVQ assignment counts must be non-negative.")
    totals = counts.sum(dim=-1)
    probabilities = checked_ratio(
        counts,
        totals.unsqueeze(-1),
        "RVQ assignment probabilities",
    )
    entropy = -torch.where(
        probabilities > 0,
        probabilities * torch.log(probabilities),
        torch.zeros_like(probabilities),
    ).sum(dim=-1)
    codebook_size = counts.shape[1]
    perplexity = torch.exp(entropy)
    utilization = (counts > 0).sum(dim=-1) / codebook_size
    return {
        "utilization": utilization,
        "perplexity": perplexity,
        "perplexity_normalized": perplexity / codebook_size,
        "entropy_normalized": entropy / math.log(codebook_size),
    }


def residual_energy_reduction(
    input_energy: torch.Tensor,
    output_energy: torch.Tensor,
) -> torch.Tensor:
    """Compute per-level fractional RVQ residual-energy reduction."""
    ratio = checked_ratio(
        output_energy,
        input_energy,
        "RVQ residual-energy ratio",
    )
    result = 1.0 - ratio
    ensure_finite(result, "RVQ residual-energy reduction")
    return result


def covariance_from_sums(
    count: torch.Tensor,
    sums: torch.Tensor,
    cross_products: torch.Tensor,
) -> torch.Tensor:
    """Return a population covariance matrix from sufficient statistics."""
    if count.numel() != 1 or count.item() <= 0:
        raise ValueError(
            "Source covariance requires a positive scalar observation count, "
            f"got {count.detach().cpu().tolist()}."
        )
    if sums.ndim != 1:
        raise ValueError(
            f"Expected source sums with shape (sources,), got {sums.shape}."
        )
    expected = (sums.numel(), sums.numel())
    if tuple(cross_products.shape) != expected:
        raise ValueError(
            f"Expected source cross-products with shape {expected}, got "
            f"{tuple(cross_products.shape)}."
        )
    covariance = cross_products / count
    covariance -= torch.outer(sums / count, sums / count)
    covariance = (covariance + covariance.transpose(0, 1)) / 2.0
    ensure_finite(covariance, "source covariance")
    return covariance


def mean_absolute_off_diagonal_correlation(
    covariance: torch.Tensor,
) -> torch.Tensor:
    """Return mean absolute off-diagonal source correlation."""
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(
            "Expected a square source covariance matrix, got "
            f"{tuple(covariance.shape)}."
        )
    if covariance.shape[0] < 2:
        raise ValueError("Source correlation requires at least two sources.")
    variance = covariance.diagonal()
    if torch.any(variance <= 0):
        raise ValueError(
            "Source correlation requires positive source variances, got "
            f"{variance.detach().cpu().tolist()}."
        )
    scale = torch.sqrt(torch.outer(variance, variance))
    correlation = covariance / scale
    mask = ~torch.eye(
        covariance.shape[0],
        dtype=torch.bool,
        device=covariance.device,
    )
    result = correlation[mask].abs().mean()
    ensure_finite(result, "mean absolute source correlation")
    return result


def effective_rank(covariance: torch.Tensor) -> torch.Tensor:
    """Return entropy-based effective rank of a source covariance matrix."""
    eigenvalues = torch.linalg.eigvalsh(covariance.double()).clamp_min(0.0)
    total = eigenvalues.sum()
    if total <= MONITOR_EPSILON:
        raise ValueError(
            "Source effective rank requires positive covariance energy."
        )
    probabilities = eigenvalues / total
    entropy = -torch.where(
        probabilities > 0,
        probabilities * torch.log(probabilities),
        torch.zeros_like(probabilities),
    ).sum()
    result = torch.exp(entropy)
    ensure_finite(result, "source effective rank")
    return result


def attention_similarity_statistics(
    attention: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sum and count of off-diagonal query cosine similarities.

    Parameters
    ----------
    attention : torch.Tensor
        Attention weights with shape
        ``(batch, windows, heads, sources, sensors)``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Scalar similarity sum and observation count.
    """
    if attention.ndim != 5 or attention.shape[-2] < 2:
        raise ValueError(
            "Expected attention with shape "
            "(batch, windows, heads, sources>=2, sensors), got "
            f"{tuple(attention.shape)}."
        )
    normalized = torch.nn.functional.normalize(
        attention.double(),
        p=2.0,
        dim=-1,
        eps=MONITOR_EPSILON,
    )
    similarity = normalized @ normalized.transpose(-2, -1)
    source_count = attention.shape[-2]
    mask = ~torch.eye(
        source_count,
        dtype=torch.bool,
        device=attention.device,
    )
    selected = similarity[..., mask]
    count = torch.tensor(
        selected.numel(),
        dtype=torch.float64,
        device=attention.device,
    )
    result = selected.sum()
    ensure_finite(result, "inter-query attention similarity sum")
    return result, count


def attention_similarity(attention: torch.Tensor) -> torch.Tensor:
    """Return mean off-diagonal cosine similarity between source queries."""
    similarity_sum, count = attention_similarity_statistics(attention)
    return checked_ratio(
        similarity_sum,
        count,
        "inter-query attention similarity",
    )


def successful_optimizer_steps(engine: Any) -> int:
    """Return the number of non-skipped DeepSpeed optimizer updates."""
    global_steps = int(engine.global_steps)
    skipped_steps = int(engine.skipped_steps)
    if skipped_steps < 0 or skipped_steps > global_steps:
        raise ValueError(
            "DeepSpeed step counters are inconsistent: "
            f"global_steps={global_steps}, skipped_steps={skipped_steps}."
        )
    return global_steps - skipped_steps


def monitor_due(
    engine: Any,
    interval: int,
    accumulation_boundary: bool,
) -> bool:
    """Return whether the next successful optimizer update is on cadence."""
    if not isinstance(interval, int) or isinstance(interval, bool):
        raise TypeError(
            "Monitoring interval must be a positive integer, got "
            f"{interval!r}."
        )
    if interval <= 0:
        raise ValueError(
            f"Monitoring interval must be positive, got {interval}."
        )
    return (
        accumulation_boundary
        and (successful_optimizer_steps(engine) + 1) % interval == 0
    )


def zero_partition_snapshot(engine: Any) -> list[torch.Tensor]:
    """Clone the local ZeRO-2 FP32 weight partitions in memory."""
    partitions = getattr(
        engine.optimizer,
        "single_partition_of_fp32_groups",
        None,
    )
    if not partitions:
        raise RuntimeError(
            "Update-to-weight monitoring requires DeepSpeed ZeRO-2 FP32 "
            "partitions, but the optimizer exposes none."
        )
    return [partition.detach().clone() for partition in partitions]


def update_to_weight_sums(
    before: list[torch.Tensor],
    engine: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local squared update and pre-update weight norms."""
    after = getattr(
        engine.optimizer,
        "single_partition_of_fp32_groups",
        None,
    )
    if after is None or len(before) != len(after):
        raise RuntimeError(
            "ZeRO-2 parameter partitions changed while measuring an update."
        )
    device = after[0].device
    update_sum = torch.zeros((), dtype=torch.float64, device=device)
    weight_sum = torch.zeros((), dtype=torch.float64, device=device)
    for old, new in zip(before, after, strict=True):
        old_device = old.to(device=device, dtype=torch.float64)
        new_device = new.detach().to(device=device, dtype=torch.float64)
        update_sum += torch.square(new_device - old_device).sum()
        weight_sum += torch.square(old_device).sum()
    return update_sum, weight_sum


def optimizer_step_values(
    engine: Any,
    learning_rates: float | Sequence[float],
    group_names: Sequence[str],
) -> dict[str, torch.Tensor | float]:
    """Return learning-rate and global-gradient scalars for one update."""
    rates = (
        list(learning_rates)
        if isinstance(learning_rates, Sequence)
        else [learning_rates]
    )
    if len(rates) != len(group_names):
        raise RuntimeError(
            f"Optimizer monitoring expected {len(group_names)} learning "
            f"rates, got {len(rates)}."
        )
    values: dict[str, torch.Tensor | float] = {}
    for name, rate in zip(group_names, rates, strict=True):
        values[
            canonical_tag(
                "train",
                "step",
                "optimization",
                "learning_rate",
                name,
            )
        ] = rate
    gradient_norm = engine.get_global_grad_norm()
    if gradient_norm is None:
        raise RuntimeError(
            "DeepSpeed did not expose a global gradient norm after a "
            "successful optimizer update."
        )
    values[
        canonical_tag(
            "train",
            "step",
            "optimization",
            "gradient_norm",
            "global",
        )
    ] = gradient_norm
    return values


def distributed_update_to_weight_ratio(
    before: list[torch.Tensor],
    engine: Any,
    reduce_sum: Callable[[torch.Tensor], None],
    device: torch.device | int,
) -> torch.Tensor:
    """Return the global ZeRO-2 update-to-pre-update-weight norm ratio."""
    update_sum, weight_sum = update_to_weight_sums(before, engine)
    norm_sums = torch.stack((update_sum, weight_sum)).to(
        device=device,
        dtype=torch.float64,
    )
    reduce_sum(norm_sums)
    result = torch.sqrt(norm_sums[0]) / (
        torch.sqrt(norm_sums[1]) + MONITOR_EPSILON
    )
    ensure_finite(result, "global update-to-weight ratio")
    return result


class TensorSums:
    """Accumulate named transient tensors and reduce them across ranks."""

    def __init__(self) -> None:
        self.values: dict[str, torch.Tensor] = {}

    def add(self, name: str, value: torch.Tensor | float) -> None:
        """Add one detached value to a named sufficient statistic."""
        tensor = torch.as_tensor(value).detach().to(
            dtype=torch.float64,
        ).contiguous()
        ensure_finite(tensor, name)
        if name not in self.values:
            self.values[name] = tensor.clone()
            return
        if self.values[name].shape != tensor.shape:
            raise ValueError(
                f"Monitor {name} expected shape "
                f"{tuple(self.values[name].shape)}, got {tuple(tensor.shape)}."
            )
        self.values[name] += tensor.to(self.values[name].device)

    def reduce_(self, reduce_sum: Callable[[torch.Tensor], None]) -> None:
        """Sum every statistic in place across distributed workers."""
        if not self.values:
            raise ValueError("Cannot reduce an empty monitor accumulator.")
        for value in self.values.values():
            reduce_sum(value)

    def require(self, name: str) -> torch.Tensor:
        """Return a statistic or raise a clear missing-input error."""
        if name not in self.values:
            raise ValueError(f"Monitor statistic {name!r} was not accumulated.")
        return self.values[name]


def _trace_pcc_sums(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    channel_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_rec = reconstruction[channel_mask]
    selected_target = target[channel_mask]
    if selected_rec.numel() == 0:
        zero = reconstruction.new_zeros((), dtype=torch.float64)
        return zero, zero
    rec = selected_rec.reshape(-1, selected_rec.shape[-1]).double()
    raw = selected_target.reshape(-1, selected_target.shape[-1]).double()
    rec_centered = rec - rec.mean(dim=-1, keepdim=True)
    raw_centered = raw - raw.mean(dim=-1, keepdim=True)
    numerator = (rec_centered * raw_centered).sum(dim=-1)
    denominator = torch.sqrt(
        torch.square(rec_centered).sum(dim=-1)
        * torch.square(raw_centered).sum(dim=-1)
    )
    valid = denominator > MONITOR_EPSILON
    if not valid.any():
        zero = reconstruction.new_zeros((), dtype=torch.float64)
        return zero, zero
    correlations = numerator[valid] / denominator[valid]
    return correlations.sum(), valid.sum().double()


def reconstruction_statistics(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    channel_masks: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return reconstruction sums and counts for named channel strata."""
    if reconstruction.shape != target.shape or reconstruction.ndim != 4:
        raise ValueError(
            "Expected reconstruction and target with equal shape "
            "(batch, channels, windows, samples), got "
            f"{tuple(reconstruction.shape)} and {tuple(target.shape)}."
        )
    expected_mask = reconstruction.shape[:2]
    result: dict[str, torch.Tensor] = {}
    difference = reconstruction.double() - target.double()
    for name, channel_mask in channel_masks.items():
        if tuple(channel_mask.shape) != expected_mask:
            raise ValueError(
                f"Reconstruction stratum {name!r} expected mask shape "
                f"{expected_mask}, got {tuple(channel_mask.shape)}."
            )
        selected = difference[channel_mask]
        result[f"{name}_absolute_sum"] = selected.abs().sum()
        result[f"{name}_squared_sum"] = torch.square(selected).sum()
        result[f"{name}_element_count"] = torch.tensor(
            selected.numel(),
            dtype=torch.float64,
            device=difference.device,
        )
        pcc_sum, pcc_count = _trace_pcc_sums(
            reconstruction,
            target,
            channel_mask,
        )
        result[f"{name}_pcc_sum"] = pcc_sum
        result[f"{name}_pcc_count"] = pcc_count
    return result


def source_statistics(source_latent: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return per-source sums, squares, and cross-products.

    Parameters
    ----------
    source_latent : torch.Tensor
        Pre-RVQ tensor with shape
        ``(batch, sources, outer_windows, inner_windows, dimension)``.

    Returns
    -------
    dict[str, torch.Tensor]
        Sufficient statistics over all non-source dimensions.
    """
    if source_latent.ndim != 5:
        raise ValueError(
            "Expected source latent shape "
            "(batch, sources, outer_windows, inner_windows, dimension), got "
            f"{tuple(source_latent.shape)}."
        )
    matrix = source_latent.double().movedim(1, 0).reshape(
        source_latent.shape[1],
        -1,
    )
    count = torch.tensor(
        matrix.shape[1],
        dtype=torch.float64,
        device=matrix.device,
    )
    return {
        "source_count": count,
        "source_sum": matrix.sum(dim=-1),
        "source_square_sum": torch.square(matrix).sum(dim=-1),
        "source_cross_product": matrix @ matrix.transpose(0, 1),
    }


def source_variance(
    count: torch.Tensor,
    sums: torch.Tensor,
    square_sums: torch.Tensor,
) -> torch.Tensor:
    """Return per-source population activation variance."""
    mean = checked_ratio(sums, count, "source activation mean")
    second_moment = checked_ratio(
        square_sums,
        count,
        "source activation second moment",
    )
    variance = second_moment - torch.square(mean)
    variance = variance.clamp_min(0.0)
    ensure_finite(variance, "source activation variance")
    return variance


def write_scalars(
    writer: Any,
    values: Mapping[str, torch.Tensor | float],
    step: int,
) -> None:
    """Write finite scalar values to TensorBoard in deterministic tag order."""
    for tag in sorted(values):
        value = torch.as_tensor(values[tag]).detach().double()
        if value.numel() != 1:
            raise ValueError(
                f"TensorBoard scalar {tag!r} expected one value, got shape "
                f"{tuple(value.shape)}."
            )
        ensure_finite(value, tag)
        writer.add_scalar(tag=tag, scalar_value=value.item(), global_step=step)

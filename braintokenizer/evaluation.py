"""Evaluate frozen BrainTokenizer weights using prepared neural tensors.

Input: a model in evaluation mode and batches containing x (B,C,T), pos
(B,C,6), and sensor_type (B,C). Metadata JSON is a nonempty list with absolute
processed-tensor path, dataset identity, and channel count for each segment.
Output: unmasked reconstruction metrics grouped as all/eeg/meg and a separate
validation_monitors mapping with the same definitions as validation epochs.
Training-only diagnostics are excluded. No optimizer or training is run.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from accessor import DataAccessor
from braintokenizer.metrics import MetricsComputer
from factory.pretraining_monitor_runtime import StageOneAccumulator
from factory.pretraining_monitors import (
    attention_similarity_statistics,
    checked_ratio,
    ensure_finite,
)
from constant import PRETRAIN_DTYPE
from pretrain_dataset import BrainDataset, collate_fn
from pretrain_config import sha256_file

INPUT_KEYS = ("x", "pos", "sensor_type")
VALIDATION_PREFIX = "validation/epoch/"
EVALUATOR_PATH = Path(__file__).resolve()
EVALUATOR_SOURCES = (
    "braintokenizer/evaluation.py",
    "braintokenizer/model.py",
    "braintokenizer/metrics.py",
    "model_utils/attn.py",
    "model_utils/conv.py",
    "model_utils/lstm.py",
    "model_utils/module.py",
    "model_utils/seanet.py",
    "model_utils/loss.py",
    "model_utils/vq.py",
    "factory/pretraining_monitor_runtime.py",
    "factory/pretraining_monitors.py",
    "pretrain_dataset.py",
    "accessor.py",
)


def evaluation_settings(
    model: torch.nn.Module,
    seed: int,
    batch_size: int,
    world_size: int,
) -> dict[str, Any]:
    """Describe the actual inference protocol and implementation identity.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded tokenizer supplying parameter dtype and channel masking ratio.
    seed : int
        Base validation-mask seed, incremented by rank for distributed use.
    batch_size : int
        Maximum input segments per rank and batch.
    world_size : int
        Number of ranks partitioning deterministic evaluation batches.

    Returns
    -------
    dict[str, Any]
        JSON-compatible settings and source digests saved with the metrics.
    """
    root = EVALUATOR_PATH.parents[1]
    return {
        "schema_version": 1,
        "seed": seed,
        "batch_size": batch_size,
        "world_size": world_size,
        "dtype": str(next(model.parameters()).dtype),
        "input_cast_dtype": str(PRETRAIN_DTYPE),
        "channel_mask_ratio": model.mask_ratio,
        "input_noise_enabled": False,
        "reconstruction_protocol": "all_channels_visible",
        "monitor_protocol": "validation_channel_masking",
        "attention_scope": "first_deterministic_batch_per_rank",
        "implementation_sha256": {
            name: sha256_file(root / name) for name in EVALUATOR_SOURCES
        },
    }


def build_evaluation_loader(
    metadata_path: Path,
    batch_size: int,
    num_workers: int,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """Partition deterministic channel-compatible batches without duplicates.

    Parameters
    ----------
    metadata_path : Path
        JSON list of prepared segment records, each with path and channels.
    batch_size : int
        Maximum segments per batch.
    num_workers : int
        Nonnegative number of tensor-reading workers.
    rank : int, optional
        Evaluation rank, default 0.
    world_size : int, optional
        Number of evaluation ranks, default 1.

    Returns
    -------
    DataLoader
        Each segment occurs on exactly one rank, including final partial
        batches. Tensor shapes are B,C,T for x and B,C,6 for pos.
    """
    if batch_size <= 0 or num_workers < 0 or not 0 <= rank < world_size:
        raise ValueError("Invalid evaluation batch size, workers, or rank.")
    rows = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected nonempty evaluation rows: {metadata_path}.")
    groups: dict[int, list[int]] = {}
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Expected metadata mapping at row {index}.")
        channels, path = row.get("channels"), row.get("path")
        if type(channels) is not int or channels <= 0:
            raise ValueError(
                f"Invalid channel count {channels!r} at row {index}."
            )
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError(f"Expected absolute tensor path at row {index}.")
        if path in seen:
            raise ValueError(f"Duplicate evaluation tensor: {path}.")
        seen.add(path)
        groups.setdefault(channels, []).append(index)
    batches = []
    for channels in sorted(groups):
        indices = sorted(groups[channels], key=lambda i: rows[i]["path"])
        batches.extend(
            indices[start:start + batch_size]
            for start in range(0, len(indices), batch_size)
        )
    if len(batches) < world_size:
        raise ValueError(
            f"Only {len(batches)} batches for {world_size} evaluation ranks. "
            "Use fewer ranks or a smaller batch size."
        )
    return DataLoader(
        BrainDataset(rows, DataAccessor()),
        batch_sampler=batches[rank::world_size],
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


@torch.no_grad()
def evaluate_tokenizer(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    codebook_size: int,
    seed: int,
    reduce_sum: Callable[[torch.Tensor], None] | None = None,
    progress: bool = False,
    reconstruction_callback: Callable[[int, Mapping], None] | None = None,
) -> dict[str, Any]:
    """Compute reconstruction and validation diagnostics without training.

    Parameters
    ----------
    model : torch.nn.Module
        Frozen BrainTokenizer in evaluation mode.
    loader : Iterable[Mapping[str, Any]]
        Batches with x (B,C,T), pos (B,C,6), sensor_type (B,C).
    codebook_size : int
        Number of entries per RVQ codebook.
    seed : int
        Seed for validation channel masking; caller RNG state is preserved.
    reduce_sum : Callable or None, optional
        In-place distributed sum, default None for single-process evaluation.
    progress : bool, optional
        Show batch progress, default False.
    reconstruction_callback : Callable or None, optional
        Receive batch index and unmasked outputs for optional visualization.
        Default None. Callback RNG draws do not affect validation masks.

    Returns
    -------
    dict[str, Any]
        Finite reconstruction groups plus validation_monitors and sample_count.
        Attention similarity uses the first deterministic batch per rank,
        matching the fixed-batch validation diagnostic. Other validation
        monitors aggregate the full dataset. Training-only diagnostics are
        excluded.
    """
    if model.training:
        raise ValueError("BrainTokenizer must be in evaluation mode.")
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    devices = [device.index] if device.type == "cuda" else []
    reconstruction = MetricsComputer()
    monitors = StageOneAccumulator(codebook_size)
    sample_count = torch.zeros((), device=device, dtype=torch.float64)
    attention_totals = torch.zeros(2, device=device, dtype=torch.float64)
    attention_samples = torch.zeros((), device=device, dtype=torch.float64)
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        for index, batch in enumerate(
            tqdm(loader, disable=not progress, unit="batch")
        ):
            inputs = {
                key: batch[key].to(
                    device=device,
                    dtype=dtype if key != "sensor_type" else batch[key].dtype,
                )
                for key in INPUT_KEYS
            }
            for key, value in inputs.items():
                ensure_finite(value, f"evaluation input {key}")
            if sample_count == 0:
                attention = model.monitor_attention(**inputs)
                attention_totals += torch.stack(
                    attention_similarity_statistics(attention)
                )
                attention_samples += inputs["x"].shape[0]
            visual = model.visualize(**inputs)
            if reconstruction_callback is not None:
                with torch.random.fork_rng(devices=devices):
                    reconstruction_callback(index, visual)
            reconstruction.step(
                visual["x_rec"], visual["x"], visual["sensor_type"],
            )
            output, _, monitor = model(**inputs, return_monitor_data=True)
            monitors.update(output, monitor)
            sample_count += inputs["x"].shape[0]
    if sample_count <= 0:
        raise ValueError("Evaluation loader produced no samples.")
    if reduce_sum is not None:
        reconstruction.reduce_(reduce_sum)
        monitors.reduce_(reduce_sum)
        reduce_sum(sample_count)
        reduce_sum(attention_totals)
        reduce_sum(attention_samples)
    values = monitors.validation_values()
    scalar_monitors = {}
    for name, value in values.items():
        if not name.startswith(VALIDATION_PREFIX):
            raise ValueError(f"Unexpected non-validation monitor {name!r}.")
        ensure_finite(value, f"held-out {name}")
        scalar_monitors[name.removeprefix(VALIDATION_PREFIX)] = value.item()
    scalar_monitors["latent_source/inter_query_attention_similarity"] = (
        checked_ratio(
            attention_totals[0], attention_totals[1],
            "held-out inter-query attention similarity",
        ).item()
    )
    return {
        "reconstruction": reconstruction.get_metrics(),
        "validation_monitors": scalar_monitors,
        "sample_count": int(sample_count.item()),
        "attention_sample_count": int(attention_samples.item()),
    }

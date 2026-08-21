"""Read canonical and historical pre-training TensorBoard scalar events.

Inputs are one or more absolute TensorBoard directories or campaign roots.
The module discovers event files recursively and returns lightweight scalar
records. CSV output is created only when explicitly requested by the caller.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from factory.pretraining_monitors import TAG_PARTS, VALID_CADENCES, VALID_SPLITS

EVENT_GLOB = "events.out.tfevents.*"
LEGACY_ENTROPY_PATTERN = re.compile(
    r"^eval_codebook_utilize_entropy_(?P<dimension>\d+|mean)$"
)
LEGACY_SCALAR_PATTERN = re.compile(
    r"^(?P<split>train|eval)_(?P<metric>"
    r"loss|judge_loss|time_loss|pcc|amp_loss|phase_loss|"
    r"commitment_loss|acc_all|acc_(?P<level>\d+))$"
)


@dataclass(frozen=True)
class MonitorEvent:
    """One normalized scalar event suitable for tables and plotting."""

    run: str
    source_file: str
    original_tag: str
    tag: str
    split: str
    cadence: str
    family: str
    metric: str
    dimension: str
    step: int
    wall_time: float
    value: float


def _legacy_scalar_tag(match: re.Match[str]) -> str:
    split = "train" if match.group("split") == "train" else "validation"
    metric = match.group("metric")
    mapping = {
        "loss": ("objective", "optimized_loss", ""),
        "judge_loss": ("objective", "optimized_loss", ""),
        "time_loss": ("reconstruction", "time_loss", ""),
        "pcc": ("reconstruction", "pcc", ""),
        "amp_loss": ("reconstruction", "amplitude_loss", ""),
        "phase_loss": ("reconstruction", "phase_loss", ""),
        "commitment_loss": ("rvq", "commitment_loss", ""),
        "acc_all": ("masked_token", "accuracy", "mean"),
    }
    if metric.startswith("acc_") and metric != "acc_all":
        family = "masked_token"
        name = "accuracy"
        dimension = f"level_{int(match.group('level')):02d}"
    else:
        family, name, dimension = mapping[metric]
    parts = [split, "epoch", family, name]
    if dimension:
        parts.append(dimension)
    return "/".join(parts)


def canonicalize_tag(tag: str) -> tuple[str, int]:
    """Return a canonical tag and duplicate-selection priority.

    Lower priority numbers win. Historical ``judge_loss`` loses to ``loss``
    because both contain the same detached scalar in Stage 1.
    """
    parts = tag.split("/")
    if (
        len(parts) >= TAG_PARTS
        and parts[0] in VALID_SPLITS
        and parts[1] in VALID_CADENCES
    ):
        return tag, 0
    entropy = LEGACY_ENTROPY_PATTERN.fullmatch(tag)
    if entropy:
        dimension = entropy.group("dimension")
        if dimension != "mean":
            dimension = f"level_{int(dimension):02d}"
        return (
            (
                "validation/epoch/rvq/assignment_entropy_normalized/"
                f"{dimension}"
            ),
            0,
        )
    scalar = LEGACY_SCALAR_PATTERN.fullmatch(tag)
    if scalar:
        priority = 1 if scalar.group("metric") == "judge_loss" else 0
        return _legacy_scalar_tag(scalar), priority
    return tag, 0


def parse_canonical_tag(tag: str) -> tuple[str, str, str, str, str]:
    """Split a canonical tag into normalized table columns."""
    parts = tag.split("/")
    if (
        len(parts) < TAG_PARTS
        or parts[0] not in VALID_SPLITS
        or parts[1] not in VALID_CADENCES
    ):
        return "", "", "", "", ""
    dimension = "/".join(parts[4:])
    return parts[0], parts[1], parts[2], parts[3], dimension


def discover_event_files(inputs: Iterable[str | Path]) -> list[Path]:
    """Discover TensorBoard event files below absolute input paths."""
    files: set[Path] = set()
    input_count = 0
    for item in inputs:
        input_count += 1
        path = Path(item)
        if not path.is_absolute():
            raise ValueError(
                f"TensorBoard input path must be absolute, got {path}."
            )
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"TensorBoard input path does not exist: {path}."
            )
        if path.is_file():
            if not path.name.startswith("events.out.tfevents."):
                raise ValueError(
                    f"Input is not a TensorBoard event file: {path}."
                )
            files.add(path)
        else:
            files.update(
                candidate.resolve()
                for candidate in path.rglob(EVENT_GLOB)
            )
    if input_count == 0:
        raise ValueError("At least one TensorBoard input path is required.")
    if not files:
        raise FileNotFoundError(
            "No TensorBoard event files were found below the requested inputs."
        )
    return sorted(files)


def _event_accumulator(path: Path) -> Any:
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as error:
        raise RuntimeError(
            "TensorBoard is required to extract monitor events. Install the "
            "versions pinned in the repository requirements before retrying."
        ) from error
    accumulator = EventAccumulator(
        str(path),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    return accumulator


def load_monitor_events(inputs: Iterable[str | Path]) -> list[MonitorEvent]:
    """Load and normalize scalar records from TensorBoard event files."""
    selected: dict[tuple[str, str, int], tuple[int, MonitorEvent]] = {}
    for event_file in discover_event_files(inputs):
        accumulator = _event_accumulator(event_file)
        run = str(event_file.parent.resolve())
        scalar_tags = accumulator.Tags().get("scalars", [])
        for original_tag in scalar_tags:
            tag, priority = canonicalize_tag(original_tag)
            split, cadence, family, metric, dimension = parse_canonical_tag(tag)
            for scalar in accumulator.Scalars(original_tag):
                record = MonitorEvent(
                    run=run,
                    source_file=str(event_file),
                    original_tag=original_tag,
                    tag=tag,
                    split=split,
                    cadence=cadence,
                    family=family,
                    metric=metric,
                    dimension=dimension,
                    step=int(scalar.step),
                    wall_time=float(scalar.wall_time),
                    value=float(scalar.value),
                )
                key = (run, tag, record.step)
                previous = selected.get(key)
                replace = (
                    previous is None
                    or priority < previous[0]
                    or (
                        priority == previous[0]
                        and record.wall_time > previous[1].wall_time
                    )
                )
                if replace:
                    selected[key] = (priority, record)
    return sorted(
        (item[1] for item in selected.values()),
        key=lambda event: (event.run, event.tag, event.step),
    )


def write_monitor_csv(
    events: Iterable[MonitorEvent],
    output: str | Path,
) -> Path:
    """Write normalized monitor events to an explicitly requested CSV."""
    output_path = Path(output)
    if not output_path.is_absolute():
        raise ValueError(
            f"Monitor CSV output path must be absolute, got {output_path}."
        )
    rows = [asdict(event) for event in events]
    if not rows:
        raise ValueError("Cannot write an empty monitor CSV.")
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path

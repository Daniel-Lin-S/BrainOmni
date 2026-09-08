"""Render Stage-1 scalar monitoring from one provenanced training attempt.

Input
-----
An absolute TensorBoard directory directly inside a campaign attempt. The
campaign's ``campaign_identity.json`` supplies stage and dataset modalities;
``model_cfg.json`` supplies the expected RVQ level count. Event files supply
canonical or legacy scalar tags via ``load_monitor_events``.

Output
------
PNG and PDF figures in losses/, reconstruction/<stratum>/, latent_source/,
and rvq/ under a sibling visualisation/ directory (or explicit destination).
manifest.json records provenance, input event files, each figure's source
curves and transformations, generated relative filenames, and missing tags.
Only prior manifest-listed generated figures may be removed on a rerun.
"""

from __future__ import annotations

import json
import math
import re
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from factory.pretraining_monitor_events import (
    MonitorEvent,
    discover_event_files,
    load_monitor_events,
)

STAGES = ("braintokenizer", "brainomni")
FORMATS = ("png", "pdf")
FIGURE_SIZE = (10, 7)
FIGURE_DPI = 180
TITLE_FONT_SIZE = 24
LABEL_FONT_SIZE = 22
TICK_FONT_SIZE = 20
LEGEND_FONT_SIZE = 18
LEGEND_COLUMNS = 2
NOTE_FONT_SIZE = 16
LINE_WIDTH = 3
TICK_LENGTH = 7
TICK_WIDTH = 1.5
TITLE_WRAP_WIDTH = 48
NOTE_WRAP_WIDTH = 65
FIGURE_MARGIN = 0.03
LineStyle = str | tuple[int, tuple[int, ...]]
LINE_STYLES: tuple[LineStyle, ...] = (
    "-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)),
)
PHASE_WEIGHT = 0.5
MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA = 1
GENERATOR = "brainomni.pretraining_visualisation"
LEVEL_PATTERN = re.compile(r"level_\d+")
FIGURE_FAMILIES = {"losses", "reconstruction", "latent_source", "rvq"}
LOSS_COMPONENTS = (
    ("objective/optimized_loss", "Optimized total", 1.0),
    ("reconstruction/time_loss", "Time", 1.0),
    ("reconstruction/amplitude_loss", "Amplitude", 1.0),
    ("reconstruction/phase_loss", "Phase × 0.5", PHASE_WEIGHT),
    ("reconstruction/pcc_loss", "PCC loss", 1.0),
    ("rvq/commitment_loss", "Commitment", 1.0),
)
RVQ_METRICS = {
    "assignment_perplexity_normalized": "Normalised assignment perplexity",
    "assignment_utilization": "Codebook utilisation",
    "quantization_error": "Quantisation error",
    "residual_energy_reduction": "Residual-energy reduction",
}
PCC_NOTE = (
    "PCC loss is exp(-aggregated PCC); component and total aggregation "
    "can differ. Weighted curves need not sum exactly to the logged total."
)


@dataclass(frozen=True)
class Curve:
    """One plotted series and its provenance; steps and values have shape N."""

    label: str
    tag: str
    steps: list[int]
    values: list[float]
    transformation: str
    colour: int
    style: LineStyle


@dataclass(frozen=True)
class FigureSpec:
    """A figure with independent series coordinates and missing-tag notes."""

    name: str
    title: str
    ylabel: str
    xlabel: str
    curves: list[Curve]
    missing: list[str]
    note: str = ""


def read_provenance(
    directory: Path,
    stage: str | None = None,
) -> dict[str, Any]:
    """Validate the stage against both campaign identity stage fields.

    Parameters
    ----------
    directory : Path
        Resolved absolute TensorBoard directory within one attempt.
    stage : str or None, optional
        Expected stage, by default None to infer it from provenance.

    Returns
    -------
    dict[str, Any]
        Parsed campaign identity. Invalid provenance, a stage mismatch,
        or an unsupported plotting stage raises ValueError.
    """
    if stage is not None and stage not in STAGES:
        raise ValueError(f"Expected stage in {STAGES}, got {stage!r}.")
    if not directory.is_absolute() or not directory.is_dir():
        raise ValueError(
            f"Expected an absolute TensorBoard directory: {directory}."
        )
    if directory.parent.parent.name != "attempts":
        raise ValueError(
            "Expected a TensorBoard directory directly inside one campaign "
            f"attempt, got {directory}."
        )
    identity_path = directory.parents[2] / "campaign_identity.json"
    try:
        identity = json.loads(identity_path.read_text())
        actual = identity["stage"]
        embedded = identity["semantic_payload"]["campaign"]["stage"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(
            f"Missing or malformed campaign provenance: {identity_path}."
        ) from error
    if actual not in STAGES or actual != embedded:
        raise ValueError(
            f"Inconsistent stage provenance in {identity_path}: "
            f"stage={actual!r}, semantic stage={embedded!r}."
        )
    if stage is not None and stage != actual:
        raise ValueError(
            f"Requested stage {stage!r}, but {identity_path} identifies "
            f"stage {actual!r}. Use --stage {actual}."
        )
    if actual == "brainomni":
        raise ValueError(
            "This visualisation command supports braintokenizer; "
            f"campaign stage is {actual!r}."
        )
    return identity


def campaign_dimensions(
    directory: Path,
    identity: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Read selected modalities and RVQ dimensions from campaign provenance.

    Returns
    -------
    tuple[bool, list[str]]
        Whether both EEG and MEG were selected, and ordered RVQ level names.
    """
    model_path = directory.parents[2] / "model_cfg.json"
    try:
        campaign = identity["semantic_payload"]["campaign"]
        data = campaign["data"]
        datasets = data["included_datasets"]
        modalities = {
            data["dataset_signal_types"][name].lower() for name in datasets
        }
        model = json.loads(model_path.read_text())
        levels = model["num_quantizers"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise ValueError(
            "Missing or malformed modality/RVQ provenance in campaign "
            f"identity or {model_path}."
        ) from error
    if not modalities or not modalities <= {"eeg", "meg", "emeg"}:
        raise ValueError(f"Invalid selected dataset modalities: {modalities}.")
    if type(levels) is not int or levels <= 0:
        raise ValueError(f"Expected positive RVQ level count, got {levels!r}.")
    joint = "emeg" in modalities or {"eeg", "meg"} <= modalities
    return joint, [f"level_{level:02d}" for level in range(levels)]


def select_curve(
    series: dict[str, list[MonitorEvent]],
    tag: str,
    label: str,
    colour: int,
    style: LineStyle,
    weight: float = 1.0,
) -> Curve | None:
    """Select a finite scalar series, allowing the exact L1-to-MAE alias.

    Parameters
    ----------
    series : dict[str, list[MonitorEvent]]
        Canonical tags mapped to events sorted by step.
    tag : str
        Requested canonical tag.
    label : str
        Legend label.
    colour : int
        Stable categorical palette index.
    style : str or tuple
        Matplotlib line style.
    weight : float, optional
        Multiplier applied to recorded values, by default 1.0.

    Returns
    -------
    Curve or None
        Selected coordinates and transformation, or None for a missing tag.
        Nonfinite values raise ValueError with the source tag and step.
    """
    source = tag
    transformation = "identity"
    if tag not in series and "/reconstruction/mae" in tag:
        source = tag.replace("/reconstruction/mae", "/reconstruction/time_loss")
        transformation = "MAE from equivalent mean absolute (L1) time loss"
    events = series.get(source)
    if not events:
        return None
    values = [event.value * weight for event in events]
    for event, value in zip(events, values):
        if not math.isfinite(value):
            raise ValueError(
                f"Nonfinite plotted value for {source} at step {event.step}: "
                f"{value}, event file {event.source_file}."
            )
    if weight != 1.0:
        transformation = f"multiply by {weight}"
    return Curve(
        label, source, [event.step for event in events], values,
        transformation, colour, style,
    )


def build_figures(
    events: list[MonitorEvent],
    joint_modalities: bool,
    levels: list[str],
) -> list[FigureSpec]:
    """Plan all Stage-1 figures without writing files.

    Parameters
    ----------
    events : list[MonitorEvent]
        Normalized scalar events for one attempt.
    joint_modalities : bool
        Include EEG and MEG strata only for a jointly selected campaign.
    levels : list[str]
        Expected RVQ level names, ordered by quantizer index.

    Returns
    -------
    list[FigureSpec]
        Figure specifications, including missing curves and empty figures
        for the manifest. Rendering omits empty figures.
    """
    series: dict[str, list[MonitorEvent]] = {}
    for event in events:
        series.setdefault(event.tag, []).append(event)
        if (
            event.family == "rvq"
            and LEVEL_PATTERN.fullmatch(event.dimension)
            and event.dimension not in levels
        ):
            raise ValueError(
                f"Event RVQ dimension {event.dimension!r} is absent from "
                f"campaign levels {levels}."
            )
    for points in series.values():
        points.sort(key=lambda point: point.step)
    figures = []

    def add(
        name: str,
        title: str,
        ylabel: str,
        requests: list[tuple[str, str, float, int, LineStyle]],
        xlabel: str = "Epoch",
        note: str = "",
    ) -> None:
        """Collect requested curves without filling missing observations."""
        curves = []
        missing = []
        for tag, label, weight, colour, style in requests:
            curve = select_curve(series, tag, label, colour, style, weight)
            if curve is None:
                missing.append(tag)
            else:
                curves.append(curve)
        figures.append(FigureSpec(
            name, title, ylabel, xlabel, curves, missing, note,
        ))

    cadence = (
        "step" if "train/step/objective/optimized_loss" in series else "epoch"
    )
    add(
        f"losses/training_{cadence}",
        "BrainTokenizer training loss and weighted components",
        "Loss",
        [
            (f"train/{cadence}/{tag}", label, weight, index,
             LINE_STYLES[index % len(LINE_STYLES)])
            for index, (tag, label, weight) in enumerate(LOSS_COMPONENTS)
        ],
        "Optimizer step" if cadence == "step" else "Epoch",
        PCC_NOTE,
    )
    strata = ["all"]
    if joint_modalities:
        strata.extend(("eeg", "meg"))
    for stratum in strata:
        suffix = "" if stratum == "all" else f"/{stratum}"
        for metric in ("pcc", "mae", "mse"):
            add(
                f"reconstruction/{stratum}/{metric}",
                f"{stratum.upper()} channels: reconstruction {metric.upper()}",
                metric.upper(),
                [
                    (f"{split}/epoch/reconstruction/{metric}{suffix}",
                     split.capitalize(), 1.0, index, LINE_STYLES[index])
                    for index, split in enumerate(("train", "validation"))
                ],
            )
    for metric in ("pcc", "mae", "mse"):
        add(
            f"reconstruction/dropped_vs_visible/{metric}",
            f"Dropped vs visible channels: {metric.upper()}",
            metric.upper(),
            [
                (f"{split}/epoch/reconstruction/{metric}/{stratum}",
                 f"{stratum.capitalize()} — {split}", 1.0,
                 group_index, LINE_STYLES[split_index])
                for group_index, stratum in enumerate(("dropped", "visible"))
                for split_index, split in enumerate(("train", "validation"))
            ],
        )
    for metric, title in (
        ("mean_absolute_correlation", "Mean absolute inter-source correlation"),
        ("effective_rank", "Latent-source effective rank"),
    ):
        add(
            f"latent_source/{metric}", title, title,
            [(f"validation/epoch/latent_source/{metric}",
              "Validation", 1.0, 1, "--")],
        )
    for metric, title in RVQ_METRICS.items():
        add(
            f"rvq/{metric}", f"Training: {title}", title,
            [
                (f"train/epoch/rvq/{metric}/{level}",
                 f"Level {int(level.removeprefix('level_'))}",
                 1.0, index, LINE_STYLES[index % len(LINE_STYLES)])
                for index, level in enumerate(levels)
            ],
        )
    if not any(figure.curves for figure in figures):
        raise ValueError("No plottable Stage-1 scalar curves were found.")
    return figures


def generated_path(root: Path, name: str) -> Path:
    """Validate a manifest-owned figure path before overwriting or deleting."""
    relative = Path(name)
    path = (root / relative).resolve()
    if (
        relative.is_absolute()
        or not relative.parts
        or relative.parts[0] not in FIGURE_FAMILIES
        or relative.suffix.removeprefix(".") not in FORMATS
        or not path.is_relative_to(root)
    ):
        raise ValueError(f"Unsafe generated figure path in manifest: {name!r}.")
    return path


def render_figure(spec: FigureSpec, root: Path) -> list[str]:
    """Save PNG/PDF versions of a populated specification using Agg."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    names = []
    try:
        palette = plt.get_cmap("tab10")
        for curve in spec.curves:
            axis.plot(
                curve.steps, curve.values, label=curve.label,
                color=palette(curve.colour % palette.N),
                linewidth=LINE_WIDTH,
                linestyle=curve.style, marker="." if len(curve.steps) == 1
                else None,
            )
        axis.set_title(
            textwrap.fill(spec.title, TITLE_WRAP_WIDTH),
            fontsize=TITLE_FONT_SIZE,
        )
        axis.set_xlabel(spec.xlabel, fontsize=LABEL_FONT_SIZE)
        axis.set_ylabel(spec.ylabel, fontsize=LABEL_FONT_SIZE)
        axis.tick_params(
            axis="both", which="both", labelsize=TICK_FONT_SIZE,
            length=TICK_LENGTH, width=TICK_WIDTH,
        )
        for coordinate in (axis.xaxis, axis.yaxis):
            coordinate.get_offset_text().set_fontsize(TICK_FONT_SIZE)
        axis.grid(alpha=0.25)
        legend = figure.legend(
            *axis.get_legend_handles_labels(),
            loc="lower center", bbox_to_anchor=(0.5, FIGURE_MARGIN),
            fontsize=LEGEND_FONT_SIZE,
            ncol=min(LEGEND_COLUMNS, len(spec.curves)),
            borderaxespad=0,
        )
        figure.canvas.draw()
        legend_height = legend.get_window_extent(
            figure.canvas.get_renderer()
        ).height / figure.dpi
        width, height = figure.get_size_inches()
        figure.set_size_inches(
            width, height + legend_height + FIGURE_MARGIN * height,
        )
        notes = [spec.note] if spec.note else []
        if spec.missing:
            missing = ", ".join(
                " ".join((tag.split("/")[0], *tag.split("/")[3:]))
                for tag in spec.missing
            )
            notes.append(f"Unavailable: {missing}")
        bottom_margin = FIGURE_MARGIN
        if notes:
            wrapped = "\n".join(
                textwrap.fill(note, NOTE_WRAP_WIDTH) for note in notes
            )
            footer = figure.text(
                FIGURE_MARGIN, FIGURE_MARGIN, wrapped,
                fontsize=NOTE_FONT_SIZE, va="bottom",
            )
            figure.canvas.draw()
            bounds = footer.get_window_extent(
                figure.canvas.get_renderer()
            ).transformed(figure.transFigure.inverted())
            bottom_margin = bounds.y1 + FIGURE_MARGIN
        legend.set_bbox_to_anchor((0.5, bottom_margin))
        figure.canvas.draw()
        legend_bounds = legend.get_window_extent(
            figure.canvas.get_renderer()
        ).transformed(figure.transFigure.inverted())
        figure.tight_layout(
            rect=(0, legend_bounds.y1 + FIGURE_MARGIN, 1, 1),
        )
        for extension in FORMATS:
            name = f"{spec.name}.{extension}"
            destination = generated_path(root, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(destination, dpi=FIGURE_DPI)
            names.append(name)
    finally:
        plt.close(figure)
    return names


def visualize_pretraining(
    tensorboard_dir: str | Path,
    stage: str | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Validate, plot, and record the visualisation manifest for one attempt.

    Parameters
    ----------
    tensorboard_dir : str or Path
        Absolute directory containing one attempt's TensorBoard events.
    stage : str or None, optional
        Expected stage, by default None to infer it from provenance.
        An explicit value must match the campaign stage.
    output_dir : str or Path or None, optional
        Absolute figure destination. By default None, selecting the sibling
        ``visualisation`` directory. Must be disjoint from the event tree.

    Returns
    -------
    Path
        Absolute manifest path. Missing curves warn and are recorded;
        wholly missing figures are omitted. Validation precedes output writes.
    """
    directory = Path(tensorboard_dir)
    if not directory.is_absolute():
        raise ValueError("TensorBoard directory must be an absolute path.")
    directory = directory.resolve()
    identity = read_provenance(directory, stage)
    joint, levels = campaign_dimensions(directory, identity)
    event_files = discover_event_files([directory])
    if any(path.parent != directory for path in event_files):
        raise ValueError(
            f"Expected one event directory, found nested runs in {directory}."
        )
    events = load_monitor_events([directory])
    figures = build_figures(events, joint, levels)
    root = Path(output_dir) if output_dir is not None else (
        directory.parent / "visualisation"
    )
    if not root.is_absolute():
        raise ValueError("Visualisation output must be an absolute path.")
    root = root.resolve()
    if root.is_relative_to(directory) or directory.is_relative_to(root):
        raise ValueError(
            f"Visualisation output {root} must be disjoint from {directory}."
        )
    manifest_path = root / MANIFEST_NAME
    previous_files = []
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if (
            previous.get("generator") != GENERATOR
            or previous.get("tensorboard_dir") != str(directory)
        ):
            raise ValueError(
                f"Output manifest belongs to another source: {manifest_path}."
            )
        previous_files = previous["generated_files"]
    for name in previous_files:
        generated_path(root, name)
    for spec in figures:
        for extension in FORMATS:
            generated_path(root, f"{spec.name}.{extension}")
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "generator": GENERATOR,
        "tensorboard_dir": str(directory),
        "stage": identity["stage"],
        "campaign_identity": identity,
        "event_files": [str(path) for path in event_files],
        "joint_modalities": joint,
        "rvq_levels": levels,
        "figures": [
            {
                "name": spec.name,
                "title": spec.title,
                "xlabel": spec.xlabel,
                "ylabel": spec.ylabel,
                "missing": spec.missing,
                "note": spec.note,
                "curves": [
                    {
                        "label": curve.label,
                        "tag": curve.tag,
                        "transformation": curve.transformation,
                        "point_count": len(curve.steps),
                        "first_step": curve.steps[0],
                        "last_step": curve.steps[-1],
                    }
                    for curve in spec.curves
                ],
            }
            for spec in figures
        ],
        "generated_files": [],
    }
    root.mkdir(parents=True, exist_ok=True)
    for spec in figures:
        if spec.missing:
            warnings.warn(
                f"Figure {root / spec.name}: missing curves: "
                + ", ".join(spec.missing)
                + (". Figure omitted." if not spec.curves else "."),
                RuntimeWarning,
                stacklevel=2,
            )
        if spec.curves:
            manifest["generated_files"].extend(render_figure(spec, root))
    stale = set(previous_files) - set(manifest["generated_files"])
    for name in stale:
        generated_path(root, name).unlink(missing_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temporary.replace(manifest_path)
    return manifest_path

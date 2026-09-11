"""Build Stage-2 figure specifications for the shared visualisation entry point.

Input: canonical scalar events, active RVQ levels, and selected modalities.
Output: specifications under optimization/ and masked_token/{ce,accuracy}/,
including per-level baseline, corruption, and modality comparisons. Values
retain their logged coordinates; accuracy fractions convert to percentages.
"""

from functools import partial

from factory.pretraining_monitor_events import MonitorEvent
from factory.pretraining_visualisation import (
    FigureSpec, LINE_STYLES, append_figure, index_series,
)

STAGE_TWO_LINE_WIDTH = 2
PERCENT = 100.0
VALIDATION = "validation/epoch/masked_token"
MODALITY_NOTE = (
    "EEG and MEG use modality-only validation inputs; overall uses complete "
    "inputs. Curves are independently logged measurements."
)


def build_stage_two_figures(
    events: list[MonitorEvent], joint_modalities: bool, levels: list[str],
) -> list[FigureSpec]:
    """Construct requested Stage-2 plots from recorded scalar observations.

    Parameters
    ----------
    events : list[MonitorEvent]
        Canonical scalar observations for a single attempt.
    joint_modalities : bool
        Whether campaign-selected datasets contain both EEG and MEG.
    levels : list[str]
        Ordered RVQ level names, restricted to num_quantizers_used.

    Returns
    -------
    list[FigureSpec]
        Plot specifications with explicit missing-tag notes. Empty figures
        are retained for the manifest and omitted by the shared renderer.
    """
    series = index_series(events, levels)
    figures = []
    add = partial(
        append_figure, figures, series, linewidth=STAGE_TWO_LINE_WIDTH,
    )
    for metric, title in (
        ("gradient_norm", "Gradient norm"),
        ("learning_rate", "Learning rate"),
        ("update_to_weight_ratio", "Update-to-weight ratio"),
    ):
        prefix = f"train/step/optimization/{metric}/"
        tags = sorted(tag for tag in series if tag.startswith(prefix))
        if not tags:
            group = "main" if metric == "learning_rate" else "global"
            tags = [prefix + group]
        sparse = [tag for tag in tags if len(series.get(tag, [])) == 1]
        note = "; ".join(
            f"{tag.rsplit('/', 1)[-1]}: one observation at optimizer step "
            f"{series[tag][0].step}." for tag in sparse
        )
        add(
            f"optimization/{metric}", title, title,
            [(tag, tag.rsplit("/", 1)[-1], 1.0, index,
              LINE_STYLES[index % len(LINE_STYLES)])
             for index, tag in enumerate(tags)],
            xlabel="Optimizer step", note=note,
        )
    for split, filename in (
        ("train", "training"), ("validation", "validation"),
    ):
        prefix = f"{split}/epoch/masked_token/cross_entropy"
        total = f"{prefix}/total"
        add(
            f"masked_token/ce/{filename}",
            f"{filename.capitalize()}: total and per-RVQ CE", "CE (nats)",
            [(total, "Total CE", 1.0, "black", "-")]
            + [(f"{prefix}/{level}", f"RVQ {int(level[6:])}", 1.0, index,
                LINE_STYLES[index % len(LINE_STYLES)])
               for index, level in enumerate(levels)],
            emphasis=(total,),
        )
    totals = tuple(
        f"{split}/epoch/masked_token/cross_entropy/total"
        for split in ("train", "validation")
    )
    add(
        "masked_token/ce/train_vs_validation_total",
        "Training and validation total CE", "CE (nats)",
        [(tag, label, 1.0, index, LINE_STYLES[index])
         for index, (tag, label) in enumerate(zip(
             totals, ("Training", "Validation"),
         ))],
        emphasis=totals,
    )
    corruptions_present = any(
        tag.endswith(("_dedicated_mask", "_random_token")) for tag in series
    )
    for level in levels:
        title = f"RVQ {int(level[6:])}"
        ce = f"{VALIDATION}/cross_entropy/{level}"
        accuracy = f"{VALIDATION}/accuracy/{level}"
        add(
            f"masked_token/ce/baseline/{level}",
            f"{title}: model CE and unigram improvement", "CE / gain (nats)",
            [(ce, "Model CE", 1.0, 0, "-"),
             (f"{VALIDATION}/cross_entropy_improvement/{level}",
              "Gain over unigram", 1.0, 1, "--")],
        )
        add(
            f"masked_token/accuracy/baseline/{level}",
            f"{title}: accuracy and majority improvement",
            "Accuracy (%) / gain (percentage points)",
            [(accuracy, "Top-1 accuracy", PERCENT, 0, "-"),
             (f"{VALIDATION}/accuracy_improvement/{level}",
              "Gain over majority", PERCENT, 1, "--")],
        )
        if corruptions_present:
            add(
                f"masked_token/ce/corruption/{level}",
                f"{title}: CE by corruption type", "CE (nats)",
                [(ce, "Overall", 1.0, "black", "-"),
                 (f"{ce}_dedicated_mask", "Mask", 1.0, 0, "--"),
                 (f"{ce}_random_token", "Random token", 1.0, 1, ":")],
            )
        if joint_modalities:
            for family, tag, metric, unit, scale in (
                ("ce", ce, "CE", "CE (nats)", 1.0),
                ("accuracy", accuracy, "top-1 accuracy",
                 "Accuracy (%)", PERCENT),
            ):
                add(
                    f"masked_token/{family}/modality/{level}",
                    f"{title}: {metric} by modality", unit,
                    [(tag, "Overall", scale, "black", "-"),
                     (f"{tag}_eeg", "EEG", scale, 0, "--"),
                     (f"{tag}_meg", "MEG", scale, 1, ":")],
                    note=MODALITY_NOTE,
                )
    if not any(figure.curves for figure in figures):
        raise ValueError("No plottable Stage-2 scalar curves were found.")
    return figures

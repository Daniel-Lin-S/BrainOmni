"""Convert completed DeepSpeed checkpoints into portable model weights."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from factory.campaign import portable_weight_name
from model_utils.conv import legacy_weight_norm_state_dict


def convert_best_checkpoint(
    campaign_directory: str | Path,
    stage: str = "braintokenizer",
    output_path: str | Path | None = None,
    allow_existing: bool = False,
) -> Path:
    """Convert a campaign's best ZeRO checkpoint to a portable state file.

    Parameters
    ----------
    campaign_directory : str or pathlib.Path
        Campaign root containing ``checkpoint/best``.
    stage : {"braintokenizer", "brainomni"}, optional
        Stage controlling the default output filename. Default is
        ``"braintokenizer"`` for backward compatibility.
    output_path : str or pathlib.Path, optional
        Exact output file. By default, the stage portable filename is written
        below ``campaign_directory``.
    allow_existing : bool, optional
        Whether an existing output may be replaced by DeepSpeed. Default is
        ``False``.

    Returns
    -------
    pathlib.Path
        Generated portable state-dictionary path.
    """
    campaign_root = Path(campaign_directory).resolve()
    best_path = campaign_root / "checkpoint" / "best"
    if not best_path.is_dir():
        raise FileNotFoundError(
            "Best DeepSpeed checkpoint does not exist: "
            f"{best_path.resolve()}. Complete at least one validation epoch "
            "before exporting portable weights."
        )
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else campaign_root / portable_weight_name(stage)
    )
    if destination.exists() and not allow_existing:
        raise FileExistsError(
            f"Refusing to overwrite portable weights: {destination}. "
            "Use campaign health repair for a verified atomic replacement."
        )
    try:
        from deepspeed.utils.zero_to_fp32 import (
            get_fp32_state_dict_from_zero_checkpoint,
        )
    except ImportError as error:
        raise RuntimeError(
            "DeepSpeed checkpoint conversion is unavailable. Install the "
            "training environment, then rerun campaign repair."
        ) from error
    state = get_fp32_state_dict_from_zero_checkpoint(
        str(campaign_root / "checkpoint"),
        tag="best",
        exclude_frozen_parameters=False,
        lazy_mode=False,
    )
    if not isinstance(state, dict) or not state:
        raise RuntimeError(
            "Best checkpoint conversion returned no tensor state at "
            f"{best_path.resolve()}."
        )
    portable_state = legacy_weight_norm_state_dict(state)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.convert"
    )
    if temporary.exists():
        temporary.unlink()
    try:
        torch.save(portable_state, temporary)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(
                "Checkpoint conversion produced no portable weights: "
                f"{temporary}."
            )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination

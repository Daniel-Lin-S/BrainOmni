"""Convert completed DeepSpeed checkpoints into portable model weights."""

from __future__ import annotations

from pathlib import Path

from factory.campaign import portable_weight_name


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
            convert_zero_checkpoint_to_fp32_state_dict,
        )
    except ImportError as error:
        raise RuntimeError(
            "DeepSpeed checkpoint conversion is unavailable. Install the "
            "training environment, then rerun campaign repair."
        ) from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    convert_zero_checkpoint_to_fp32_state_dict(
        str(campaign_root / "checkpoint"),
        str(destination),
        tag="best",
    )
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(
            "Checkpoint conversion produced no portable weights: "
            f"{destination}. Inspect checkpoint/best and rerun repair."
        )
    return destination

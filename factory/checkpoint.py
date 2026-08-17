"""DeepSpeed checkpoint conversion used by completed pre-training runs."""

from __future__ import annotations

from pathlib import Path


def convert_best_checkpoint(run_directory: str | Path) -> Path:
    """Convert the best DeepSpeed checkpoint to portable tokenizer weights.

    Returns the generated ``BrainTokenizer.pt`` file.
    """
    run_path = Path(run_directory)
    checkpoint_path = run_path / "checkpoint" / "best"
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(
            "Best DeepSpeed checkpoint does not exist: "
            f"{checkpoint_path.resolve()}"
        )
    output_path = run_path / "BrainTokenizer.pt"
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite tokenizer weights: {output_path.resolve()}"
        )
    try:
        from deepspeed.utils.zero_to_fp32 import (
            convert_zero_checkpoint_to_fp32_state_dict,
        )
    except ImportError as error:
        raise RuntimeError(
            "DeepSpeed checkpoint conversion is unavailable."
        ) from error
    convert_zero_checkpoint_to_fp32_state_dict(
        str(run_path / "checkpoint"),
        str(output_path),
        tag="best",
    )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(
            "Checkpoint conversion produced no weights: "
            f"{output_path.resolve()}"
        )
    return output_path

"""Convert checkpoint/best shards into an atomic portable state dictionary.

Input is a campaign directory with DeepSpeed model/optimizer shards. Output
is BrainTokenizer.pt or BrainOmni.pt, containing named parameters and buffers.
Real floating tensors use FP32; complex, integer and boolean buffers retain
their original values and dtypes from the saved model-state shard.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from factory.campaign import PORTABLE_FLOAT_DTYPE, portable_weight_name
from model_utils.conv import legacy_weight_norm_state_dict


def _original_buffers(model_path: Path) -> dict[str, torch.Tensor]:
    """Read saved buffers before DeepSpeed's unconditional FP32 conversion.

    Parameters
    ----------
    model_path : Path
        First model shard in DeepSpeed's checkpoint ordering. DeepSpeed also
        selects this shard's replicated buffers when consolidating weights.

    Returns
    -------
    dict[str, torch.Tensor]
        Named buffers with original shapes, dtypes and values. The trusted
        training shard includes non-tensor metadata and needs full loading.
        Missing, non-tensor or nonfinite buffers raise ValueError.
    """
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    if (
        not isinstance(state, dict)
        or not isinstance(state.get("buffer_names"), (list, tuple))
        or not isinstance(state.get("module"), dict)
    ):
        raise ValueError(f"Malformed buffer metadata in {model_path}.")
    buffers = {}
    for name in state["buffer_names"]:
        value = state["module"].get(name)
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"Expected saved tensor buffer {name!r} in {model_path}."
            )
        if not torch.isfinite(value).all().item():
            raise ValueError(
                f"Nonfinite saved buffer {name!r} in {model_path}."
            )
        buffers[name] = value
    return buffers


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
        Generated portable state-dictionary path. Floating tensors, including
        frozen parameters retained in training precision by DeepSpeed, use
        FP32; non-floating tensor dtypes are preserved.
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
            get_model_state_files,
        )
    except ImportError as error:
        raise RuntimeError(
            "DeepSpeed checkpoint conversion is unavailable. Install the "
            "training environment, then rerun campaign repair."
        ) from error
    model_path = Path(get_model_state_files(str(best_path))[0])
    buffers = _original_buffers(model_path)
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
    # DeepSpeed casts even complex/integer buffers to float; restore
    # saved values before applying our real-floating-only FP32 policy.
    state.update(buffers)
    portable_state = {
        name: tensor.to(dtype=PORTABLE_FLOAT_DTYPE)
        if tensor.is_floating_point() else tensor
        for name, tensor in legacy_weight_norm_state_dict(state).items()
    }
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

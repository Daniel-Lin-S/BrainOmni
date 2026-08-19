"""Construct a health-checked BrainOmni model for downstream evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from brainomni.model import BrainOmni
from factory.campaign import ensure_campaign_health


def get_brainomni(
    pretrained: bool = True,
    ckpt_path: str | None = None,
    *args: Any,
    **kwargs: Any,
) -> tuple[BrainOmni, int]:
    """Load BrainOmni from a verified semantic campaign root.

    Parameters
    ----------
    pretrained : bool, optional
        Whether to load completed portable weights. Default is ``True``.
    ckpt_path : str, optional
        BrainOmni semantic campaign root. A direct weight-file path is not
        accepted. Default is ``None``.
    *args : Any
        Unused compatibility positional arguments.
    **kwargs : Any
        Unused compatibility keyword arguments.

    Returns
    -------
    model : brainomni.model.BrainOmni
        Constructed BrainOmni model.
    lm_dim : int
        Latent model dimension.
    """
    del args, kwargs
    if ckpt_path is None:
        raise ValueError(
            "ckpt_path must name a BrainOmni semantic campaign root."
        )
    campaign_root = Path(ckpt_path).resolve()
    if campaign_root.is_file():
        raise ValueError(
            f"Expected a BrainOmni campaign directory, got file: "
            f"{campaign_root}. Pass the directory containing "
            "campaign_identity.json."
        )
    health = ensure_campaign_health(
        campaign_root,
        expected_stage="brainomni",
        repair=True,
    )
    model_config_path = campaign_root / "model_cfg.json"
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    model = BrainOmni(**model_config)
    if pretrained:
        checkpoint = torch.load(
            health.portable_path,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(checkpoint, strict=True)
        for parameter in model.tokenizer.parameters():
            parameter.requires_grad = False
    return model, model.lm_dim

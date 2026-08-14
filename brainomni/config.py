"""BrainOmni trainer configuration derived from resolved settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from pretrain_config import (
    ConfigError,
    build_deepspeed_config,
    metadata_directory,
    sha256_file,
)


class BrainOmniTrainerConfig:
    """Runtime view of a validated BrainOmni pre-training configuration."""

    def __init__(self, settings: Mapping[str, Any], world_size: int):
        campaign = settings["campaign"]
        invocation = settings["invocation"]
        tokenizer_directory = Path(invocation["tokenizer_path"])
        self.settings = settings
        self.signal_type = campaign["data"]["signal_type"]
        self.exp_name = invocation["run_name"]
        self.pretrain_metadata_path = str(metadata_directory(settings))
        self.tokenizer_ckpt_path = str(tokenizer_directory / "BrainTokenizer.pt")
        model_path = tokenizer_directory / "model_cfg.json"
        if not model_path.is_file() or not Path(self.tokenizer_ckpt_path).is_file():
            raise ConfigError(
                "tokenizer_path must contain model_cfg.json and BrainTokenizer.pt."
            )
        self.tokenizer_identity = self._tokenizer_identity(
            model_path, Path(self.tokenizer_ckpt_path), invocation
        )
        tokenizer_model = json.loads(model_path.read_text(encoding="utf-8"))
        for key, value in tokenizer_model.items():
            setattr(self, key, value)
        model = campaign["model"]
        objective = campaign["objective"]
        optimizer = campaign["optimizer"]
        self.overlap_ratio = objective["overlap_ratio"]
        self.mask_ratio = objective["mask_ratio"]
        self.num_quantizers_used = objective["num_quantizers_used"]
        self.lm_dim = model["lm_dim"]
        self.lm_head = model["lm_head"]
        self.lm_depth = model["lm_depth"]
        self.lm_dropout = model["lm_dropout"]
        self.batch_size = invocation["batch_size_per_gpu"]
        self.num_workers = invocation["num_workers"]
        self.epoch = campaign["training"]["epochs"]
        self.train_data_ratio = 1.0
        self.valid_data_ratio = 1.0
        self.test_data_ratio = 1.0
        self.lr = optimizer["lr"]
        self.weight_decay = optimizer["weight_decay"]
        self.scheduler_warm_ratio = campaign["scheduler"]["warmup_ratio"]
        self.ds_config, self.gradient_accumulation_steps = build_deepspeed_config(
            settings, world_size
        )
        self.evaluation_modes = invocation["evaluation_modes"]
        self.checkpoint_interval_epochs = invocation["checkpoint_interval_epochs"]

    @staticmethod
    def _tokenizer_identity(
        model_path: Path,
        weights_path: Path,
        invocation: Mapping[str, Any],
    ) -> dict[str, str]:
        identity = {
            "model_config_sha256": sha256_file(model_path),
            "weights_sha256": sha256_file(weights_path),
        }
        expected = {
            "model_config_sha256": invocation[
                "expected_tokenizer_model_config_sha256"
            ],
            "weights_sha256": invocation["expected_tokenizer_weights_sha256"],
        }
        for key, digest in expected.items():
            if digest is not None and digest != identity[key]:
                raise ConfigError(
                    f"Configured tokenizer {key} does not match the supplied file."
                )
        return identity

    def get_model_cfg(self) -> dict[str, Any]:
        """Return the unchanged portable BrainOmni constructor configuration."""
        tokenizer_keys = {
            "window_length", "n_filters", "ratios", "kernel_size",
            "last_kernel_size", "n_dim", "n_head", "n_neuro", "dropout",
            "codebook_dim", "codebook_size", "num_quantizers",
            "rotation_trick", "quantize_optimize_method",
        }
        config = {key: getattr(self, key) for key in tokenizer_keys}
        config.update(
            {
                "overlap_ratio": self.overlap_ratio,
                "lm_dim": self.lm_dim,
                "lm_head": self.lm_head,
                "lm_depth": self.lm_depth,
                "lm_dropout": self.lm_dropout,
                "mask_ratio": self.mask_ratio,
                "num_quantizers_used": self.num_quantizers_used,
            }
        )
        return config

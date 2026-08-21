"""BrainOmni trainer configuration derived from resolved settings."""

from __future__ import annotations

import json
from typing import Any, Mapping

from factory.campaign import ensure_campaign_health
from pretrain_config import (
    ConfigError,
    build_deepspeed_config,
    metadata_directory,
)


class BrainOmniTrainerConfig:
    """Runtime view of a validated BrainOmni pre-training configuration."""

    def __init__(self, settings: Mapping[str, Any], world_size: int):
        campaign = settings["campaign"]
        invocation = settings["invocation"]
        tokenizer_health = ensure_campaign_health(
            invocation["tokenizer_path"],
            expected_stage="braintokenizer",
            repair=True,
        )
        self.tokenizer_health = tokenizer_health
        tokenizer_directory = tokenizer_health.root
        self.settings = settings
        self.exp_name = invocation["run_name"]
        self.pretrain_metadata_path = str(metadata_directory(settings))
        self.tokenizer_ckpt_path = str(tokenizer_health.portable_path)
        model_path = tokenizer_directory / "model_cfg.json"
        self.tokenizer_identity = {
            "campaign_sha256": tokenizer_health.campaign_sha256,
            "model_config_sha256": tokenizer_health.model_config_sha256,
            "model_state_sha256": tokenizer_health.model_state_sha256,
        }
        expected = {
            "model_config_sha256": invocation[
                "expected_tokenizer_model_config_sha256"
            ],
            "model_state_sha256": invocation[
                "expected_tokenizer_weights_sha256"
            ],
        }
        for key, digest in expected.items():
            if digest is not None and digest != self.tokenizer_identity[key]:
                raise ConfigError(
                    f"Configured tokenizer {key} expected {digest}, but "
                    f"campaign health resolved {self.tokenizer_identity[key]} "
                    f"at {tokenizer_directory.resolve()}. Update the expected "
                    "digest or select the intended tokenizer campaign root."
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
        self.scheduler_warmup_min_lr_ratio = campaign["scheduler"][
            "warmup_min_lr_ratio"
        ]
        self.scheduler_cosine_min_ratio = campaign["scheduler"][
            "cosine_min_ratio"
        ]
        self.ds_config, self.gradient_accumulation_steps = (
            build_deepspeed_config(settings, world_size)
        )
        self.evaluation_datasets = [
            "test",
            *invocation["held_out_evaluation_datasets"],
        ]
        self.held_out_evaluation_datasets = invocation[
            "held_out_evaluation_datasets"
        ]
        self.checkpoint_interval_epochs = invocation[
            "checkpoint_interval_epochs"
        ]
        monitoring = invocation["monitoring"]
        self.lightweight_monitor_interval_steps = monitoring[
            "lightweight_interval_steps"
        ]
        self.diagnostic_monitor_interval_steps = monitoring[
            "diagnostic_interval_steps"
        ]

    def get_model_cfg(self) -> dict[str, Any]:
        """Return the unchanged portable BrainOmni constructor configuration."""
        tokenizer_keys = {
            "window_length",
            "n_filters",
            "ratios",
            "kernel_size",
            "last_kernel_size",
            "n_dim",
            "n_head",
            "n_neuro",
            "dropout",
            "codebook_dim",
            "codebook_size",
            "num_quantizers",
            "rotation_trick",
            "quantize_optimize_method",
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

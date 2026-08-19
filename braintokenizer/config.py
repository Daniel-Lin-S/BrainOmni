"""BrainTokenizer trainer configuration derived from resolved settings."""

from __future__ import annotations

from typing import Any, Mapping

from pretrain_config import build_deepspeed_config, metadata_directory


class BrainTokenizerTrainerConfig:
    """Runtime view of a validated BrainTokenizer pre-training configuration."""

    def __init__(self, settings: Mapping[str, Any], world_size: int):
        campaign = settings["campaign"]
        invocation = settings["invocation"]
        model = campaign["model"]
        optimizer = campaign["optimizer"]
        self.settings = settings
        self.exp_name = invocation["run_name"]
        self.pretrain_metadata_path = str(metadata_directory(settings))
        self.window_length = model["window_length"]
        self.n_filters = model["n_filters"]
        self.ratios = model["ratios"]
        self.kernel_size = model["kernel_size"]
        self.last_kernel_size = model["last_kernel_size"]
        self.n_dim = model["n_dim"]
        self.n_neuro = model["n_neuro"]
        self.n_head = model["n_head"]
        self.dropout = model["dropout"]
        self.codebook_dim = model["codebook_dim"]
        self.codebook_size = model["codebook_size"]
        self.num_quantizers = model["num_quantizers"]
        self.rotation_trick = model["rotation_trick"]
        self.quantize_optimize_method = model["quantize_optimize_method"]
        self.channel_mask_ratio = campaign["objective"]["channel_mask_ratio"]
        self.batch_size = invocation["batch_size_per_gpu"]
        self.num_workers = invocation["num_workers"]
        self.epoch = campaign["training"]["epochs"]
        self.train_data_ratio = 1.0
        self.val_data_ratio = 1.0
        self.lr = optimizer["lr"]
        self.codebook_lr = optimizer["codebook_lr"]
        self.weight_decay = optimizer["weight_decay"]
        self.scheduler_warm_ratio = campaign["scheduler"]["warmup_ratio"]
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
        self.visualization_interval_steps = invocation[
            "visualization_interval_steps"
        ]

    def get_model_cfg(self) -> dict[str, Any]:
        """Return the unchanged portable model constructor configuration."""
        return {
            "window_length": self.window_length,
            "n_filters": self.n_filters,
            "ratios": self.ratios,
            "kernel_size": self.kernel_size,
            "last_kernel_size": self.last_kernel_size,
            "n_dim": self.n_dim,
            "n_neuro": self.n_neuro,
            "n_head": self.n_head,
            "dropout": self.dropout,
            "codebook_dim": self.codebook_dim,
            "codebook_size": self.codebook_size,
            "num_quantizers": self.num_quantizers,
            "rotation_trick": self.rotation_trick,
            "quantize_optimize_method": self.quantize_optimize_method,
        }

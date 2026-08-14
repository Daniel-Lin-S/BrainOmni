"""Strict pre-training configuration and artifact helpers."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

CONFIGURATION_KEYS = {
    "schema_version", "campaign.stage", "campaign.seed",
    "campaign.data", "campaign.model", "campaign.objective",
    "campaign.optimizer", "campaign.scheduler", "campaign.training",
    "invocation.raw_root", "invocation.processed_root",
    "invocation.metadata_root", "invocation.output_root",
    "invocation.run_name", "invocation.batch_size_per_gpu",
    "invocation.num_workers", "invocation.preprocess_workers",
    "invocation.evaluation_modes", "invocation.checkpoint_interval_epochs",
    "invocation.deepspeed",
}

COMMON_CAMPAIGN = {
    "stage", "seed", "data", "model", "objective", "optimizer",
    "scheduler", "training",
}
DATA_KEYS = {
    "signal_type", "included_datasets", "split_ratios", "preprocessing",
}
PREPROCESSING_KEYS = {
    "sample_rate_hz", "low_frequency_hz", "high_frequency_hz",
    "segment_seconds", "stride_seconds",
}
BASE_INVOCATION = {
    "raw_root", "processed_root", "metadata_root", "output_root", "run_name",
    "batch_size_per_gpu", "num_workers", "preprocess_workers",
    "evaluation_modes", "checkpoint_interval_epochs", "deepspeed",
}
DEEPSPEED_KEYS = {"bf16", "zero_optimization"}
BF16_KEYS = {
    "enabled", "auto_cast", "loss_scale", "initial_scale_power",
    "loss_scale_window", "hysteresis", "min_loss_scale",
}
ZERO_KEYS = {
    "stage", "offload_optimizer_device", "offload_optimizer_pin_memory",
    "overlap_comm", "allgather_partitions", "allgather_bucket_size",
    "reduce_scatter", "reduce_bucket_size",
}
TOKENIZER_MODEL_KEYS = {
    "window_length", "n_filters", "ratios", "kernel_size", "last_kernel_size",
    "n_dim", "n_neuro", "n_head", "dropout", "codebook_dim",
    "codebook_size", "num_quantizers", "rotation_trick",
    "quantize_optimize_method",
}


class ConfigError(ValueError):
    """Raised when a configuration is incomplete or inconsistent."""


def load_pretrain_config(
    config_path: str | Path,
    local_config_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Load a YAML/JSON base config, local overlay, and CLI overrides."""
    config = _load_mapping(Path(config_path))
    if local_config_path is not None:
        config = _merge(config, _load_mapping(Path(local_config_path)))
    for override in overrides or []:
        _apply_override(config, override)
    validate_pretrain_config(config)
    return config


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Configuration file does not exist: {path.resolve()}")
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ConfigError(f"Unsupported configuration extension: {path.suffix}")
    if not isinstance(value, dict):
        raise ConfigError(f"Expected mapping in configuration: {path.resolve()}")
    return value


def _merge(base: Mapping[str, Any], local: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge a local config while rejecting unknown override keys."""
    merged = deepcopy(dict(base))
    for key, value in local.items():
        if key not in merged:
            raise ConfigError(f"Local override contains unknown key: {key}")
        if isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _apply_override(config: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ConfigError("Override must use section.key=<JSON value>.")
    dotted_path, value_text = override.split("=", 1)
    try:
        value = json.loads(value_text)
    except json.JSONDecodeError as error:
        raise ConfigError("Override value must be valid JSON.") from error
    target = config
    keys = dotted_path.split(".")
    for key in keys[:-1]:
        if not isinstance(target.get(key), dict):
            raise ConfigError(f"Override path is unknown: {dotted_path}")
        target = target[key]
    if not keys[-1] or keys[-1] not in target:
        raise ConfigError(f"Override path is unknown: {dotted_path}")
    target[keys[-1]] = value


def _mapping(value: Any, keys: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"Expected mapping at {path}, got {type(value).__name__}.")
    missing = keys - set(value)
    unknown = set(value) - keys
    if missing:
        raise ConfigError(f"Missing keys at {path}: {sorted(missing)}")
    if unknown:
        raise ConfigError(f"Unknown keys at {path}: {sorted(unknown)}")
    return value


def _number(value: Any, path: str, minimum: float = 0.0) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"Expected numeric {path}, got {value!r}.")
    if value < minimum:
        raise ConfigError(f"Expected {path} >= {minimum}, got {value}.")


def validate_pretrain_config(config: dict[str, Any]) -> None:
    """Validate full configuration shape, values, and stage-specific settings."""
    root = _mapping(config, {"schema_version", "campaign", "invocation"}, "root")
    if root["schema_version"] != 1:
        raise ConfigError("schema_version must equal 1.")
    campaign = _mapping(root["campaign"], COMMON_CAMPAIGN, "campaign")
    stage = campaign["stage"]
    if stage not in {"braintokenizer", "brainomni"}:
        raise ConfigError("campaign.stage must be braintokenizer or brainomni.")
    _number(campaign["seed"], "campaign.seed", 0)
    _validate_data(campaign["data"])
    _validate_optimizer(campaign["optimizer"], stage)
    _mapping(campaign["scheduler"], {"warmup_ratio", "cosine_min_ratio"},
             "campaign.scheduler")
    for key, value in campaign["scheduler"].items():
        _number(value, f"campaign.scheduler.{key}", 0)
        if value > 1:
            raise ConfigError(f"campaign.scheduler.{key} must be <= 1.")
    _mapping(campaign["training"], {"epochs", "global_batch_size"},
             "campaign.training")
    _number(campaign["training"]["epochs"], "campaign.training.epochs", 1)
    _number(campaign["training"]["global_batch_size"],
            "campaign.training.global_batch_size", 1)
    _validate_stage(campaign, root["invocation"])
    _validate_invocation(root["invocation"], stage)


def _validate_data(data: Any) -> None:
    values = _mapping(data, DATA_KEYS, "campaign.data")
    if values["signal_type"] not in {"eeg", "meg", "both"}:
        raise ConfigError("campaign.data.signal_type must be eeg, meg, or both.")
    if not isinstance(values["included_datasets"], list):
        raise ConfigError("campaign.data.included_datasets must be a list.")
    ratios = _mapping(values["split_ratios"], {"train", "validation", "test"},
                      "campaign.data.split_ratios")
    for key, value in ratios.items():
        _number(value, f"campaign.data.split_ratios.{key}")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ConfigError("campaign.data.split_ratios must sum to 1.0.")
    pre = _mapping(values["preprocessing"], PREPROCESSING_KEYS,
                   "campaign.data.preprocessing")
    for key, value in pre.items():
        _number(value, f"campaign.data.preprocessing.{key}")
    if pre["low_frequency_hz"] >= pre["high_frequency_hz"]:
        raise ConfigError("low_frequency_hz must be below high_frequency_hz.")
    if pre["high_frequency_hz"] >= pre["sample_rate_hz"] / 2:
        raise ConfigError("high_frequency_hz must be below the Nyquist frequency.")


def _validate_optimizer(optimizer: Any, stage: str) -> None:
    keys = {"type", "lr", "betas", "eps", "weight_decay"}
    if stage == "braintokenizer":
        keys.add("codebook_lr")
    values = _mapping(optimizer, keys, "campaign.optimizer")
    if values["type"] != "AdamW":
        raise ConfigError("campaign.optimizer.type must be AdamW.")
    for key in keys - {"type", "betas"}:
        _number(values[key], f"campaign.optimizer.{key}")
    if not isinstance(values["betas"], list) or len(values["betas"]) != 2:
        raise ConfigError("campaign.optimizer.betas must contain two values.")
    for beta in values["betas"]:
        _number(beta, "campaign.optimizer.betas")
        if beta >= 1:
            raise ConfigError("campaign.optimizer.betas values must be below 1.")


def _validate_stage(campaign: dict[str, Any], invocation: Any) -> None:
    if campaign["stage"] == "braintokenizer":
        _mapping(campaign["model"], TOKENIZER_MODEL_KEYS, "campaign.model")
        objective = _mapping(campaign["objective"],
                             {"channel_mask_ratio", "noise_std"},
                             "campaign.objective")
        _number(objective["channel_mask_ratio"],
                "campaign.objective.channel_mask_ratio")
        _number(objective["noise_std"], "campaign.objective.noise_std")
        if objective["channel_mask_ratio"] > 1:
            raise ConfigError("campaign.objective.channel_mask_ratio must be <= 1.")
        return
    _mapping(campaign["model"],
             {"size", "lm_dim", "lm_head", "lm_depth", "lm_dropout"},
             "campaign.model")
    objective = _mapping(campaign["objective"],
                         {"overlap_ratio", "mask_ratio", "num_quantizers_used"},
                         "campaign.objective")
    for key, value in objective.items():
        _number(value, f"campaign.objective.{key}")
    if objective["overlap_ratio"] > 1 or objective["mask_ratio"] > 1:
        raise ConfigError("campaign.objective ratios must be <= 1.")


def _validate_invocation(invocation: Any, stage: str) -> None:
    keys = set(BASE_INVOCATION)
    if stage == "braintokenizer":
        keys.add("visualization_interval_steps")
    else:
        keys |= {
            "tokenizer_path", "expected_tokenizer_model_config_sha256",
            "expected_tokenizer_weights_sha256",
        }
    values = _mapping(invocation, keys, "invocation")
    for key in {"raw_root", "processed_root", "metadata_root", "output_root",
                "run_name"}:
        if not isinstance(values[key], str) or not values[key]:
            raise ConfigError(f"invocation.{key} must be a non-empty string.")
    for key in {"batch_size_per_gpu", "num_workers", "preprocess_workers",
                "checkpoint_interval_epochs"}:
        _number(values[key], f"invocation.{key}", 1)
    if not isinstance(values["evaluation_modes"], list):
        raise ConfigError("invocation.evaluation_modes must be a list.")
    deep = _mapping(values["deepspeed"], DEEPSPEED_KEYS, "invocation.deepspeed")
    _mapping(deep["bf16"], BF16_KEYS, "invocation.deepspeed.bf16")
    zero = _mapping(deep["zero_optimization"], ZERO_KEYS,
                    "invocation.deepspeed.zero_optimization")
    if zero["stage"] != 2:
        raise ConfigError("DeepSpeed zero_optimization.stage must equal 2.")
    if stage == "brainomni":
        for key in {"expected_tokenizer_model_config_sha256",
                    "expected_tokenizer_weights_sha256"}:
            value = values[key]
            if value is not None and (not isinstance(value, str) or len(value) != 64):
                raise ConfigError(f"invocation.{key} must be a SHA-256 or null.")


def metadata_directory(config: Mapping[str, Any]) -> Path:
    """Return the metadata directory selected by semantic preprocessing."""
    pre = config["campaign"]["data"]["preprocessing"]
    name = (
        f"sfreq_{pre['sample_rate_hz']}_low_{pre['low_frequency_hz']}_high_"
        f"{pre['high_frequency_hz']}_time_{pre['segment_seconds']}_stride_"
        f"{pre['stride_seconds']}"
    )
    return Path(config["invocation"]["metadata_root"]) / name


def build_deepspeed_config(
    config: Mapping[str, Any], world_size: int
) -> tuple[dict[str, Any], int]:
    """Build DeepSpeed settings and enforce effective batch divisibility."""
    campaign = config["campaign"]
    invocation = config["invocation"]
    denominator = invocation["batch_size_per_gpu"] * world_size
    total = campaign["training"]["global_batch_size"]
    if total % denominator:
        raise ConfigError(
            "campaign.training.global_batch_size must be divisible by the "
            "per-GPU batch size times world_size."
        )
    accumulation = total // denominator
    zero = invocation["deepspeed"]["zero_optimization"]
    return {
        "train_micro_batch_size_per_gpu": invocation["batch_size_per_gpu"],
        "gradient_accumulation_steps": accumulation,
        "bf16": deepcopy(invocation["deepspeed"]["bf16"]),
        "optimizer": {
            "type": campaign["optimizer"]["type"],
            "params": {
                "betas": campaign["optimizer"]["betas"],
                "eps": campaign["optimizer"]["eps"],
            },
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "total_num_steps": None, "warmup_num_steps": None,
                "warmup_min_ratio": campaign["scheduler"]["warmup_ratio"],
                "cos_min_ratio": campaign["scheduler"]["cosine_min_ratio"],
            },
        },
        "zero_optimization": {
            "stage": zero["stage"],
            "offload_optimizer": {
                "device": zero["offload_optimizer_device"],
                "pin_memory": zero["offload_optimizer_pin_memory"],
            },
            "overlap_comm": zero["overlap_comm"],
            "allgather_partitions": zero["allgather_partitions"],
            "allgather_bucket_size": zero["allgather_bucket_size"],
            "reduce_scatter": zero["reduce_scatter"],
            "reduce_bucket_size": zero["reduce_bucket_size"],
        },
    }, accumulation


def sha256_file(path: str | Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_artifacts(
    run_directory: str | Path,
    config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    tokenizer_identity: Mapping[str, str] | None = None,
) -> None:
    """Write atomic resolved sidecars for a newly created run."""
    directory = Path(run_directory)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "model_cfg.json"
    _write_json(model_path, model_config)
    setting = {"schema_version": config["schema_version"],
               "campaign": deepcopy(config["campaign"])}
    setting["campaign"].pop("model")
    setting_path = directory / "pretrain_setting.yaml"
    _write_yaml(setting_path, setting)
    manifest: dict[str, Any] = {
        "schema_version": config["schema_version"],
        "stage": config["campaign"]["stage"],
        "model_config_sha256": sha256_file(model_path),
        "pretrain_setting_sha256": sha256_file(setting_path),
    }
    if tokenizer_identity is not None:
        manifest["tokenizer_identity"] = dict(tokenizer_identity)
    _write_json(directory / "pretrain_setting.json", manifest)
    _write_yaml(directory / "invocation.yaml", config["invocation"])


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(path, yaml.safe_dump(dict(value), sort_keys=False))


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)

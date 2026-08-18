"""Strict pre-training configuration and artifact helpers."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

COMMON_CAMPAIGN = {
    "stage",
    "seed",
    "data",
    "model",
    "objective",
    "optimizer",
    "scheduler",
    "training",
}
DATA_KEYS = {
    "signal_type",
    "included_datasets",
    "split_ratios",
    "preprocessing",
}
PREPROCESSING_KEYS = {
    "sample_rate_hz",
    "low_frequency_hz",
    "high_frequency_hz",
    "segment_seconds",
    "stride_seconds",
}
BASE_INVOCATION = {
    "raw_root",
    "processed_root",
    "metadata_root",
    "output_root",
    "run_name",
    "batch_size_per_gpu",
    "num_workers",
    "preprocess_workers",
    "evaluation_modes",
    "checkpoint_interval_epochs",
    "deepspeed",
}
DEEPSPEED_KEYS = {"bf16", "zero_optimization"}
BF16_KEYS = {
    "enabled",
    "auto_cast",
    "loss_scale",
    "initial_scale_power",
    "loss_scale_window",
    "hysteresis",
    "min_loss_scale",
}
ZERO_KEYS = {
    "stage",
    "offload_optimizer_device",
    "offload_optimizer_pin_memory",
    "overlap_comm",
    "allgather_partitions",
    "allgather_bucket_size",
    "reduce_scatter",
    "reduce_bucket_size",
}
TOKENIZER_MODEL_KEYS = {
    "window_length",
    "n_filters",
    "ratios",
    "kernel_size",
    "last_kernel_size",
    "n_dim",
    "n_neuro",
    "n_head",
    "dropout",
    "codebook_dim",
    "codebook_size",
    "num_quantizers",
    "rotation_trick",
    "quantize_optimize_method",
}


class ConfigError(ValueError):
    """Raised when a configuration is incomplete or inconsistent."""


def load_pretrain_config(
    config_paths: str | Path | Sequence[str | Path],
    local_config_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Load layered configs, a local overlay, and CLI overrides."""
    paths = _config_paths(config_paths)
    config: dict[str, Any] = {}
    for path in paths:
        config = _merge(config, _load_mapping(path))
    if local_config_path is not None:
        config = _merge(config, _load_mapping(Path(local_config_path)))
    for override in overrides or []:
        _apply_override(config, override)
    validate_pretrain_config(config)
    return config


def _config_paths(
    config_paths: str | Path | Sequence[str | Path],
) -> list[Path]:
    if isinstance(config_paths, (str, Path)):
        paths = [Path(config_paths)]
    else:
        paths = [Path(path) for path in config_paths]
    if not paths:
        raise ConfigError("At least one --config file is required.")
    return paths


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(
            f"Configuration file does not exist: {path.resolve()}"
        )
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ConfigError(
            f"Unsupported configuration extension: {path.suffix}"
        )
    if not isinstance(value, dict):
        raise ConfigError(
            f"Expected mapping in configuration: {path.resolve()}"
        )
    return value


def _merge(
    base: Mapping[str, Any], local: Mapping[str, Any]
) -> dict[str, Any]:
    """Recursively merge one configuration layer over another."""
    merged = deepcopy(dict(base))
    for key, value in local.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
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
        raise ConfigError(
            f"Expected mapping at {path}, got {type(value).__name__}."
        )
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


def _integer(value: Any, path: str, minimum: int = 0) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"Expected integer {path}, got {value!r}.")
    if value < minimum:
        raise ConfigError(f"Expected {path} >= {minimum}, got {value}.")


def _positive_number(value: Any, path: str) -> None:
    _number(value, path)
    if value == 0:
        raise ConfigError(f"Expected {path} > 0, got {value}.")


def _fraction(value: Any, path: str) -> None:
    _number(value, path)
    if value > 1:
        raise ConfigError(f"Expected {path} <= 1, got {value}.")


def _boolean(value: Any, path: str) -> None:
    if not isinstance(value, bool):
        raise ConfigError(f"Expected boolean {path}, got {value!r}.")


def validate_pretrain_config(config: dict[str, Any]) -> None:
    """Validate configuration shape, values, and stage-specific settings."""
    root = _mapping(
        config,
        {"schema_version", "campaign", "invocation"},
        "root",
    )
    if root["schema_version"] != 1:
        raise ConfigError("schema_version must equal 1.")
    campaign = _mapping(root["campaign"], COMMON_CAMPAIGN, "campaign")
    stage = campaign["stage"]
    if stage not in {"braintokenizer", "brainomni"}:
        raise ConfigError(
            "campaign.stage must be braintokenizer or brainomni."
        )
    _integer(campaign["seed"], "campaign.seed", 0)
    _validate_data(campaign["data"])
    _validate_optimizer(campaign["optimizer"], stage)
    scheduler = _mapping(
        campaign["scheduler"],
        {"warmup_ratio", "cosine_min_ratio"},
        "campaign.scheduler",
    )
    for key, value in scheduler.items():
        _fraction(value, f"campaign.scheduler.{key}")
    training = _mapping(
        campaign["training"],
        {"epochs", "global_batch_size"},
        "campaign.training",
    )
    _integer(training["epochs"], "campaign.training.epochs", 1)
    _integer(
        training["global_batch_size"],
        "campaign.training.global_batch_size",
        1,
    )
    _validate_stage(campaign)
    _validate_invocation(root["invocation"], stage)


def _validate_data(data: Any) -> None:
    values = _mapping(data, DATA_KEYS, "campaign.data")
    if values["signal_type"] not in {"eeg", "meg", "both"}:
        raise ConfigError(
            "campaign.data.signal_type must be eeg, meg, or both."
        )
    datasets = values["included_datasets"]
    if not isinstance(datasets, list) or not datasets:
        raise ConfigError(
            "campaign.data.included_datasets must be a non-empty list."
        )
    if any(
        not isinstance(item, str) or not item.strip() for item in datasets
    ):
        raise ConfigError(
            "campaign.data.included_datasets contains an invalid name."
        )
    if "*" in datasets and datasets != ["*"]:
        raise ConfigError(
            "campaign.data.included_datasets may use '*' only as the "
            "sole item."
        )
    if len(set(datasets)) != len(datasets):
        raise ConfigError(
            "campaign.data.included_datasets must not contain duplicates."
        )
    ratios = _mapping(
        values["split_ratios"],
        {"train", "validation", "test"},
        "campaign.data.split_ratios",
    )
    for key, value in ratios.items():
        _fraction(value, f"campaign.data.split_ratios.{key}")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ConfigError("campaign.data.split_ratios must sum to 1.0.")
    preprocessing = _mapping(
        values["preprocessing"],
        PREPROCESSING_KEYS,
        "campaign.data.preprocessing",
    )
    _positive_number(
        preprocessing["sample_rate_hz"],
        "campaign.data.preprocessing.sample_rate_hz",
    )
    _number(
        preprocessing["low_frequency_hz"],
        "campaign.data.preprocessing.low_frequency_hz",
    )
    for key in ("high_frequency_hz", "segment_seconds", "stride_seconds"):
        _positive_number(
            preprocessing[key],
            f"campaign.data.preprocessing.{key}",
        )
    if preprocessing["low_frequency_hz"] >= preprocessing["high_frequency_hz"]:
        raise ConfigError("low_frequency_hz must be below high_frequency_hz.")
    if (
        preprocessing["high_frequency_hz"]
        >= preprocessing["sample_rate_hz"] / 2
    ):
        raise ConfigError(
            "high_frequency_hz must be below the Nyquist frequency."
        )


def _validate_optimizer(optimizer: Any, stage: str) -> None:
    keys = {"type", "lr", "betas", "eps", "weight_decay"}
    if stage == "braintokenizer":
        keys.add("codebook_lr")
    values = _mapping(optimizer, keys, "campaign.optimizer")
    if values["type"] != "AdamW":
        raise ConfigError("campaign.optimizer.type must be AdamW.")
    _positive_number(values["lr"], "campaign.optimizer.lr")
    _positive_number(values["eps"], "campaign.optimizer.eps")
    _number(
        values["weight_decay"],
        "campaign.optimizer.weight_decay",
    )
    if stage == "braintokenizer":
        _positive_number(
            values["codebook_lr"],
            "campaign.optimizer.codebook_lr",
        )
    if not isinstance(values["betas"], list) or len(values["betas"]) != 2:
        raise ConfigError("campaign.optimizer.betas must contain two values.")
    for beta in values["betas"]:
        _number(beta, "campaign.optimizer.betas")
        if beta >= 1:
            raise ConfigError(
                "campaign.optimizer.betas values must be below 1."
            )


def _validate_stage(campaign: dict[str, Any]) -> None:
    model = campaign["model"]
    objective = campaign["objective"]
    if campaign["stage"] == "braintokenizer":
        model = _mapping(model, TOKENIZER_MODEL_KEYS, "campaign.model")
        integer_keys = {
            "window_length",
            "n_filters",
            "kernel_size",
            "last_kernel_size",
            "n_dim",
            "n_neuro",
            "n_head",
            "codebook_dim",
            "codebook_size",
            "num_quantizers",
        }
        for key in integer_keys:
            _integer(model[key], f"campaign.model.{key}", 1)
        ratios = model["ratios"]
        if not isinstance(ratios, list) or not ratios:
            raise ConfigError(
                "campaign.model.ratios must be a non-empty list."
            )
        for ratio in ratios:
            _integer(ratio, "campaign.model.ratios", 1)
        _fraction(model["dropout"], "campaign.model.dropout")
        _boolean(model["rotation_trick"], "campaign.model.rotation_trick")
        if model["quantize_optimize_method"] != "ema":
            raise ConfigError(
                "campaign.model.quantize_optimize_method must be ema."
            )
        objective = _mapping(
            objective,
            {"channel_mask_ratio"},
            "campaign.objective",
        )
        _fraction(
            objective["channel_mask_ratio"],
            "campaign.objective.channel_mask_ratio",
        )
        return
    model = _mapping(
        model,
        {"lm_dim", "lm_head", "lm_depth", "lm_dropout"},
        "campaign.model",
    )
    for key in ("lm_dim", "lm_head", "lm_depth"):
        _integer(model[key], f"campaign.model.{key}", 1)
    _fraction(model["lm_dropout"], "campaign.model.lm_dropout")
    objective = _mapping(
        objective,
        {"overlap_ratio", "mask_ratio", "num_quantizers_used"},
        "campaign.objective",
    )
    _fraction(objective["overlap_ratio"], "campaign.objective.overlap_ratio")
    _fraction(objective["mask_ratio"], "campaign.objective.mask_ratio")
    _integer(
        objective["num_quantizers_used"],
        "campaign.objective.num_quantizers_used",
        1,
    )


def _validate_invocation(invocation: Any, stage: str) -> None:
    keys = set(BASE_INVOCATION)
    if stage == "braintokenizer":
        keys.add("visualization_interval_steps")
    else:
        keys |= {
            "tokenizer_path",
            "expected_tokenizer_model_config_sha256",
            "expected_tokenizer_weights_sha256",
        }
    values = _mapping(invocation, keys, "invocation")
    path_keys = {
        "raw_root",
        "processed_root",
        "metadata_root",
        "output_root",
    }
    if stage == "brainomni":
        path_keys.add("tokenizer_path")
    for key in path_keys:
        if not isinstance(values[key], str) or not values[key]:
            raise ConfigError(
                f"invocation.{key} must be set in a local --config layer; "
                "public configurations do not contain machine-local paths."
            )
    if not isinstance(values["run_name"], str) or not values["run_name"]:
        raise ConfigError("invocation.run_name must be a non-empty string.")
    for key in (
        "batch_size_per_gpu",
        "num_workers",
        "preprocess_workers",
        "checkpoint_interval_epochs",
    ):
        _integer(values[key], f"invocation.{key}", 1)
    if (
        not isinstance(values["evaluation_modes"], list)
        or not values["evaluation_modes"]
    ):
        raise ConfigError(
            "invocation.evaluation_modes must be a non-empty list."
        )
    if any(
        not isinstance(mode, str) or not mode.strip()
        for mode in values["evaluation_modes"]
    ):
        raise ConfigError(
            "invocation.evaluation_modes contains an invalid mode."
        )
    deep = _mapping(
        values["deepspeed"],
        DEEPSPEED_KEYS,
        "invocation.deepspeed",
    )
    bf16 = _mapping(deep["bf16"], BF16_KEYS, "invocation.deepspeed.bf16")
    for key in ("enabled", "auto_cast"):
        _boolean(bf16[key], f"invocation.deepspeed.bf16.{key}")
    _number(bf16["loss_scale"], "invocation.deepspeed.bf16.loss_scale")
    _integer(
        bf16["initial_scale_power"],
        "invocation.deepspeed.bf16.initial_scale_power",
        0,
    )
    _integer(
        bf16["loss_scale_window"],
        "invocation.deepspeed.bf16.loss_scale_window",
        1,
    )
    _integer(
        bf16["hysteresis"],
        "invocation.deepspeed.bf16.hysteresis",
        0,
    )
    _positive_number(
        bf16["min_loss_scale"],
        "invocation.deepspeed.bf16.min_loss_scale",
    )
    zero = _mapping(
        deep["zero_optimization"],
        ZERO_KEYS,
        "invocation.deepspeed.zero_optimization",
    )
    _integer(
        zero["stage"],
        "invocation.deepspeed.zero_optimization.stage",
        2,
    )
    if zero["stage"] != 2:
        raise ConfigError("DeepSpeed zero_optimization.stage must equal 2.")
    if (
        not isinstance(zero["offload_optimizer_device"], str)
        or not zero["offload_optimizer_device"]
    ):
        raise ConfigError(
            "invocation.deepspeed.zero_optimization.offload_optimizer_device "
            "must be a non-empty string."
        )
    _boolean(
        zero["offload_optimizer_pin_memory"],
        "invocation.deepspeed.zero_optimization.offload_optimizer_pin_memory",
    )
    for key in (
        "overlap_comm",
        "allgather_partitions",
        "reduce_scatter",
    ):
        _boolean(
            zero[key],
            f"invocation.deepspeed.zero_optimization.{key}",
        )
    for key in ("allgather_bucket_size", "reduce_bucket_size"):
        value = zero[key]
        if value != "auto":
            _integer(
                value,
                f"invocation.deepspeed.zero_optimization.{key}",
                1,
            )
    if stage == "brainomni":
        for key in (
            "expected_tokenizer_model_config_sha256",
            "expected_tokenizer_weights_sha256",
        ):
            value = values[key]
            if value is None:
                continue
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(
                    character not in "0123456789abcdef" for character in value
                )
            ):
                raise ConfigError(
                    f"invocation.{key} must be a SHA-256 or null."
                )


def repository_log_directory(run_directory: str | Path, stage: str) -> Path:
    """Return the repository-managed text-log directory for one run.

    Parameters
    ----------
    run_directory : str or pathlib.Path
        Run artifact directory with ``<run_name>/exp_<timestamp>`` layout.
    stage : str
        Pre-training stage name.

    Returns
    -------
    pathlib.Path
        Directory for rank-zero text logs.
    """
    run_path = Path(run_directory)
    if stage not in {"braintokenizer", "brainomni"}:
        raise ConfigError(f"Unsupported pre-training log stage: {stage}")
    return (
        Path(__file__).resolve().parent / "logs" / stage
        / run_path.parent.name / run_path.name
    )


def metadata_directory(config: Mapping[str, Any]) -> Path:
    """Return the metadata directory selected by semantic preprocessing."""
    pre = config["campaign"]["data"]["preprocessing"]
    name = (
        f"sfreq_{pre['sample_rate_hz']}_low_{pre['low_frequency_hz']}_high_"
        f"{pre['high_frequency_hz']}_time_{pre['segment_seconds']}_stride_"
        f"{pre['stride_seconds']}"
    )
    return Path(config["invocation"]["metadata_root"]) / name


def resolve_dataset_identities(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the source dataset selection from processed metadata."""
    info_path = metadata_directory(config) / "info.json"
    if not info_path.is_file():
        raise ConfigError(
            f"Preprocessing metadata does not exist: {info_path.resolve()}"
        )
    metadata = json.loads(info_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, list) or not metadata:
        raise ConfigError(
            f"Preprocessing metadata is empty: {info_path.resolve()}"
        )
    if any(not isinstance(item, dict) for item in metadata):
        raise ConfigError(
            "Preprocessing metadata must contain dataset mappings: "
            f"{info_path.resolve()}"
        )
    dataset_set = {item.get("dataset") for item in metadata}
    if not dataset_set or any(
        not isinstance(item, str) or not item for item in dataset_set
    ):
        raise ConfigError(
            "Preprocessing metadata has invalid dataset identities: "
            f"{info_path.resolve()}"
        )
    datasets = sorted(dataset_set)
    selected = config["campaign"]["data"]["included_datasets"]
    if selected != ["*"]:
        unexpected = set(datasets) - set(selected)
        missing = set(selected) - set(datasets)
        if unexpected:
            raise ConfigError(
                "Preprocessing metadata contains datasets outside "
                "included_datasets: "
                f"{sorted(unexpected)}"
            )
        if missing:
            raise ConfigError(
                "No generated preprocessing metadata exists for "
                "included_datasets: "
                f"{sorted(missing)}"
            )
    resolved = deepcopy(dict(config))
    resolved["campaign"]["data"]["included_datasets"] = datasets
    return resolved


def build_deepspeed_config(
    config: Mapping[str, Any], world_size: int
) -> tuple[dict[str, Any], int]:
    """Build DeepSpeed settings and enforce effective batch divisibility."""
    _integer(world_size, "world_size", 1)
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
                "total_num_steps": None,
                "warmup_num_steps": None,
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
    datasets = config["campaign"]["data"]["included_datasets"]
    if not isinstance(datasets, list) or not datasets:
        raise ConfigError(
            "Run artifacts require resolved non-empty "
            "campaign.data.included_datasets."
        )
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "model_cfg.json"
    _write_json(model_path, model_config)
    setting = {
        "schema_version": config["schema_version"],
        "campaign": deepcopy(config["campaign"]),
    }
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

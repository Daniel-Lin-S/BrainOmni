"""Strict pre-training configuration and artifact helpers."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import re
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
    "processed_root",
    "metadata_root",
    "output_root",
    "run_name",
    "data_catalog",
    "batch_size_per_gpu",
    "num_workers",
    "preprocess_workers",
    "held_out_evaluation_datasets",
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


DATA_CATALOG_PATH = (
    Path(__file__).resolve().parent / "configs/data/datasets.local.yaml"
)


def load_pretrain_config(
    config_paths: str | Path | Sequence[str | Path],
    overrides: list[str] | None = None,
    data_catalog_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load configuration layers and CLI overrides."""
    paths = _config_paths(config_paths)
    config: dict[str, Any] = {}
    for path in paths:
        config = _merge(config, _load_mapping(path))
    for override in overrides or []:
        _apply_override(config, override)
    if data_catalog_path is not None:
        invocation = config.get("invocation")
        if not isinstance(invocation, dict):
            raise ConfigError(
                "invocation must be a mapping before catalog loading."
            )
        invocation["data_catalog"] = load_data_catalog(data_catalog_path)
    validate_pretrain_config(config)
    return config


def load_pretrain_launch_config(
    config_paths: str | Path | Sequence[str | Path],
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Load launch configuration with the required local dataset catalog."""
    return load_pretrain_config(
        config_paths,
        overrides=overrides,
        data_catalog_path=DATA_CATALOG_PATH,
    )


def load_data_catalog(path: str | Path) -> dict[str, dict[str, str]]:
    """Load and validate one local dataset-path catalog."""
    catalog_path = Path(path)
    if not catalog_path.is_file():
        raise ConfigError(
            "Local dataset catalog does not exist: " f"{catalog_path.resolve()}"
        )
    value = _load_data_catalog_mapping(catalog_path)
    root = _mapping(value, {"datasets"}, "data catalog")
    datasets = root["datasets"]
    if not isinstance(datasets, dict) or not datasets:
        raise ConfigError("data catalog datasets must be a non-empty mapping.")
    _validate_data_catalog(datasets)
    return deepcopy(datasets)


def selected_data_catalog(
    config: Mapping[str, Any],
    include_held_out: bool = False,
) -> dict[str, dict[str, str]]:
    """Return selected dataset definitions from a validated catalog."""
    catalog = config["invocation"]["data_catalog"]
    selected = config["campaign"]["data"]["included_datasets"]
    identities = sorted(catalog) if selected == ["*"] else list(selected)
    if include_held_out:
        identities.extend(
            config["invocation"].get("held_out_evaluation_datasets", [])
        )
        identities = sorted(set(identities))
    missing = sorted(set(identities) - set(catalog))
    if missing:
        raise ConfigError(f"Data catalog lacks included_datasets: {missing}")
    return {identity: deepcopy(catalog[identity]) for identity in identities}


def _load_data_catalog_mapping(path: Path) -> dict[str, Any]:
    """Load a catalog while rejecting duplicate YAML mapping keys."""
    text = path.read_text(encoding="utf-8")
    try:
        document = yaml.compose(text, Loader=yaml.SafeLoader)
        _reject_duplicate_yaml_keys(document)
        value = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigError(
            f"Could not parse data catalog: {path.resolve()}"
        ) from error
    if not isinstance(value, dict):
        raise ConfigError(f"Expected mapping in data catalog: {path.resolve()}")
    return value


def _reject_duplicate_yaml_keys(node: yaml.Node | None) -> None:
    """Raise when any YAML mapping declares a key more than once."""
    if node is None:
        return
    if isinstance(node, yaml.MappingNode):
        keys: set[str] = set()
        for key_node, value_node in node.value:
            if isinstance(key_node, yaml.ScalarNode):
                key = key_node.value
                if key in keys:
                    raise ConfigError(
                        "Data catalog contains a duplicate mapping key: "
                        f"{key!r}."
                    )
                keys.add(key)
            _reject_duplicate_yaml_keys(value_node)
        return
    if isinstance(node, yaml.SequenceNode):
        for item in node.value:
            _reject_duplicate_yaml_keys(item)


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
        raise ConfigError(f"Unsupported configuration extension: {path.suffix}")
    if not isinstance(value, dict):
        raise ConfigError(
            f"Expected mapping in configuration: {path.resolve()}"
        )
    return value


def _merge(base: Mapping[str, Any], local: Mapping[str, Any]) -> dict[str, Any]:
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
        raise ConfigError("campaign.stage must be braintokenizer or brainomni.")
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
    training_datasets = set(selected_data_catalog(root))
    held_out = set(root["invocation"]["held_out_evaluation_datasets"])
    overlap = sorted(training_datasets & held_out)
    if overlap:
        raise ConfigError(
            "Held-out evaluation datasets are included in this campaign: "
            f"{overlap}. Remove them from campaign.data.included_datasets "
            "before preprocessing or training."
        )


def _validate_data(data: Any) -> None:
    values = _mapping(data, DATA_KEYS, "campaign.data")
    datasets = values["included_datasets"]
    if not isinstance(datasets, list) or not datasets:
        raise ConfigError(
            "campaign.data.included_datasets must be a non-empty list."
        )
    if any(not isinstance(item, str) or not item.strip() for item in datasets):
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
            raise ConfigError("campaign.model.ratios must be a non-empty list.")
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


def _validate_data_catalog(catalog: Mapping[str, Any]) -> None:
    """Validate local paths and declared modalities for all datasets."""
    if not isinstance(catalog, Mapping):
        raise ConfigError("data catalog datasets must be a mapping.")
    if not catalog:
        raise ConfigError("data catalog datasets must not be empty.")
    for dataset, definition in catalog.items():
        if (
            not isinstance(dataset, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", dataset) is None
        ):
            raise ConfigError(
                "Data catalog dataset IDs must start with a letter or digit "
                "and contain only letters, digits, periods, underscores, or "
                f"hyphens; got {dataset!r}. Update "
                f"{DATA_CATALOG_PATH.resolve()} and retry."
            )
        entry = _mapping(definition, {"path", "signal_type"}, dataset)
        path = entry["path"]
        if not isinstance(path, str) or not path:
            raise ConfigError(
                f"Data catalog {dataset}.path must be non-empty. Update "
                f"{DATA_CATALOG_PATH.resolve()} and retry."
            )
        resolved_path = Path(path).resolve()
        if not resolved_path.is_dir():
            raise ConfigError(
                f"Data catalog path for {dataset} is not a directory: "
                f"{resolved_path}. Correct the path in "
                f"{DATA_CATALOG_PATH.resolve()} and retry."
            )
        signal_type = entry["signal_type"]
        if signal_type not in {"eeg", "meg", "both"}:
            raise ConfigError(
                f"Data catalog {dataset}.signal_type must be eeg, meg, or "
                f"both; got {signal_type!r}. Correct it in "
                f"{DATA_CATALOG_PATH.resolve()} and retry."
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
    _validate_data_catalog(values["data_catalog"])
    path_keys = {
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
    held_out = values["held_out_evaluation_datasets"]
    if not isinstance(held_out, list):
        raise ConfigError(
            "invocation.held_out_evaluation_datasets must be a list."
        )
    if any(
        not isinstance(dataset, str) or not dataset.strip()
        for dataset in held_out
    ):
        raise ConfigError(
            "invocation.held_out_evaluation_datasets contains an invalid "
            "dataset ID. Remove empty or non-string values and retry."
        )
    if len(set(held_out)) != len(held_out):
        raise ConfigError(
            "invocation.held_out_evaluation_datasets must not contain "
            "duplicates. Remove duplicate dataset IDs and retry."
        )
    unknown = sorted(set(held_out) - set(values["data_catalog"]))
    if unknown:
        raise ConfigError(
            "Held-out evaluation datasets are absent from the local data "
            f"catalog: {unknown}. Add them to "
            f"{DATA_CATALOG_PATH.resolve()} and retry."
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


def preprocessing_directory(config: Mapping[str, Any]) -> Path:
    """Return the reusable recording cache for preprocessing semantics."""
    pre = config["campaign"]["data"]["preprocessing"]
    name = (
        f"sfreq_{pre['sample_rate_hz']}_low_"
        f"{pre['low_frequency_hz']}_high_"
        f"{pre['high_frequency_hz']}_time_"
        f"{pre['segment_seconds']}_stride_"
        f"{pre['stride_seconds']}"
    )
    return Path(config["invocation"]["metadata_root"]) / name


def metadata_directory(config: Mapping[str, Any]) -> Path:
    """Return the isolated split directory for training semantics."""
    catalog = selected_data_catalog(config)
    split_identity = {
        "datasets": {
            dataset: catalog[dataset]["signal_type"]
            for dataset in sorted(catalog)
        },
        "seed": config["campaign"]["seed"],
        "split_ratios": config["campaign"]["data"]["split_ratios"],
    }
    digest = canonical_config_sha256(split_identity)[:12]
    return preprocessing_directory(config) / f"splits_{digest}"


def canonical_config_sha256(value: Any) -> str:
    """Return a canonical digest for path-free configuration values."""
    text = json.dumps(
        value,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(text.encode("utf-8")).hexdigest()


def resolve_dataset_identities(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the source dataset selection from processed metadata."""
    info_path = preprocessing_directory(config) / "info.json"
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
    selected = set(selected_data_catalog(config))
    catalog = set(config["invocation"]["data_catalog"])
    unexpected = dataset_set - catalog
    missing = selected - dataset_set
    if unexpected:
        raise ConfigError(
            "Preprocessing metadata contains datasets outside catalog: "
            f"{sorted(unexpected)}"
        )
    if missing:
        raise ConfigError(
            "No generated preprocessing metadata exists for catalog datasets: "
            f"{sorted(missing)}"
        )

    resolved = deepcopy(dict(config))
    resolved["campaign"]["data"]["included_datasets"] = sorted(selected)
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

"""Semantic pre-training campaign artifacts, recovery, and health checks.

A campaign root contains portable semantic sidecars, DeepSpeed checkpoints,
and one completed portable state dictionary. Invocation details and transient
TensorBoard events are isolated below ``attempts/<attempt-id>``.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import shlex
import shutil
import time
from typing import Any, Iterator, Mapping
import warnings

import numpy as np
import torch
import yaml

from pretrain_config import ConfigError, selected_data_catalog, sha256_file

ARTIFACT_SCHEMA_VERSION = 1
CAMPAIGN_HASH_LENGTH = 20
CAMPAIGN_IDENTITY_FILE = "campaign_identity.json"
CAMPAIGN_STATUS_FILE = "campaign_status.json"
CHECKPOINT_MANIFEST_FILE = "manifest.json"
PORTABLE_WEIGHT_NAMES = {
    "braintokenizer": "BrainTokenizer.pt",
    "brainomni": "BrainOmni.pt",
}
STAGE_LAUNCHERS = {
    "braintokenizer": "script/train_braintokenizer.sh",
    "brainomni": "script/train_brainomni.sh",
}


class CampaignHealthError(RuntimeError):
    """Raised when a campaign cannot provide verified portable weights."""


@dataclass(frozen=True)
class CampaignContext:
    """Resolved paths and state for one semantic campaign attempt."""

    root: Path
    attempt_root: Path
    attempt_id: str
    stage: str
    identity_sha256: str
    training_required: bool

    @property
    def checkpoint_root(self) -> Path:
        """Return the campaign DeepSpeed checkpoint directory."""
        return self.root / "checkpoint"

    @property
    def portable_path(self) -> Path:
        """Return the completed portable weight path for this stage."""
        return self.root / PORTABLE_WEIGHT_NAMES[self.stage]


@dataclass(frozen=True)
class CampaignHealth:
    """Verified identity information returned to campaign consumers."""

    root: Path
    stage: str
    campaign_sha256: str
    model_config_sha256: str
    model_state_sha256: str
    portable_path: Path
    repaired: bool


def canonical_json_sha256(value: Any) -> str:
    """Return SHA-256 for a canonical JSON representation."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def portable_weight_name(stage: str) -> str:
    """Return the portable checkpoint filename for a pre-training stage."""
    try:
        return PORTABLE_WEIGHT_NAMES[stage]
    except KeyError as error:
        raise ConfigError(
            f"Unsupported pre-training stage {stage!r}. Expected one of "
            f"{sorted(PORTABLE_WEIGHT_NAMES)}."
        ) from error


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise CampaignHealthError(
            f"{description} does not exist: {path.resolve()}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignHealthError(
            f"Could not read {description}: {path.resolve()}. "
            f"Original error: {error}"
        ) from error
    if not isinstance(value, dict):
        raise CampaignHealthError(
            f"Expected a mapping in {description}, but got "
            f"{type(value).__name__}: {path.resolve()}"
        )
    return value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically write a JSON mapping."""
    _atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True) + "\n",
    )


def atomic_yaml(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically write a YAML mapping."""
    _atomic_text(path, yaml.safe_dump(dict(value), sort_keys=False))


@contextmanager
def campaign_lock(root: str | Path) -> Iterator[None]:
    """Hold the exclusive filesystem lock for one campaign root."""
    campaign_root = Path(root).resolve()
    campaign_root.mkdir(parents=True, exist_ok=True)
    lock_path = campaign_root / ".campaign.lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def build_split_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build digests for exact train, validation, and test splits."""
    from pretrain_config import metadata_directory

    metadata_root = metadata_directory(config)
    manifest: dict[str, Any] = {"partitions": {}}
    for partition in ("train", "val", "test"):
        path = metadata_root / f"{partition}.json"
        if not path.is_file():
            raise ConfigError(
                f"Required {partition} metadata does not exist: "
                f"{path.resolve()}. Run the preprocessing launcher with the "
                "same configuration layers, then retry training."
            )
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ConfigError(
                f"Could not read {partition} metadata: {path.resolve()}. "
                f"Rerun preprocessing. Original error: {error}"
            ) from error
        if not isinstance(rows, list) or not rows:
            raise ConfigError(
                f"The {partition} metadata partition is empty: "
                f"{path.resolve()}. Adjust dataset selection or split ratios "
                "and rerun preprocessing."
            )
        datasets = sorted(
            {
                row.get("dataset")
                for row in rows
                if isinstance(row, dict) and isinstance(row.get("dataset"), str)
            }
        )
        manifest["partitions"][partition] = {
            "records": len(rows),
            "datasets": datasets,
            "sha256": sha256_file(path),
        }
    manifest["sha256"] = canonical_json_sha256(manifest["partitions"])
    return manifest


def _semantic_payload(
    config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    tokenizer_identity: Mapping[str, str] | None,
) -> dict[str, Any]:
    campaign = deepcopy(config["campaign"])
    campaign.pop("model")
    catalog = selected_data_catalog(config)
    datasets = campaign["data"]["included_datasets"]
    campaign["data"]["dataset_signal_types"] = {
        dataset: catalog[dataset]["signal_type"] for dataset in datasets
    }
    payload: dict[str, Any] = {
        "configuration_schema_version": config["schema_version"],
        "campaign": campaign,
        "model_config_sha256": canonical_json_sha256(model_config),
        "split_manifest_sha256": split_manifest["sha256"],
    }
    if tokenizer_identity is not None:
        payload["tokenizer_identity"] = dict(tokenizer_identity)
    return payload


def _attempt_id() -> str:
    provided = os.environ.get("BRAINOMNI_ATTEMPT_ID")
    if provided:
        return provided
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{os.getppid()}"


def _write_invariant(path: Path, value: Mapping[str, Any]) -> None:
    expected = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        observed = path.read_text(encoding="utf-8")
        if observed != expected:
            raise ConfigError(
                f"Invariant campaign artifact differs from the resolved "
                f"semantics: {path.resolve()}. Do not edit campaign artifacts; "
                "restore the original file or use a different output root."
            )
        return
    _atomic_text(path, expected)


def _write_yaml_invariant(path: Path, value: Mapping[str, Any]) -> None:
    expected = yaml.safe_dump(dict(value), sort_keys=False)
    if path.exists():
        observed = path.read_text(encoding="utf-8")
        if observed != expected:
            raise ConfigError(
                f"Invariant campaign artifact differs from the resolved "
                f"semantics: {path.resolve()}. Restore the original file "
                "before retrying."
            )
        return
    _atomic_text(path, expected)


def prepare_campaign(
    config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    tokenizer_identity: Mapping[str, str] | None = None,
    config_paths: list[str] | None = None,
    overrides: list[str] | None = None,
    world_size: int | None = None,
    rank: int = 0,
) -> CampaignContext:
    """Resolve and initialize one exact-semantic campaign attempt."""
    split_manifest = build_split_manifest(config)
    payload = _semantic_payload(
        config,
        model_config,
        split_manifest,
        tokenizer_identity,
    )
    identity_sha256 = canonical_json_sha256(payload)
    stage = config["campaign"]["stage"]
    root = (
        Path(config["invocation"]["output_root"])
        / stage
        / identity_sha256[:CAMPAIGN_HASH_LENGTH]
    ).resolve()
    attempt_id = _attempt_id()
    attempt_root = root / "attempts" / attempt_id
    identity = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "campaign_sha256": identity_sha256,
        "stage": stage,
        "semantic_payload": payload,
    }
    portable_setting = {
        "schema_version": config["schema_version"],
        "campaign": deepcopy(config["campaign"]),
    }
    portable_setting["campaign"].pop("model")
    portable_setting["campaign"]["data"]["dataset_signal_types"] = payload[
        "campaign"
    ]["data"]["dataset_signal_types"]
    with campaign_lock(root):
        root.mkdir(parents=True, exist_ok=True)
        _write_invariant(root / CAMPAIGN_IDENTITY_FILE, identity)
        _write_invariant(root / "model_cfg.json", dict(model_config))
        _write_yaml_invariant(
            root / "pretrain_setting.yaml",
            portable_setting,
        )
        sidecar = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "stage": stage,
            "campaign_sha256": identity_sha256,
            "model_config_sha256": sha256_file(root / "model_cfg.json"),
            "pretrain_setting_sha256": sha256_file(
                root / "pretrain_setting.yaml"
            ),
        }
        if tokenizer_identity is not None:
            sidecar["tokenizer_identity"] = dict(tokenizer_identity)
        _write_invariant(root / "pretrain_setting.json", sidecar)
        _write_invariant(root / "split_manifest.json", split_manifest)
        if rank == 0:
            attempt_root.mkdir(parents=True, exist_ok=True)
        invocation_artifact = {
            "invocation": deepcopy(config["invocation"]),
            "launch": {
                "config_paths": [
                    str(Path(path).resolve()) for path in config_paths or []
                ],
                "overrides": list(overrides or []),
                "world_size": world_size,
            },
        }
        if rank == 0:
            atomic_yaml(
                attempt_root / "invocation.yaml",
                invocation_artifact,
            )
            atomic_json(
                attempt_root / "status.json",
                {
                    "attempt_id": attempt_id,
                    "campaign_sha256": identity_sha256,
                    "state": "started",
                },
            )
        status_path = root / CAMPAIGN_STATUS_FILE
        if status_path.exists():
            status = _load_json(status_path, "campaign status")
        else:
            status = {
                "campaign_sha256": identity_sha256,
                "state": "incomplete",
            }
            atomic_json(status_path, status)
        if status.get("campaign_sha256") != identity_sha256:
            raise ConfigError(
                f"Campaign status identity mismatch at "
                f"{status_path.resolve()}. Expected {identity_sha256}, got "
                f"{status.get('campaign_sha256')!r}. Restore the original "
                "status file before retrying."
            )
    training_required = status.get("state") != "complete"
    return CampaignContext(
        root=root,
        attempt_root=attempt_root,
        attempt_id=attempt_id,
        stage=stage,
        identity_sha256=identity_sha256,
        training_required=training_required,
    )


def repository_attempt_log_directory(context: CampaignContext) -> Path:
    """Return the active repository log directory for one attempt."""
    terminal_log = os.environ.get("BRAINOMNI_TERMINAL_LOG_PATH")
    if terminal_log:
        directory = Path(terminal_log).resolve().parent
        if directory.name != context.attempt_id:
            raise ConfigError(
                "Terminal-log attempt identifier does not match the "
                "resolved campaign attempt."
            )
        return directory
    return (
        Path(__file__).resolve().parents[1]
        / "logs"
        / context.stage
        / "pending"
        / context.attempt_id
    )


def _checkpoint_files(directory: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(directory)),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not files:
        raise CampaignHealthError(
            f"Checkpoint contains no files: {directory.resolve()}"
        )
    return files


def record_checkpoint(
    root: str | Path,
    tag: str,
    runtime_identity: Mapping[str, Any] | None = None,
    acquire_lock: bool = True,
) -> None:
    """Record file sizes and digests for one completed checkpoint tag."""
    campaign_root = Path(root).resolve()
    tag_path = campaign_root / "checkpoint" / tag
    files = _checkpoint_files(tag_path)
    manifest_path = campaign_root / "checkpoint" / CHECKPOINT_MANIFEST_FILE
    lock = campaign_lock(campaign_root) if acquire_lock else nullcontext()
    with lock:
        if manifest_path.exists():
            manifest = _load_json(manifest_path, "checkpoint manifest")
        else:
            manifest = {"tags": {}}
        tags = manifest.setdefault("tags", {})
        tags[tag] = {
            "files": files,
            "runtime_recovery_identity": (
                dict(runtime_identity) if runtime_identity is not None else None
            ),
            "sha256": canonical_json_sha256(files),
        }
        atomic_json(manifest_path, manifest)


def validate_checkpoint(
    root: str | Path,
    tag: str,
    runtime_identity: Mapping[str, Any] | None = None,
) -> str:
    """Validate one DeepSpeed checkpoint tag and return its manifest digest."""
    campaign_root = Path(root).resolve()
    manifest_path = campaign_root / "checkpoint" / CHECKPOINT_MANIFEST_FILE
    manifest = _load_json(manifest_path, "checkpoint manifest")
    expected = manifest.get("tags", {}).get(tag)
    if not isinstance(expected, dict):
        raise CampaignHealthError(
            f"Checkpoint manifest has no {tag!r} tag: "
            f"{manifest_path.resolve()}"
        )
    if runtime_identity is not None and expected.get(
        "runtime_recovery_identity"
    ) != dict(runtime_identity):
        raise CampaignHealthError(
            f"Checkpoint {tag!r} runtime is incompatible at "
            f"{manifest_path.resolve()}. Expected "
            f"{expected.get('runtime_recovery_identity')!r}, got "
            f"{dict(runtime_identity)!r}. Rerun with the original GPU count, "
            "per-GPU batch, precision, accumulation, and ZeRO stage."
        )
    tag_path = campaign_root / "checkpoint" / tag
    observed_files = _checkpoint_files(tag_path)
    observed_sha256 = canonical_json_sha256(observed_files)
    if observed_sha256 != expected.get("sha256"):
        raise CampaignHealthError(
            f"Checkpoint {tag!r} failed integrity validation at "
            f"{tag_path.resolve()}. Expected digest "
            f"{expected.get('sha256')!r}, got {observed_sha256}. Run the "
            "campaign repair command shown below."
        )
    return observed_sha256


def load_portable_state(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a portable tensor state dictionary from disk."""
    weights_path = Path(path).resolve()
    if not weights_path.is_file():
        raise CampaignHealthError(
            f"Portable weights do not exist: {weights_path}"
        )
    try:
        state = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise CampaignHealthError(
            f"Could not load portable weights: {weights_path}. "
            f"Original error: {error}"
        ) from error
    if not isinstance(state, dict) or not state:
        raise CampaignHealthError(
            f"Expected a non-empty state dictionary in {weights_path}, got "
            f"{type(state).__name__}."
        )
    invalid = [
        key for key, value in state.items() if not torch.is_tensor(value)
    ]
    if invalid:
        raise CampaignHealthError(
            f"Portable weights contain non-tensor entries {invalid}: "
            f"{weights_path}"
        )
    return state


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Return a serialization-independent digest for a tensor state mapping."""
    digest = sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        metadata = {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        digest.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_state_schema(
    state: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    """Return portable tensor names, dtypes, and shapes."""
    return {
        name: {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        for name, tensor in sorted(state.items())
    }


def _expected_model_state(
    stage: str,
    model_config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    if stage == "braintokenizer":
        from braintokenizer.model import BrainTokenizer

        model = BrainTokenizer(**model_config)
    elif stage == "brainomni":
        from brainomni.model import BrainOmni

        model = BrainOmni(**model_config)
    else:
        raise CampaignHealthError(
            f"Unsupported campaign stage {stage!r}. Expected one of "
            f"{sorted(PORTABLE_WEIGHT_NAMES)}."
        )
    return model.state_dict()


def validate_portable_state(
    path: str | Path,
    stage: str,
    model_config: Mapping[str, Any],
    expected_sha256: str | None = None,
    expected_schema: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """Validate portable tensors against model keys, shapes, and digest."""
    state = load_portable_state(path)
    if expected_schema is None:
        expected = _expected_model_state(stage, model_config)
        expected_schema = tensor_state_schema(expected)
    observed_schema = tensor_state_schema(state)
    missing = sorted(set(expected_schema) - set(observed_schema))
    unexpected = sorted(set(observed_schema) - set(expected_schema))
    if missing or unexpected:
        raise CampaignHealthError(
            f"Portable state keys do not match {stage}. Missing keys: "
            f"{missing}; unexpected keys: {unexpected}. Affected file: "
            f"{Path(path).resolve()}"
        )
    mismatched_schema = {
        key: {
            "expected": expected_schema[key],
            "observed": observed_schema[key],
        }
        for key in expected_schema
        if expected_schema[key] != observed_schema[key]
    }
    if mismatched_schema:
        raise CampaignHealthError(
            f"Portable tensor dtypes or shapes do not match {stage}: "
            f"{mismatched_schema}. "
            f"Affected file: {Path(path).resolve()}"
        )
    non_finite = [
        key
        for key, tensor in state.items()
        if (tensor.is_floating_point() or tensor.is_complex())
        and not torch.isfinite(tensor).all().item()
    ]
    if non_finite:
        raise CampaignHealthError(
            f"Portable weights contain non-finite tensors {non_finite}: "
            f"{Path(path).resolve()}"
        )
    observed_sha256 = tensor_state_sha256(state)
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise CampaignHealthError(
            f"Portable model-state digest mismatch at "
            f"{Path(path).resolve()}. Expected {expected_sha256}, got "
            f"{observed_sha256}."
        )
    return observed_sha256


def _validate_sidecars(
    root: Path,
    expected_stage: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    identity = _load_json(
        root / CAMPAIGN_IDENTITY_FILE,
        "campaign identity",
    )
    sidecar = _load_json(root / "pretrain_setting.json", "semantic sidecar")
    status = _load_json(root / CAMPAIGN_STATUS_FILE, "campaign status")
    if identity.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise CampaignHealthError(
            "Unsupported campaign artifact schema at "
            f"{(root / CAMPAIGN_IDENTITY_FILE).resolve()}. Expected "
            f"{ARTIFACT_SCHEMA_VERSION}, got "
            f"{identity.get('artifact_schema_version')!r}."
        )
    semantic_payload = identity.get("semantic_payload")
    observed_identity = canonical_json_sha256(semantic_payload)
    if observed_identity != identity.get("campaign_sha256"):
        raise CampaignHealthError(
            f"Campaign semantic identity is corrupt at "
            f"{(root / CAMPAIGN_IDENTITY_FILE).resolve()}. Expected "
            f"{identity.get('campaign_sha256')!r}, got "
            f"{observed_identity}. Restore the original identity file."
        )
    stage = identity.get("stage")
    if stage not in PORTABLE_WEIGHT_NAMES:
        raise CampaignHealthError(
            f"Campaign identity has unsupported stage {stage!r}: "
            f"{(root / CAMPAIGN_IDENTITY_FILE).resolve()}"
        )
    if expected_stage is not None and stage != expected_stage:
        raise CampaignHealthError(
            f"Expected a {expected_stage} campaign root, but found {stage}: "
            f"{root.resolve()}"
        )
    campaign_sha256 = identity.get("campaign_sha256")
    if sidecar.get("stage") != stage:
        raise CampaignHealthError(
            f"Semantic sidecar stage differs from campaign stage at "
            f"{root.resolve()}. Expected {stage!r}, got "
            f"{sidecar.get('stage')!r}. Restore the original sidecar."
        )
    if sidecar.get("campaign_sha256") != campaign_sha256:
        raise CampaignHealthError(
            f"Campaign identity differs between semantic sidecars at "
            f"{root.resolve()}. Restore the original campaign metadata."
        )
    if status.get("campaign_sha256") != campaign_sha256:
        raise CampaignHealthError(
            f"Campaign status identity differs from campaign metadata at "
            f"{root.resolve()}. Restore the original campaign status."
        )
    model_path = root / "model_cfg.json"
    setting_path = root / "pretrain_setting.yaml"
    for artifact_path, description in (
        (model_path, "model configuration"),
        (setting_path, "pre-training setting"),
    ):
        if not artifact_path.is_file():
            raise CampaignHealthError(
                f"Campaign {description} does not exist: "
                f"{artifact_path.resolve()}. Restore the original semantic "
                "artifact before retrying health repair."
            )
    observed_model = sha256_file(model_path)
    observed_setting = sha256_file(setting_path)
    if observed_model != sidecar.get("model_config_sha256"):
        raise CampaignHealthError(
            f"Model configuration digest mismatch at {model_path.resolve()}. "
            f"Expected {sidecar.get('model_config_sha256')!r}, got "
            f"{observed_model}. Restore the original model configuration."
        )
    if observed_setting != sidecar.get("pretrain_setting_sha256"):
        raise CampaignHealthError(
            f"Pre-training setting digest mismatch at "
            f"{setting_path.resolve()}. Expected "
            f"{sidecar.get('pretrain_setting_sha256')!r}, got "
            f"{observed_setting}. Restore the original semantic settings."
        )
    return identity, sidecar, status


def _repair_commands(root: Path, stage: str) -> str:
    repository = Path(__file__).resolve().parents[1]
    repair_script = repository / "script" / "repair_pretrain_campaign.sh"
    repair = " ".join(
        shlex.quote(part)
        for part in (
            "bash",
            str(repair_script),
            "--campaign-root",
            str(root),
        )
    )
    attempts = sorted((root / "attempts").glob("*/invocation.yaml"))
    retrain = (
        f"rerun {STAGE_LAUNCHERS[stage]} with the original configuration "
        "layers"
    )
    if attempts:
        artifact = yaml.safe_load(attempts[-1].read_text(encoding="utf-8"))
        launch = (
            artifact.get("launch", {}) if isinstance(artifact, dict) else {}
        )
        paths = launch.get("config_paths", [])
        world_size = launch.get("world_size")
        overrides = launch.get("overrides", [])
        if paths and isinstance(world_size, int):
            parts = [
                "bash",
                str(repository / STAGE_LAUNCHERS[stage]),
                "--num-gpus",
                str(world_size),
                "--config",
                *paths,
            ]
            for override in overrides:
                parts.extend(["--set", override])
            retrain = " ".join(shlex.quote(part) for part in parts)
    return f"Repair command: {repair}\nFull retraining command: {retrain}"


def _convert_best(
    root: Path,
    stage: str,
    destination: Path,
) -> None:
    from factory.checkpoint import convert_best_checkpoint

    convert_best_checkpoint(
        root,
        stage=stage,
        output_path=destination,
        allow_existing=False,
    )


def ensure_campaign_health(
    root: str | Path,
    expected_stage: str | None = None,
    repair: bool = True,
) -> CampaignHealth:
    """Validate a completed campaign and optionally repair portable weights."""
    campaign_root = Path(root).resolve()
    if not campaign_root.is_dir():
        raise CampaignHealthError(
            f"Campaign root is not a directory: {campaign_root}. Pass the "
            "directory containing campaign_identity.json, not a direct "
            "weight-file path."
        )
    with campaign_lock(campaign_root):
        identity, sidecar, status = _validate_sidecars(
            campaign_root,
            expected_stage,
        )
        stage = identity["stage"]
        if status.get("state") != "complete":
            raise CampaignHealthError(
                f"Campaign is not complete: {campaign_root}. Observed state "
                f"{status.get('state')!r}. Resume it with the "
                "training command.\n"
                f"{_repair_commands(campaign_root, stage)}"
            )
        try:
            validate_checkpoint(campaign_root, "best")
        except CampaignHealthError as checkpoint_error:
            raise CampaignHealthError(
                f"Best checkpoint health validation failed: "
                f"{checkpoint_error}\n"
                f"{_repair_commands(campaign_root, stage)}"
            ) from checkpoint_error
        expected_sha256 = status.get("portable_model_state_sha256")
        if not isinstance(expected_sha256, str):
            raise CampaignHealthError(
                f"Campaign status lacks portable model-state identity: "
                f"{(campaign_root / CAMPAIGN_STATUS_FILE).resolve()}.\n"
                f"{_repair_commands(campaign_root, stage)}"
            )
        model_config = _load_json(
            campaign_root / "model_cfg.json",
            "model configuration",
        )
        expected_schema = status.get("portable_state_schema")
        if expected_schema is not None:
            observed_schema_sha256 = canonical_json_sha256(expected_schema)
            expected_schema_sha256 = status.get("portable_state_schema_sha256")
            if observed_schema_sha256 != expected_schema_sha256:
                raise CampaignHealthError(
                    "Portable state schema identity is corrupt at "
                    f"{(campaign_root / CAMPAIGN_STATUS_FILE).resolve()}. "
                    f"Expected {expected_schema_sha256!r}, got "
                    f"{observed_schema_sha256}. Run full campaign retraining."
                )
        portable_path = campaign_root / portable_weight_name(stage)
        try:
            observed_sha256 = validate_portable_state(
                portable_path,
                stage,
                model_config,
                expected_sha256,
                expected_schema=expected_schema,
            )
            repaired = False
        except CampaignHealthError as original_error:
            if not repair:
                raise CampaignHealthError(
                    f"Campaign health validation failed: {original_error}\n"
                    f"{_repair_commands(campaign_root, stage)}"
                ) from original_error
            temporary = portable_path.with_name(
                f".{portable_path.name}.{os.getpid()}.repair"
            )
            try:
                validate_checkpoint(campaign_root, "best")
                if temporary.exists():
                    temporary.unlink()
                _convert_best(campaign_root, stage, temporary)
                observed_sha256 = validate_portable_state(
                    temporary,
                    stage,
                    model_config,
                    expected_sha256,
                    expected_schema=expected_schema,
                )
                temporary.replace(portable_path)
            except Exception as repair_error:
                if temporary.exists():
                    temporary.unlink()
                raise CampaignHealthError(
                    f"Campaign health validation failed: {original_error}. "
                    f"Automatic repair from "
                    f"{(campaign_root / 'checkpoint' / 'best').resolve()} "
                    f"also failed: {repair_error}.\n"
                    f"{_repair_commands(campaign_root, stage)}"
                ) from repair_error
            status.setdefault("repair_history", []).append(
                {
                    "portable_path": str(portable_path.resolve()),
                    "repaired_from": str(
                        (campaign_root / "checkpoint" / "best").resolve()
                    ),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            atomic_json(campaign_root / CAMPAIGN_STATUS_FILE, status)
            warnings.warn(
                f"Repaired {portable_path.resolve()} from "
                f"{(campaign_root / 'checkpoint' / 'best').resolve()}.",
                RuntimeWarning,
                stacklevel=2,
            )
            repaired = True
        return CampaignHealth(
            root=campaign_root,
            stage=stage,
            campaign_sha256=identity["campaign_sha256"],
            model_config_sha256=sidecar["model_config_sha256"],
            model_state_sha256=observed_sha256,
            portable_path=portable_path,
            repaired=repaired,
        )


def export_completed_weights(
    context: CampaignContext,
    expected_state: Mapping[str, torch.Tensor] | None = None,
) -> CampaignHealth:
    """Create and validate a stage portable checkpoint from the best tag."""
    with campaign_lock(context.root):
        validate_checkpoint(context.root, "best")
        model_config = _load_json(
            context.root / "model_cfg.json",
            "model configuration",
        )
        temporary = context.portable_path.with_name(
            f".{context.portable_path.name}.{os.getpid()}.export"
        )
        if temporary.exists():
            temporary.unlink()
        _convert_best(context.root, context.stage, temporary)
        expected_schema = (
            tensor_state_schema(expected_state)
            if expected_state is not None
            else None
        )
        digest = validate_portable_state(
            temporary,
            context.stage,
            model_config,
            expected_schema=expected_schema,
        )
        if expected_schema is None:
            expected_schema = tensor_state_schema(
                load_portable_state(temporary)
            )
        temporary.replace(context.portable_path)
        status = _load_json(
            context.root / CAMPAIGN_STATUS_FILE,
            "campaign status",
        )
        status.update(
            {
                "state": "complete",
                "portable_model_state_sha256": digest,
                "portable_state_schema": expected_schema,
                "portable_state_schema_sha256": (
                    canonical_json_sha256(expected_schema)
                ),
                "completed_attempt_id": context.attempt_id,
            }
        )
        failed_path_text = status.pop("active_failed_recovery", None)
        atomic_json(context.root / CAMPAIGN_STATUS_FILE, status)
        attempt_status = _load_json(
            context.attempt_root / "status.json",
            "attempt status",
        )
        attempt_status["state"] = "complete"
        attempt_status["portable_model_state_sha256"] = digest
        atomic_json(context.attempt_root / "status.json", attempt_status)
        if failed_path_text is not None:
            failed_path = Path(failed_path_text).resolve()
            checkpoint_root = (context.root / "checkpoint").resolve()
            if failed_path.parent != checkpoint_root:
                raise CampaignHealthError(
                    f"Refusing to remove failed recovery outside checkpoint "
                    f"root: {failed_path}"
                )
            if failed_path.is_dir():
                shutil.rmtree(failed_path)
                warnings.warn(
                    f"Removed recovered checkpoint quarantine: {failed_path}",
                    RuntimeWarning,
                    stacklevel=2,
                )
    return ensure_campaign_health(
        context.root,
        expected_stage=context.stage,
        repair=False,
    )


def quarantine_failed_recovery(
    context: CampaignContext,
    error: Exception,
) -> Path:
    """Preserve unusable state and reset one exact campaign for retraining."""
    checkpoint_root = context.checkpoint_root
    failed_path = checkpoint_root / (f"failed_recovery_{context.attempt_id}")
    with campaign_lock(context.root):
        if failed_path.exists():
            raise CampaignHealthError(
                f"Failed-recovery directory already exists: "
                f"{failed_path.resolve()}. Inspect or move it before retrying."
            )
        failed_path.mkdir(parents=True)
        for path in sorted(checkpoint_root.iterdir()):
            if path == failed_path or path.name.startswith("failed_recovery_"):
                continue
            shutil.move(str(path), str(failed_path / path.name))
        if context.portable_path.exists():
            shutil.move(
                str(context.portable_path),
                str(failed_path / context.portable_path.name),
            )
        status = _load_json(
            context.root / CAMPAIGN_STATUS_FILE,
            "campaign status",
        )
        status.update(
            {
                "state": "incomplete",
                "active_failed_recovery": str(failed_path.resolve()),
                "recovery_error": str(error),
            }
        )
        status.pop("portable_model_state_sha256", None)
        atomic_json(context.root / CAMPAIGN_STATUS_FILE, status)
    warnings.warn(
        f"Automatic repair failed for {context.root.resolve()}: {error}. "
        f"Preserved unusable artifacts at {failed_path.resolve()} and will "
        "restart this exact semantic campaign from scratch.",
        RuntimeWarning,
        stacklevel=2,
    )
    return failed_path


def _wait_for_attempt_decision(context: CampaignContext) -> bool:
    """Wait for rank zero to publish the shared training decision."""
    status_path = context.attempt_root / "status.json"
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        if status_path.is_file():
            status = _load_json(status_path, "attempt status")
            required = status.get("training_required")
            if isinstance(required, bool):
                return required
        time.sleep(0.1)
    raise CampaignHealthError(
        f"Timed out waiting for rank zero to publish campaign state at "
        f"{status_path.resolve()}. Stop the distributed launch and retry."
    )


def ensure_training_campaign(
    context: CampaignContext,
    rank: int = 0,
) -> bool:
    """Repair a completed campaign or report that training is needed."""
    if rank != 0:
        return _wait_for_attempt_decision(context)
    status = _load_json(
        context.root / CAMPAIGN_STATUS_FILE,
        "campaign status",
    )
    health = None
    training_required = status.get("state") != "complete"
    if not training_required:
        try:
            health = ensure_campaign_health(
                context.root,
                expected_stage=context.stage,
                repair=True,
            )
            if health.repaired:
                record_attempt_repair(context, health)
        except CampaignHealthError as error:
            quarantine_failed_recovery(context, error)
            training_required = True
    attempt_status = _load_json(
        context.attempt_root / "status.json",
        "attempt status",
    )
    attempt_status["training_required"] = training_required
    if training_required:
        attempt_status["state"] = "training"
    else:
        attempt_status["state"] = "evaluation_only"
        attempt_status["portable_repaired"] = health.repaired
    atomic_json(context.attempt_root / "status.json", attempt_status)
    return training_required


def rng_client_state(
    epoch: int,
    best_eval_loss: float,
    train_step_counter: int,
) -> dict[str, Any]:
    """Return exact trainer and random state for DeepSpeed client storage."""
    state: dict[str, Any] = {
        "epoch": epoch,
        "best_eval_loss": best_eval_loss,
        "train_step_counter": train_step_counter,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random_state"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_client_state(
    state: Mapping[str, Any],
) -> tuple[int, float, int]:
    """Restore trainer progress and every recorded random-number generator."""
    required = {
        "epoch",
        "best_eval_loss",
        "train_step_counter",
        "python_random_state",
        "numpy_random_state",
        "torch_random_state",
    }
    missing = sorted(required - set(state))
    if missing:
        raise CampaignHealthError(
            f"Checkpoint client state is missing required recovery fields: "
            f"{missing}. Restart the exact campaign from scratch."
        )
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_random_state"])
    cuda_state = state.get("torch_cuda_random_state")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
    return (
        int(state["epoch"]),
        float(state["best_eval_loss"]),
        int(state["train_step_counter"]),
    )


def update_attempt_status(
    context: CampaignContext,
    state: str,
    error: Exception | None = None,
) -> None:
    """Record the final state of one invocation attempt."""
    status_path = context.attempt_root / "status.json"
    with campaign_lock(context.root):
        status = _load_json(status_path, "attempt status")
        status["state"] = state
        if error is not None:
            status["error"] = str(error)
        atomic_json(status_path, status)


def record_attempt_repair(
    context: CampaignContext,
    health: CampaignHealth,
) -> None:
    """Record a successful campaign repair in the invoking attempt."""
    status_path = context.attempt_root / "status.json"
    with campaign_lock(context.root):
        status = _load_json(status_path, "attempt status")
        repairs = status.setdefault("repairs", [])
        repair = {
            "campaign_root": str(health.root.resolve()),
            "portable_path": str(health.portable_path.resolve()),
            "repaired_from": str(
                (health.root / "checkpoint" / "best").resolve()
            ),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        repairs.append(repair)
        atomic_json(status_path, status)

"""Shared distributed checkpoint and evaluation artifact operations."""

from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import shlex
from typing import Any, Mapping
from urllib.parse import quote
import warnings

import torch.distributed as dist
import torch

from factory.campaign import (
    CampaignContext,
    CampaignHealthError,
    atomic_json,
    campaign_lock,
    load_portable_state,
    record_checkpoint,
    restore_rng_client_state,
    rng_client_state,
    validate_checkpoint,
)
from pretrain_config import canonical_config_sha256, sha256_file


def _engine_recovery_identity(engine: Any) -> dict[str, Any]:
    """Return DeepSpeed runtime fields required for exact state recovery."""
    return {
        "world_size": int(engine.world_size),
        "micro_batch_size_per_gpu": int(
            engine.train_micro_batch_size_per_gpu()
        ),
        "gradient_accumulation_steps": int(
            engine.gradient_accumulation_steps()
        ),
        "zero_optimization_stage": int(engine.zero_optimization_stage()),
        "bfloat16_enabled": bool(engine.bfloat16_enabled()),
    }


def save_distributed_checkpoint(
    engine: Any,
    context: CampaignContext,
    tag: str,
    epoch: int,
    best_eval_loss: float,
    train_step_counter: int,
    rank: int,
) -> None:
    """Save one collective DeepSpeed checkpoint and record its manifest."""
    state = rng_client_state(
        epoch,
        best_eval_loss,
        train_step_counter,
    )
    state["runtime_recovery_identity"] = _engine_recovery_identity(engine)
    lock = campaign_lock(context.root) if rank == 0 else nullcontext()
    with lock:
        dist.barrier()
        engine.save_checkpoint(
            save_dir=str(context.checkpoint_root),
            tag=tag,
            client_state=state,
            save_latest=False,
        )
        dist.barrier()
        if rank == 0:
            record_checkpoint(
                context.root,
                tag,
                runtime_identity=_engine_recovery_identity(engine),
                acquire_lock=False,
            )
        dist.barrier()


def resume_distributed_checkpoint(
    engine: Any,
    context: CampaignContext,
) -> tuple[int, float, int] | None:
    """Restore the exact campaign's latest valid DeepSpeed state if present."""
    latest = context.checkpoint_root / "latest"
    if not latest.is_dir():
        return None
    rank = dist.get_rank()
    lock = campaign_lock(context.root) if rank == 0 else nullcontext()
    with lock:
        dist.barrier()
        validate_checkpoint(
            context.root,
            "latest",
            runtime_identity=_engine_recovery_identity(engine),
        )
        load_path, client_state = engine.load_checkpoint(
            load_dir=str(context.checkpoint_root),
            tag="latest",
        )
        dist.barrier()
    if load_path is None or client_state is None:
        raise CampaignHealthError(
            f"DeepSpeed could not restore latest checkpoint at "
            f"{latest.resolve()}. Run campaign repair or move the invalid "
            "checkpoint before restarting this exact campaign."
        )
    expected_runtime = _engine_recovery_identity(engine)
    observed_runtime = client_state.get("runtime_recovery_identity")
    if observed_runtime != expected_runtime:
        raise CampaignHealthError(
            "Latest checkpoint runtime is incompatible with this invocation. "
            f"Expected {observed_runtime!r}, got {expected_runtime!r}. "
            "Rerun with the original GPU count, per-GPU batch, precision, "
            "gradient accumulation, and ZeRO stage."
        )
    return restore_rng_client_state(client_state)


def load_completed_portable(engine: Any, context: CampaignContext) -> None:
    """Load a verified completed portable state into a DeepSpeed module."""
    state = load_portable_state(context.portable_path)
    engine.module.load_state_dict(state, strict=True)


def evaluation_metrics_path(
    context: CampaignContext,
    dataset: str,
) -> Path:
    """Return a flat, reversible filename for one evaluation dataset."""
    if dataset == "test":
        filename = "metrics_test_set.json"
    else:
        encoded = quote(dataset, safe="._-")
        if not encoded:
            raise ValueError("Evaluation dataset ID must not be empty.")
        filename = f"metrics_heldout_{encoded}.json"
    return context.root / "evaluations" / filename


def _evaluation_identity(
    context: CampaignContext,
    dataset: str,
    evaluator_path: Path,
    metadata_path: Path,
) -> dict[str, str]:
    status = json.loads(
        (context.root / "campaign_status.json").read_text(encoding="utf-8")
    )
    campaign_identity = json.loads(
        (context.root / "campaign_identity.json").read_text(encoding="utf-8")
    )
    preprocessing = campaign_identity["semantic_payload"]["campaign"][
        "data"
    ]["preprocessing"]
    return {
        "campaign_sha256": context.identity_sha256,
        "dataset": dataset,
        "dataset_metadata_sha256": sha256_file(metadata_path),
        "evaluator_sha256": sha256_file(evaluator_path),
        "model_state_sha256": status["portable_model_state_sha256"],
        "preprocessing_sha256": canonical_config_sha256(preprocessing),
    }


def existing_evaluation_matches(
    context: CampaignContext,
    dataset: str,
    evaluator_path: Path,
    metadata_path: Path,
) -> bool:
    """Return whether an immutable matching evaluation already exists."""
    path = evaluation_metrics_path(context, dataset)
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignHealthError(
            f"Existing evaluation metrics are unreadable: {path.resolve()}. "
            "Do not overwrite them; restore or move the file before retrying. "
            f"Original error: {error}"
        ) from error
    expected = _evaluation_identity(
        context,
        dataset,
        evaluator_path,
        metadata_path,
    )
    if value.get("identity") != expected:
        raise CampaignHealthError(
            f"Existing evaluation identity conflicts at {path.resolve()}. "
            f"Expected {expected}, got {value.get('identity')!r}. Preserve the "
            "existing result and use a new evaluator schema or campaign."
        )
    return True


def write_evaluation_metrics(
    context: CampaignContext,
    dataset: str,
    metrics: Mapping[str, Any],
    evaluator_path: Path,
    metadata_path: Path,
) -> Path:
    """Create one immutable flat evaluation metrics artifact."""
    path = evaluation_metrics_path(context, dataset)
    identity = _evaluation_identity(
        context,
        dataset,
        evaluator_path,
        metadata_path,
    )
    with campaign_lock(context.root):
        if existing_evaluation_matches(
            context,
            dataset,
            evaluator_path,
            metadata_path,
        ):
            return path
        atomic_json(
            path,
            {
                "identity": identity,
                "metrics": dict(metrics),
            },
        )
        _append_evaluation_record(
            context,
            {
                "dataset": dataset,
                "metrics_file": path.name,
                "state": "complete",
                **identity,
            },
        )
    return path


def _append_evaluation_record(
    context: CampaignContext,
    record: Mapping[str, Any],
) -> None:
    index_path = context.root / "evaluations" / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {"records": []}
    records = index.get("records")
    if not isinstance(records, list):
        raise CampaignHealthError(
            f"Evaluation index has invalid records at {index_path.resolve()}. "
            "Restore the original index before retrying."
        )
    if dict(record) not in records:
        records.append(dict(record))
        atomic_json(index_path, index)


def record_skipped_evaluation(
    context: CampaignContext,
    dataset: str,
    reason: str,
) -> None:
    """Append a durable skipped-evaluation outcome without failing training."""
    with campaign_lock(context.root):
        _append_evaluation_record(
            context,
            {
                "campaign_sha256": context.identity_sha256,
                "dataset": dataset,
                "reason": reason,
                "state": "skipped",
            },
        )


def evaluation_metadata_available(
    context: CampaignContext,
    metadata_root: str | Path,
    dataset: str,
    rank: int,
) -> bool:
    """Validate evaluation metadata or record an actionable optional skip."""
    metadata_path = evaluation_metadata_path(metadata_root, dataset)
    available = False
    if metadata_path.is_file():
        try:
            rows = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            if dataset == "test":
                raise CampaignHealthError(
                    f"Test metadata is unreadable: {metadata_path.resolve()}. "
                    f"Rerun preprocessing. Original error: {error}"
                ) from error
        else:
            available = isinstance(rows, list) and bool(rows)
    if available:
        return True
    if dataset == "test":
        raise CampaignHealthError(
            f"Mandatory test metadata is missing or empty: "
            f"{metadata_path.resolve()}. Run preprocessing with the same "
            "campaign configuration before training."
        )
    command = _preprocessing_command(context, dataset)
    reason = (
        f"Compatible held-out metadata is missing or empty at "
        f"{metadata_path.resolve()}. Run: {command}. Then rerun the same "
        "training command; completed weights will not be retrained."
    )
    if rank == 0:
        warnings.warn(reason, RuntimeWarning, stacklevel=2)
        record_skipped_evaluation(context, dataset, reason)
    return False


def evaluation_metadata_path(
    metadata_root: str | Path,
    dataset: str,
) -> Path:
    """Return the metadata file for one test or held-out evaluation."""
    filename = "test.json" if dataset == "test" else f"{dataset}.json"
    return Path(metadata_root) / filename


def _preprocessing_command(
    context: CampaignContext,
    dataset: str,
) -> str:
    invocation_path = context.attempt_root / "invocation.yaml"
    import yaml

    artifact = yaml.safe_load(invocation_path.read_text(encoding="utf-8"))
    launch = artifact.get("launch", {})
    repository = Path(__file__).resolve().parents[1]
    parts = [
        "bash",
        str(repository / "script/pretrain_preprocess.sh"),
        "--config",
        *launch.get("config_paths", []),
    ]
    for override in launch.get("overrides", []):
        if not override.startswith("invocation.held_out_evaluation_datasets="):
            parts.extend(["--set", override])
    held_out = json.dumps([dataset], separators=(",", ":"))
    parts.extend(
        [
            "--set",
            f"invocation.held_out_evaluation_datasets={held_out}",
        ]
    )
    return " ".join(shlex.quote(part) for part in parts)

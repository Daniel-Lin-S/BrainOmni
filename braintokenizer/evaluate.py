"""Evaluate an exported BrainTokenizer without launching training.

Usage: python -m braintokenizer.evaluate --attempt-dir ABSOLUTE_ATTEMPT_PATH
       --datasets DATASET [DATASET ...] --device cpu
Input: campaign sidecars and portable weights above the source attempt;
invocation.yaml supplies the original metadata location. --metadata-dir can
select separately prepared held-out metadata using the campaign preprocessing.
Output: campaign evaluations/metrics_heldout_<dataset>.json, plus the source
attempt's heldout_evaluation_status.json recording completed or unavailable
results. Missing data produces an explicit status, never invented metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from braintokenizer.evaluation import (
    EVALUATOR_PATH,
    build_evaluation_loader,
    evaluate_tokenizer,
    evaluation_settings,
)
from braintokenizer.model import BrainTokenizer
from factory.campaign import (
    CampaignContext,
    atomic_json,
    ensure_campaign_health,
    load_portable_state,
)
from factory.training_runtime import (
    evaluation_metrics_path,
    existing_evaluation_matches,
    write_evaluation_metrics,
)
from pretrain_config import metadata_directory

DEFAULT_BATCH_SIZE = 4
DEFAULT_WORKERS = 0
STATUS_FILENAME = "heldout_evaluation_status.json"


def parse_args() -> argparse.Namespace:
    """Parse the source attempt, datasets, and evaluation runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS)
    return parser.parse_args()


def resolve_catalog_id(dataset: str, catalog: dict) -> str:
    """Resolve an exact ID or unique underscore/hyphen catalog spelling."""
    if dataset in catalog:
        return dataset
    normalized = dataset.replace("_", "-").casefold()
    matches = [
        name for name in catalog
        if name.replace("_", "-").casefold() == normalized
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Dataset {dataset!r} has no unique saved catalog identity."
        )
    return matches[0]


def main() -> None:
    """Write held-out metrics or explicit missing-metadata status."""
    args = parse_args()
    attempt = args.attempt_dir.resolve()
    root = attempt.parent.parent
    health = ensure_campaign_health(root, expected_stage="braintokenizer")
    invocation = yaml.safe_load((attempt / "invocation.yaml").read_text())
    identity = json.loads((root / "campaign_identity.json").read_text())
    campaign = identity["semantic_payload"]["campaign"]
    config = {"campaign": campaign, "invocation": invocation["invocation"]}
    default_metadata_root = metadata_directory(config).resolve()
    metadata_root = (
        args.metadata_dir.resolve() if args.metadata_dir is not None
        else default_metadata_root
    )
    if metadata_root != default_metadata_root:
        manifest_path = metadata_root / "preprocessing.json"
        if not manifest_path.is_file():
            raise ValueError(
                f"External evaluation metadata requires {manifest_path} "
                "with the actual preprocessing configuration."
            )
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("preprocessing") != campaign["data"]["preprocessing"]:
            raise ValueError(
                f"Preprocessing in {manifest_path} does not match the campaign."
            )
    context = CampaignContext(
        root=root, attempt_root=attempt, attempt_id=attempt.name,
        stage="braintokenizer", identity_sha256=health.campaign_sha256,
        training_required=False,
    )
    report = {
        "campaign_sha256": health.campaign_sha256,
        "model_state_sha256": health.model_state_sha256,
        "source_attempt": str(attempt),
        "datasets": {},
    }
    model = None
    for dataset in args.datasets:
        source_id = resolve_catalog_id(
            dataset, invocation["invocation"]["data_catalog"],
        )
        if source_id in campaign["data"]["included_datasets"]:
            raise ValueError(f"Dataset {dataset!r} was used for training.")
        candidates = {metadata_root / f"{name}.json"
                      for name in (dataset, source_id)}
        existing = sorted(path for path in candidates if path.is_file())
        if len(existing) > 1:
            raise ValueError(f"Ambiguous metadata aliases: {existing}.")
        entry = {
            "source_catalog_id": source_id,
            "source_root": invocation["invocation"]["data_catalog"][
                source_id
            ]["path"],
        }
        report["datasets"][dataset] = entry
        if not existing:
            entry.update(
                state="missing_preprocessed_metadata",
                searched_paths=sorted(str(path) for path in candidates),
                reason="Held-out tensors must be prepared using the saved "
                       "campaign preprocessing before evaluation.",
            )
            print(f"Missing evaluation metadata for {dataset}: {metadata_root}")
            continue
        metadata_path = existing[0]
        rows = json.loads(metadata_path.read_text())
        if not isinstance(rows, list) or not rows or any(
            not isinstance(row, dict)
            or row.get("dataset") not in {dataset, source_id} for row in rows
        ):
            raise ValueError(f"Invalid held-out dataset rows: {metadata_path}.")
        if model is None:
            model_config = json.loads((root / "model_cfg.json").read_text())
            objective = campaign["objective"]
            model = BrainTokenizer(
                **model_config,
                channel_mask_ratio=objective["channel_mask_ratio"],
                noise_std=objective["noise_std"],
            )
            model.load_state_dict(load_portable_state(health.portable_path))
            model.to(torch.device(args.device)).eval()
            model.requires_grad_(False)
        settings = evaluation_settings(
            model, campaign["seed"], args.batch_size, 1,
        )
        if not existing_evaluation_matches(
            context, dataset, EVALUATOR_PATH, metadata_path, settings,
        ):
            loader = build_evaluation_loader(
                metadata_path, args.batch_size, args.num_workers,
            )
            metrics = evaluate_tokenizer(
                model, loader, model_config["codebook_size"],
                campaign["seed"], progress=True,
            )
            write_evaluation_metrics(
                context, dataset, metrics, EVALUATOR_PATH,
                metadata_path, settings,
            )
        path = evaluation_metrics_path(context, dataset)
        entry.update(state="complete", metrics_file=str(path))
        print(f"Evaluation metrics: {path}")
    report_path = attempt / STATUS_FILENAME
    atomic_json(report_path, report)
    print(f"Held-out evaluation status: {report_path}")


if __name__ == "__main__":
    main()

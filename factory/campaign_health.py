"""Validate or repair one completed pre-training campaign.

Usage
-----
``python -m factory.campaign_health --campaign-root CAMPAIGN [--check-only]``

Input is a BrainTokenizer or BrainOmni semantic campaign root. Successful
output reports the verified stage, campaign identity, portable path, and
canonical tensor-state digest. Repair uses the campaign's verified best
DeepSpeed checkpoint and never starts training. ``--finish-export`` also
finishes an interrupted export after verifying final-epoch completion,
recording an export-only attempt without running evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from factory.campaign import (
    CampaignHealthError,
    ensure_campaign_health,
    finish_campaign_export,
)


def parse_args() -> argparse.Namespace:
    """Parse the campaign health command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--check-only",
        action="store_true",
        help="validate without repairing damaged portable weights",
    )
    action.add_argument(
        "--finish-export",
        action="store_true",
        help="export after verified final-epoch completion without training",
    )
    return parser.parse_args()


def main() -> None:
    """Validate the requested campaign and print verified identities."""
    args = parse_args()
    root = Path(args.campaign_root).resolve()
    try:
        if args.finish_export:
            health = finish_campaign_export(root)
        else:
            health = ensure_campaign_health(
                root,
                repair=not args.check_only,
            )
    except CampaignHealthError as error:
        raise SystemExit(f"Campaign health check failed: {error}") from error
    action = "repaired and verified" if health.repaired else "verified"
    print(f"Campaign {action}: {health.root}")
    print(f"Stage: {health.stage}")
    print(f"Campaign SHA-256: {health.campaign_sha256}")
    print(f"Model-state SHA-256: {health.model_state_sha256}")
    print(f"Portable weights: {health.portable_path}")


if __name__ == "__main__":
    main()

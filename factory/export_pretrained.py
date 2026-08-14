"""Export a portable pre-training checkpoint with semantic settings."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

REQUIRED_SIDECARS = {
    "model_cfg.json",
    "pretrain_setting.yaml",
    "pretrain_setting.json",
}


def parse_args() -> argparse.Namespace:
    """Parse run, weights, and portable output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
    return parser.parse_args()


def export_checkpoint(
    run_directory: str | Path,
    weights_path: str | Path,
    output_directory: str | Path,
    output_name: str,
) -> None:
    """Copy weights and semantic sidecars without invocation configuration."""
    run_path = Path(run_directory)
    weights = Path(weights_path)
    output_path = Path(output_directory)
    missing = [
        name for name in REQUIRED_SIDECARS if not (run_path / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Run directory is missing required sidecars: {sorted(missing)}"
        )
    if not weights.is_file():
        raise FileNotFoundError(f"Weights file does not exist: {weights.resolve()}")
    if output_path.exists():
        raise FileExistsError(
            f"Portable output directory already exists: {output_path.resolve()}"
        )
    output_path.mkdir(parents=True)
    for name in REQUIRED_SIDECARS:
        shutil.copy2(run_path / name, output_path / name)
    shutil.copy2(weights, output_path / output_name)


def main() -> None:
    """Export one portable checkpoint."""
    args = parse_args()
    export_checkpoint(
        args.run_dir,
        args.weights,
        args.output_dir,
        args.output_name,
    )


if __name__ == "__main__":
    main()

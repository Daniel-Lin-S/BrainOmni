"""Save pre-training figures beside an attempt's TensorBoard directory.

Input: --tensorboard-dir ABSOLUTE_PATH
Output: PNG/PDF figures grouped by monitor family and a provenance manifest
under the sibling visualisation/ directory, or --output-dir ABSOLUTE_PATH.
See the monitor documentation for missing-metric handling and the output
directory structure.
"""

from __future__ import annotations

import argparse

from factory.pretraining_visualisation import STAGES, visualize_pretraining


def parse_args() -> argparse.Namespace:
    """Parse an attempt directory with optional stage and destination."""
    parser = argparse.ArgumentParser(
        description="Plot pre-training scalar monitors for one attempt."
    )
    parser.add_argument("--tensorboard-dir", required=True)
    parser.add_argument(
        "--stage", choices=STAGES, default=None,
        help="Validate this stage against campaign provenance; "
        "inferred from provenance when omitted.",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    """Render available monitoring figures and print the absolute manifest."""
    arguments = parse_args()
    manifest = visualize_pretraining(
        arguments.tensorboard_dir, arguments.stage, arguments.output_dir,
    )
    print(f"Saved pre-training visualisation manifest to {manifest}.")


if __name__ == "__main__":
    main()

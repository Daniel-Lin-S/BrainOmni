"""Export pre-training TensorBoard scalar events as a long-form CSV.

Input
-----
One or more absolute event directories, campaign roots, or event files.

Output
------
An explicitly requested CSV containing run, source file, original and
canonical tags, tag components, step, wall time, and scalar value.
"""

from __future__ import annotations

import argparse

from factory.pretraining_monitor_events import (
    load_monitor_events,
    write_monitor_csv,
)


def parse_args() -> argparse.Namespace:
    """Parse monitor export arguments."""
    parser = argparse.ArgumentParser(
        description="Export normalized pre-training TensorBoard scalars."
    )
    parser.add_argument(
        "--event-dir",
        action="append",
        required=True,
        help="Absolute event directory, campaign root, or event file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Absolute destination CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    """Load, normalize, and export requested monitor events."""
    arguments = parse_args()
    events = load_monitor_events(arguments.event_dir)
    output = write_monitor_csv(events, arguments.output)
    print(f"Saved normalized monitor events to {output.resolve()}.")


if __name__ == "__main__":
    main()

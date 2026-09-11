"""Verify dataset-scoped progress and independent preprocessing summaries.

Synthetic completion/window metadata covers included, evaluation and unrelated
cached datasets. JSON artifacts are written only under temporary log folders.
"""

import json
import logging
from pathlib import Path
import tempfile
import unittest

from factory.process import (
    build_grouped_dataset_summary,
    select_pending_recordings,
    write_dataset_summary,
)
from pretrain_config import ConfigError


class PreprocessingReportingTests(unittest.TestCase):
    """Keep cache status explicit and summary groups disjoint."""

    def metadata(self):
        """Build distinct dataset sizes and channel counts for aggregation."""
        records, windows = [], []
        for name, channels, count in [
            ("TRAIN-A", 2, 1), ("TRAIN-B", 4, 2),
            ("EVAL", 8, 3), ("UNSELECTED", 16, 4),
        ]:
            records.append({
                "dataset": name, "source_recording": "raw.fif",
                "raw_duration_seconds": count * 10.0,
                "preprocessed_duration_seconds": count * 10.0,
                "generated_windows": count,
            })
            windows.extend({
                "dataset": name, "source_recording": "raw.fif",
                "window_modality": "meg", "channels": channels,
                "eeg_channels": 0, "meg_channels": channels,
                "grad_channels": 0,
            } for _ in range(count))
        config = {
            "campaign": {"data": {
                "included_datasets": ["TRAIN-A", "TRAIN-B"],
                "preprocessing": {"sample_rate_hz": 256},
            }},
            "invocation": {
                "held_out_evaluation_datasets": ["EVAL"],
                "data_catalog": {r["dataset"]: {} for r in records},
            },
        }
        return records, windows, config

    def test_summary_groups_aggregate_only_selected_datasets(self):
        records, windows, config = self.metadata()
        summary = build_grouped_dataset_summary(records, windows, config)
        training = summary["included_datasets"]
        evaluation = summary["evaluation_datasets"]
        self.assertEqual(training["aggregate"]["completed_recordings"], 2)
        self.assertEqual(training["aggregate"]["generated_windows"], 3)
        self.assertEqual(training["aggregate"]["raw_duration_seconds"], 30)
        self.assertEqual(training["aggregate"]["window_channel_count"], {
            "min": 2, "max": 4, "median": 4,
        })
        self.assertEqual(evaluation["aggregate"]["completed_recordings"], 1)
        self.assertEqual(evaluation["aggregate"]["generated_windows"], 3)
        self.assertEqual(set(evaluation["datasets"]), {"EVAL"})
        self.assertNotIn("UNSELECTED", json.dumps(summary))
        with tempfile.TemporaryDirectory() as temporary:
            path = write_dataset_summary(summary, Path(temporary))
            self.assertEqual(path.name, "dataset_summary.json")
            self.assertEqual(json.loads(path.read_text()), summary)

    def test_empty_and_missing_evaluation_are_distinguished(self):
        records, windows, config = self.metadata()
        config["invocation"]["held_out_evaluation_datasets"] = []
        summary = build_grouped_dataset_summary(records, windows, config)
        self.assertEqual(summary["evaluation_datasets"], {
            "state": "not_requested", "aggregate": None, "datasets": {},
        })
        config["invocation"]["held_out_evaluation_datasets"] = ["MISSING"]
        with self.assertRaisesRegex(ConfigError, "No completion records"):
            build_grouped_dataset_summary(records, windows, config)
        config["invocation"]["held_out_evaluation_datasets"] = ["TRAIN-A"]
        with self.assertRaisesRegex(ConfigError, "overlap"):
            build_grouped_dataset_summary(records, windows, config)

    def test_completed_and_partial_dataset_cache_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                {"dataset": dataset, "path": str(root / str(index))}
                for index, dataset in enumerate(["CACHED", "PART", "PART"])
            ]
            completed = {row["path"] for row in rows[:2]}
            logger = logging.getLogger("preprocessing-status-test")
            with self.assertLogs(logger, level="INFO") as messages:
                pending = select_pending_recordings(rows, completed, logger)
            self.assertEqual(pending, [rows[2]])
            text = "\n".join(messages.output)
            self.assertIn("dataset=CACHED: skipped signal preprocessing", text)
            self.assertIn("dataset=PART: 1 recordings to preprocess", text)
            self.assertIn("1/2 already complete and skipped", text)


if __name__ == "__main__":
    unittest.main()

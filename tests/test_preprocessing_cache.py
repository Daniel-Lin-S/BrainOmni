"""Verify cache reuse avoids raw reads while discovering new recordings.

All fixtures use temporary catalog roots and in-memory completion records.
No production tensors or preprocessing metadata are modified.
"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from factory.channel_selection import channel_selection_provenance
from factory.process import discover_catalog_recordings
from pretrain_config import ConfigError


class PreprocessingCacheTests(unittest.TestCase):
    """Exercise complete, partial and incompatible recording caches."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        definition = {"path": str(self.root), "signal_type": "meg"}
        self.config = {
            "campaign": {"data": {"included_datasets": ["MEG"]}},
            "invocation": {"data_catalog": {"MEG": definition}},
        }
        self.recording = {"dataset": "MEG", "path": str(self.root / "a.fif")}
        self.completion = {
            "dataset": "MEG", "recording_path": self.recording["path"],
            "channel_selection": channel_selection_provenance(
                "MEG", definition,
            ),
        }
        self.accessor = Mock()
        self.accessor.search_brain_files.return_value = [self.recording]
        raw = self.accessor.read_brain_file.return_value
        raw.info = {"sfreq": 256, "comps": []}
        raw.n_times = 2560

    def test_completed_recordings_never_open_headers(self) -> None:
        with self.assertLogs("processor", level="INFO") as messages:
            result = discover_catalog_recordings(
                self.accessor, self.config, [self.completion],
            )
        self.assertEqual(result, [self.recording])
        self.accessor.read_brain_file.assert_not_called()
        self.assertIn(
            "skipped header validation for 1", "\n".join(messages.output),
        )

    def test_partial_cache_only_reads_new_headers(self) -> None:
        new = {"dataset": "MEG", "path": str(self.root / "new.fif")}
        self.accessor.search_brain_files.return_value.append(new)
        with patch("factory.process.describe_recording_channels",
                   return_value=(0, 2, 0)), patch(
            "factory.process.infer_signal_type", return_value="meg",
        ):
            result = discover_catalog_recordings(
                self.accessor, self.config, [self.completion],
            )
        self.assertEqual(len(result), 2)
        self.accessor.read_brain_file.assert_called_once_with(
            new["path"], preload=False,
        )
        self.assertEqual(result[1]["raw_samples"], 2560)

    def test_changed_selection_rejected_before_discovery(self) -> None:
        self.completion["channel_selection"]["exclude_channel_types"] = ["grad"]
        with self.assertRaisesRegex(ConfigError, "Cached channel selection"):
            discover_catalog_recordings(
                self.accessor, self.config, [self.completion],
            )
        self.accessor.search_brain_files.assert_not_called()
        self.accessor.read_brain_file.assert_not_called()

    def test_unrecorded_selection_is_not_trusted(self) -> None:
        self.completion.pop("channel_selection")
        with self.assertRaisesRegex(ConfigError, "Cached channel selection"):
            discover_catalog_recordings(
                self.accessor, self.config, [self.completion],
            )

    def test_without_versioned_cache_headers_are_validated(self) -> None:
        with patch("factory.process.describe_recording_channels",
                   return_value=(0, 2, 0)), patch(
            "factory.process.infer_signal_type", return_value="meg",
        ):
            result = discover_catalog_recordings(self.accessor, self.config)
        self.accessor.read_brain_file.assert_called_once()
        self.assertEqual(result[0]["signal_type"], "meg")


if __name__ == "__main__":
    unittest.main()

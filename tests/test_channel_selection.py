"""Exercise catalog channel selection, cache reuse, and portable provenance.

Tests use temporary catalogs and synthetic MNE recordings; no external data
or campaign outputs are modified.
"""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mne
import numpy as np
import yaml

from factory.campaign import _semantic_payload
from factory.channel_selection import channel_selection_provenance
from factory.process import (
    discover_catalog_recordings, validate_cached_channel_selection,
)
from factory.utils import filter_channel
from pretrain_config import ConfigError, load_data_catalog, metadata_directory


class RecordingAccessor:
    """Supply fresh synthetic recordings for catalog discovery."""

    def __init__(self, raw: mne.io.BaseRaw) -> None:
        self.raw = raw

    def search_brain_files(self, root: str, dataset: str) -> list[dict]:
        """Return one recording identified by its configured catalog root."""
        return [{"path": str(Path(root) / "raw.fif"), "dataset": dataset}]

    def read_brain_file(self, path: str, preload: bool) -> mne.io.BaseRaw:
        """Isolate in-place channel selection between discovery calls."""
        return self.raw.copy()


class ChannelSelectionTests(unittest.TestCase):
    """Configured types must preserve existing fixed channel-name rules."""

    def config(self, root: Path, policy: dict) -> dict:
        """Build a minimal configuration for one training dataset."""
        return {
            "schema_version": 1,
            "campaign": {
                "seed": 42, "model": {},
                "data": {
                    "included_datasets": ["DATA"],
                    "split_ratios": {"train": 0.8},
                    "preprocessing": {
                        "sample_rate_hz": 256, "low_frequency_hz": 0.1,
                        "high_frequency_hz": 96, "segment_seconds": 10,
                        "stride_seconds": 10,
                    },
                },
            },
            "invocation": {
                "metadata_root": str(root),
                "held_out_evaluation_datasets": [],
                "data_catalog": {
                    "DATA": {"path": str(root), "signal_type": "meg",
                             **policy},
                },
            },
        }

    def raw(self) -> mne.io.BaseRaw:
        """Return four channels including an auxiliary channel typed EEG."""
        info = mne.create_info(
            ["MAG", "GRAD", "EEG", "AUX"], 256,
            ["mag", "grad", "eeg", "eeg"],
        )
        return mne.io.RawArray(np.ones((4, 512)), info, verbose=False)

    def test_fixed_names_and_configurable_types_are_combined(self):
        self.assertEqual(len(filter_channel(self.raw(), "DATA").ch_names), 4)
        with patch.dict("factory.channel_selection.EXCLUDE_DICT", {
            "DATA": ["AUX", "OPTIONAL"],
        }):
            selected = filter_channel(self.raw(), "DATA", {
                "exclude_channel_types": ["grad"],
            })
        self.assertEqual(selected.ch_names, ["MAG", "EEG"])
        with self.assertRaisesRegex(ValueError, "No EEG or MEG"):
            filter_channel(self.raw(), "DATA", {
                "exclude_channel_types": ["eeg", "mag", "grad"],
            })

    def test_original_name_exclusions_remain_active(self):
        info = mne.create_info(
            ["Cz2", "Cpz", "HEO", "VEO", "EKG", "EMG", "KEEP"],
            256, ["eeg"] * 7,
        )
        raw = mne.io.RawArray(np.ones((7, 512)), info, verbose=False)
        self.assertEqual(
            filter_channel(raw, "ds000117-1.0.6").ch_names, ["KEEP"],
        )

    def test_catalog_defaults_and_invalid_lists(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "catalog.yaml"
            entry = {"path": str(root), "signal_type": "both"}
            path.write_text(yaml.safe_dump({"datasets": {"DATA": entry}}))
            loaded = load_data_catalog(path)["DATA"]
            self.assertNotIn("exclude_channels", loaded)
            self.assertEqual(loaded["exclude_channel_types"], [])
            for policy in [
                {"exclude_channels": ["AUX"]},
                {"exclude_channel_types": "eeg"},
                {"exclude_channel_types": [""]},
                {"exclude_channel_types": ["eeg", "eeg"]},
                {"exclude_channel_types": ["typo"]},
                {"exclude_channel_types": None},
            ]:
                with self.subTest(policy=policy):
                    path.write_text(yaml.safe_dump({
                        "datasets": {"DATA": {**entry, **policy}},
                    }))
                    with self.assertRaises(ConfigError):
                        load_data_catalog(path)

    def test_discovery_passes_catalog_policy_to_recording(self):
        raw = self.raw()

        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary), {
                "exclude_channel_types": ["eeg"],
            })
            records = discover_catalog_recordings(
                RecordingAccessor(raw), config,
            )
            self.assertEqual(records[0]["signal_type"], "meg")
            self.assertEqual(records[0]["channel_selection"], {
                "exclude_channels": ["EKG", "EMG", "HEO", "VEO"],
                "exclude_channel_types": ["eeg"],
            })

    def test_discovery_excludes_names_after_renaming(self):
        raw = self.raw()

        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary), {})
            with patch("factory.utils.RENAME_DICT", {
                "DATA": {"EEG": "RENAMED"},
            }), patch.dict("factory.channel_selection.EXCLUDE_DICT", {
                "DATA": ["RENAMED", "AUX"],
            }):
                records = discover_catalog_recordings(
                    RecordingAccessor(raw), config,
                )
            self.assertEqual(records[0]["signal_type"], "meg")
            self.assertEqual(records[0]["eeg_channels"], 0)

    def test_cache_rejects_changed_or_missing_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            policy = channel_selection_provenance("DATA", {
                "exclude_channel_types": ["eeg"],
            })
            config = self.config(Path(temporary), policy)
            record = {"dataset": "DATA", "channel_selection": policy}
            validate_cached_channel_selection([record], config)
            for observed in [None, channel_selection_provenance("DATA", {})]:
                with self.assertRaisesRegex(ConfigError, "Cached channel"):
                    validate_cached_channel_selection([
                        {"dataset": "DATA", "channel_selection": observed},
                    ], config)

    def test_policy_affects_split_and_portable_training_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary), {})
            changed = deepcopy(config)
            changed["invocation"]["data_catalog"]["DATA"][
                "exclude_channel_types"
            ] = ["eeg"]
            self.assertNotEqual(
                metadata_directory(config), metadata_directory(changed),
            )
            manifest = {"sha256": "split"}
            first = _semantic_payload(config, {}, manifest, None)
            second = _semantic_payload(changed, {}, manifest, None)
            self.assertNotEqual(first, second)
            self.assertEqual(
                first["campaign"]["data"]["dataset_channel_selection"]["DATA"],
                {"exclude_channels": ["EKG", "EMG", "HEO", "VEO"],
                 "exclude_channel_types": []},
            )
            self.assertNotIn(temporary, str(second))


if __name__ == "__main__":
    unittest.main()

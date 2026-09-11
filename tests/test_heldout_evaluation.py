"""Protect held-out preprocessing boundaries and frozen-model evaluation.

Temporary recording trees and tensor metadata represent training, requested
held-out, and unrelated cached datasets. No external dataset files are used.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import tempfile
import unittest
import warnings

import mne
import numpy as np
import torch

from accessor import DataAccessor
from braintokenizer.evaluate import resolve_catalog_id
from braintokenizer.evaluation import (
    build_evaluation_loader,
    evaluate_tokenizer,
)
from braintokenizer.metrics import MetricsComputer
from braintokenizer.model import BrainTokenizer
from factory.utils import (
    filter_channel, split_pretrain_metadata, split_to_segments_save,
)
from pretrain_config import metadata_directory

SMALL_CONFIG = {
    "window_length": 32, "n_filters": 4, "ratios": [2, 2],
    "kernel_size": 5, "last_kernel_size": 5, "n_dim": 8,
    "n_neuro": 4, "n_head": 2, "dropout": 0.0, "codebook_dim": 8,
    "codebook_size": 8, "num_quantizers": 2, "rotation_trick": True,
    "quantize_optimize_method": "ema",
}


class HeldOutEvaluationTests(unittest.TestCase):
    """Keep requested held-out examples separate and metrics finite."""

    def test_raw_discovery_avoids_derivatives_storage_and_split_duplicates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = (
                "sub-01/meg/sub-01_split-01_meg.fif",
                "sub-01/meg/sub-01_split-02_meg.fif",
                "derivatives/sub-01/cleaned.fif",
                ".git/annex/objects/duplicate.fif",
                "sub-02/meg/sub-02_meg.ds/internal.fif",
            )
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            found = DataAccessor().search_brain_files(str(root), "MEG")
            self.assertEqual(
                {Path(row["path"]).relative_to(root).as_posix()
                 for row in found},
                {names[0], "sub-02/meg/sub-02_meg.ds"},
            )

    def test_annex_symlink_keeps_bids_source_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sub-01" / "meg" / "sub-01_meg.fif"
            source.parent.mkdir(parents=True)
            stored = root / ".git" / "annex" / "objects" / "content.fif"
            stored.parent.mkdir(parents=True)
            stored.touch()
            source.symlink_to(stored)
            mask = np.ones(2, dtype=bool)
            rows = split_to_segments_save(
                DataAccessor(read_only=False),
                np.arange(10, dtype=float).reshape(2, 5),
                np.zeros((2, 6)), np.ones(2, dtype=int),
                ~mask, mask, ~mask, mask, str(source), "HELD", str(root),
                str(root / "processed"), 2.5, 2.5, "meg", 2, 1, 1,
            )
            self.assertTrue(rows)
            for row in rows:
                self.assertEqual(
                    row["source_recording"],
                    source.relative_to(root).as_posix(),
                )
                self.assertNotIn(".git", row["path"])
                self.assertTrue(Path(row["path"]).is_file())

    def test_bids_discovery_requires_subject_recordings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "dataset_description.json").write_text("{}")
            valid = root / "sub-01" / "meg" / "sub-01_meg.ds"
            valid.mkdir(parents=True)
            stray = root / "meg" / "sub-01_meg.ds"
            stray.mkdir(parents=True)
            found = DataAccessor().search_brain_files(str(root), "MEG")
            self.assertEqual([row["path"] for row in found], [str(valid)])

    def test_catalog_can_exclude_auxiliary_eeg(self) -> None:
        info = mne.create_info(
            ["MEG", "EEG", "REF"], 256, ["mag", "eeg", "ref_meg"],
        )
        raw = mne.io.RawArray(np.ones((3, 10)), info, verbose=False)
        selected = filter_channel(
            raw, "DATA", {"exclude_channel_types": ["eeg"]},
        )
        self.assertEqual(selected.ch_names, ["MEG"])

    def test_heldout_selection_preserves_splits_and_ignores_unrequested_cache(
        self,
    ) -> None:
        training = [{"dataset": "TRAIN", "path": str(i)} for i in range(20)]
        extra = [{"dataset": name, "path": name}
                 for name in ("HELD", "UNREQUESTED")]
        ratios = {"train": 0.8, "validation": 0.1, "test": 0.1}
        random.seed(42)
        original = split_pretrain_metadata(training, ratios, ["TRAIN"], [])
        random.seed(42)
        extended = split_pretrain_metadata(
            training + extra, ratios, ["TRAIN"], ["HELD"],
        )
        self.assertEqual(original[:3], extended[:3])
        self.assertEqual(set(extended[3]), {"HELD"})
        with self.assertRaisesRegex(ValueError, "No preprocessed windows"):
            split_pretrain_metadata(training, ratios, ["TRAIN"], ["HELD"])
        with self.assertRaisesRegex(ValueError, "overlap"):
            split_pretrain_metadata(training, ratios, ["TRAIN"], ["TRAIN"])

    def test_heldout_changes_do_not_change_training_split_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = {
                "campaign": {
                    "seed": 42,
                    "data": {
                        "included_datasets": ["TRAIN"],
                        "split_ratios": {"train": 0.8},
                        "preprocessing": {
                            "sample_rate_hz": 256, "low_frequency_hz": 0.1,
                            "high_frequency_hz": 96, "segment_seconds": 10,
                            "stride_seconds": 10,
                        },
                    },
                },
                "invocation": {
                    "metadata_root": temporary,
                    "data_catalog": {"TRAIN": {"signal_type": "meg"}},
                    "held_out_evaluation_datasets": [],
                },
            }
            original = metadata_directory(config)
            config["invocation"]["data_catalog"]["HELD"] = {
                "signal_type": "meg",
            }
            config["invocation"]["held_out_evaluation_datasets"] = ["HELD"]
            self.assertEqual(original, metadata_directory(config))

    def test_evaluation_batches_cover_each_segment_once_across_ranks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                {"path": str(root / f"{i}.pt"), "dataset": "HELD",
                 "channels": channels}
                for i, channels in enumerate([3, 3, 3, 3, 3, 4, 4])
            ]
            metadata = root / "held.json"
            metadata.write_text(json.dumps(rows))
            indices = []
            for rank in range(2):
                loader = build_evaluation_loader(metadata, 2, 0, rank, 2)
                indices.extend(
                    i for batch in loader.batch_sampler for i in batch
                )
            self.assertEqual(sorted(indices), list(range(len(rows))))

    def test_reconstruction_metrics_weight_traces_and_omit_empty_modality(self):
        computer = MetricsComputer()
        for size, difference in ((2, 1.0), (1, 4.0)):
            target = torch.arange(8).float().reshape(1, 1, 1, 8)
            target = target.expand(size, 2, 1, 8)
            computer.step(
                target + difference, target,
                torch.ones((size, 2), dtype=torch.long),
            )
        with warnings.catch_warnings(record=True) as caught:
            metrics = computer.get_metrics()
        self.assertEqual(set(metrics), {"all", "meg"})
        self.assertAlmostEqual(metrics["all"]["mae"], 2.0)
        self.assertAlmostEqual(metrics["all"]["mse"], 6.0)
        self.assertTrue(caught)
        with self.assertRaises(ValueError):
            computer.step(
                torch.full((1, 2, 1, 8), float("nan")),
                torch.ones((1, 2, 1, 8)), torch.ones((1, 2)),
            )

    def test_validation_monitors_are_reproducible_and_leave_weights_unchanged(
        self,
    ) -> None:
        torch.set_num_threads(1)
        model = BrainTokenizer(**SMALL_CONFIG).eval()
        batch = {
            "x": torch.randn(1, 4, 64), "pos": torch.randn(1, 4, 6),
            "sensor_type": torch.ones((1, 4), dtype=torch.long),
        }
        # Initialize the disposable codebooks before verifying frozen inference.
        with torch.no_grad():
            model(**batch)
        before = {
            name: value.clone() for name, value in model.state_dict().items()
        }
        random_state = torch.random.get_rng_state()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            first = evaluate_tokenizer(model, [batch], 8, 42)
            second = evaluate_tokenizer(model, [batch], 8, 42)
            callback_batches = []

            def visualize(index, output):
                callback_batches.append(index)
                torch.rand(10)
                self.assertEqual(output["x"].shape[0], 1)

            with_visuals = evaluate_tokenizer(
                model, [batch], 8, 42,
                reconstruction_callback=visualize,
            )
        self.assertEqual(callback_batches, [0])
        self.assertEqual(first, with_visuals)
        self.assertEqual(first, second)
        torch.testing.assert_close(torch.random.get_rng_state(), random_state)
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, before[name])
        values = first["validation_monitors"]
        self.assertIn("latent_source/effective_rank", values)
        self.assertIn("latent_source/inter_query_attention_similarity", values)
        self.assertEqual(first["attention_sample_count"], 1)
        self.assertIn("reconstruction/mae/dropped", values)
        self.assertFalse(any("optimization" in name for name in values))
        self.assertFalse(any("exposure" in name for name in values))
        json.dumps(first, allow_nan=False)

    def test_dataset_alias_is_unique_and_keeps_source_identity(self):
        catalog = {"Gloups_MEG": {"signal_type": "meg"}}
        self.assertEqual(
            resolve_catalog_id("Gloups-MEG", catalog), "Gloups_MEG",
        )
        with self.assertRaises(ValueError):
            resolve_catalog_id("Groups-MEG", catalog)


if __name__ == "__main__":
    unittest.main()

"""Check provenance, plotting data, historical gaps, and epoch statistics.

All event files, campaign identities and rendered figures use disposable
absolute paths. No training campaigns or user artifacts are modified.
"""

from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import torch
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from factory.pretraining_monitor_events import (
    MonitorEvent,
    load_monitor_events,
)
from factory.pretraining_monitor_runtime import StageOneAccumulator
from factory.pretraining_visualisation import (
    build_figures,
    campaign_dimensions,
    read_provenance,
    visualize_pretraining,
)
from script.visualize_pretraining import parse_args


def scalar(tag: str, value: float, step: int = 1) -> MonitorEvent:
    """Build a canonical in-memory event for curve-selection assertions."""
    split, cadence, family, metric, *dimension = tag.split("/")
    return MonitorEvent(
        "fixture", "fixture", tag, tag, split, cadence, family, metric,
        "/".join(dimension), step, float(step), value,
    )


def write_events(directory: Path, values: list[tuple[str, float]]) -> None:
    """Write genuine scalar events with deterministic duplicate ordering."""
    writer = EventFileWriter(str(directory))
    try:
        for index, (tag, value) in enumerate(values):
            writer.add_event(Event(
                wall_time=float(index + 1), step=1,
                summary=Summary(value=[Summary.Value(
                    tag=tag, simple_value=value,
                )]),
            ))
    finally:
        writer.close()


class VisualisationTests(unittest.TestCase):
    """Exercise one-attempt provenance and the public plotting entry point."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.campaign = self.root / "campaign"
        self.directory = self.campaign / "attempts" / "attempt" / "tensorboard"
        self.directory.mkdir(parents=True)
        self.identity_path = self.campaign / "campaign_identity.json"
        self.identity = {
            "stage": "braintokenizer",
            "semantic_payload": {"campaign": {
                "stage": "braintokenizer",
                "data": {
                    "included_datasets": ["selected"],
                    "dataset_signal_types": {
                        "selected": "meg", "unused": "eeg",
                    },
                },
            }},
        }
        self.save_identity()
        (self.campaign / "model_cfg.json").write_text(
            json.dumps({"num_quantizers": 2})
        )

    def save_identity(self) -> None:
        self.identity_path.write_text(json.dumps(self.identity))

    def test_stage_validation_precedes_output(self) -> None:
        self.assertEqual(
            read_provenance(self.directory, "braintokenizer")["stage"],
            "braintokenizer",
        )
        with self.assertRaisesRegex(ValueError, "Requested stage.*brainomni"):
            visualize_pretraining(self.directory, "brainomni")
        self.identity["semantic_payload"]["campaign"]["stage"] = "brainomni"
        self.save_identity()
        with self.assertRaisesRegex(ValueError, "Inconsistent stage"):
            visualize_pretraining(self.directory, "braintokenizer")
        self.identity["stage"] = "brainomni"
        self.save_identity()
        with self.assertRaisesRegex(ValueError, "supports braintokenizer"):
            visualize_pretraining(self.directory, "brainomni")
        with self.assertRaisesRegex(ValueError, "supports braintokenizer"):
            visualize_pretraining(self.directory)
        self.identity_path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing or malformed"):
            visualize_pretraining(self.directory, "braintokenizer")
        self.assertFalse((self.directory.parent / "visualisation").exists())

    def test_optional_stage_cli(self) -> None:
        arguments = [
            "visualize_pretraining", "--tensorboard-dir", str(self.directory),
        ]
        with patch("sys.argv", arguments):
            self.assertIsNone(parse_args().stage)
        with patch("sys.argv", arguments + ["--stage", "braintokenizer"]):
            self.assertEqual(parse_args().stage, "braintokenizer")
        self.assertEqual(
            read_provenance(self.directory)["stage"], "braintokenizer",
        )

    def test_empty_and_nested_inputs(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "No TensorBoard"):
            visualize_pretraining(self.directory, "braintokenizer")
        write_events(self.directory / "other", [("train_loss", 1.0)])
        with self.assertRaisesRegex(ValueError, "nested runs"):
            visualize_pretraining(self.directory, "braintokenizer")
        with self.assertRaisesRegex(ValueError, "absolute"):
            visualize_pretraining("relative", "braintokenizer")

    def test_selected_modalities_and_level_count(self) -> None:
        joint, levels = campaign_dimensions(self.directory, self.identity)
        self.assertFalse(joint)
        self.assertEqual(levels, ["level_00", "level_01"])
        data = self.identity["semantic_payload"]["campaign"]["data"]
        data["dataset_signal_types"]["selected"] = "emeg"
        self.assertTrue(campaign_dimensions(self.directory, self.identity)[0])
        (self.campaign / "model_cfg.json").write_text(
            json.dumps({"num_quantizers": 0})
        )
        with self.assertRaisesRegex(ValueError, "positive RVQ"):
            campaign_dimensions(self.directory, self.identity)

    def test_loss_weights_cadence_and_independent_coordinates(self) -> None:
        events = [
            scalar("train/step/objective/optimized_loss", 10.0, 100),
            scalar("train/step/reconstruction/phase_loss", 4.0, 100),
            scalar("train/epoch/reconstruction/time_loss", 7.0, 2),
            scalar("validation/epoch/reconstruction/mae", 3.0, 1),
        ]
        figures = build_figures(events, False, ["level_00"])
        loss = figures[0]
        self.assertEqual(loss.xlabel, "Optimizer step")
        self.assertEqual([curve.values for curve in loss.curves], [[10], [2]])
        self.assertTrue(all("/step/" in curve.tag for curve in loss.curves))
        mae = next(f for f in figures if f.name == "reconstruction/all/mae")
        self.assertEqual([curve.steps for curve in mae.curves], [[2], [1]])
        self.assertIn("equivalent", mae.curves[0].transformation)
        self.assertFalse(any("/eeg/" in f.name for f in figures))
        epoch = build_figures(
            [scalar("train/epoch/objective/optimized_loss", 1)],
            False, ["level_00"],
        )
        self.assertEqual(epoch[0].xlabel, "Epoch")

    def test_dropped_visible_comparison(self) -> None:
        events = [
            scalar(f"{split}/epoch/reconstruction/{metric}/{stratum}", 0.5)
            for split in ("train", "validation")
            for stratum in ("dropped", "visible")
            for metric in ("pcc", "mae", "mse")
        ]
        figures = build_figures(events, False, ["level_00"])
        comparisons = [
            f for f in figures if "/dropped_vs_visible/" in f.name
        ]
        self.assertEqual(len(comparisons), 3)
        for figure in comparisons:
            self.assertEqual(len(figure.curves), 4)
            self.assertFalse(figure.missing)
            dropped_train, dropped_val, visible_train, visible_val = (
                figure.curves
            )
            self.assertEqual(dropped_train.colour, dropped_val.colour)
            self.assertEqual(visible_train.colour, visible_val.colour)
            self.assertNotEqual(dropped_train.colour, visible_train.colour)
            self.assertEqual(dropped_train.style, visible_train.style)
            self.assertEqual(dropped_val.style, visible_val.style)
            self.assertNotEqual(dropped_train.style, dropped_val.style)
        self.assertFalse(any(
            "/dropped/" in f.name or "/visible/" in f.name for f in figures
        ))

    def test_levels_missing_series_and_nonfinite_values(self) -> None:
        events = [
            scalar("train/epoch/rvq/assignment_utilization/level_01", 0.4),
            scalar("train/epoch/rvq/assignment_utilization/level_00", 0.6),
            scalar(
                "validation/epoch/rvq/assignment_entropy_normalized/level_00",
                0.9,
            ),
        ]
        figures = build_figures(events, True, ["level_00", "level_01"])
        usage = next(f for f in figures if f.name.endswith("utilization"))
        self.assertEqual(
            [c.label for c in usage.curves], ["Level 0", "Level 1"],
        )
        perplexity = next(f for f in figures if "perplexity" in f.name)
        self.assertFalse(perplexity.curves)
        self.assertEqual(len(perplexity.missing), 2)
        self.assertTrue(any("/eeg/" in f.name for f in figures))
        for value in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "Nonfinite.*step 1"):
                build_figures(
                    [scalar("train/epoch/objective/optimized_loss", value)],
                    False, ["level_00"],
                )
        with self.assertRaisesRegex(ValueError, "absent from campaign"):
            build_figures(events, False, ["level_00"])
        with self.assertRaisesRegex(ValueError, "No plottable"):
            build_figures([], False, ["level_00"])

    def test_render_duplicate_resolution_and_safe_rerun(self) -> None:
        write_events(self.directory, [
            ("train_judge_loss", 100.0),
            ("train_loss", 2.0),
            ("train_loss", 3.0),
        ])
        events = load_monitor_events([self.directory])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].value, 3.0)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            manifest_path = visualize_pretraining(
                self.directory,
            )
        self.assertTrue(any(
            "missing curves" in str(w.message) for w in recorded
        ))
        self.assertEqual(manifest_path.parent.name, "visualisation")
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["stage"], "braintokenizer")
        self.assertEqual(len(manifest["generated_files"]), 2)
        for name in manifest["generated_files"]:
            self.assertGreater((manifest_path.parent / name).stat().st_size, 0)
        png = manifest_path.parent / "losses/training_epoch.png"
        self.assertTrue(png.read_bytes().startswith(b"\x89PNG"))
        stale = manifest_path.parent / "losses/stale.png"
        stale.write_bytes(b"owned")
        unrelated = manifest_path.parent / "notes.txt"
        unrelated.write_text("retain")
        manifest["generated_files"].append("losses/stale.png")
        manifest_path.write_text(json.dumps(manifest))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            visualize_pretraining(self.directory, "braintokenizer")
        self.assertFalse(stale.exists())
        self.assertEqual(unrelated.read_text(), "retain")
        manifest = json.loads(manifest_path.read_text())
        manifest["generated_files"].append("../outside.png")
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Unsafe generated"):
            visualize_pretraining(self.directory, "braintokenizer")

    def test_nonfinite_and_bad_output_do_not_write(self) -> None:
        write_events(self.directory, [("train_loss", float("nan"))])
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            visualize_pretraining(self.directory, "braintokenizer")
        self.assertFalse((self.directory.parent / "visualisation").exists())
        finite = [scalar("train/epoch/objective/optimized_loss", 1.0)]
        with patch(
            "factory.pretraining_visualisation.load_monitor_events",
            return_value=finite,
        ), self.assertRaisesRegex(ValueError, "disjoint"):
            visualize_pretraining(
                self.directory, "braintokenizer", self.directory,
            )


class ReconstructionEpochTests(unittest.TestCase):
    """Check finite count-weighted reconstruction values and empty strata."""

    @staticmethod
    def add_batch(
        accumulator: StageOneAccumulator,
        batch_size: int,
        error: float,
        dropped: bool = True,
        mixed: bool = True,
    ) -> None:
        """Accumulate unequal batches with known per-element errors."""
        target = torch.tensor([[[[1., 2., 3.]], [[3., 2., 1.]]]])
        target = target.repeat(batch_size, 1, 1, 1)
        output = {
            name: torch.tensor(1.0) for name in (
                "loss", "time_loss", "pcc", "amp_loss", "phase_loss",
                "commitment_loss",
            )
        }
        monitor = {
            "target": target,
            "reconstruction": target + error,
            "dropped_channel_mask": torch.tensor(
                [[dropped, False]]
            ).repeat(batch_size, 1),
            "sensor_type": torch.tensor(
                [[0, 1 if mixed else 0]]
            ).repeat(batch_size, 1),
            "source_latent": torch.tensor(
                [[[[[1., 2.]]], [[[2., 1.]]]]]
            ).repeat(batch_size, 1, 1, 1, 1),
            "indices": torch.zeros(batch_size, 2, 1, 1, 2).long(),
            "quantization_error_sum": torch.tensor([2., 4.]),
            "quantization_count": torch.tensor([2., 2.]),
            "residual_input_energy_sum": torch.tensor([4., 2.]),
            "residual_output_energy_sum": torch.tensor([2., 1.]),
        }
        accumulator.update(output, monitor)

    def test_both_splits_count_weighted_metrics_and_aliases(self) -> None:
        accumulator = StageOneAccumulator(codebook_size=2)
        self.add_batch(accumulator, 1, 1.0)
        self.add_batch(accumulator, 2, 3.0)
        for split, values in (
            ("train", accumulator.training_values("epoch")),
            ("validation", accumulator.validation_values()),
        ):
            for stratum in ("", "/dropped", "/visible", "/eeg", "/meg"):
                base = f"{split}/epoch/reconstruction/"
                self.assertAlmostEqual(values[base + "mae" + stratum], 7 / 3)
                self.assertAlmostEqual(values[base + "mse" + stratum], 19 / 3)
                self.assertAlmostEqual(values[base + "pcc" + stratum], 1.0)
                if stratum:
                    self.assertEqual(
                        values[base + "time_loss" + stratum],
                        values[base + "mae" + stratum],
                    )
        steps = accumulator.training_values("step")
        self.assertFalse(any("/mae" in tag or "/mse" in tag for tag in steps))

    def test_empty_strata_and_pure_modality(self) -> None:
        accumulator = StageOneAccumulator(codebook_size=2)
        self.add_batch(accumulator, 1, 1.0, dropped=False, mixed=False)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            values = accumulator.training_values("epoch")
        self.assertTrue(any("No train reconstruction elements" in str(w.message)
                            for w in recorded))
        self.assertFalse(any(tag.endswith("/dropped") for tag in values))
        self.assertFalse(any(tag.endswith(("/eeg", "/meg")) for tag in values))
        self.assertIn("train/epoch/reconstruction/mse/visible", values)

    def test_undefined_pcc_and_nonfinite_statistics(self) -> None:
        accumulator = StageOneAccumulator(codebook_size=2)
        self.add_batch(accumulator, 1, 1.0)
        accumulator.sums.values["all_pcc_count"].zero_()
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            values = accumulator.validation_values()
        self.assertTrue(any("No valid validation all traces" in str(w.message)
                            for w in recorded))
        self.assertNotIn("validation/epoch/reconstruction/pcc", values)
        self.assertIn("validation/epoch/reconstruction/mae", values)
        accumulator.sums.values["all_squared_sum"].fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "finite"):
            accumulator.training_values("epoch")


if __name__ == "__main__":
    unittest.main()

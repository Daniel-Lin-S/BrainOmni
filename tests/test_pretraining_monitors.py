"""Focused tests for pre-training monitor mathematics and event schemas."""

from __future__ import annotations

import ast
import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from factory.pretraining_monitor_events import (
    MonitorEvent,
    canonicalize_tag,
    discover_event_files,
    load_monitor_events,
    parse_canonical_tag,
    write_monitor_csv,
)
from factory.pretraining_monitor_runtime import (
    StageOneAccumulator,
    StageTwoAccumulator,
)
from factory.pretraining_monitors import (
    TensorSums,
    assignment_metrics,
    attention_similarity,
    canonical_tag,
    covariance_from_sums,
    distributed_update_to_weight_ratio,
    effective_rank,
    mean_absolute_off_diagonal_correlation,
    modality_channel_groups,
    monitor_due,
    optimizer_step_values,
    reconstruction_statistics,
    residual_energy_reduction,
    source_statistics,
    source_variance,
    successful_optimizer_steps,
    update_to_weight_sums,
    zero_partition_snapshot,
)
from pretrain_dataset import build_fixed_monitor_batch


class FakeAccessor:
    """Return deterministic in-memory processed tensors by absolute path."""

    def __init__(self, payloads: dict[str, dict[str, torch.Tensor]]) -> None:
        self.payloads = payloads

    def read(self, path, loader):
        del loader
        return dict(self.payloads[path])


class FakeZeroOptimizer:
    """Expose the FP32 ZeRO partition interface used by monitoring."""

    def __init__(self, partitions: list[torch.Tensor]) -> None:
        self.single_partition_of_fp32_groups = partitions


class FakeEngine:
    """Expose DeepSpeed counters and optimizer partitions."""

    def __init__(self, partitions: list[torch.Tensor]) -> None:
        self.global_steps = 8
        self.skipped_steps = 2
        self.optimizer = FakeZeroOptimizer(partitions)

    def get_global_grad_norm(self) -> torch.Tensor:
        return torch.tensor(5.0)


class PretrainingMonitorTest(unittest.TestCase):
    """Validate monitor reductions against hand-computed quantities."""

    def test_canonical_and_legacy_tags(self) -> None:
        tag = canonical_tag(
            "validation",
            "epoch",
            "reconstruction",
            "pcc",
        )
        self.assertEqual(tag, "validation/epoch/reconstruction/pcc")
        self.assertEqual(
            parse_canonical_tag(tag),
            ("validation", "epoch", "reconstruction", "pcc", ""),
        )
        self.assertEqual(
            canonicalize_tag("train_acc_2"),
            ("train/epoch/masked_token/accuracy/level_02", 0),
        )
        self.assertEqual(
            canonicalize_tag("eval_codebook_utilize_entropy_1"),
            (
                (
                    "validation/epoch/rvq/"
                    "assignment_entropy_normalized/level_01"
                ),
                0,
            ),
        )
        self.assertEqual(canonicalize_tag("train_judge_loss")[1], 1)
        self.assertEqual(
            canonicalize_tag("train_judge_loss")[0],
            canonicalize_tag("train_loss")[0],
        )

    def test_monitor_payloads_are_opt_in_at_model_boundaries(self) -> None:
        root = Path(__file__).resolve().parents[1]
        expectations = {
            root / "braintokenizer/model.py": "(output, indices)",
            root / "brainomni/model.py": "output_dict",
        }
        for path, expected_return in expectations.items():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            forward = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
                and node.name == "forward"
                and any(
                    argument.arg == "return_monitor_data"
                    for argument in node.args.args
                )
            )
            arguments = [argument.arg for argument in forward.args.args]
            default_offset = len(arguments) - len(forward.args.defaults)
            default_index = arguments.index("return_monitor_data")
            default = forward.args.defaults[default_index - default_offset]
            self.assertIsInstance(default, ast.Constant)
            self.assertIs(default.value, False)
            final_return = next(
                statement
                for statement in reversed(forward.body)
                if isinstance(statement, ast.Return)
            )
            self.assertEqual(ast.unparse(final_return.value), expected_return)

    def test_assignment_and_residual_metrics(self) -> None:
        counts = torch.tensor([[2.0, 2.0, 0.0, 0.0]])
        metrics = assignment_metrics(counts)
        self.assertAlmostEqual(metrics["utilization"].item(), 0.5)
        self.assertAlmostEqual(metrics["perplexity"].item(), 2.0)
        self.assertAlmostEqual(
            metrics["perplexity_normalized"].item(),
            0.5,
        )
        reduction = residual_energy_reduction(
            torch.tensor([4.0, 2.0]),
            torch.tensor([1.0, 1.0]),
        )
        torch.testing.assert_close(reduction, torch.tensor([0.75, 0.5]))

    def test_source_covariance_rank_and_correlation(self) -> None:
        source = torch.tensor(
            [[[[[1.0, 2.0]]], [[[2.0, 1.0]]]]]
        )
        statistics = source_statistics(source)
        variance = source_variance(
            statistics["source_count"],
            statistics["source_sum"],
            statistics["source_square_sum"],
        )
        torch.testing.assert_close(
            variance,
            torch.tensor([0.25, 0.25], dtype=torch.float64),
        )
        covariance = covariance_from_sums(
            statistics["source_count"],
            statistics["source_sum"],
            statistics["source_cross_product"],
        )
        self.assertAlmostEqual(
            mean_absolute_off_diagonal_correlation(covariance).item(),
            1.0,
        )
        self.assertAlmostEqual(effective_rank(covariance).item(), 1.0)

    def test_attention_similarity(self) -> None:
        attention = torch.tensor(
            [[[[[1.0, 0.0], [0.0, 1.0]]]]]
        )
        self.assertAlmostEqual(attention_similarity(attention).item(), 0.0)

    def test_reconstruction_mixed_emeg_strata(self) -> None:
        target = torch.tensor(
            [[[[1.0, 2.0, 3.0]], [[3.0, 2.0, 1.0]]]]
        )
        reconstruction = target + 1.0
        masks = {
            "eeg": torch.tensor([[True, False]]),
            "meg": torch.tensor([[False, True]]),
        }
        statistics = reconstruction_statistics(
            reconstruction,
            target,
            masks,
        )
        self.assertEqual(statistics["eeg_element_count"].item(), 3)
        self.assertEqual(statistics["meg_element_count"].item(), 3)
        self.assertEqual(statistics["eeg_absolute_sum"].item(), 3)
        self.assertAlmostEqual(statistics["meg_pcc_sum"].item(), 1.0)

    def test_modality_groups_cover_pure_and_mixed_emeg(self) -> None:
        sensor_type = torch.tensor(
            [
                [0, 1, 2],
                [0, 0, 0],
                [1, 2, 1],
            ]
        )
        groups = modality_channel_groups(sensor_type)
        eeg_samples = torch.cat(
            [sample_indices for sample_indices, _ in groups["eeg"]]
        )
        meg_samples = torch.cat(
            [sample_indices for sample_indices, _ in groups["meg"]]
        )
        self.assertEqual(set(eeg_samples.tolist()), {0, 1})
        self.assertEqual(set(meg_samples.tolist()), {0, 2})
        mixed_eeg_mask = next(
            mask
            for sample_indices, mask in groups["eeg"]
            if 0 in sample_indices.tolist()
        )
        mixed_meg_mask = next(
            mask
            for sample_indices, mask in groups["meg"]
            if 0 in sample_indices.tolist()
        )
        torch.testing.assert_close(
            mixed_eeg_mask,
            torch.tensor([True, False, False]),
        )
        torch.testing.assert_close(
            mixed_meg_mask,
            torch.tensor([False, True, True]),
        )

    def test_stage_one_accumulator(self) -> None:
        target = torch.tensor(
            [[[[1.0, 2.0, 3.0]], [[3.0, 2.0, 1.0]]]]
        )
        output = {
            "loss": torch.tensor(5.0),
            "time_loss": torch.tensor(1.0),
            "pcc": torch.tensor(0.5),
            "amp_loss": torch.tensor(1.0),
            "phase_loss": torch.tensor(1.0),
            "commitment_loss": torch.tensor(0.5),
        }
        monitor = {
            "target": target,
            "reconstruction": target + 1.0,
            "dropped_channel_mask": torch.tensor([[True, False]]),
            "sensor_type": torch.tensor([[0, 1]]),
            "source_latent": torch.tensor(
                [[[[[1.0, 2.0]]], [[[2.0, 1.0]]]]]
            ),
            "indices": torch.tensor([[[[[0, 1]]], [[[1, 1]]]]]),
            "quantization_error_sum": torch.tensor([2.0, 4.0]),
            "quantization_count": torch.tensor([2.0, 2.0]),
            "residual_input_energy_sum": torch.tensor([4.0, 2.0]),
            "residual_output_energy_sum": torch.tensor([2.0, 1.0]),
        }
        accumulator = StageOneAccumulator(codebook_size=2)
        accumulator.update(output, monitor)
        values = accumulator.training_values("epoch")
        self.assertIn(
            "train/epoch/rvq/assignment_utilization/level_00",
            values,
        )
        self.assertAlmostEqual(
            values[
                "train/epoch/rvq/quantization_error/level_01"
            ].item(),
            2.0,
        )
        validation = accumulator.validation_values()
        self.assertIn(
            "validation/epoch/reconstruction/time_loss/eeg",
            validation,
        )
        self.assertIn(
            "validation/epoch/reconstruction/time_loss/dropped",
            validation,
        )
        self.assertIn(
            "validation/epoch/reconstruction/pcc/visible",
            validation,
        )
        for sensor_type in (
            torch.tensor([[0, 0]]),
            torch.tensor([[1, 2]]),
        ):
            pure_monitor = dict(monitor)
            pure_monitor["sensor_type"] = sensor_type
            pure = StageOneAccumulator(codebook_size=2)
            pure.update(output, pure_monitor)
            pure_values = pure.validation_values()
            self.assertFalse(
                any(tag.endswith("/eeg") for tag in pure_values)
            )
            self.assertFalse(
                any(tag.endswith("/meg") for tag in pure_values)
            )

    def test_stage_two_accumulator(self) -> None:
        output = {"loss": torch.tensor(1.5)}
        monitor = {
            "cross_entropy_sum": torch.tensor([2.0, 4.0]),
            "correct_sum": torch.tensor([1.0, 2.0]),
            "masked_count": torch.tensor(2.0),
            "label_counts": torch.tensor(
                [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]
            ),
            "corruption_cross_entropy_sum": torch.tensor(
                [[1.0, 2.0], [1.0, 2.0]]
            ),
            "corruption_count": torch.tensor([1.0, 1.0]),
            "source_cross_entropy_sum": torch.tensor(
                [[1.0, 1.0], [2.0, 2.0]]
            ),
            "source_count": torch.tensor([1.0, 1.0]),
        }
        accumulator = StageTwoAccumulator()
        accumulator.update(output, monitor)
        accumulator.add_modality("eeg", monitor)
        accumulator.add_modality("meg", monitor)
        values = accumulator.validation_values()
        self.assertAlmostEqual(
            values[
                "validation/epoch/masked_token/"
                "cross_entropy/total"
            ].item(),
            3.0,
        )
        self.assertIn(
            "validation/epoch/masked_token/"
            "cross_entropy/level_00_random_token",
            values,
        )
        self.assertIn(
            "validation/epoch/masked_token/cross_entropy/level_01_meg",
            values,
        )
        for modality in ("eeg", "meg"):
            pure = StageTwoAccumulator()
            pure.update(output, monitor)
            pure.add_modality(modality, monitor)
            pure_values = pure.validation_values()
            self.assertFalse(
                any(
                    tag.endswith(("/eeg", "/meg"))
                    for tag in pure_values
                )
            )

    def test_sufficient_statistics_use_sum_reduction(self) -> None:
        statistics = TensorSums()
        statistics.add("sum", torch.tensor(3.0))
        statistics.add("count", torch.tensor(2.0))

        def add_remote_rank(value: torch.Tensor) -> None:
            value.add_(value)

        statistics.reduce_(add_remote_rank)
        self.assertEqual(statistics.require("sum").item(), 6.0)
        self.assertEqual(statistics.require("count").item(), 4.0)

    def test_optimizer_step_and_update_ratio(self) -> None:
        engine = FakeEngine([torch.tensor([3.0, 4.0])])
        self.assertEqual(successful_optimizer_steps(engine), 6)
        self.assertFalse(
            monitor_due(
                engine,
                interval=7,
                accumulation_boundary=False,
            )
        )
        self.assertTrue(
            monitor_due(
                engine,
                interval=7,
                accumulation_boundary=True,
            )
        )
        before = zero_partition_snapshot(engine)
        engine.optimizer.single_partition_of_fp32_groups[0].add_(1.0)
        update_sum, weight_sum = update_to_weight_sums(before, engine)
        self.assertEqual(update_sum.item(), 2.0)
        self.assertEqual(weight_sum.item(), 25.0)
        ratio = distributed_update_to_weight_ratio(
            before,
            engine,
            lambda value: None,
            torch.device("cpu"),
        )
        self.assertAlmostEqual(ratio.item(), 2.0**0.5 / 5.0)
        values = optimizer_step_values(
            engine,
            [0.1, 0.2],
            ("main", "no_decay"),
        )
        self.assertEqual(
            values[
                "train/step/optimization/learning_rate/no_decay"
            ],
            0.2,
        )
        self.assertEqual(
            values[
                "train/step/optimization/gradient_norm/global"
            ].item(),
            5.0,
        )

    def test_cadence_handles_skips_and_resume(self) -> None:
        engine = FakeEngine([torch.tensor([1.0])])
        engine.global_steps = 103
        engine.skipped_steps = 4
        self.assertTrue(monitor_due(engine, 100, True))
        engine.global_steps = 104
        engine.skipped_steps = 5
        self.assertTrue(monitor_due(engine, 100, True))
        engine.global_steps = 104
        engine.skipped_steps = 4
        self.assertFalse(monitor_due(engine, 100, True))

    def test_fixed_monitor_batch_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            paths = [str(root / f"sample_{index}.pt") for index in range(4)]
            metadata = [
                {"path": path, "channels": 2}
                for path in reversed(paths)
            ]
            (root / "val.json").write_text(json.dumps(metadata))
            payloads = {
                path: {
                    "x": torch.full((2, 4), index, dtype=torch.float32),
                    "pos": torch.zeros(2, 6),
                    "sensor_type": torch.zeros(2, dtype=torch.long),
                }
                for index, path in enumerate(paths)
            }
            accessor = FakeAccessor(payloads)
            first = build_fixed_monitor_batch(
                str(root),
                accessor,
                rank=0,
                world_size=2,
                batch_size=2,
            )
            second = build_fixed_monitor_batch(
                str(root),
                accessor,
                rank=0,
                world_size=2,
                batch_size=2,
            )
            other_rank = build_fixed_monitor_batch(
                str(root),
                accessor,
                rank=1,
                world_size=2,
                batch_size=2,
            )
            self.assertEqual(first["path"], second["path"])
            self.assertNotEqual(first["path"], other_rank["path"])
            torch.testing.assert_close(first["x"], second["x"])

    def test_event_discovery_and_csv_export(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be absolute"):
            discover_event_files(["relative/events"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            event_path = root / "events.out.tfevents.test"
            event_path.write_bytes(b"placeholder")
            self.assertEqual(discover_event_files([root]), [event_path])
            event = MonitorEvent(
                run=str(root),
                source_file=str(event_path),
                original_tag="train_loss",
                tag="train/epoch/objective/optimized_loss",
                split="train",
                cadence="epoch",
                family="objective",
                metric="optimized_loss",
                dimension="",
                step=1,
                wall_time=2.0,
                value=3.0,
            )
            output = root / "monitors.csv"
            write_monitor_csv([event], output)
            with output.open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["tag"], event.tag)

    def test_synthetic_events_are_normalized_and_deduplicated(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ModuleNotFoundError:
            self.skipTest("The pinned TensorBoard dependency is unavailable.")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            writer = SummaryWriter(str(root / "nested"))
            writer.add_scalar("train_judge_loss", 9.0, 4)
            writer.add_scalar("train_loss", 3.0, 4)
            writer.add_scalar(
                "validation/epoch/reconstruction/pcc",
                0.75,
                2,
            )
            writer.close()
            events = load_monitor_events([root])
        optimized = [
            event
            for event in events
            if event.tag == "train/epoch/objective/optimized_loss"
        ]
        self.assertEqual(len(optimized), 1)
        self.assertEqual(optimized[0].original_tag, "train_loss")
        self.assertAlmostEqual(optimized[0].value, 3.0)
        self.assertTrue(
            any(
                event.tag == "validation/epoch/reconstruction/pcc"
                for event in events
            )
        )


if __name__ == "__main__":
    unittest.main()

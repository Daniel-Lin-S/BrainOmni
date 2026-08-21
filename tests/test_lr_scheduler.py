"""Regression tests for multi-group pre-training LR initialization."""

from __future__ import annotations

import unittest
from unittest import mock

import torch
from deepspeed.runtime.lr_schedules import WarmupCosineLR

from factory.lr_scheduler import warmup_cosine_scheduler_factory


BASE_LEARNING_RATES = (2e-4, 3e-4)
TOTAL_STEPS = 100
WARMUP_STEPS = 10
COSINE_MIN_RATIO = 0.05


def build_optimizer() -> torch.optim.AdamW:
    """Return a two-group optimizer with distinct base learning rates."""
    first = torch.nn.Parameter(torch.tensor(1.0))
    second = torch.nn.Parameter(torch.tensor(2.0))
    return torch.optim.AdamW(
        [
            {"params": [first], "lr": BASE_LEARNING_RATES[0]},
            {"params": [second], "lr": BASE_LEARNING_RATES[1]},
        ]
    )


def build_scheduler(
    optimizer: torch.optim.AdamW,
) -> WarmupCosineLR:
    """Return the configured scheduler used by the focused tests."""
    factory = warmup_cosine_scheduler_factory(
        total_num_steps=TOTAL_STEPS,
        warmup_ratio=WARMUP_STEPS / TOTAL_STEPS,
        warmup_min_lr_ratio=0.0,
        cosine_min_ratio=COSINE_MIN_RATIO,
    )
    return factory(optimizer)


class LearningRateSchedulerTest(unittest.TestCase):
    """Protect startup, progression, and recovery behavior."""

    def test_all_groups_start_at_zero_without_warning(self) -> None:
        """Initialize every optimizer group at the same zero ratio."""
        optimizer = build_optimizer()
        with mock.patch(
            "deepspeed.runtime.lr_schedules.logger.warning"
        ) as warning:
            scheduler = build_scheduler(optimizer)
        warning.assert_not_called()
        self.assertEqual(scheduler.last_batch_iteration, 0)
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [0.0, 0.0],
        )

    def test_factory_rejects_invalid_schedule_boundaries(self) -> None:
        """Reject schedules that cannot represent a training campaign."""
        with self.assertRaisesRegex(ValueError, "at least one"):
            warmup_cosine_scheduler_factory(
                total_num_steps=0,
                warmup_ratio=0.1,
                warmup_min_lr_ratio=0.0,
                cosine_min_ratio=COSINE_MIN_RATIO,
            )
        with self.assertRaisesRegex(ValueError, "warmup_min_lr_ratio"):
            warmup_cosine_scheduler_factory(
                total_num_steps=TOTAL_STEPS,
                warmup_ratio=0.1,
                warmup_min_lr_ratio=-0.1,
                cosine_min_ratio=COSINE_MIN_RATIO,
            )

    def test_first_update_uses_initialized_rates(self) -> None:
        """Keep parameters fixed while the initial LR ratio is zero."""
        optimizer = build_optimizer()
        build_scheduler(optimizer)
        before = [
            group["params"][0].detach().clone()
            for group in optimizer.param_groups
        ]
        for group in optimizer.param_groups:
            group["params"][0].grad = torch.ones_like(group["params"][0])
        optimizer.step()
        for expected, group in zip(
            before,
            optimizer.param_groups,
            strict=True,
        ):
            torch.testing.assert_close(expected, group["params"][0])

    def test_progression_preserves_group_proportions(self) -> None:
        """Scale every base LR by the same warmup and cosine ratio."""
        optimizer = build_optimizer()
        scheduler = build_scheduler(optimizer)
        scheduler.step()
        rates = [group["lr"] for group in optimizer.param_groups]
        self.assertGreater(rates[0], 0.0)
        self.assertAlmostEqual(
            rates[1] / rates[0],
            BASE_LEARNING_RATES[1] / BASE_LEARNING_RATES[0],
        )
        for _ in range(TOTAL_STEPS - 2):
            scheduler.step()
        self.assertAlmostEqual(
            optimizer.param_groups[0]["lr"],
            BASE_LEARNING_RATES[0] * COSINE_MIN_RATIO,
        )

    def test_optimizer_and_scheduler_state_round_trip(self) -> None:
        """Continue with identical rates after checkpoint restoration."""
        optimizer = build_optimizer()
        scheduler = build_scheduler(optimizer)
        for _ in range(7):
            scheduler.step()
        optimizer_state = optimizer.state_dict()
        scheduler_state = scheduler.state_dict()

        restored_optimizer = build_optimizer()
        restored_scheduler = build_scheduler(restored_optimizer)
        restored_optimizer.load_state_dict(optimizer_state)
        restored_scheduler.load_state_dict(scheduler_state)
        scheduler.step()
        restored_scheduler.step()
        self.assertEqual(
            [group["lr"] for group in restored_optimizer.param_groups],
            [group["lr"] for group in optimizer.param_groups],
        )


if __name__ == "__main__":
    unittest.main()

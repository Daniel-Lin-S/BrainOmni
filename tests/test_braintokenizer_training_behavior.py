"""Regression tests for BrainTokenizer training semantics and compatibility."""

from __future__ import annotations

import warnings
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn.utils import weight_norm as legacy_weight_norm

from brainomni.model import BrainOmni
from braintokenizer.model import BrainTokenizer
from factory.campaign import tensor_state_sha256
from factory.training_runtime import destroy_distributed_process_group
from model_utils.conv import (
    LEGACY_WEIGHT_NORM_DIRECTION_KEY,
    LEGACY_WEIGHT_NORM_MAGNITUDE_KEY,
    WEIGHT_NORM_DIRECTION_KEY,
    WEIGHT_NORM_MAGNITUDE_KEY,
    apply_parametrization_norm,
    legacy_weight_norm_state_dict,
)
from model_utils.module import BackWardSolution


SMALL_TOKENIZER_CONFIG = {
    "window_length": 32,
    "n_filters": 4,
    "ratios": [2, 2],
    "kernel_size": 5,
    "last_kernel_size": 5,
    "n_dim": 8,
    "n_neuro": 4,
    "n_head": 2,
    "dropout": 0.0,
    "codebook_dim": 8,
    "codebook_size": 16,
    "num_quantizers": 2,
    "rotation_trick": True,
    "quantize_optimize_method": "ema",
}


class BrainTokenizerTrainingBehaviorTest(unittest.TestCase):
    """Protect Stage-1 training behavior and released checkpoint schemas."""

    def _model_shell(
        self,
        noise_std: float = 0.1,
        mask_ratio: float = 0.25,
    ) -> BrainTokenizer:
        """Create a minimal model shell for stateless behavior tests."""
        model = BrainTokenizer.__new__(BrainTokenizer)
        nn.Module.__init__(model)
        model.noise_std = noise_std
        model.mask_ratio = mask_ratio
        return model

    def test_additive_noise_is_training_only(self) -> None:
        """Apply configured noise in training without perturbing evaluation."""
        inputs = torch.ones(2, 3, 4, 5)
        model = self._model_shell(noise_std=0.1)
        model.train()
        torch.manual_seed(7)
        noisy = model.add_noise(inputs)
        self.assertFalse(torch.equal(noisy, inputs))
        model.eval()
        torch.testing.assert_close(model.add_noise(inputs), inputs)

        model.train()
        model.noise_std = 0.0
        state_before = torch.random.get_rng_state()
        torch.testing.assert_close(model.add_noise(inputs), inputs)
        torch.testing.assert_close(torch.random.get_rng_state(), state_before)

    def test_channel_masking_boundaries(self) -> None:
        """Keep visible channels for every accepted masking configuration."""
        model = self._model_shell(mask_ratio=0.0)
        self.assertEqual(model.masked_channel_count(1), 0)
        self.assertEqual(model.masked_channel_count(8), 0)

        model.mask_ratio = 0.01
        self.assertEqual(model.masked_channel_count(8), 1)
        with self.assertRaisesRegex(ValueError, "at least two"):
            model.masked_channel_count(1)

        model.mask_ratio = 0.99
        self.assertEqual(model.masked_channel_count(8), 7)
        with self.assertRaisesRegex(ValueError, "at least one"):
            model.masked_channel_count(0)

    def test_model_rejects_invalid_objective_values(self) -> None:
        """Reject direct construction that bypasses config validation."""
        with self.assertRaisesRegex(ValueError, "channel_mask_ratio"):
            BrainTokenizer(
                **SMALL_TOKENIZER_CONFIG,
                channel_mask_ratio=1.0,
            )
        with self.assertRaisesRegex(ValueError, "noise_std"):
            BrainTokenizer(
                **SMALL_TOKENIZER_CONFIG,
                noise_std=-0.1,
            )

    def test_optimizer_groups_omit_empty_ema_codebook(self) -> None:
        """Name every non-empty optimizer group exactly once."""
        model = BrainTokenizer(**SMALL_TOKENIZER_CONFIG)
        groups = model.get_named_parameter_groups(
            lr=2e-4,
            codebook_lr=3e-4,
            weight_decay=0.01,
        )
        self.assertEqual(tuple(groups), ("main", "no_decay"))
        grouped_ids = [
            id(parameter)
            for group in groups.values()
            for parameter in group["params"]
        ]
        trainable_ids = [
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertCountEqual(grouped_ids, trainable_ids)
        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))

    def test_optimizer_groups_include_trainable_quantizer(self) -> None:
        """Apply codebook LR when quantizer projections are trainable."""
        config = dict(SMALL_TOKENIZER_CONFIG)
        config["codebook_dim"] = 4
        model = BrainTokenizer(**config)
        groups = model.get_named_parameter_groups(
            lr=2e-4,
            codebook_lr=3e-4,
            weight_decay=0.01,
        )
        self.assertEqual(
            tuple(groups),
            ("main", "no_decay", "codebook"),
        )
        self.assertEqual(groups["codebook"]["lr"], 3e-4)
        self.assertTrue(groups["codebook"]["params"])

    def test_monitor_attention_disables_only_its_dropout(self) -> None:
        """Use zero dropout for monitor attention without changing training."""
        module = BackWardSolution(n_dim=4, n_head=1, dropout=0.3)
        neuros = torch.randn(2, 3, 4)
        keys = torch.randn(2, 5, 4)
        values = torch.randn(2, 5, 4)
        observed: list[float] = []

        def fake_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            dropout_p: float,
            is_causal: bool,
        ) -> torch.Tensor:
            del key, value, is_causal
            observed.append(dropout_p)
            return torch.zeros_like(query)

        with mock.patch(
            "model_utils.module.F.scaled_dot_product_attention",
            side_effect=fake_attention,
        ):
            module(neuros, keys, values)
            module(neuros, keys, values, return_attention=True)
        self.assertEqual(observed, [0.3, 0.0])

    def test_modern_weight_norm_preserves_legacy_state(self) -> None:
        """Load and save released weight-normalization tensor names."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            legacy = legacy_weight_norm(nn.Conv1d(2, 3, 3))
        legacy_state = {
            name: tensor.detach().clone()
            for name, tensor in legacy.state_dict().items()
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            modern = apply_parametrization_norm(
                nn.Conv1d(2, 3, 3),
                "weight_norm",
            )
        modern.load_state_dict(legacy_state, strict=True)
        saved = modern.state_dict()
        self.assertEqual(set(saved), set(legacy_state))
        for name in legacy_state:
            torch.testing.assert_close(saved[name], legacy_state[name])
        self.assertEqual(
            tensor_state_sha256(saved),
            tensor_state_sha256(legacy_state),
        )

    def test_tokenizer_model_has_no_weight_norm_warning(self) -> None:
        """Instantiate the full tokenizer with the released state schema."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            model = BrainTokenizer(**SMALL_TOKENIZER_CONFIG)
        state = model.state_dict()
        self.assertTrue(any(name.endswith("weight_g") for name in state))
        self.assertFalse(
            any("parametrizations.weight" in name for name in state)
        )
        restored = BrainTokenizer(**SMALL_TOKENIZER_CONFIG)
        restored.load_state_dict(state, strict=True)

    def test_consolidated_brainomni_state_uses_portable_norm_names(
        self,
    ) -> None:
        """Canonicalize nested tokenizer weights in Stage-2 exports."""
        config = {
            **SMALL_TOKENIZER_CONFIG,
            "lm_dim": 8,
            "lm_head": 2,
            "lm_depth": 1,
            "lm_dropout": 0.0,
            "overlap_ratio": 0.25,
            "mask_ratio": 0.5,
            "num_quantizers_used": 2,
        }
        model = BrainOmni(**config)
        portable = model.state_dict()
        reconstructed = {}
        for key, tensor in portable.items():
            key = key.replace(
                LEGACY_WEIGHT_NORM_MAGNITUDE_KEY,
                WEIGHT_NORM_MAGNITUDE_KEY,
            )
            key = key.replace(
                LEGACY_WEIGHT_NORM_DIRECTION_KEY,
                WEIGHT_NORM_DIRECTION_KEY,
            )
            reconstructed[key] = tensor
        converted = legacy_weight_norm_state_dict(reconstructed)
        self.assertEqual(set(converted), set(portable))
        self.assertEqual(
            tensor_state_sha256(converted),
            tensor_state_sha256(portable),
        )
        restored = BrainOmni(**config)
        restored.load_state_dict(converted, strict=True)

    def test_distributed_cleanup_is_safe_and_conditional(self) -> None:
        """Destroy initialized process groups without touching absent groups."""
        with (
            mock.patch(
                "factory.training_runtime.dist.is_available",
                return_value=True,
            ),
            mock.patch(
                "factory.training_runtime.dist.is_initialized",
                return_value=True,
            ),
            mock.patch(
                "factory.training_runtime.dist.destroy_process_group"
            ) as destroy,
        ):
            destroy_distributed_process_group()
        destroy.assert_called_once_with()

        with (
            mock.patch(
                "factory.training_runtime.dist.is_available",
                return_value=True,
            ),
            mock.patch(
                "factory.training_runtime.dist.is_initialized",
                return_value=False,
            ),
            mock.patch(
                "factory.training_runtime.dist.destroy_process_group"
            ) as destroy,
        ):
            destroy_distributed_process_group()
        destroy.assert_not_called()


if __name__ == "__main__":
    unittest.main()

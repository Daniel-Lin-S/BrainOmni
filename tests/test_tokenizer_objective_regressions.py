"""Regressions for EMA state consistency and circular spectral objectives.

Synthetic vectors and waveforms exercise numerical invariants without data
files or training artifacts. Temporary config files test semantic provenance.
"""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import yaml

from braintokenizer.metrics import compute_amp, compute_phase
from braintokenizer.model import BrainTokenizer
from tests.test_braintokenizer_training_behavior import SMALL_TOKENIZER_CONFIG
from factory.campaign import _semantic_payload
from model_utils.loss import get_frequency_domain_loss
from model_utils.vq import EuclideanCodebook
from tests.test_pretrain_config import write_local_overlay
from pretrain_config import (
    ConfigError,
    canonical_config_sha256,
    load_pretrain_config,
    validate_pretrain_config,
)

ROOT = Path(__file__).resolve().parents[1]


class EmaRegressionTest(unittest.TestCase):
    """Preserve centroid and count invariants across actual forward calls."""

    def test_kmeans_centroids_remain_fixed_for_exact_cluster_data(self) -> None:
        """An EMA update at its fixed point must not shrink centroids."""
        centers = torch.eye(2)
        counts = torch.tensor([30.0, 10.0])
        data = torch.repeat_interleave(centers, counts.long(), dim=0)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                codebook = EuclideanCodebook(
                    dim=2, codebook_size=2, kmeans_init=True,
                    threshold_ema_dead_code=0,
                ).to(dtype)
                with mock.patch(
                    "model_utils.vq.kmeans", return_value=(centers, counts)
                ):
                    codebook.init_embed_(data.to(dtype))
                torch.testing.assert_close(
                    codebook.embed_avg.float(), centers * counts[:, None]
                )
                for _ in range(3):
                    output, indices = codebook(data.to(dtype))
                    torch.testing.assert_close(output.float(), data)
                    self.assertEqual(indices.unique().numel(), 2)
                    torch.testing.assert_close(
                        codebook.embed.float(), centers,
                        atol=0.01 if dtype == torch.bfloat16 else 1e-6,
                        rtol=0,
                    )

    def test_replacement_survives_and_is_selectable_next_batch(self) -> None:
        """Revival must reset sufficient statistics and survive the update."""
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                codebook = EuclideanCodebook(dim=1, codebook_size=2)
                codebook = codebook.to(dtype)
                codebook.embed.copy_(torch.tensor([[1.0], [99.0]]))
                codebook.embed_avg.copy_(torch.tensor([[10.0], [99.0]]))
                codebook.cluster_size.copy_(torch.tensor([10.0, 0.0]))
                data = torch.tensor([[1.0], [2.0]], dtype=dtype)
                replacements = torch.full((2, 1), 2.0, dtype=dtype)
                with mock.patch(
                    "model_utils.vq.sample_vectors",
                    return_value=replacements,
                ), mock.patch("model_utils.vq.broadcast_tensors") as sync:
                    codebook(data)
                self.assertEqual(sync.call_count, 1)
                self.assertEqual(codebook.embed[1].item(), 2.0)
                self.assertEqual(codebook.cluster_size[1].item(), 2.0)
                self.assertEqual(codebook.embed_avg[1].item(), 4.0)
                codebook.eval()
                _, indices = codebook(data[1:])
                self.assertEqual(indices.item(), 1)

    def test_evaluation_does_not_update_initialized_state(self) -> None:
        """An initialized checkpoint stays immutable during evaluation."""
        codebook = EuclideanCodebook(dim=2, codebook_size=2).eval()
        before = deepcopy(codebook.state_dict())
        codebook(torch.ones(3, 2))
        for name, value in codebook.state_dict().items():
            torch.testing.assert_close(value, before[name])

    def test_random_initialization_has_consistent_pseudocounts(self) -> None:
        """Non-kmeans initialization must also use consistent EMA state."""
        codebook = EuclideanCodebook(dim=2, codebook_size=4)
        torch.testing.assert_close(
            codebook.embed_avg,
            codebook.embed * codebook.cluster_size[:, None],
        )

    def test_invalid_vectors_fail_clearly(self) -> None:
        """Reject empty, malformed and non-finite quantizer inputs."""
        codebook = EuclideanCodebook(dim=2, codebook_size=2)
        for data in (
            torch.empty(0, 2), torch.ones(3, 4),
            torch.full((2, 2), float("nan")),
        ):
            with self.subTest(shape=data.shape):
                with self.assertRaises(ValueError):
                    codebook(data)

    def test_tokenizer_forward_backward_remains_finite(self) -> None:
        """Exercise corrected EMA and phase loss together with rotation VQ."""
        torch.manual_seed(42)
        model = BrainTokenizer(**SMALL_TOKENIZER_CONFIG).train()
        inputs = {
            "x": torch.randn(2, 4, 64),
            "pos": torch.randn(2, 4, 6),
            "sensor_type": torch.full((2, 4), 2, dtype=torch.long),
        }
        output, _, monitor = model(**inputs, return_monitor_data=True)
        output["loss"].backward()
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertEqual(monitor["reconstruction"].shape, (2, 4, 2, 32))
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for name, buffer in model.named_buffers():
            self.assertTrue(torch.isfinite(buffer).all(), name)



class SpectralRegressionTest(unittest.TestCase):
    """Keep the training objective and reported spectral errors identical."""

    def test_phase_wrap_uses_shortest_distance(self) -> None:
        """Near-identical angles across the branch cut incur small error."""
        length = 8
        target_spectrum = torch.ones(length // 2 + 1, dtype=torch.complex64)
        predicted_spectrum = target_spectrum.clone()
        target_spectrum[1] = torch.polar(
            torch.tensor(1.0), torch.tensor(torch.pi - 0.01)
        )
        predicted_spectrum[1] = target_spectrum[1].conj()
        window = torch.hamming_window(length)
        target = torch.fft.irfft(target_spectrum, norm="ortho") / window
        predicted = torch.fft.irfft(predicted_spectrum, norm="ortho") / window
        predicted.requires_grad_()
        amplitude, phase = get_frequency_domain_loss(predicted, target)
        self.assertAlmostEqual(phase.item(), 0.02 / 5, places=6)
        self.assertLess(amplitude.item(), 1e-6)
        phase.backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())
        self.assertGreater(predicted.grad.abs().sum().item(), 0)
        torch.testing.assert_close(compute_phase(predicted, target), phase)
        torch.testing.assert_close(compute_amp(predicted, target), amplitude)
        torch.testing.assert_close(compute_phase(target, predicted), phase)

    def test_identical_and_zero_predictions_have_finite_gradients(self) -> None:
        """Zero FFT coefficients must not create non-finite gradients."""
        torch.manual_seed(1)
        target = torch.randn(2, 3, 4, 32)
        amplitude, phase = get_frequency_domain_loss(target, target)
        self.assertEqual(amplitude.item(), 0)
        self.assertEqual(phase.item(), 0)
        predicted = torch.zeros_like(target, requires_grad=True)
        amplitude, phase = get_frequency_domain_loss(predicted, target)
        (amplitude + phase).backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())
        self.assertLessEqual(phase.item(), torch.pi)

    def test_invalid_waveforms_raise(self) -> None:
        """Undefined inputs must not silently produce empty or NaN losses."""
        for predicted, target in (
            (torch.empty(0), torch.empty(0)),
            (torch.ones(2), torch.ones(3)),
            (torch.full((2,), float("nan")), torch.ones(2)),
        ):
            with self.assertRaises(ValueError):
                get_frequency_domain_loss(predicted, target)


class ImplementationIdentityTest(unittest.TestCase):
    """Separate corrected training from campaigns with historical semantics."""

    def test_version_is_materialized_and_changes_campaign_identity(
        self,
    ) -> None:
        """Old YAML receives the version; explicitly old versions fail."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = load_pretrain_config([
                ROOT / "configs/pretrain/braintokenizer.yaml",
                write_local_overlay(directory, "braintokenizer"),
            ])
            objective = config["campaign"]["objective"]
            self.assertEqual(objective["implementation_version"], 2)
            source = deepcopy(config)
            del source["campaign"]["objective"]["implementation_version"]
            path = directory / "config.yaml"
            path.write_text(yaml.safe_dump(source), encoding="utf-8")
            loaded = load_pretrain_config(path)
        self.assertEqual(loaded["campaign"]["objective"], objective)
        config["campaign"]["data"]["included_datasets"] = ["test"]
        config["invocation"]["data_catalog"] = {
            "test": {"path": str(ROOT), "signal_type": "eeg"}
        }
        legacy = deepcopy(config)
        del legacy["campaign"]["objective"]["implementation_version"]
        payloads = [
            _semantic_payload(
                settings, settings["campaign"]["model"],
                {"sha256": "test-split"}, None,
            )
            for settings in (config, legacy)
        ]
        self.assertNotEqual(
            canonical_config_sha256(payloads[0]),
            canonical_config_sha256(payloads[1]),
        )
        for version in (1, True, 2.0):
            invalid = deepcopy(loaded)
            invalid["campaign"]["objective"][
                "implementation_version"
            ] = version
            with self.assertRaises(ConfigError):
                validate_pretrain_config(invalid)


if __name__ == "__main__":
    unittest.main()

"""Focused tests for semantic campaign health and immutable artifacts."""

from __future__ import annotations

from copy import deepcopy
import json
import importlib
import sys
from types import ModuleType
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import yaml

from factory.campaign import (
    ARTIFACT_SCHEMA_VERSION,
    CampaignContext,
    CampaignHealthError,
    atomic_json,
    canonical_json_sha256,
    ensure_campaign_health,
    ensure_training_campaign,
    export_completed_weights,
    prepare_campaign,
    record_checkpoint,
    tensor_state_sha256,
    validate_portable_state,
)
from factory.training_runtime import (
    evaluation_metrics_path,
    write_evaluation_metrics,
)
from pretrain_config import (
    load_pretrain_config,
    metadata_directory,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]


class CampaignHealthTest(unittest.TestCase):
    """Validate stage-wide campaign health, repair, and identity behavior."""

    def _campaign(
        self,
        root: Path,
        stage: str,
        state: dict[str, torch.Tensor],
    ) -> Path:
        payload = {
            "campaign": {
                "data": {"preprocessing": {"sample_rate_hz": 256}},
                "epochs": 2,
                "stage": stage,
            }
        }
        identity_sha256 = canonical_json_sha256(payload)
        campaign_root = root / identity_sha256[:20]
        campaign_root.mkdir()
        model_path = campaign_root / "model_cfg.json"
        model_path.write_text("{}\n", encoding="utf-8")
        setting_path = campaign_root / "pretrain_setting.yaml"
        setting_path.write_text("campaign: {}\n", encoding="utf-8")
        atomic_json(
            campaign_root / "campaign_identity.json",
            {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "campaign_sha256": identity_sha256,
                "stage": stage,
                "semantic_payload": payload,
            },
        )
        atomic_json(
            campaign_root / "pretrain_setting.json",
            {
                "campaign_sha256": identity_sha256,
                "stage": stage,
                "model_config_sha256": sha256_file(model_path),
                "pretrain_setting_sha256": sha256_file(setting_path),
            },
        )
        weight_name = (
            "BrainTokenizer.pt" if stage == "braintokenizer" else "BrainOmni.pt"
        )
        torch.save(state, campaign_root / weight_name)
        atomic_json(
            campaign_root / "campaign_status.json",
            {
                "campaign_sha256": identity_sha256,
                "stage": stage,
                "portable_model_state_sha256": tensor_state_sha256(state),
                "state": "complete",
            },
        )
        best = campaign_root / "checkpoint" / "best"
        best.mkdir(parents=True)
        (best / "model_states.pt").write_bytes(b"checkpoint")
        record_checkpoint(campaign_root, "best")
        return campaign_root

    def test_health_and_repair_cover_both_stages(self) -> None:
        state = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
        for stage in ("braintokenizer", "brainomni"):
            with self.subTest(
                stage=stage
            ), tempfile.TemporaryDirectory() as tmp:
                campaign = self._campaign(Path(tmp), stage, state)
                with mock.patch(
                    "factory.campaign._expected_model_state",
                    return_value=deepcopy(state),
                ):
                    healthy = ensure_campaign_health(
                        campaign,
                        expected_stage=stage,
                    )
                    self.assertFalse(healthy.repaired)
                    healthy.portable_path.write_bytes(b"corrupt")

                    def convert(
                        root: Path,
                        observed_stage: str,
                        destination: Path,
                    ) -> None:
                        self.assertEqual(root, campaign)
                        self.assertEqual(observed_stage, stage)
                        torch.save(state, destination)

                    with mock.patch(
                        "factory.campaign._convert_best",
                        side_effect=convert,
                    ):
                        repaired = ensure_campaign_health(
                            campaign,
                            expected_stage=stage,
                        )
                self.assertTrue(repaired.repaired)
                status = json.loads(
                    (campaign / "campaign_status.json").read_text()
                )
                self.assertEqual(len(status["repair_history"]), 1)

    def test_unrepairable_campaign_reports_repair_and_training_commands(
        self,
    ) -> None:
        state = {"weight": torch.ones(2, 3)}
        with tempfile.TemporaryDirectory() as temporary:
            campaign = self._campaign(
                Path(temporary),
                "brainomni",
                state,
            )
            (campaign / "BrainOmni.pt").write_bytes(b"corrupt")
            best_file = campaign / "checkpoint" / "best" / "model_states.pt"
            best_file.write_bytes(b"also-corrupt")
            with mock.patch(
                "factory.campaign._expected_model_state",
                return_value=state,
            ):
                with self.assertRaises(CampaignHealthError) as raised:
                    ensure_campaign_health(
                        campaign,
                        expected_stage="brainomni",
                    )
            message = str(raised.exception)
            self.assertIn("Repair command:", message)
            self.assertIn("Full retraining command:", message)
            self.assertFalse(
                any(
                    path.name.startswith("failed_recovery_")
                    for path in (campaign / "checkpoint").iterdir()
                )
            )

    def test_health_rejects_wrong_campaign_stage(self) -> None:
        state = {"weight": torch.ones(2, 3)}
        with tempfile.TemporaryDirectory() as temporary:
            campaign = self._campaign(
                Path(temporary),
                "braintokenizer",
                state,
            )
            with self.assertRaisesRegex(
                CampaignHealthError,
                "Expected a brainomni campaign root",
            ):
                ensure_campaign_health(
                    campaign,
                    expected_stage="brainomni",
                )

    def test_portable_validation_rejects_corruption_modes(self) -> None:
        expected = {"weight": torch.ones(2, 3)}
        cases = {
            "wrong-key": {"other": torch.ones(2, 3)},
            "wrong-shape": {"weight": torch.ones(3, 2)},
            "non-finite": {"weight": torch.full((2, 3), float("nan"))},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, state in cases.items():
                with self.subTest(name=name):
                    path = root / f"{name}.pt"
                    torch.save(state, path)
                    with mock.patch(
                        "factory.campaign._expected_model_state",
                        return_value=expected,
                    ):
                        with self.assertRaises(CampaignHealthError):
                            validate_portable_state(
                                path,
                                "braintokenizer",
                                {},
                            )

    def test_successful_restart_export_removes_its_quarantine(self) -> None:
        state = {"weight": torch.ones(2, 3)}
        with tempfile.TemporaryDirectory() as temporary:
            campaign = self._campaign(
                Path(temporary),
                "brainomni",
                state,
            )
            failed = campaign / "checkpoint" / "failed_recovery_attempt"
            failed.mkdir()
            (failed / "diagnostic.pt").write_bytes(b"old")
            status_path = campaign / "campaign_status.json"
            status = json.loads(status_path.read_text())
            status["state"] = "incomplete"
            status["active_failed_recovery"] = str(failed.resolve())
            atomic_json(status_path, status)
            attempt_root = campaign / "attempts" / "attempt"
            attempt_root.mkdir(parents=True)
            atomic_json(
                attempt_root / "status.json",
                {"state": "started"},
            )
            context = CampaignContext(
                root=campaign,
                attempt_root=attempt_root,
                attempt_id="attempt",
                stage="brainomni",
                identity_sha256=status["campaign_sha256"],
                training_required=True,
            )

            def convert(
                root: Path,
                stage: str,
                destination: Path,
            ) -> None:
                del root, stage
                torch.save(state, destination)

            with mock.patch(
                "factory.campaign._expected_model_state",
                return_value=state,
            ), mock.patch(
                "factory.campaign._convert_best",
                side_effect=convert,
            ):
                export_completed_weights(context)
            self.assertFalse(failed.exists())
            self.assertTrue((campaign / "BrainOmni.pt").is_file())

    def test_failed_training_repair_is_quarantined_until_success(self) -> None:
        state = {"weight": torch.ones(2, 3)}
        with tempfile.TemporaryDirectory() as temporary:
            campaign = self._campaign(
                Path(temporary),
                "braintokenizer",
                state,
            )
            portable = campaign / "BrainTokenizer.pt"
            portable.write_bytes(b"corrupt")
            attempt_root = campaign / "attempts" / "attempt"
            attempt_root.mkdir(parents=True)
            atomic_json(
                attempt_root / "status.json",
                {"state": "started"},
            )
            context = CampaignContext(
                root=campaign,
                attempt_root=attempt_root,
                attempt_id="attempt",
                stage="braintokenizer",
                identity_sha256=json.loads(
                    (campaign / "campaign_identity.json").read_text()
                )["campaign_sha256"],
                training_required=False,
            )
            with mock.patch(
                "factory.campaign._convert_best",
                side_effect=RuntimeError("conversion failed"),
            ), self.assertWarns(RuntimeWarning):
                required = ensure_training_campaign(context)
            quarantine = (
                campaign
                / "checkpoint"
                / "failed_recovery_attempt"
            )
            self.assertTrue(required)
            self.assertTrue(quarantine.is_dir())
            self.assertTrue(
                (quarantine / "BrainTokenizer.pt").is_file()
            )
            self.assertFalse(portable.exists())

    def test_checkpoint_manifest_corruption_blocks_consumers(self) -> None:
        state = {"weight": torch.ones(2, 3)}
        with tempfile.TemporaryDirectory() as temporary:
            campaign = self._campaign(
                Path(temporary),
                "brainomni",
                state,
            )
            checkpoint = campaign / "checkpoint" / "best" / "model_states.pt"
            checkpoint.write_bytes(b"changed")
            with self.assertRaisesRegex(
                CampaignHealthError,
                "Best checkpoint health validation failed",
            ):
                ensure_campaign_health(campaign, expected_stage="brainomni")

    def test_epochs_change_semantic_campaign_without_checkpoint_reuse(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            overlay = root / "local.yaml"
            raw = root / "raw"
            raw.mkdir()
            invocation = {
                "data_catalog": {
                    "dataset": {
                        "path": str(raw),
                        "signal_type": "eeg",
                    }
                },
                "metadata_root": str(root / "metadata"),
                "output_root": str(root / "output"),
                "processed_root": str(root / "processed"),
            }
            overlay.write_text(
                yaml.safe_dump({"invocation": invocation}),
                encoding="utf-8",
            )
            config = load_pretrain_config(
                [
                    ROOT / "configs/pretrain/braintokenizer.yaml",
                    overlay,
                ]
            )
            metadata = metadata_directory(config)
            metadata.mkdir(parents=True)
            row = {"dataset": "dataset", "path": "portable-id"}
            for partition in ("train", "val", "test"):
                (metadata / f"{partition}.json").write_text(
                    json.dumps([row]),
                    encoding="utf-8",
                )
            config["campaign"]["data"]["included_datasets"] = ["dataset"]
            first = prepare_campaign(
                config,
                config["campaign"]["model"],
            )
            extended = deepcopy(config)
            extended["campaign"]["training"]["epochs"] += 1
            second = prepare_campaign(
                extended,
                extended["campaign"]["model"],
            )
            self.assertNotEqual(first.root, second.root)
            self.assertFalse(second.checkpoint_root.joinpath("latest").exists())

    def test_downstream_loader_health_checks_campaign_root(self) -> None:
        class FakeTokenizer:
            def parameters(self) -> list[torch.Tensor]:
                return []

        class FakeModel:
            def __init__(self, **config: object) -> None:
                self.lm_dim = int(config["lm_dim"])
                self.tokenizer = FakeTokenizer()
                self.loaded_strict: bool | None = None

            def load_state_dict(
                self,
                state: dict[str, torch.Tensor],
                strict: bool,
            ) -> None:
                self.loaded_strict = strict
                self.state = state

        brainomni_module = ModuleType("brainomni.model")
        brainomni_module.BrainOmni = FakeModel
        with mock.patch.dict(
            sys.modules,
            {"brainomni.model": brainomni_module},
        ):
            downstream_brainomni = importlib.import_module(
                "downstream.model_collection.BrainOmni"
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model_cfg.json").write_text(
                json.dumps({"lm_dim": 4}),
                encoding="utf-8",
            )
            portable = root / "BrainOmni.pt"
            torch.save({"weight": torch.ones(1)}, portable)
            health = mock.Mock(
                root=root,
                stage="brainomni",
                campaign_sha256="a" * 64,
                model_config_sha256="b" * 64,
                model_state_sha256="c" * 64,
                portable_path=portable,
                repaired=False,
            )
            with mock.patch.object(
                downstream_brainomni,
                "ensure_campaign_health",
                return_value=health,
            ) as health_check, mock.patch.object(
                downstream_brainomni,
                "BrainOmni",
                FakeModel,
            ):
                model, dimension = downstream_brainomni.get_brainomni(
                    ckpt_path=str(root)
                )
            health_check.assert_called_once_with(
                root.resolve(),
                expected_stage="brainomni",
                repair=True,
            )
            self.assertEqual(dimension, 4)
            self.assertTrue(model.loaded_strict)

    def test_downstream_health_failure_does_not_construct_model(self) -> None:
        brainomni_module = ModuleType("brainomni.model")
        brainomni_module.BrainOmni = mock.Mock
        with mock.patch.dict(
            sys.modules,
            {"brainomni.model": brainomni_module},
        ):
            downstream_brainomni = importlib.import_module(
                "downstream.model_collection.BrainOmni"
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            error = CampaignHealthError(
                "repair failed; Repair command: repair; "
                "Full retraining command: train"
            )
            with mock.patch.object(
                downstream_brainomni,
                "ensure_campaign_health",
                side_effect=error,
            ), mock.patch.object(
                downstream_brainomni,
                "BrainOmni",
            ) as model_constructor:
                with self.assertRaisesRegex(
                    CampaignHealthError,
                    "Repair command",
                ):
                    downstream_brainomni.get_brainomni(ckpt_path=str(root))
            model_constructor.assert_not_called()

    def test_evaluation_metrics_are_immutable(self) -> None:
        state = {"weight": torch.ones(2, 3)}
        with tempfile.TemporaryDirectory() as temporary:
            campaign = self._campaign(
                Path(temporary),
                "brainomni",
                state,
            )
            attempt_root = campaign / "attempts" / "attempt"
            attempt_root.mkdir(parents=True)
            context = CampaignContext(
                root=campaign,
                attempt_root=attempt_root,
                attempt_id="attempt",
                stage="brainomni",
                identity_sha256=json.loads(
                    (campaign / "campaign_identity.json").read_text()
                )["campaign_sha256"],
                training_required=False,
            )
            evaluator = Path(temporary) / "evaluator.py"
            evaluator.write_text("version = 1\n", encoding="utf-8")
            metadata = Path(temporary) / "test.json"
            metadata.write_text('[{"segment": 1}]\n', encoding="utf-8")
            path = write_evaluation_metrics(
                context,
                "test",
                {"loss": 1.0},
                evaluator,
                metadata,
            )
            original = path.read_bytes()
            write_evaluation_metrics(
                context,
                "test",
                {"loss": 2.0},
                evaluator,
                metadata,
            )
            self.assertEqual(path.read_bytes(), original)
            evaluator.write_text("version = 2\n", encoding="utf-8")
            with self.assertRaises(CampaignHealthError):
                write_evaluation_metrics(
                    context,
                    "test",
                    {"loss": 3.0},
                    evaluator,
                    metadata,
                )
            self.assertEqual(path.read_bytes(), original)
            evaluator.write_text("version = 1\n", encoding="utf-8")
            metadata.write_text('[{"segment": 2}]\n', encoding="utf-8")
            with self.assertRaises(CampaignHealthError):
                write_evaluation_metrics(
                    context,
                    "test",
                    {"loss": 3.0},
                    evaluator,
                    metadata,
                )
            self.assertEqual(path.read_bytes(), original)

    def test_flat_evaluation_filenames(self) -> None:
        context = CampaignContext(
            root=Path("campaign"),
            attempt_root=Path("campaign/attempts/a"),
            attempt_id="a",
            stage="brainomni",
            identity_sha256="a" * 64,
            training_required=False,
        )
        self.assertEqual(
            evaluation_metrics_path(context, "test").name,
            "metrics_test_set.json",
        )
        self.assertEqual(
            evaluation_metrics_path(context, "MEG-MASC").name,
            "metrics_heldout_MEG-MASC.json",
        )


if __name__ == "__main__":
    unittest.main()

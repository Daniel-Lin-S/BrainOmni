"""Focused tests for the pre-training configuration contract."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

import yaml

from brainomni.config import BrainOmniTrainerConfig
from factory.checkpoint import convert_best_checkpoint
from factory.export_pretrained import export_checkpoint
from pretrain_config import (
    ConfigError,
    build_deepspeed_config,
    load_pretrain_config,
    metadata_directory,
    resolve_dataset_identities,
    repository_log_directory,
    validate_pretrain_config,
    write_run_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]


def write_local_overlay(directory: Path, stage: str) -> Path:
    """Create a disposable local overlay without machine-specific paths."""
    invocation = {
        "raw_root": str(directory / "raw"),
        "processed_root": str(directory / "processed"),
        "metadata_root": str(directory / "metadata"),
        "output_root": str(directory / "output"),
    }
    if stage == "brainomni":
        invocation["tokenizer_path"] = str(directory / "tokenizer")
    overlay_path = directory / "pretrain.local.json"
    overlay_path.write_text(
        json.dumps(
            {
                "campaign": {"data": {"included_datasets": ["test"]}},
                "invocation": invocation,
            }
        )
    )
    return overlay_path


def write_metadata(config: dict[str, object], datasets: list[str]) -> None:
    """Write minimal generated preprocessing metadata for one test."""
    directory = metadata_directory(config)
    directory.mkdir(parents=True)
    metadata = [
        {"dataset": dataset, "path": f"{dataset}_{index}.pt"}
        for index, dataset in enumerate(datasets)
    ]
    (directory / "info.json").write_text(json.dumps(metadata))


def leaf_paths(value: object, prefix: str = "") -> set[str]:
    """Return dotted paths for every mapping leaf in a resolved config."""
    if not isinstance(value, dict):
        return {prefix}
    paths: set[str] = set()
    for key, child in value.items():
        child_prefix = f"{prefix}.{key}" if prefix else key
        paths |= leaf_paths(child, child_prefix)
    return paths


class PretrainConfigTest(unittest.TestCase):
    """Validate resolution, artifacts, and documentation coverage."""

    def test_default_tokenizer_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = yaml.safe_load(
                (ROOT / "configs/pretrain/braintokenizer.yaml").read_text()
            )
            self.assertEqual(
                source["campaign"]["data"]["included_datasets"],
                ["*"],
            )
            config = load_pretrain_config(
                ROOT / "configs/pretrain/braintokenizer.yaml",
                write_local_overlay(root, "braintokenizer"),
            )
            self.assertEqual(config["campaign"]["training"]["epochs"], 16)
            self.assertEqual(config["campaign"]["model"]["codebook_size"], 512)
            _, accumulation = build_deepspeed_config(config, world_size=8)
            json_path = root / "braintokenizer.json"
            json_path.write_text(json.dumps(config))
            json_config = load_pretrain_config(json_path)
        self.assertEqual(json_config, config)
        self.assertEqual(accumulation, 4)
        with self.assertRaises(ConfigError):
            build_deepspeed_config(config, world_size=3)

    def test_layered_config_and_cli_precedence(self) -> None:
        base = ROOT / "configs/pretrain/braintokenizer.yaml"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared_path = root / "shared.yaml"
            shared_path.write_text("campaign:\n  training:\n    epochs: 18\n")
            config = load_pretrain_config(
                [base, shared_path],
                write_local_overlay(root, "braintokenizer"),
                ["campaign.training.epochs=20"],
            )
        self.assertEqual(config["campaign"]["training"]["epochs"], 20)

    def test_invalid_unknown_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_pretrain_config(
                ROOT / "configs/pretrain/braintokenizer.yaml",
                write_local_overlay(root, "braintokenizer"),
            )
            invalid = deepcopy(config)
            invalid["campaign"]["unknown"] = True
            path = root / "invalid.json"
            path.write_text(json.dumps(invalid))
            with self.assertRaises(ConfigError):
                load_pretrain_config(path)
            with self.assertRaises(ConfigError):
                load_pretrain_config(
                    ROOT / "configs/pretrain/braintokenizer.yaml",
                    write_local_overlay(root, "braintokenizer"),
                    ["campaign.training.epochs=not-json"],
                )
            wildcard = deepcopy(config)
            wildcard["campaign"]["data"]["included_datasets"] = ["*"]
            validate_pretrain_config(wildcard)
            mixed_wildcard = deepcopy(config)
            mixed_wildcard["campaign"]["data"]["included_datasets"] = [
                "*",
                "test",
            ]
            with self.assertRaises(ConfigError):
                validate_pretrain_config(mixed_wildcard)
            non_integer = deepcopy(config)
            non_integer["campaign"]["training"]["epochs"] = 1.5
            with self.assertRaises(ConfigError):
                validate_pretrain_config(non_integer)

    def test_dataset_resolution_and_artifact_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            overlay_path = write_local_overlay(root, "braintokenizer")
            overlay = json.loads(overlay_path.read_text())
            overlay["campaign"] = {"data": {"included_datasets": ["*"]}}
            overlay_path.write_text(json.dumps(overlay))
            config = load_pretrain_config(
                ROOT / "configs/pretrain/braintokenizer.yaml",
                overlay_path,
            )
            write_metadata(config, ["meg_b", "meg_a", "meg_b"])
            resolved = resolve_dataset_identities(config)
            self.assertEqual(
                resolved["campaign"]["data"]["included_datasets"],
                ["meg_a", "meg_b"],
            )
            run_path = root / "run"
            write_run_artifacts(run_path, resolved, {"window_length": 512})
            artifact_paths = [
                run_path / "model_cfg.json",
                run_path / "pretrain_setting.yaml",
                run_path / "pretrain_setting.json",
                run_path / "invocation.yaml",
            ]
            for artifact_path in artifact_paths:
                self.assertNotIn("*", artifact_path.read_text())
            setting = yaml.safe_load(
                (run_path / "pretrain_setting.yaml").read_text()
            )
            invocation = yaml.safe_load(
                (run_path / "invocation.yaml").read_text()
            )
        self.assertEqual(
            setting["campaign"]["data"]["included_datasets"],
            ["meg_a", "meg_b"],
        )
        self.assertEqual(invocation, resolved["invocation"])

    def test_dataset_resolution_rejects_empty_and_mismatched_metadata(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_pretrain_config(
                ROOT / "configs/pretrain/braintokenizer.yaml",
                write_local_overlay(root, "braintokenizer"),
            )
            write_metadata(config, ["other"])
            with self.assertRaises(ConfigError):
                resolve_dataset_identities(config)
            info_path = metadata_directory(config) / "info.json"
            info_path.write_text("[]")
            with self.assertRaises(ConfigError):
                resolve_dataset_identities(config)

    def test_tiny_and_base_paper_values_and_legacy_absence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tiny = load_pretrain_config(
                ROOT / "configs/pretrain/brainomni_tiny.yaml",
                write_local_overlay(root, "brainomni"),
            )
            base = load_pretrain_config(
                ROOT / "configs/pretrain/brainomni_base.yaml",
                write_local_overlay(root, "brainomni"),
            )
        self.assertEqual(tiny["campaign"]["model"]["lm_dim"], 256)
        self.assertEqual(tiny["campaign"]["model"]["lm_head"], 8)
        self.assertEqual(tiny["campaign"]["model"]["lm_depth"], 12)
        self.assertEqual(tiny["campaign"]["optimizer"]["lr"], 0.0005)
        self.assertEqual(base["campaign"]["model"]["lm_dim"], 512)
        self.assertEqual(base["campaign"]["model"]["lm_head"], 16)
        self.assertEqual(base["campaign"]["model"]["lm_depth"], 12)
        self.assertEqual(base["campaign"]["optimizer"]["lr"], 0.0004)
        self.assertFalse((ROOT / "configs/pretrain/brainomni.yaml").exists())

    def test_repository_log_directory_is_outside_run_artifacts(self) -> None:
        run_path = ROOT / "temporary-artifacts" / "braintokenizer" / "exp_1"
        log_path = repository_log_directory(run_path, "braintokenizer")
        self.assertEqual(
            log_path,
            ROOT / "logs" / "braintokenizer" / "braintokenizer" / "exp_1",
        )
        with self.assertRaises(ConfigError):
            repository_log_directory(run_path, "unknown")

    def test_stage_two_records_actual_tokenizer_identity(self) -> None:
        tokenizer_model = {
            "window_length": 512,
            "n_filters": 32,
            "ratios": [8, 4, 2],
            "kernel_size": 5,
            "last_kernel_size": 5,
            "n_dim": 256,
            "n_neuro": 16,
            "n_head": 4,
            "dropout": 0.0,
            "codebook_dim": 256,
            "codebook_size": 512,
            "num_quantizers": 4,
            "rotation_trick": True,
            "quantize_optimize_method": "ema",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_pretrain_config(
                ROOT / "configs/pretrain/brainomni_tiny.yaml",
                write_local_overlay(root, "brainomni"),
            )
            tokenizer_path = root / "tokenizer"
            tokenizer_path.mkdir()
            (tokenizer_path / "model_cfg.json").write_text(
                json.dumps(tokenizer_model)
            )
            (tokenizer_path / "BrainTokenizer.pt").write_bytes(b"weights")
            write_metadata(config, ["test"])
            config = resolve_dataset_identities(config)
            trainer_config = BrainOmniTrainerConfig(config, world_size=8)
            model_config = trainer_config.get_model_cfg()
            self.assertEqual(model_config["lm_dim"], 256)
            self.assertEqual(model_config["mask_ratio"], 0.5)
            run_path = root / "run"
            write_run_artifacts(
                run_path,
                config,
                model_config,
                trainer_config.tokenizer_identity,
            )
            manifest = json.loads(
                (run_path / "pretrain_setting.json").read_text()
            )
            json_path = root / "brainomni_tiny.json"
            json_path.write_text(json.dumps(config))
            json_config = load_pretrain_config(json_path)
        self.assertEqual(json_config, config)
        self.assertEqual(
            manifest["tokenizer_identity"],
            trainer_config.tokenizer_identity,
        )

    def test_portable_export_excludes_invocation_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_pretrain_config(
                ROOT / "configs/pretrain/braintokenizer.yaml",
                write_local_overlay(root, "braintokenizer"),
            )
            write_metadata(config, ["test"])
            config = resolve_dataset_identities(config)
            run_path = root / "run"
            weights_path = root / "weights.pt"
            weights_path.write_bytes(b"weights")
            write_run_artifacts(run_path, config, {"window_length": 512})
            export_checkpoint(
                run_path,
                weights_path,
                root / "portable",
                "BrainTokenizer.pt",
            )
            self.assertFalse((root / "portable" / "invocation.yaml").exists())

    def test_mocked_best_checkpoint_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_path = Path(temporary) / "run"
            (run_path / "checkpoint" / "best").mkdir(parents=True)
            zero_module = ModuleType("deepspeed.utils.zero_to_fp32")

            def convert(checkpoint: str, output: str, tag: str) -> None:
                self.assertEqual(Path(checkpoint), run_path / "checkpoint")
                self.assertEqual(tag, "best")
                Path(output).write_bytes(b"checkpoint")

            zero_module.convert_zero_checkpoint_to_fp32_state_dict = convert
            utils_module = ModuleType("deepspeed.utils")
            deepspeed_module = ModuleType("deepspeed")
            utils_module.zero_to_fp32 = zero_module
            deepspeed_module.utils = utils_module
            modules = {
                "deepspeed": deepspeed_module,
                "deepspeed.utils": utils_module,
                "deepspeed.utils.zero_to_fp32": zero_module,
            }
            with mock.patch.dict(sys.modules, modules):
                output_path = convert_best_checkpoint(run_path)
            self.assertEqual(output_path, run_path / "BrainTokenizer.pt")
            self.assertEqual(output_path.read_bytes(), b"checkpoint")

    def test_documentation_covers_schema_and_public_files_have_no_paths(
        self,
    ) -> None:
        documentation = (
            ROOT / "docs/pretraining_configuration.md"
        ).read_text()
        self.assertIn(
            "docs/pretraining_configuration.md",
            (ROOT / "README.md").read_text(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer = load_pretrain_config(
                ROOT / "configs/pretrain/braintokenizer.yaml",
                write_local_overlay(root, "braintokenizer"),
            )
            brainomni = load_pretrain_config(
                ROOT / "configs/pretrain/brainomni_tiny.yaml",
                write_local_overlay(root, "brainomni"),
            )
        keys = leaf_paths(tokenizer) | leaf_paths(brainomni)
        missing = [key for key in keys if key not in documentation]
        self.assertFalse(missing)
        public_files = [
            ROOT / "docs/pretraining_configuration.md",
            ROOT / "configs/pretrain/braintokenizer.yaml",
            ROOT / "configs/pretrain/brainomni_tiny.yaml",
            ROOT / "configs/pretrain/brainomni_base.yaml",
            ROOT / "braintokenizer/launcher.py",
            ROOT / "brainomni/launcher.py",
            ROOT / "tests/test_pretrain_config.py",
        ]
        data_prefix = "/" + "data" + "/"
        home_prefix = "/" + "home" + "/"
        for path in public_files:
            text = path.read_text()
            self.assertNotIn(data_prefix, text)
            self.assertNotIn(home_prefix, text)


if __name__ == "__main__":
    unittest.main()

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
from factory.campaign import CampaignHealth
from factory.export_pretrained import export_checkpoint
from factory.process import discover_catalog_recordings
from factory.utils import split_pretrain_metadata
from pretrain_config import (
    ConfigError,
    build_deepspeed_config,
    load_data_catalog,
    load_pretrain_config,
    load_pretrain_launch_config,
    metadata_directory,
    preprocessing_directory,
    resolve_dataset_identities,
    selected_data_catalog,
    validate_pretrain_config,
)

ROOT = Path(__file__).resolve().parents[1]


def write_local_overlay(directory: Path, stage: str) -> Path:
    """Create a disposable local overlay without machine-specific paths."""
    (directory / "raw").mkdir(parents=True, exist_ok=True)
    invocation = {
        "processed_root": str(directory / "processed"),
        "metadata_root": str(directory / "metadata"),
        "data_catalog": {
            "test": {
                "path": str(directory / "raw"),
                "signal_type": "eeg",
            }
        },
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
    directory = preprocessing_directory(config)
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
            self.assertNotIn("signal_type", source["campaign"]["data"])
            self.assertNotIn("raw_root", source["invocation"])
            config = load_pretrain_config(
                [
                    ROOT / "configs/pretrain/braintokenizer.yaml",
                    write_local_overlay(root, "braintokenizer"),
                ]
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
                [
                    base,
                    shared_path,
                    write_local_overlay(root, "braintokenizer"),
                ],
                overrides=["campaign.training.epochs=20"],
            )
        self.assertEqual(config["campaign"]["training"]["epochs"], 20)

    def test_invalid_unknown_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_pretrain_config(
                [
                    ROOT / "configs/pretrain/braintokenizer.yaml",
                    write_local_overlay(root, "braintokenizer"),
                ]
            )
            invalid = deepcopy(config)
            invalid["campaign"]["unknown"] = True
            path = root / "invalid.json"
            path.write_text(json.dumps(invalid))
            with self.assertRaises(ConfigError):
                load_pretrain_config(path)
            with self.assertRaises(ConfigError):
                load_pretrain_config(
                    [
                        ROOT / "configs/pretrain/braintokenizer.yaml",
                        write_local_overlay(root, "braintokenizer"),
                    ],
                    overrides=["campaign.training.epochs=not-json"],
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

    def test_data_catalog_loading_and_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog_path = root / "datasets.local.yaml"
            (root / "eeg").mkdir()
            (root / "meg").mkdir()
            catalog_path.write_text(
                "datasets:\n"
                "  EEG-ONE:\n"
                f"    path: {root / 'eeg'}\n"
                "    signal_type: eeg\n"
                "  MEG-ONE:\n"
                f"    path: {root / 'meg'}\n"
                "    signal_type: meg\n"
            )
            catalog = load_data_catalog(catalog_path)
            self.assertEqual(sorted(catalog), ["EEG-ONE", "MEG-ONE"])
            overlay_path = write_local_overlay(root, "braintokenizer")
            overlay = json.loads(overlay_path.read_text())
            overlay["campaign"] = {"data": {"included_datasets": ["*"]}}
            overlay_path.write_text(json.dumps(overlay))
            config = load_pretrain_config(
                [
                    ROOT / "configs/pretrain/braintokenizer.yaml",
                    overlay_path,
                ],
                data_catalog_path=catalog_path,
            )
            self.assertEqual(
                list(selected_data_catalog(config)),
                ["EEG-ONE", "MEG-ONE"],
            )
            with mock.patch(
                "pretrain_config.DATA_CATALOG_PATH",
                catalog_path,
            ):
                automatic = load_pretrain_launch_config(
                    [
                        ROOT / "configs/pretrain/braintokenizer.yaml",
                        overlay_path,
                    ]
                )
            self.assertEqual(
                automatic["invocation"]["data_catalog"],
                catalog,
            )
            overlay["campaign"] = {"data": {"included_datasets": ["UNKNOWN"]}}
            overlay_path.write_text(json.dumps(overlay))
            with self.assertRaises(ConfigError):
                load_pretrain_config(
                    [
                        ROOT / "configs/pretrain/braintokenizer.yaml",
                        overlay_path,
                    ],
                    data_catalog_path=catalog_path,
                )

            catalog_path.write_text(
                "datasets:\n"
                "  DUPLICATE:\n"
                f"    path: {root / 'one'}\n"
                "    signal_type: eeg\n"
                "  DUPLICATE:\n"
                f"    path: {root / 'two'}\n"
                "    signal_type: meg\n"
            )
            with self.assertRaises(ConfigError):
                load_data_catalog(catalog_path)
            catalog_path.write_text(
                "datasets:\n"
                "  INVALID:\n"
                "    path: ''\n"
                "    signal_type: ecog\n"
            )
            with self.assertRaises(ConfigError):
                load_data_catalog(catalog_path)

    def test_catalog_discovery_validates_roots_and_modalities(self) -> None:
        class FakeRaw:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        class FakeAccessor:
            def __init__(self) -> None:
                self.roots: list[tuple[str, str]] = []

            def search_brain_files(
                self,
                root_path: str,
                dataset: str,
            ) -> list[dict[str, str]]:
                self.roots.append((root_path, dataset))
                return [
                    {
                        "path": str(Path(root_path) / f"{dataset}.fif"),
                        "dataset": dataset,
                    }
                ]

            def read_brain_file(
                self,
                path: str,
                preload: bool,
            ) -> FakeRaw:
                del path, preload
                return FakeRaw()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            eeg_root = root / "eeg"
            meg_root = root / "meg"
            eeg_root.mkdir()
            meg_root.mkdir()
            config = {
                "campaign": {"data": {"included_datasets": ["EEG", "MEG"]}},
                "invocation": {
                    "data_catalog": {
                        "EEG": {
                            "path": str(eeg_root),
                            "signal_type": "eeg",
                        },
                        "MEG": {
                            "path": str(meg_root),
                            "signal_type": "meg",
                        },
                    }
                },
            }
            accessor = FakeAccessor()
            with mock.patch(
                "factory.process.infer_signal_type",
                side_effect=["eeg", "meg"],
            ):
                recordings = discover_catalog_recordings(accessor, config)
            self.assertEqual(
                accessor.roots,
                [(str(eeg_root), "EEG"), (str(meg_root), "MEG")],
            )
            self.assertEqual(
                [recording["signal_type"] for recording in recordings],
                ["eeg", "meg"],
            )
            with mock.patch(
                "factory.process.infer_signal_type",
                return_value="eeg",
            ):
                with self.assertRaises(ConfigError):
                    discover_catalog_recordings(accessor, config)
            config["invocation"]["data_catalog"]["MEG"]["path"] = str(
                root / "missing"
            )
            with self.assertRaises(ConfigError):
                discover_catalog_recordings(accessor, config)

    def test_split_excludes_all_nontraining_catalog_datasets(self) -> None:
        rows = [
            {"dataset": "TRAIN", "path": f"train-{index}"}
            for index in range(20)
        ]
        rows.extend(
            [
                {"dataset": "HELD-A", "path": "held-a"},
                {"dataset": "HELD-B", "path": "held-b"},
            ]
        )
        train, validation, test, held_out = split_pretrain_metadata(
            rows,
            {"train": 0.8, "validation": 0.1, "test": 0.1},
            ["TRAIN"],
        )
        split_rows = [*train, *validation, *test]
        self.assertEqual({row["dataset"] for row in split_rows}, {"TRAIN"})
        self.assertEqual(sorted(held_out), ["HELD-A", "HELD-B"])

    def test_dataset_resolution_records_sorted_catalog_identities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for dataset in ("meg_a", "meg_b"):
                (root / dataset).mkdir()
            overlay_path = write_local_overlay(root, "braintokenizer")
            overlay = json.loads(overlay_path.read_text())
            overlay["campaign"] = {"data": {"included_datasets": ["*"]}}
            overlay["invocation"]["data_catalog"] = {
                "meg_a": {
                    "path": str(root / "meg_a"),
                    "signal_type": "meg",
                },
                "meg_b": {
                    "path": str(root / "meg_b"),
                    "signal_type": "meg",
                },
            }
            overlay_path.write_text(json.dumps(overlay))
            config = load_pretrain_config(
                [
                    ROOT / "configs/pretrain/braintokenizer.yaml",
                    overlay_path,
                ]
            )
            write_metadata(config, ["meg_b", "meg_a", "meg_b"])
            resolved = resolve_dataset_identities(config)
        self.assertEqual(
            resolved["campaign"]["data"]["included_datasets"],
            ["meg_a", "meg_b"],
        )
        self.assertNotIn(
            str(root / "meg_a"),
            yaml.safe_dump(resolved["campaign"]),
        )

    def test_dataset_resolution_rejects_empty_and_mismatched_metadata(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_pretrain_config(
                [
                    ROOT / "configs/pretrain/braintokenizer.yaml",
                    write_local_overlay(root, "braintokenizer"),
                ]
            )
            write_metadata(config, ["other"])
            with self.assertRaises(ConfigError):
                resolve_dataset_identities(config)
            info_path = preprocessing_directory(config) / "info.json"
            info_path.write_text("[]")
            with self.assertRaises(ConfigError):
                resolve_dataset_identities(config)

    def test_tiny_and_base_paper_values_and_legacy_absence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tiny = load_pretrain_config(
                [
                    ROOT / "configs/pretrain/brainomni_tiny.yaml",
                    write_local_overlay(root, "brainomni"),
                ]
            )
            base = load_pretrain_config(
                [
                    ROOT / "configs/pretrain/brainomni_base.yaml",
                    write_local_overlay(root, "brainomni"),
                ]
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
                [
                    ROOT / "configs/pretrain/brainomni_tiny.yaml",
                    write_local_overlay(root, "brainomni"),
                ]
            )
            tokenizer_path = root / "tokenizer"
            tokenizer_path.mkdir()
            (tokenizer_path / "model_cfg.json").write_text(
                json.dumps(tokenizer_model)
            )
            (tokenizer_path / "BrainTokenizer.pt").write_bytes(b"weights")
            write_metadata(config, ["test"])
            config = resolve_dataset_identities(config)
            tokenizer_health = CampaignHealth(
                root=tokenizer_path,
                stage="braintokenizer",
                campaign_sha256="a" * 64,
                model_config_sha256="b" * 64,
                model_state_sha256="c" * 64,
                portable_path=tokenizer_path / "BrainTokenizer.pt",
                repaired=False,
            )
            with mock.patch(
                "brainomni.config.ensure_campaign_health",
                return_value=tokenizer_health,
            ):
                trainer_config = BrainOmniTrainerConfig(
                    config,
                    world_size=8,
                )
            model_config = trainer_config.get_model_cfg()
            self.assertEqual(model_config["lm_dim"], 256)
            self.assertEqual(model_config["mask_ratio"], 0.5)
            self.assertEqual(
                trainer_config.tokenizer_identity,
                {
                    "campaign_sha256": "a" * 64,
                    "model_config_sha256": "b" * 64,
                    "model_state_sha256": "c" * 64,
                },
            )
            json_path = root / "brainomni_tiny.json"
            json_path.write_text(json.dumps(config))
            json_config = load_pretrain_config(json_path)
        self.assertEqual(json_config, config)

    def test_portable_export_excludes_invocation_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_pretrain_config(
                [
                    ROOT / "configs/pretrain/braintokenizer.yaml",
                    write_local_overlay(root, "braintokenizer"),
                ]
            )
            write_metadata(config, ["test"])
            config = resolve_dataset_identities(config)
            run_path = root / "run"
            weights_path = root / "weights.pt"
            weights_path.write_bytes(b"weights")
            run_path.mkdir()
            (run_path / "model_cfg.json").write_text("{}")
            (run_path / "pretrain_setting.yaml").write_text("campaign: {}")
            (run_path / "pretrain_setting.json").write_text("{}")
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
        documentation = (ROOT / "docs/pretraining_configuration.md").read_text()
        self.assertIn(
            "docs/pretraining_configuration.md",
            (ROOT / "README.md").read_text(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer = load_pretrain_config(
                [
                    ROOT / "configs/pretrain/braintokenizer.yaml",
                    write_local_overlay(root, "braintokenizer"),
                ]
            )
            brainomni = load_pretrain_config(
                [
                    ROOT / "configs/pretrain/brainomni_tiny.yaml",
                    write_local_overlay(root, "brainomni"),
                ]
            )
        keys = leaf_paths(tokenizer) | leaf_paths(brainomni)
        missing = [
            key
            for key in keys
            if not key.startswith("invocation.data_catalog.")
            and key not in documentation
        ]
        self.assertFalse(missing)
        public_files = [
            ROOT / "docs/pretraining_configuration.md",
            ROOT / "configs/pretrain/braintokenizer.yaml",
            ROOT / "configs/pretrain/brainomni_tiny.yaml",
            ROOT / "configs/pretrain/brainomni_base.yaml",
            ROOT / "configs/data/datasets.yaml",
            ROOT / "braintokenizer/launcher.py",
            ROOT / "brainomni/launcher.py",
            ROOT / "tests/test_pretrain_config.py",
        ]
        template = yaml.safe_load(
            (ROOT / "configs/data/datasets.yaml").read_text()
        )
        self.assertEqual(
            template,
            {
                "datasets": {
                    "DATASET_ID": {
                        "path": None,
                        "signal_type": "eeg",
                    }
                }
            },
        )
        for path in public_files:
            text = path.read_text()
            self.assertNotIn("\n    path: /", text)
            self.assertNotIn("/" + "home" + "/", text)


if __name__ == "__main__":
    unittest.main()

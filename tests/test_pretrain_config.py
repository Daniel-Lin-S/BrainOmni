"""Focused tests for the pre-training configuration contract."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from factory.export_pretrained import export_checkpoint
from pretrain_config import (
    CONFIGURATION_KEYS,
    ConfigError,
    build_deepspeed_config,
    load_pretrain_config,
    write_run_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]


class PretrainConfigTest(unittest.TestCase):
    """Validate resolution, artifacts, and documentation coverage."""

    def test_default_tokenizer_configuration(self) -> None:
        config = load_pretrain_config(ROOT / "configs/pretrain/braintokenizer.yaml")
        self.assertEqual(config["campaign"]["training"]["epochs"], 16)
        self.assertEqual(config["campaign"]["model"]["codebook_size"], 512)
        _, accumulation = build_deepspeed_config(config, world_size=8)
        with tempfile.TemporaryDirectory() as temporary:
            json_path = Path(temporary) / "braintokenizer.json"
            json_path.write_text(json.dumps(config))
            json_config = load_pretrain_config(json_path)
        self.assertEqual(json_config, config)
        self.assertEqual(accumulation, 4)

    def test_local_override_and_cli_override(self) -> None:
        base = ROOT / "configs/pretrain/braintokenizer.yaml"
        with tempfile.TemporaryDirectory() as temporary:
            local_path = Path(temporary) / "local.yaml"
            local_path.write_text("campaign:\n  training:\n    epochs: 18\n")
            config = load_pretrain_config(
                base,
                local_path,
                ["campaign.training.epochs=20"],
            )
        self.assertEqual(config["campaign"]["training"]["epochs"], 20)

    def test_invalid_unknown_key_is_rejected(self) -> None:
        config = load_pretrain_config(ROOT / "configs/pretrain/braintokenizer.yaml")
        invalid = deepcopy(config)
        invalid["campaign"]["unknown"] = True
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            path.write_text(json.dumps(invalid))
            with self.assertRaises(ConfigError):
                load_pretrain_config(path)

    def test_portable_export_excludes_invocation_settings(self) -> None:
        config = load_pretrain_config(ROOT / "configs/pretrain/braintokenizer.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
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

    def test_documentation_mentions_configuration_keys(self) -> None:
        documentation = (ROOT / "docs/pretraining_configuration.md").read_text()
        self.assertFalse([key for key in CONFIGURATION_KEYS if key not in documentation])


if __name__ == "__main__":
    unittest.main()

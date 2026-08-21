"""Regression tests for state-first pre-training terminal logs."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FAKE_FAILURE_STATUS = 23


class LauncherLogLifecycleTest(unittest.TestCase):
    """Validate training launchers move complete and failed attempt logs."""

    def _copy_launchers(self, root: Path) -> None:
        """Copy the launcher scripts into an isolated repository layout."""
        script_directory = root / "script"
        script_directory.mkdir()
        for name in (
            "launch_common.sh",
            "train_braintokenizer.sh",
            "train_brainomni.sh",
        ):
            shutil.copy2(ROOT / "script" / name, script_directory / name)

    def _write_fake_command(self, path: Path, content: str) -> None:
        """Write one executable fake command for a launcher subprocess."""
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)

    def _run_launcher(self, stage: str, outcome: str) -> tuple[Path, int]:
        """Run one isolated launcher and return its root and exit status."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self._copy_launchers(root)
        config_path = root / "config.yaml"
        config_path.write_text("campaign: {}\n", encoding="utf-8")
        command_directory = root / "bin"
        command_directory.mkdir()
        self._write_fake_command(
            command_directory / "nvidia-smi",
            "#!/usr/bin/env bash\nprintf '%s\\n' '0, 1024, 2048, 0'\n",
        )
        self._write_fake_command(
            command_directory / "deepspeed",
            "#!/usr/bin/env bash\n"
            "set -eu\n"
            "[[ -z \"${CUDA_VISIBLE_DEVICES+x}\" ]] || exit 97\n"
            "printf '%s\\n' \"$@\" > \"${FAKE_ARGUMENTS_PATH}\"\n"
            "attempt_directory=$(dirname \"${BRAINOMNI_TERMINAL_LOG_PATH}\")\n"
            "printf 'trainer log\\n' > \"${attempt_directory}/logs.txt\"\n"
            "if [[ \"${FAKE_OUTCOME}\" == failed_after ]]; then\n"
            "    mkdir -p \"${FAKE_CAMPAIGN_ATTEMPT}\"\n"
            "    printf '%s\\n' '{\"state\": \"failed\"}' "
            "> \"${FAKE_CAMPAIGN_ATTEMPT}/status.json\"\n"
            "    printf 'post-campaign failure\\n'\n"
            "    exit \"${FAKE_EXIT_STATUS}\"\n"
            "fi\n"
            "if [[ \"${FAKE_OUTCOME}\" == failed_before ]]; then\n"
            "    printf 'pre-campaign failure\\n'\n"
            "    exit \"${FAKE_EXIT_STATUS}\"\n"
            "fi\n"
            "printf 'training complete\\n'\n",
        )
        environment = os.environ.copy()
        environment["FAKE_CAMPAIGN_ATTEMPT"] = str(
            root / "artifacts" / stage / "campaign" / "attempts" / "attempt"
        )
        environment["FAKE_EXIT_STATUS"] = str(FAKE_FAILURE_STATUS)
        environment["FAKE_OUTCOME"] = outcome
        environment["FAKE_ARGUMENTS_PATH"] = str(root / "arguments.txt")
        environment["CUDA_VISIBLE_DEVICES"] = "7"
        environment["NVIDIA_SMI_BIN"] = str(command_directory / "nvidia-smi")
        environment["PATH"] = f"{command_directory}:{environment['PATH']}"
        result = subprocess.run(
            [
                "bash",
                str(root / "script" / f"train_{stage}.sh"),
                "--num-gpus",
                "1",
                "--config",
                str(config_path),
            ],
            cwd=root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        return root, result.returncode

    def test_launchers_move_attempts_to_terminal_state(self) -> None:
        """Move stage outcomes out of pending and preserve command status."""
        for stage in ("braintokenizer", "brainomni"):
            for outcome in (
                "complete",
                "failed_before",
                "failed_after",
            ):
                with self.subTest(stage=stage, outcome=outcome):
                    root, status = self._run_launcher(stage, outcome)
                    expected_status = (
                        0
                        if outcome == "complete"
                        else FAKE_FAILURE_STATUS
                    )
                    self.assertEqual(status, expected_status)
                    state = (
                        "complete" if outcome == "complete" else "failed"
                    )
                    final_directory = root / "logs" / stage / state
                    attempt_directories = list(final_directory.iterdir())
                    self.assertEqual(len(attempt_directories), 1)
                    attempt_directory = attempt_directories[0]
                    terminal_log_path = attempt_directory / "terminal.log"
                    self.assertTrue(terminal_log_path.is_file())
                    self.assertTrue((attempt_directory / "logs.txt").is_file())
                    arguments = (root / "arguments.txt").read_text(
                        encoding="utf-8"
                    ).splitlines()
                    self.assertIn("--include=localhost:0", arguments)
                    pending_directory = root / "logs" / stage / "pending"
                    self.assertFalse(
                        pending_directory.exists()
                        and any(pending_directory.iterdir())
                    )
                    terminal_log = terminal_log_path.read_text(
                        encoding="utf-8"
                    )
                    if outcome == "failed_after":
                        status_path = (
                            root
                            / "artifacts"
                            / stage
                            / "campaign"
                            / "attempts"
                            / "attempt"
                            / "status.json"
                        )
                        self.assertTrue(status_path.is_file())
                        self.assertIn("post-campaign failure", terminal_log)
                    if outcome == "failed_before":
                        self.assertIn("pre-campaign failure", terminal_log)


if __name__ == "__main__":
    unittest.main()

"""CLI commands, run in-process."""

from __future__ import annotations

import io
import json
import shutil
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from longrun.cli import EXIT_DONE, EXIT_STOPPED, EXIT_USAGE, main

from .helpers import TempDirs

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


class CliTests(TempDirs):
    def setUp(self) -> None:
        super().setUp()
        shutil.copytree(EXAMPLES / "buggy-project", self.workspace, dirs_exist_ok=True)
        self.script = self.state_dir.parent / "fast.json"
        turns = json.loads((EXAMPLES / "fix-bug.script.json").read_text())["turns"]
        self.script.write_text(json.dumps({"turns": [{**t, "delay_s": 0} for t in turns]}))

    def cli(self, *args: str) -> tuple[int, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["--state-dir", str(self.state_dir), *args])
        return code, out.getvalue()

    def run_demo(self) -> None:
        code, out = self.cli(
            "run", "fix it", "--run-id", "r1", "--workspace", str(self.workspace), "--script", str(self.script)
        )
        self.assertEqual(code, EXIT_DONE)
        self.assertIn("run r1: done after 8 steps", out)

    def test_run_status_log_verify(self) -> None:
        self.run_demo()
        code, out = self.cli("status")
        self.assertIn("r1", out)
        self.assertIn("done", out)
        code, out = self.cli("status", "r1")
        self.assertEqual(json.loads(out)["status"], "done")
        code, out = self.cli("log", "r1")
        self.assertIn("tool.denied", out)
        code, out = self.cli("log", "r1", "--json")
        self.assertEqual(json.loads(out.splitlines()[0])["event"], "run.start")
        code, out = self.cli("verify", "r1")
        self.assertEqual(code, EXIT_DONE)

    def test_resume_of_finished_run_reports_it(self) -> None:
        self.run_demo()
        code, out = self.cli("resume", "r1", "--script", str(self.script))
        self.assertEqual(code, EXIT_DONE)
        self.assertIn("done after 8 steps", out)

    def test_max_steps_exit_code(self) -> None:
        code, _ = self.cli(
            "run", "fix it", "--workspace", str(self.workspace), "--script", str(self.script), "--max-steps", "2"
        )
        self.assertEqual(code, EXIT_STOPPED)

    def test_usage_errors(self) -> None:
        self.assertEqual(self.cli("resume", "missing")[0], EXIT_USAGE)
        inside = self.workspace / ".state"
        code, _ = self.cli(
            "--state-dir", str(inside), "run", "x", "--workspace", str(self.workspace), "--script", str(self.script)
        )
        self.assertEqual(code, EXIT_USAGE)  # state dir inside the jail is refused

    def test_empty_status(self) -> None:
        self.assertIn("no runs yet", self.cli("status")[1])


if __name__ == "__main__":
    unittest.main()

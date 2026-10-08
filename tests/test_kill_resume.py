"""End to end: SIGKILL the real CLI process mid-run, then resume it.

This is the property the whole harness exists for, so it is tested with a
real ``kill -9`` against a real process rather than a simulated exception.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path

from longrun.audit import read_records, verify

from .helpers import TempDirs

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SCRIPT = EXAMPLES / "fix-bug.script.json"


@unittest.skipUnless(os.name == "posix", "uses SIGKILL")
class KillResumeTests(TempDirs):
    def setUp(self) -> None:
        super().setUp()
        shutil.copytree(EXAMPLES / "buggy-project", self.workspace, dirs_exist_ok=True)
        self.env = {**os.environ, "LONGRUN_STATE_DIR": str(self.state_dir)}
        self.checkpoint = self.state_dir / "runs" / "kill-test" / "checkpoint.json"

    def longrun(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "longrun", *args],
            capture_output=True,
            text=True,
            env=self.env,
            timeout=60,
        )

    def wait_for_step(self, step: int, timeout: float = 30) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if json.loads(self.checkpoint.read_text())["step"] >= step:
                    return
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(0.05)
        self.fail(f"run never reached step {step}")

    def test_sigkill_then_resume_finishes_the_job(self) -> None:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "longrun",
                "run",
                "Fix the failing test",
                "--run-id",
                "kill-test",
                "--workspace",
                str(self.workspace),
                "--script",
                str(SCRIPT),
            ],
            env=self.env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.wait_for_step(3)
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
        self.assertEqual(proc.returncode, -signal.SIGKILL)

        killed_at = json.loads(self.checkpoint.read_text())
        self.assertEqual(killed_at["status"], "running")
        self.assertIn("- 1", (self.workspace / "calc.py").read_text())  # bug not fixed yet

        resumed = self.longrun("resume", "kill-test", "--script", str(SCRIPT))
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertIn("done after 8 steps", resumed.stdout)

        self.assertNotIn("- 1", (self.workspace / "calc.py").read_text())
        tests = subprocess.run([sys.executable, "-m", "unittest"], cwd=self.workspace, capture_output=True)
        self.assertEqual(tests.returncode, 0)

        audit = self.state_dir / "runs" / "kill-test" / "audit.jsonl"
        self.assertTrue(verify(audit).ok)
        events = [r["event"] for r in read_records(audit)]
        self.assertEqual(events.count("run.start"), 1)
        self.assertEqual(events.count("run.resume"), 1)
        self.assertEqual(events.count("tool.denied"), 1)  # the curl to prod

        verify_cli = self.longrun("verify", "kill-test")
        self.assertEqual(verify_cli.returncode, 0)
        self.assertIn("[ok ] kill-test", verify_cli.stdout)

    def test_tampered_audit_log_fails_verify(self) -> None:
        run = self.longrun(
            "run",
            "Fix the failing test",
            "--run-id",
            "kill-test",
            "--workspace",
            str(self.workspace),
            "--script",
            str(SCRIPT),
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        audit = self.state_dir / "runs" / "kill-test" / "audit.jsonl"
        audit.write_text(audit.read_text().replace("tool.denied", "tool.end", 1))  # hide the evidence
        result = self.longrun("verify", "kill-test")
        self.assertEqual(result.returncode, 1)
        self.assertIn("BAD", result.stdout)


if __name__ == "__main__":
    unittest.main()

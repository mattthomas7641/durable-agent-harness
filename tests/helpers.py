from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from longrun.model import ModelTurn, ScriptedModel


class Crash(BaseException):
    """Stands in for the process dying: not caught by any harness error handling."""


class TempDirs(unittest.TestCase):
    """Gives each test a fresh workspace and a separate state directory."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name).resolve()
        self.workspace = base / "workspace"
        self.state_dir = base / "state"
        self.outside = base / "outside"
        for d in (self.workspace, self.state_dir, self.outside):
            d.mkdir()
        self.addCleanup(self._tmp.cleanup)


class CrashingModel:
    """Wraps a model and 'kills the process' right before returning turn ``crash_at``."""

    def __init__(self, inner: ScriptedModel, crash_at: int) -> None:
        self.inner = inner
        self.crash_at = crash_at
        self.name = inner.name
        self.calls = 0

    def next_turn(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn:
        self.calls += 1
        index = sum(1 for m in messages if m["role"] == "assistant")
        if index == self.crash_at:
            raise Crash(f"simulated crash before turn {index}")
        return self.inner.next_turn(system, messages, tools)


def call(name: str, **args: Any) -> dict[str, Any]:
    return {"name": name, "input": args}

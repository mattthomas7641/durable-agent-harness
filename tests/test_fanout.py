"""Subagent fan-out through the Harness, including crash/resume across the fan-out."""

from __future__ import annotations

import json
import threading
import unittest
from collections import Counter
from typing import Any

from longrun.audit import read_records, verify
from longrun.model import Model, ModelTurn, ScriptedModel
from longrun.runtime import Harness, HarnessConfig

from .helpers import Crash, TempDirs, call


class ChildModel:
    """Writes ``<task>.txt`` then reports done. Can be told to crash once on one task."""

    name = "child"

    def __init__(self, calls: Counter[str], crash_task: list[str]) -> None:
        self.calls = calls
        self.crash_task = crash_task
        self.lock = threading.Lock()

    def next_turn(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn:
        task = messages[0]["content"]
        with self.lock:
            self.calls[task] += 1
            if self.crash_task and self.crash_task[0] == task and len(messages) == 3:
                self.crash_task.clear()
                raise Crash(f"subagent {task} died")
        if len(messages) == 1:
            return ModelTurn(
                [
                    {
                        "type": "tool_use",
                        "id": f"toolu_{task}",
                        "name": "write_file",
                        "input": {"path": f"{task}.txt", "content": task},
                    }
                ],
                "tool_use",
            )
        return ModelTurn([{"type": "text", "text": f"wrote {task}.txt"}], "end_turn")


PARENT = [
    {"tool_calls": [call("spawn_subagents", tasks=["a", "b", "c"])]},
    {"text": "All three parts are done."},
]


class FanOutTests(TempDirs):
    def setUp(self) -> None:
        super().setUp()
        self.child_calls: Counter[str] = Counter()
        self.crash_task: list[str] = []

    def harness(self, max_subagents: int = 4) -> Harness:
        child = ChildModel(self.child_calls, self.crash_task)

        def factory(role: str) -> Model:
            return child if role == "subagent" else ScriptedModel(PARENT)

        config = HarnessConfig(workspace=self.workspace, state_dir=self.state_dir, max_subagents=max_subagents)
        return Harness(config, factory)

    def test_fan_out_runs_children_and_returns_summaries(self) -> None:
        harness = self.harness()
        state = harness.start("build it", run_id="p")
        self.assertEqual(state.status, "done")
        for task in "abc":
            self.assertEqual((self.workspace / f"{task}.txt").read_text(), task)

        summary = json.loads(state.messages[2]["content"][0]["content"])
        self.assertEqual([s["status"] for s in summary], ["done"] * 3)
        self.assertEqual([s["summary"] for s in summary], ["wrote a.txt", "wrote b.txt", "wrote c.txt"])

        children = [s for s in harness.store.list_runs() if s.parent_id == "p"]
        self.assertEqual(len(children), 3)
        for child in children:
            self.assertTrue(child.run_id.startswith("p.sub-"))
            self.assertTrue(verify(harness.audit_path(child.run_id)).ok)

    def test_subagents_cannot_spawn_subagents(self) -> None:
        harness = self.harness()
        harness.start("build it", run_id="p")
        child_id = next(s.run_id for s in harness.store.list_runs() if s.parent_id == "p")
        start = next(r for r in read_records(harness.audit_path(child_id)) if r["event"] == "run.start")
        self.assertNotIn("spawn_subagents", start["data"]["tools"])
        parent_start = next(r for r in read_records(harness.audit_path("p")) if r["event"] == "run.start")
        self.assertIn("spawn_subagents", parent_start["data"]["tools"])

    def test_fan_out_limit_is_enforced(self) -> None:
        harness = self.harness(max_subagents=2)
        state = harness.start("build it", run_id="p")
        result = state.messages[2]["content"][0]
        self.assertTrue(result["is_error"])
        self.assertIn("invalid input", result["content"])
        self.assertEqual(sum(self.child_calls.values()), 0)

    def test_crash_mid_fan_out_resumes_only_unfinished_children(self) -> None:
        self.crash_task.append("b")
        harness = self.harness()
        with self.assertRaises(Crash):
            harness.start("build it", run_id="p")
        self.assertFalse(harness.store.load("p").finished)
        calls_before = dict(self.child_calls)  # a and c finished (2 turns each), b crashed on turn 2

        state = self.harness().resume("p")
        self.assertEqual(state.status, "done")
        self.assertEqual(self.child_calls["a"], calls_before["a"])  # finished children not re-run
        self.assertEqual(self.child_calls["c"], calls_before["c"])
        self.assertEqual(self.child_calls["b"], calls_before["b"] + 1)  # b picked up at its last step
        summary = json.loads(state.messages[2]["content"][0]["content"])
        self.assertEqual([s["status"] for s in summary], ["done"] * 3)


if __name__ == "__main__":
    unittest.main()

"""The agent loop: checkpoint/resume semantics under simulated crashes."""

from __future__ import annotations

import json
import re
import unittest
from itertools import pairwise
from typing import Any

from longrun.agent import INTERRUPTED, TRUNCATED, Agent
from longrun.audit import AuditLog, verify
from longrun.checkpoint import CheckpointStore, RunState
from longrun.isolation import Sandbox
from longrun.memory import MemoryStore
from longrun.model import Model, ScriptedModel
from longrun.tools import Tool, ToolContext, ToolRegistry, builtin_tools

from .helpers import Crash, CrashingModel, TempDirs, call

OBJ: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}


class AgentTestCase(TempDirs):
    def setUp(self) -> None:
        super().setUp()
        self.store = CheckpointStore(self.state_dir)
        self.sandbox = Sandbox(self.workspace)
        self.side_effects: list[str] = []
        self.crash_in: str | None = None  # name of a tool that "kills the process" mid-call, once

    def registry(self) -> ToolRegistry:
        """Built-in tools plus two test tools that count their real executions."""

        def maybe_crash(name: str) -> None:
            if self.crash_in == name:
                self.crash_in = None
                raise Crash(f"died inside {name}")

        def deploy(args: dict[str, Any], ctx: ToolContext) -> str:
            self.side_effects.append(ctx.call_id)
            maybe_crash("deploy")
            return "deployed"

        def probe(args: dict[str, Any], ctx: ToolContext) -> str:
            self.side_effects.append(f"probe:{ctx.call_id}")
            maybe_crash("probe")
            return "healthy"

        tools = ToolRegistry(builtin_tools(self.sandbox, MemoryStore(self.state_dir / "memory.json")))
        tools.register(Tool("deploy", "side effect", OBJ, deploy, idempotent=False))
        tools.register(Tool("probe", "read only", OBJ, probe, idempotent=True))
        return tools

    def agent(self, model: Model, run_id: str = "r1", max_steps: int = 50) -> Agent:
        audit = AuditLog(self.store.run_dir(run_id) / "audit.jsonl", run_id)
        return Agent(model=model, tools=self.registry(), store=self.store, audit=audit, max_steps=max_steps)

    def start(self, model: Model, run_id: str = "r1", max_steps: int = 50) -> RunState:
        state = RunState(run_id=run_id, task="ship it")
        return self.agent(model, run_id, max_steps).run(state)

    def resume(self, model: Model, run_id: str = "r1") -> RunState:
        return self.agent(model, run_id).run(self.store.load(run_id))

    def audit_events(self, run_id: str = "r1") -> list[str]:
        return [r["event"] for r in AuditLog(self.store.run_dir(run_id) / "audit.jsonl", run_id).records()]

    def assert_well_formed(self, state: RunState) -> None:
        """Roles alternate and every tool_use is answered by the very next message."""
        roles = [m["role"] for m in state.messages]
        self.assertEqual(roles[0], "user")
        for a, b in pairwise(roles):
            self.assertNotEqual(a, b, roles)
        for i, msg in enumerate(state.messages):
            if msg["role"] != "assistant" or not isinstance(msg["content"], list):
                continue
            uses = [b["id"] for b in msg["content"] if b["type"] == "tool_use"]
            if uses:
                answers = [b["tool_use_id"] for b in state.messages[i + 1]["content"]]
                self.assertEqual(uses, answers)
        self.assertTrue(verify(self.store.run_dir(state.run_id) / "audit.jsonl").ok)


SCRIPT = [
    {"tool_calls": [call("write_file", path="app.py", content="print('v1')\n")]},
    {"tool_calls": [call("deploy")]},
    {"tool_calls": [call("run_command", argv=["python", "app.py"]), call("probe")]},
    {"tool_calls": [call("deploy")]},
    {"text": "Shipped v1 and verified it runs."},
]


class AgentLoopTests(AgentTestCase):
    def test_scripted_run_completes(self) -> None:
        state = self.start(ScriptedModel(SCRIPT))
        self.assertEqual(state.status, "done")
        self.assertEqual(state.step, 5)
        self.assertEqual(state.final_text, "Shipped v1 and verified it runs.")
        self.assertEqual((self.workspace / "app.py").read_text(), "print('v1')\n")
        self.assertIn("v1", state.messages[6]["content"][0]["content"])  # run_command output
        self.assertEqual(len(self.side_effects), 3)
        self.assert_well_formed(state)
        self.assertEqual(self.audit_events()[0], "run.start")
        self.assertEqual(self.audit_events()[-1], "run.end")

    def test_denials_are_returned_to_the_model_and_audited(self) -> None:
        script = [
            {"tool_calls": [call("read_file", path="../../etc/passwd"), call("run_command", argv=["curl", "prod"])]},
            {"text": "Couldn't, both were denied."},
        ]
        state = self.start(ScriptedModel(script))
        results = state.messages[2]["content"]
        self.assertTrue(all(r["is_error"] for r in results))
        self.assertTrue(all(r["content"].startswith("denied:") for r in results))
        self.assertEqual(self.audit_events().count("tool.denied"), 2)
        self.assertEqual(state.status, "done")

    def test_max_steps_stops_a_runaway_agent(self) -> None:
        state = self.start(ScriptedModel([{"tool_calls": [call("probe")]}] * 100), max_steps=3)
        self.assertEqual(state.status, "max_steps")
        self.assertEqual(state.step, 3)
        self.assert_well_formed(state)

    def test_refusal_fails_the_run(self) -> None:
        state = self.start(ScriptedModel([{"text": "", "stop_reason": "refusal"}]))
        self.assertEqual(state.status, "failed")
        self.assertIn("refusal", state.error)

    def test_truncated_tool_inputs_are_not_executed(self) -> None:
        script = [{"tool_calls": [call("deploy")], "stop_reason": "max_tokens"}, {"text": "ok"}]
        state = self.start(ScriptedModel(script))
        self.assertEqual(self.side_effects, [])
        self.assertEqual(state.messages[2]["content"][0]["content"], TRUNCATED)

    def test_truncated_text_asks_model_to_continue(self) -> None:
        state = self.start(ScriptedModel([{"text": "partial", "stop_reason": "max_tokens"}, {"text": "done"}]))
        self.assertEqual(state.final_text, "done")
        self.assertEqual(state.messages[2]["role"], "user")


def transcript(state: RunState) -> str:
    """The conversation with command timings blanked out, for run-to-run comparison."""
    return re.sub(r"\d+\.\d+s\]", "Xs]", json.dumps(state.messages))


class CrashResumeTests(AgentTestCase):
    def test_resume_after_crash_between_steps_matches_clean_run(self) -> None:
        clean = self.start(ScriptedModel(SCRIPT), run_id="clean")
        clean_effects = len(self.side_effects)
        self.side_effects.clear()

        for crash_at in range(1, len(SCRIPT)):
            with self.subTest(crash_at=crash_at):
                run_id = f"crash-{crash_at}"
                self.side_effects.clear()
                with self.assertRaises(Crash):
                    self.start(CrashingModel(ScriptedModel(SCRIPT), crash_at), run_id=run_id)
                self.assertFalse(self.store.load(run_id).finished)

                resumed = self.resume(ScriptedModel(SCRIPT), run_id=run_id)
                self.assertEqual(resumed.status, "done")
                self.assertEqual(transcript(resumed), transcript(clean))  # identical conversation
                self.assertEqual(len(self.side_effects), clean_effects)  # nothing ran twice
                self.assert_well_formed(resumed)
                self.assertIn("run.resume", self.audit_events(run_id))

    def test_side_effecting_tool_killed_mid_call_is_not_replayed(self) -> None:
        script = [{"tool_calls": [call("probe"), call("deploy"), call("probe")]}, {"text": "done"}]
        self.crash_in = "deploy"
        with self.assertRaises(Crash):
            self.start(ScriptedModel(script))
        self.assertEqual(self.side_effects, ["probe:toolu_000_0", "toolu_000_1"])  # deploy died mid-call

        resumed = self.resume(ScriptedModel(script))
        self.assertEqual(resumed.status, "done")
        results = resumed.messages[2]["content"]
        self.assertEqual(results[0]["content"], "healthy")  # saved before the crash, not re-run
        self.assertEqual(results[1]["content"], INTERRUPTED)  # unknown outcome: reported, not replayed
        self.assertTrue(results[1]["is_error"])
        self.assertEqual(results[2]["content"], "healthy")
        self.assertEqual(self.side_effects, ["probe:toolu_000_0", "toolu_000_1", "probe:toolu_000_2"])
        self.assertIn("tool.interrupted", self.audit_events())
        self.assert_well_formed(resumed)

    def test_idempotent_tool_killed_mid_call_is_rerun(self) -> None:
        script = [{"tool_calls": [call("probe")]}, {"text": "done"}]
        self.crash_in = "probe"
        with self.assertRaises(Crash):
            self.start(ScriptedModel(script))
        resumed = self.resume(ScriptedModel(script))
        self.assertEqual(resumed.messages[2]["content"][0]["content"], "healthy")
        self.assertEqual(len(self.side_effects), 2)  # ran, died, ran again
        self.assertNotIn("tool.interrupted", self.audit_events())

    def test_resuming_a_finished_run_is_a_no_op(self) -> None:
        state = self.start(ScriptedModel(SCRIPT))
        again = self.resume(ScriptedModel(SCRIPT))
        self.assertEqual(again.messages, state.messages)


if __name__ == "__main__":
    unittest.main()

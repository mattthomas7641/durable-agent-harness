"""The agent loop: model asks for a tool -> path jail -> checkpoint -> audit log.

The loop is a small state machine over ``RunState``::

    no pending calls  --model turn-->  pending calls  --each tool-->  results
          ^                                                             |
          +---------------- tool_result message appended <--------------+

The state is checkpointed after the model turn and after *every* tool call,
so the process can be killed at any instant and ``run()`` picks up exactly
where it stopped. The one thing that can't be known after a crash is whether a
side-effecting tool that was mid-flight actually finished; such calls are
reported back to the model as ``interrupted`` instead of being re-executed.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable
from typing import Any

from .audit import AuditLog, redact
from .checkpoint import CheckpointStore, RunState
from .model import Model, ModelTurn
from .tools import ToolContext, ToolRegistry, ToolResult

log = logging.getLogger("longrun")

INTERRUPTED = (
    "interrupted: the harness stopped while this call was running, so its side effects are "
    "unknown. Inspect the workspace before retrying."
)
TRUNCATED = "not executed: your response hit max_tokens before this tool input was complete. Retry with less input."
CONTINUE = "Your previous response was cut off by the output token limit. Continue from where you stopped."
PREVIEW_CHARS = 500


class Agent:
    def __init__(
        self,
        *,
        model: Model,
        tools: ToolRegistry,
        store: CheckpointStore,
        audit: AuditLog,
        max_steps: int = 200,
    ) -> None:
        self.model = model
        self.tools = tools
        self.store = store
        self.audit = audit
        self.max_steps = max_steps
        self._in_flight: set[str] = set()

    # ------------------------------------------------------------------ driver

    def run(self, state: RunState) -> RunState:
        resuming = bool(state.messages)
        if resuming:
            self._in_flight = in_flight_calls(self.audit.records())
            self.audit.append(
                "run.resume", step=state.step, pending=len(state.pending), in_flight=sorted(self._in_flight)
            )
            log.info("[%s] resuming at step %d", state.run_id, state.step)
        else:
            state.messages = [{"role": "user", "content": state.task}]
            self.audit.append(
                "run.start", task=state.task, model=self.model.name, tools=self.tools.names(), max_steps=self.max_steps
            )
            self.store.save(state)

        while not state.finished:
            if state.pending:
                self._finish_tool_batch(state)
            elif state.step >= self.max_steps:
                state.status = "max_steps"
                state.error = f"stopped after {self.max_steps} model turns"
                self.store.save(state)
            else:
                self._model_step(state)

        self.audit.append("run.end", status=state.status, steps=state.step, usage=state.usage, error=state.error)
        log.info("[%s] %s after %d steps", state.run_id, state.status, state.step)
        return state

    # ------------------------------------------------------------- model turn

    def _model_step(self, state: RunState) -> None:
        self.audit.append("model.request", step=state.step, messages=len(state.messages))
        turn: ModelTurn = self.model.next_turn(state.system, state.messages, self.tools.specs())
        state.step += 1
        state.add_usage(turn.usage)
        state.messages.append({"role": "assistant", "content": turn.content})
        calls = turn.tool_calls
        self.audit.append(
            "model.response",
            step=state.step,
            stop_reason=turn.stop_reason,
            tool_calls=[c["name"] for c in calls],
            usage=turn.usage,
        )
        log.info("[%s] step %d: %s %s", state.run_id, state.step, turn.stop_reason, ", ".join(c["name"] for c in calls))

        if turn.stop_reason == "refusal":
            state.status, state.error = "failed", "model declined the request (stop_reason=refusal)"
        elif calls:
            state.pending, state.results = calls, {}
            if turn.stop_reason == "max_tokens":
                # Inputs may be cut off mid-JSON; never execute them.
                for call in calls:
                    state.results[call["id"]] = _result_block(call["id"], ToolResult(TRUNCATED, is_error=True))
        elif turn.stop_reason == "max_tokens":
            state.messages.append({"role": "user", "content": CONTINUE})
        else:
            state.status, state.final_text = "done", turn.text
        self.store.save(state)

    # ------------------------------------------------------------- tool calls

    def _finish_tool_batch(self, state: RunState) -> None:
        for call in state.pending:
            call_id = call["id"]
            if call_id in state.results:
                continue  # answered before a crash; checkpoint already has it
            self._execute(state, call)

        # All results go back in ONE user message, in the order they were requested.
        state.messages.append({"role": "user", "content": [state.results[c["id"]] for c in state.pending]})
        state.pending, state.results = [], {}
        self._in_flight.clear()
        self.store.save(state)

    def _execute(self, state: RunState, call: dict[str, Any]) -> None:
        call_id, name, args = call["id"], call["name"], call.get("input", {})
        tool = self.tools.get(name)
        if call_id in self._in_flight and not (tool and tool.idempotent):
            state.results[call_id] = _result_block(call_id, ToolResult(INTERRUPTED, is_error=True))
            self.store.save(state)
            self.audit.append("tool.interrupted", call_id=call_id, tool=name)
            return

        self.audit.append("tool.start", call_id=call_id, tool=name, input=redact(args))
        result = self.tools.execute(name, args, ToolContext(run_id=state.run_id, call_id=call_id))
        # Checkpoint BEFORE closing the audit record. A crash in between leaves an
        # unclosed tool.start, but the result is already saved, so it's never re-run.
        state.results[call_id] = _result_block(call_id, result)
        self.store.save(state)
        self.audit.append(
            "tool.denied" if result.denied else "tool.end",
            call_id=call_id,
            tool=name,
            is_error=result.is_error,
            output_sha256=hashlib.sha256(result.content.encode()).hexdigest(),
            output_preview=result.content[:PREVIEW_CHARS],
        )


def in_flight_calls(records: Iterable[dict[str, Any]]) -> set[str]:
    """Tool calls that were started but never finished (the process died mid-call)."""
    open_calls: set[str] = set()
    for record in records:
        call_id = record["data"].get("call_id")
        if record["event"] == "tool.start":
            open_calls.add(call_id)
        elif record["event"] in ("tool.end", "tool.denied", "tool.interrupted"):
            open_calls.discard(call_id)
    return open_calls


def _result_block(call_id: str, result: ToolResult) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": call_id, "content": result.content}
    if result.is_error:
        block["is_error"] = True
    return block

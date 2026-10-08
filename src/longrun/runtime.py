"""Wiring: builds the sandbox, stores, tools and agents for a run, and fans out subagents.

Fan-out is checkpointed the same way as everything else. Child run ids are
derived from the parent's ``tool_use_id``, so if the parent is killed while
children are working, resuming the parent re-enters the same tool call, finds
the children's checkpoints, skips the ones that finished, and resumes the rest.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agent import Agent
from .audit import AuditLog
from .checkpoint import CheckpointStore, RunState
from .isolation import ALLOWED_ARGV, Sandbox
from .memory import MemoryStore
from .model import Model
from .tools import Tool, ToolContext, ToolRegistry, builtin_tools

ModelFactory = Callable[[str], Model]
"""Called with ``"agent"`` or ``"subagent"``; lets subagents use a cheaper model or effort."""

SYSTEM_PROMPT = """\
You are a long-running coding agent working inside a sandboxed workspace.

How this environment works:
- All paths are relative to the workspace root. Paths outside it, secrets files, and .git are denied.
- run_command takes an argv list (no shell) and only allow-listed programs run. The environment is
  scrubbed: there are no credentials and you must not try to reach production systems.
- A denial is a policy decision, not a bug. Adapt your approach rather than retrying the same call.
- Your progress is checkpointed after every step and every tool call is audited. You may be stopped
  and resumed at any time. If a tool result says "interrupted", inspect the workspace before retrying.
- Use `remember` for durable facts a future run will need (build/test commands, conventions, open
  issues). Don't store secrets.

Work in small, verifiable steps and run the tests after changes. When the task is complete, reply
with a short summary of what changed and how you verified it.
{role}
Long-term memory from earlier runs:
{memory}
"""

PARENT_ROLE = """
You can delegate independent sub-tasks with `spawn_subagents`. Each subagent gets the same tools
(except spawning) and works in the same workspace in parallel, so give them non-overlapping files.
"""

CHILD_ROLE = """
You are a subagent working on one part of a larger task. Other subagents may be editing other files
in the same workspace concurrently: only touch files that your task is about. Finish with a concise
summary; it is returned to the agent that spawned you.
"""


@dataclass
class HarnessConfig:
    workspace: Path
    state_dir: Path
    max_steps: int = 200
    subagent_max_steps: int = 60
    max_subagents: int = 4
    allowed_argv: Sequence[Sequence[str]] = ALLOWED_ARGV
    launcher: Sequence[str] = ()
    command_timeout_s: float = 120.0
    extra_path: Sequence[str] = field(default_factory=tuple)


class Harness:
    def __init__(self, config: HarnessConfig, model_factory: ModelFactory) -> None:
        self.config = config
        self.model_factory = model_factory
        workspace = Path(config.workspace).resolve()
        state_dir = Path(config.state_dir).resolve()
        if state_dir.is_relative_to(workspace):
            # Otherwise the agent could rewrite its own checkpoints and audit log.
            raise ValueError("state_dir must be outside the workspace")
        self.sandbox = Sandbox(
            workspace,
            allowed_argv=config.allowed_argv,
            launcher=config.launcher,
            timeout_s=config.command_timeout_s,
            extra_path=config.extra_path,
        )
        self.store = CheckpointStore(state_dir)
        self.memory = MemoryStore(state_dir / "memory.json")

    # ---------------------------------------------------------------- public

    @staticmethod
    def new_run_id() -> str:
        return f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"

    def start(self, task: str, run_id: str | None = None) -> RunState:
        state = RunState(run_id=run_id or self.new_run_id(), task=task)
        state.system = self._system_prompt(child=False)
        state.workspace = str(self.sandbox.root)
        self.store.save(state)
        return self._drive(state)

    def resume(self, run_id: str) -> RunState:
        state = self.store.load(run_id)
        if state.finished:
            return state
        return self._drive(state)

    def audit_path(self, run_id: str) -> Path:
        return self.store.run_dir(run_id) / "audit.jsonl"

    def mcp_tools(self) -> ToolRegistry:
        """Tools exposed over MCP: the same jailed tools, without fan-out."""
        return ToolRegistry(builtin_tools(self.sandbox, self.memory))

    # --------------------------------------------------------------- private

    def _drive(self, state: RunState) -> RunState:
        is_child = state.parent_id is not None
        tools = ToolRegistry(builtin_tools(self.sandbox, self.memory))
        if not is_child:
            tools.register(self._spawn_tool())
        agent = Agent(
            model=self.model_factory("subagent" if is_child else "agent"),
            tools=tools,
            store=self.store,
            audit=AuditLog(self.audit_path(state.run_id), state.run_id),
            max_steps=self.config.subagent_max_steps if is_child else self.config.max_steps,
        )
        return agent.run(state)

    def _system_prompt(self, *, child: bool) -> str:
        return SYSTEM_PROMPT.format(
            role=CHILD_ROLE if child else PARENT_ROLE,
            memory=self.memory.render() or "(none yet)",
        )

    def _spawn_tool(self) -> Tool:
        limit = self.config.max_subagents

        def spawn_subagents(args: dict[str, Any], ctx: ToolContext) -> str:
            tasks: list[str] = args["tasks"]
            tag = hashlib.sha256(ctx.call_id.encode()).hexdigest()[:8]
            child_ids = [f"{ctx.run_id}.sub-{tag}-{i}" for i in range(len(tasks))]

            def run_child(index: int) -> RunState:
                child_id = child_ids[index]
                if self.store.exists(child_id):
                    child = self.store.load(child_id)  # resumed parent: reuse prior progress
                else:
                    child = RunState(run_id=child_id, task=tasks[index], parent_id=ctx.run_id)
                    child.system = self._system_prompt(child=True)
                    child.workspace = str(self.sandbox.root)
                    self.store.save(child)
                return child if child.finished else self._drive(child)

            with ThreadPoolExecutor(max_workers=min(limit, len(tasks)), thread_name_prefix="subagent") as pool:
                children = list(pool.map(run_child, range(len(tasks))))

            return json.dumps(
                [
                    {
                        "run_id": c.run_id,
                        "task": c.task,
                        "status": c.status,
                        "summary": c.final_text or c.error,
                    }
                    for c in children
                ],
                indent=1,
            )

        return Tool(
            name="spawn_subagents",
            description=(
                "Run independent sub-tasks in parallel, each in its own checkpointed agent. "
                f"At most {limit} tasks per call. Returns each subagent's status and summary."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 4000},
                        "minItems": 1,
                        "maxItems": limit,
                        "description": "Self-contained instructions, one per subagent.",
                    }
                },
                "required": ["tasks"],
                "additionalProperties": False,
            },
            handler=spawn_subagents,
            # Safe to re-enter after a crash: children resume from their own checkpoints.
            idempotent=True,
        )

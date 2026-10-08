"""Durable run state, saved after every step.

A checkpoint is the *entire* state needed to continue a run: the conversation,
tool calls that were requested but not yet answered, and results gathered so
far for the current batch. Saves are atomic (write temp file, fsync,
``os.replace``, fsync directory), so a ``kill -9`` at any instant leaves either
the previous checkpoint or the new one on disk, never a half-written file.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .errors import CheckpointError

SCHEMA_VERSION = 1

Status = Literal["running", "done", "failed", "max_steps"]


@dataclass
class RunState:
    run_id: str
    task: str
    workspace: str = ""
    system: str = ""
    """System prompt, frozen at start so a resumed run sees exactly the same prefix."""
    status: Status = "running"
    step: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    pending: list[dict[str, Any]] = field(default_factory=list)
    """``tool_use`` blocks from the last model turn that still need answers."""
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    """``tool_use_id -> tool_result`` for the pending batch, filled in one at a time."""
    final_text: str = ""
    error: str = ""
    parent_id: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    schema_version: int = SCHEMA_VERSION

    @property
    def finished(self) -> bool:
        return self.status != "running"

    def add_usage(self, usage: dict[str, int]) -> None:
        for key, value in usage.items():
            self.usage[key] = self.usage.get(key, 0) + int(value or 0)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunState:
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise CheckpointError(f"unsupported checkpoint schema {version!r} (expected {SCHEMA_VERSION})")
        return cls(**data)


class CheckpointStore:
    """One directory per run under ``state_dir/runs/<run_id>/``."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self.runs_dir = self.state_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        if not run_id or "/" in run_id or run_id.startswith("."):
            raise CheckpointError(f"invalid run id: {run_id!r}")
        return self.runs_dir / run_id

    def path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "checkpoint.json"

    def exists(self, run_id: str) -> bool:
        return self.path(run_id).exists()

    def save(self, state: RunState) -> None:
        state.updated_at = time.time()
        directory = self.run_dir(state.run_id)
        directory.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(state), ensure_ascii=False, indent=1).encode()
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".checkpoint.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path(state.run_id))
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        _fsync_dir(directory)

    def load(self, run_id: str) -> RunState:
        path = self.path(run_id)
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            raise CheckpointError(f"no checkpoint for run {run_id!r}") from None
        except json.JSONDecodeError as exc:
            raise CheckpointError(f"corrupt checkpoint {path}: {exc}") from exc
        return RunState.from_dict(data)

    def list_runs(self) -> list[RunState]:
        states = []
        for child in sorted(self.runs_dir.iterdir()):
            if (child / "checkpoint.json").exists():
                try:
                    states.append(self.load(child.name))
                except CheckpointError:
                    continue
        return sorted(states, key=lambda s: s.created_at)


def _fsync_dir(directory: Path) -> None:
    """Make the rename itself durable (no-op where directories can't be opened)."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)

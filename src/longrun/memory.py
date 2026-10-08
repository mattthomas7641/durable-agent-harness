"""Long-term memory shared across runs and subagents.

A small key/value store the agent writes to deliberately (``remember``) and
reads back (``recall``). It lives in the state directory, outside the
workspace jail, so the agent can only change it through these tools.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

MAX_KEY_CHARS = 128
MAX_VALUE_CHARS = 4000
MAX_ENTRIES = 500


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self, data: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".memory.", suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    def remember(self, key: str, value: str, run_id: str) -> None:
        key = key.strip()
        if not key or len(key) > MAX_KEY_CHARS:
            raise ValueError(f"key must be 1-{MAX_KEY_CHARS} characters")
        if len(value) > MAX_VALUE_CHARS:
            raise ValueError(f"value must be at most {MAX_VALUE_CHARS} characters")
        with self._lock:
            data = self._load()
            if key not in data and len(data) >= MAX_ENTRIES:
                raise ValueError(f"memory is full ({MAX_ENTRIES} entries); overwrite or forget a key")
            data[key] = {"value": value, "run_id": run_id, "updated_at": time.time()}
            self._save(data)

    def forget(self, key: str) -> bool:
        with self._lock:
            data = self._load()
            existed = data.pop(key, None) is not None
            if existed:
                self._save(data)
            return existed

    def recall(self, query: str = "") -> dict[str, str]:
        """Entries whose key or value contains ``query`` (case-insensitive)."""
        needle = query.lower()
        with self._lock:
            data = self._load()
        return {k: v["value"] for k, v in sorted(data.items()) if needle in k.lower() or needle in v["value"].lower()}

    def render(self, limit_chars: int = 6000) -> str:
        """A compact block for the system prompt at the start of a run."""
        lines, used = [], 0
        for key, value in self.recall().items():
            line = f"- {key}: {value}"
            if used + len(line) > limit_chars:
                lines.append("- ... (more entries; use the recall tool)")
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines)

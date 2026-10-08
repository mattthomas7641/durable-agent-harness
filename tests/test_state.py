"""Checkpoint store, memory store and tool-input validation."""

from __future__ import annotations

import json
import threading
import unittest
from typing import Any, ClassVar

from longrun.checkpoint import CheckpointStore, RunState
from longrun.errors import CheckpointError, ToolInputError
from longrun.memory import MemoryStore
from longrun.tools import validate

from .helpers import TempDirs


class CheckpointTests(TempDirs):
    def setUp(self) -> None:
        super().setUp()
        self.store = CheckpointStore(self.state_dir)

    def test_round_trip(self) -> None:
        state = RunState(run_id="r1", task="do it", step=3, messages=[{"role": "user", "content": "do it"}])
        state.add_usage({"input_tokens": 10})
        state.add_usage({"input_tokens": 5, "output_tokens": 2})
        self.store.save(state)
        loaded = self.store.load("r1")
        self.assertEqual(loaded.step, 3)
        self.assertEqual(loaded.messages, state.messages)
        self.assertEqual(loaded.usage, {"input_tokens": 15, "output_tokens": 2})

    def test_save_leaves_no_temp_files(self) -> None:
        for step in range(5):
            self.store.save(RunState(run_id="r1", task="t", step=step))
        self.assertEqual([p.name for p in self.store.run_dir("r1").iterdir()], ["checkpoint.json"])

    def test_leftover_temp_file_from_crash_is_ignored(self) -> None:
        self.store.save(RunState(run_id="r1", task="t", step=1))
        (self.store.run_dir("r1") / ".checkpoint.abc.tmp").write_text('{"half": ')
        self.assertEqual(self.store.load("r1").step, 1)

    def test_missing_and_corrupt_checkpoints(self) -> None:
        with self.assertRaises(CheckpointError):
            self.store.load("nope")
        self.store.run_dir("bad").mkdir()
        self.store.path("bad").write_text("{not json")
        with self.assertRaises(CheckpointError):
            self.store.load("bad")

    def test_schema_version_is_enforced(self) -> None:
        self.store.save(RunState(run_id="r1", task="t"))
        data = json.loads(self.store.path("r1").read_text())
        data["schema_version"] = 99
        self.store.path("r1").write_text(json.dumps(data))
        with self.assertRaises(CheckpointError):
            self.store.load("r1")

    def test_run_ids_cannot_escape_runs_dir(self) -> None:
        for bad in ("../x", "a/b", ".hidden", ""):
            with self.subTest(run_id=bad), self.assertRaises(CheckpointError):
                self.store.run_dir(bad)


class MemoryTests(TempDirs):
    def test_remember_recall_forget(self) -> None:
        memory = MemoryStore(self.state_dir / "memory.json")
        memory.remember("test-command", "python -m unittest", "r1")
        memory.remember("style", "use type hints", "r1")
        self.assertEqual(memory.recall("unittest"), {"test-command": "python -m unittest"})
        self.assertEqual(len(memory.recall()), 2)
        self.assertIn("- style: use type hints", memory.render())
        self.assertTrue(memory.forget("style"))
        self.assertFalse(memory.forget("style"))
        # Persisted: a new instance (a later run) sees it.
        self.assertEqual(MemoryStore(self.state_dir / "memory.json").recall(), {"test-command": "python -m unittest"})

    def test_limits(self) -> None:
        memory = MemoryStore(self.state_dir / "memory.json")
        with self.assertRaises(ValueError):
            memory.remember("", "x", "r1")
        with self.assertRaises(ValueError):
            memory.remember("k", "x" * 5000, "r1")

    def test_concurrent_writers_lose_nothing(self) -> None:
        memory = MemoryStore(self.state_dir / "memory.json")
        threads = [threading.Thread(target=memory.remember, args=(f"key-{i}", "v", f"sub-{i}")) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(memory.recall()), 20)


class ValidateTests(unittest.TestCase):
    schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "maxLength": 5},
            "argv": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "n": {"type": "integer", "minimum": 0},
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def test_accepts_valid_input(self) -> None:
        validate({"path": "a.py", "argv": ["ls"], "n": 2}, self.schema)

    def test_rejects_invalid_input(self) -> None:
        bad = [
            "not an object",
            {},
            {"path": 1},
            {"path": "too-long.py"},
            {"path": "a", "argv": []},
            {"path": "a", "argv": [1]},
            {"path": "a", "n": -1},
            {"path": "a", "n": True},
            {"path": "a", "extra": 1},
        ]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ToolInputError):
                validate(value, self.schema)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import threading
import unittest

from longrun.audit import AuditLog, redact, require_valid, verify
from longrun.errors import AuditIntegrityError

from .helpers import TempDirs


class AuditLogTests(TempDirs):
    def setUp(self) -> None:
        super().setUp()
        self.path = self.state_dir / "audit.jsonl"
        self.log = AuditLog(self.path, "run-1")

    def write_events(self, n: int) -> None:
        for i in range(n):
            self.log.append("tool.start", call_id=f"c{i}", tool="read_file")

    def lines(self) -> list[dict]:
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def rewrite(self, records: list[dict]) -> None:
        self.path.write_text("".join(json.dumps(r) + "\n" for r in records))

    def test_chain_verifies(self) -> None:
        self.write_events(5)
        result = verify(self.path)
        self.assertTrue(result.ok)
        self.assertEqual(result.records, 5)

    def test_editing_a_record_is_detected(self) -> None:
        self.write_events(5)
        records = self.lines()
        records[2]["data"]["tool"] = "write_file"
        self.rewrite(records)
        result = verify(self.path)
        self.assertFalse(result.ok)
        self.assertEqual(result.first_bad_seq, 2)

    def test_deleting_a_record_is_detected(self) -> None:
        self.write_events(5)
        records = self.lines()
        del records[1]
        self.rewrite(records)
        self.assertEqual(verify(self.path).first_bad_seq, 1)

    def test_rehashing_an_edited_record_still_breaks_the_next_link(self) -> None:
        from longrun.audit import _digest

        self.write_events(4)
        records = self.lines()
        records[1]["data"]["tool"] = "evil"
        records[1]["hash"] = _digest(records[1])  # attacker recomputes this record's hash...
        self.rewrite(records)
        result = verify(self.path)
        self.assertEqual(result.first_bad_seq, 2)  # ...but record 2 still points at the old one

    def test_require_valid_raises(self) -> None:
        self.write_events(2)
        self.path.write_text(self.path.read_text().replace("read_file", "rm_rf"))
        with self.assertRaises(AuditIntegrityError):
            require_valid(self.path)

    def test_torn_tail_is_repaired_and_chain_continues(self) -> None:
        self.write_events(3)
        with self.path.open("ab") as fh:
            fh.write(b'{"seq": 3, "event": "tool.st')  # process died mid-write
        reopened = AuditLog(self.path, "run-1")
        reopened.append("run.resume")
        result = verify(self.path)
        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.records, 4)

    def test_reopen_continues_sequence(self) -> None:
        self.write_events(2)
        AuditLog(self.path, "run-1").append("run.resume")
        self.assertEqual([r["seq"] for r in self.lines()], [0, 1, 2])
        self.assertTrue(verify(self.path).ok)

    def test_concurrent_appends_keep_chain_valid(self) -> None:
        threads = [threading.Thread(target=self.write_events, args=(25,)) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        result = verify(self.path)
        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.records, 200)

    def test_redact_bounds_large_inputs(self) -> None:
        out = redact({"path": "a.py", "content": "x" * 5000, "argv": ["python"]})
        self.assertEqual(out["path"], "a.py")
        self.assertEqual(out["content"]["chars"], 5000)
        self.assertEqual(len(out["content"]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()

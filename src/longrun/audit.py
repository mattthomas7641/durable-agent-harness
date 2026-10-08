"""Append-only, hash-chained audit log.

Every tool *attempt* is written here before it executes and again when it
finishes (or is denied). Each record carries the SHA-256 of the previous
record, so editing, reordering, or deleting any line breaks the chain and
``verify()`` reports the first bad sequence number.

The log is JSON Lines and is fsync'd after every append: if the process is
killed, at most the final line can be torn, and ``open`` repairs that by
truncating back to the last complete record.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import AuditIntegrityError

GENESIS = "0" * 64


def canonical(obj: Any) -> bytes:
    """Deterministic JSON encoding used for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _digest(record: dict[str, Any]) -> str:
    body = {k: v for k, v in record.items() if k != "hash"}
    return hashlib.sha256(canonical(body)).hexdigest()


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    records: int
    first_bad_seq: int | None = None
    reason: str = ""


class AuditLog:
    """Thread-safe writer for one run's audit trail."""

    def __init__(self, path: Path, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq, self._prev = self._recover_tail()

    def _recover_tail(self) -> tuple[int, str]:
        if not self.path.exists():
            return 0, GENESIS
        with self.path.open("rb+") as fh:
            data = fh.read()
            good_end = data.rfind(b"\n") + 1  # 0 if no complete line
            if good_end < len(data):
                # A crash mid-write left a partial last line; drop it.
                fh.truncate(good_end)
                fh.flush()
                os.fsync(fh.fileno())
        last = data[:good_end].splitlines()[-1:] if good_end else []
        if not last:
            return 0, GENESIS
        record = json.loads(last[0])
        return int(record["seq"]) + 1, str(record["hash"])

    def append(self, event: str, **data: Any) -> dict[str, Any]:
        with self._lock:
            record: dict[str, Any] = {
                "seq": self._seq,
                "ts": round(time.time(), 6),
                "run_id": self.run_id,
                "event": event,
                "data": data,
                "prev": self._prev,
            }
            record["hash"] = _digest(record)
            line = canonical(record) + b"\n"
            with self.path.open("ab") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            self._seq += 1
            self._prev = record["hash"]
            return record

    def records(self) -> Iterator[dict[str, Any]]:
        yield from read_records(self.path)

    def verify(self) -> VerifyResult:
        return verify(self.path)


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    if not Path(path).exists():
        return
    with Path(path).open("rb") as fh:
        for line in fh:
            if line.endswith(b"\n"):
                yield json.loads(line)


def verify(path: Path) -> VerifyResult:
    """Walk the chain and report the first record that doesn't link up."""
    prev, expected_seq, count = GENESIS, 0, 0
    try:
        for record in read_records(path):
            seq = record.get("seq")
            if seq != expected_seq:
                return VerifyResult(False, count, expected_seq, f"expected seq {expected_seq}, got {seq}")
            if record.get("prev") != prev:
                return VerifyResult(False, count, seq, "prev-hash does not match previous record")
            if record.get("hash") != _digest(record):
                return VerifyResult(False, count, seq, "record contents do not match its hash")
            prev, expected_seq, count = record["hash"], expected_seq + 1, count + 1
    except json.JSONDecodeError as exc:
        return VerifyResult(False, count, expected_seq, f"unparseable line: {exc}")
    return VerifyResult(True, count)


def require_valid(path: Path) -> VerifyResult:
    result = verify(path)
    if not result.ok:
        raise AuditIntegrityError(f"{path}: seq {result.first_bad_seq}: {result.reason}")
    return result


def redact(value: Any, limit: int = 2000) -> Any:
    """Keep audit records bounded: long strings are replaced by a prefix + hash."""
    if isinstance(value, str) and len(value) > limit:
        digest = hashlib.sha256(value.encode()).hexdigest()
        return {"prefix": value[:200], "chars": len(value), "sha256": digest}
    if isinstance(value, dict):
        return {k: redact(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, limit) for v in value]
    return value

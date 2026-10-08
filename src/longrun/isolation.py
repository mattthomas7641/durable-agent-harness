"""The jail: every filesystem path and every subprocess goes through here.

The sandbox is deliberately small and deny-by-default:

* **Paths** are resolved (following symlinks) and must stay inside ``root``.
  ``..`` traversal, absolute paths elsewhere, and symlinks that point out of
  the workspace are all rejected *in code*, before any I/O happens.
* **Commands** run without a shell, only if their argv starts with an
  allow-listed prefix, with a scrubbed environment (no inherited secrets),
  a wall-clock timeout that kills the whole process group, and POSIX
  resource limits (CPU seconds, file size, open files).

It is a policy layer, not a kernel boundary. For untrusted workloads, pass a
``launcher`` (e.g. ``bwrap``/``docker run``/``sandbox-exec``) so the process
also runs inside an OS-level sandbox. See README "Threat model".
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .errors import Denied

ALLOWED_ARGV: tuple[tuple[str, ...], ...] = (
    ("python",),
    ("python3",),
    ("git", "status"),
    ("git", "diff"),
    ("git", "log"),
    ("ls",),
    ("grep",),
    ("wc",),
)
"""Default allow-list. A command is allowed if its argv *starts with* one of these."""

PROTECTED_NAMES = frozenset({".git", ".ssh"})
"""Directory names that may be read but never written."""

SECRET_PREFIXES = (".env",)
"""File names that may be neither read nor written (``.env``, ``.env.prod``, ...)."""

_DEFAULT_PATH = ("/usr/local/bin", "/usr/bin", "/bin")

# Runs in the child *after* fork and before exec, but as a fresh interpreter, so
# it is safe to use from a multi-threaded parent (unlike ``preexec_fn``).
_LIMITS_TRAMPOLINE = (
    "import os,sys,resource as r\n"
    "cpu,fsz,nof=map(int,sys.argv[1:4])\n"
    "r.setrlimit(r.RLIMIT_CPU,(cpu,cpu))\n"
    "r.setrlimit(r.RLIMIT_FSIZE,(fsz,fsz))\n"
    "r.setrlimit(r.RLIMIT_NOFILE,(nof,nof))\n"
    "os.execvp(sys.argv[4],sys.argv[4:])\n"
)


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    exit_code: int
    output: str
    truncated: bool
    timed_out: bool
    duration_s: float

    def to_text(self) -> str:
        status = "timed out" if self.timed_out else f"exit {self.exit_code}"
        note = "\n[output truncated]" if self.truncated else ""
        return f"$ {' '.join(self.argv)}\n[{status}, {self.duration_s:.2f}s]\n{self.output}{note}"


@dataclass
class Sandbox:
    """Confines file access and subprocesses to ``root``."""

    root: Path
    allowed_argv: Iterable[Sequence[str]] = ALLOWED_ARGV
    timeout_s: float = 60.0
    cpu_limit_s: int = 120
    max_file_bytes: int = 32 * 1024 * 1024
    max_output_bytes: int = 64 * 1024
    max_open_files: int = 256
    launcher: Sequence[str] = ()
    extra_path: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"sandbox root is not a directory: {self.root}")
        self.allowed_argv = tuple(tuple(p) for p in self.allowed_argv)
        self.launcher = tuple(self.launcher)
        python_bin = str(Path(sys.executable).parent)
        self._path_env = os.pathsep.join([python_bin, *self.extra_path, *_DEFAULT_PATH])

    # ------------------------------------------------------------------ paths

    def resolve(self, path: str | os.PathLike[str]) -> Path:
        """Map a model-supplied path to an absolute path inside the jail."""
        raw = os.fspath(path)
        if "\x00" in raw:
            raise Denied("path contains a NUL byte")
        target = (self.root / raw).resolve()
        if not target.is_relative_to(self.root):
            raise Denied(f"path escapes workspace: {raw}")
        if any(target.name.startswith(p) for p in SECRET_PREFIXES):
            raise Denied(f"path is a secrets file: {raw}")
        return target

    def relative(self, target: Path) -> str:
        rel = target.relative_to(self.root).as_posix()
        return rel or "."

    def _writable(self, path: str) -> Path:
        target = self.resolve(path)
        parts = target.relative_to(self.root).parts
        if target == self.root or PROTECTED_NAMES.intersection(parts):
            raise Denied(f"path is read-only: {path}")
        return target

    def read_text(self, path: str, max_bytes: int | None = None) -> tuple[str, bool]:
        """Return ``(text, truncated)``. Refuses to follow a swapped-in final symlink."""
        target = self.resolve(path)
        limit = max_bytes or self.max_output_bytes
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(target, flags)
        except FileNotFoundError:
            raise Denied(f"no such file: {path}") from None
        except IsADirectoryError:
            raise Denied(f"is a directory: {path}") from None
        with os.fdopen(fd, "rb") as fh:
            data = fh.read(limit + 1)
        return data[:limit].decode("utf-8", errors="replace"), len(data) > limit

    def write_text(self, path: str, content: str) -> int:
        """Atomically write ``content``; returns bytes written."""
        target = self._writable(path)
        data = content.encode("utf-8")
        if len(data) > self.max_file_bytes:
            raise Denied(f"file too large: {len(data)} bytes > {self.max_file_bytes}")
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            # os.replace swaps the directory entry, so a symlink at ``target`` is
            # replaced rather than followed.
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return len(data)

    def list_dir(self, path: str = ".", limit: int = 500) -> list[str]:
        target = self.resolve(path)
        if not target.is_dir():
            raise Denied(f"not a directory: {path}")
        entries = sorted(target.iterdir(), key=lambda p: p.name)
        return [p.name + ("/" if p.is_dir() else "") for p in entries[:limit]]

    # --------------------------------------------------------------- commands

    def check_argv(self, argv: Sequence[str]) -> tuple[str, ...]:
        if not argv or not all(isinstance(a, str) for a in argv):
            raise Denied("argv must be a non-empty list of strings")
        argv = tuple(argv)
        if any("\x00" in a for a in argv):
            raise Denied("argv contains a NUL byte")
        if not any(argv[: len(p)] == p for p in self.allowed_argv):
            raise Denied(f"command not allowed: {' '.join(argv[:3])}")
        for arg in argv[1:]:
            for piece in arg.split("=")[-1:] if arg.startswith("-") else (arg,):
                if _looks_like_path(piece):
                    self.resolve(piece)
        return argv

    def run(self, argv: Sequence[str]) -> CommandResult:
        argv = self.check_argv(argv)
        exe = shutil.which(argv[0], path=self._path_env)
        if exe is None:
            raise Denied(f"executable not found on sandbox PATH: {argv[0]}")
        cmd = [*self.launcher, exe, *argv[1:]]
        if os.name == "posix":
            limits = (str(self.cpu_limit_s), str(self.max_file_bytes), str(self.max_open_files))
            cmd = [sys.executable, "-I", "-c", _LIMITS_TRAMPOLINE, *limits, *cmd]

        start = time.monotonic()
        timed_out = False
        # Output goes to a temp file (bounded by RLIMIT_FSIZE) rather than a pipe,
        # so a runaway process can't exhaust the harness's memory.
        with tempfile.TemporaryFile() as out:
            proc = subprocess.Popen(
                cmd,
                cwd=self.root,
                env=self._env(),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                exit_code = proc.wait(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_group(proc)
                exit_code = proc.wait()
            out.seek(0)
            data = out.read(self.max_output_bytes + 1)

        return CommandResult(
            argv=argv,
            exit_code=exit_code,
            output=data[: self.max_output_bytes].decode("utf-8", errors="replace"),
            truncated=len(data) > self.max_output_bytes,
            timed_out=timed_out,
            duration_s=time.monotonic() - start,
        )

    def _env(self) -> dict[str, str]:
        """A minimal environment: nothing from the parent leaks in (API keys, cloud creds)."""
        return {
            "PATH": self._path_env,
            "HOME": str(self.root),
            "TMPDIR": str(self.root),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }


def _looks_like_path(arg: str) -> bool:
    return arg.startswith("/") or ".." in Path(arg).parts


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()

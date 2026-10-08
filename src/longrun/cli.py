"""Command-line entry point: ``longrun run | resume | status | log | verify | mcp``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .audit import AuditLog, read_records, verify
from .checkpoint import CheckpointStore, RunState
from .errors import CheckpointError
from .isolation import ALLOWED_ARGV
from .mcp_server import McpServer
from .model import DEFAULT_MODEL, AnthropicModel, Model, ScriptedModel
from .runtime import Harness, HarnessConfig, ModelFactory

EXIT_DONE, EXIT_STOPPED, EXIT_USAGE, EXIT_INTERRUPTED = 0, 1, 2, 3


def default_state_dir() -> Path:
    if env := os.environ.get("LONGRUN_STATE_DIR"):
        return Path(env)
    base = os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state"
    return Path(base) / "longrun"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="longrun", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=default_state_dir(),
        help="where checkpoints, audit logs and memory live (default: %(default)s)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_agent_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", default=DEFAULT_MODEL, help="Claude model id (default: %(default)s)")
        p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
        p.add_argument("--subagent-effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
        p.add_argument("--script", type=Path, help="replay a scripted model from JSON instead of calling the API")
        p.add_argument("--max-steps", type=int, default=200)
        p.add_argument(
            "--allow",
            action="append",
            default=[],
            metavar="CMD",
            help='extra allow-listed command prefix, e.g. --allow "git commit" (repeatable)',
        )
        p.add_argument("--launcher", default="", help='OS sandbox prefix, e.g. "bwrap --unshare-net ..."')

    run = sub.add_parser("run", help="start a new run")
    run.add_argument("task", help="what the agent should do")
    run.add_argument("--workspace", type=Path, default=Path.cwd())
    run.add_argument("--run-id")
    add_agent_options(run)

    resume = sub.add_parser("resume", help="continue a run from its last checkpoint")
    resume.add_argument("run_id")
    resume.add_argument("--workspace", type=Path, help="override the workspace recorded in the checkpoint")
    add_agent_options(resume)

    status = sub.add_parser("status", help="list runs, or show one run")
    status.add_argument("run_id", nargs="?")

    log = sub.add_parser("log", help="print a run's audit trail")
    log.add_argument("run_id")
    log.add_argument("--json", action="store_true", help="raw JSON lines")

    ver = sub.add_parser("verify", help="check the audit hash chain of a run and its subagents")
    ver.add_argument("run_id")

    mcp = sub.add_parser("mcp", help="serve the jailed tools over MCP (stdio)")
    mcp.add_argument("--workspace", type=Path, default=Path.cwd())
    mcp.add_argument("--allow", action="append", default=[], metavar="CMD")
    return parser


def model_factory(args: argparse.Namespace) -> ModelFactory:
    if args.script:
        script = args.script
        return lambda role: ScriptedModel.from_file(script, "subagent_turns" if role == "subagent" else "turns")

    def factory(role: str) -> Model:
        effort = args.subagent_effort if role == "subagent" else args.effort
        return AnthropicModel(args.model, effort=effort)

    return factory


def make_harness(args: argparse.Namespace, workspace: Path) -> Harness:
    allowed = (*ALLOWED_ARGV, *(tuple(shlex.split(a)) for a in args.allow))
    config = HarnessConfig(
        workspace=workspace,
        state_dir=args.state_dir,
        max_steps=getattr(args, "max_steps", 200),
        allowed_argv=allowed,
        launcher=tuple(shlex.split(getattr(args, "launcher", "") or "")),
    )
    factory = model_factory(args) if hasattr(args, "script") else (lambda role: ScriptedModel([]))
    return Harness(config, factory)


def report(state: RunState) -> int:
    print(f"\nrun {state.run_id}: {state.status} after {state.step} steps")
    if any(state.usage.values()):
        print("usage: " + ", ".join(f"{k}={v}" for k, v in sorted(state.usage.items())))
    if state.final_text:
        print("\n" + state.final_text)
    if state.error:
        print(f"error: {state.error}", file=sys.stderr)
    return EXIT_DONE if state.status == "done" else EXIT_STOPPED


def drive(harness: Harness, run_id: str, start_task: str | None) -> int:
    try:
        state = harness.start(start_task, run_id) if start_task is not None else harness.resume(run_id)
    except KeyboardInterrupt:
        print(f"\ninterrupted. Progress is checkpointed; continue with:\n  longrun resume {run_id}", file=sys.stderr)
        return 130
    except Exception as exc:  # API outage, rate limit after retries, ...
        logging.getLogger("longrun").exception("run %s stopped", run_id)
        print(f"\nrun stopped: {type(exc).__name__}: {exc}\ncontinue with:\n  longrun resume {run_id}", file=sys.stderr)
        return EXIT_INTERRUPTED
    return report(state)


def cmd_status(store: CheckpointStore, run_id: str | None) -> int:
    if run_id:
        state = store.load(run_id)
        print(
            json.dumps({k: v for k, v in vars(state).items() if k not in ("messages", "system")}, indent=2, default=str)
        )
        return EXIT_DONE
    runs = store.list_runs()
    if not runs:
        print("no runs yet")
    for s in runs:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(s.created_at))
        indent = "  └ " if s.parent_id else ""
        print(f"{indent}{s.run_id:<40} {s.status:<10} step {s.step:<4} {when}  {s.task[:50]!r}")
    return EXIT_DONE


def cmd_log(store: CheckpointStore, run_id: str, raw: bool) -> int:
    for record in read_records(store.run_dir(run_id) / "audit.jsonl"):
        if raw:
            print(json.dumps(record))
            continue
        when = time.strftime("%H:%M:%S", time.localtime(record["ts"]))
        data = record["data"]
        detail = data.get("tool") or data.get("stop_reason") or data.get("status") or ""
        extra = ""
        if record["event"] == "tool.start":
            extra = json.dumps(data.get("input"))[:100]
        elif record["event"] in ("tool.end", "tool.denied"):
            extra = (data.get("output_preview") or "").splitlines()[0][:100] if data.get("output_preview") else ""
        print(f"{record['seq']:>5} {when} {record['event']:<17} {detail:<16} {extra}")
    return EXIT_DONE


def cmd_verify(store: CheckpointStore, run_id: str) -> int:
    ok = True
    run_ids = [run_id, *sorted(p.name for p in store.runs_dir.glob(f"{run_id}.sub-*"))]
    for rid in run_ids:
        result = verify(store.run_dir(rid) / "audit.jsonl")
        mark = "ok " if result.ok else "BAD"
        detail = "" if result.ok else f" at seq {result.first_bad_seq}: {result.reason}"
        print(f"[{mark}] {rid}: {result.records} records{detail}")
        ok &= result.ok
    return EXIT_DONE if ok else EXIT_STOPPED


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    if not args.verbose:
        for noisy in ("httpx", "anthropic"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    store = CheckpointStore(args.state_dir)
    try:
        if args.command == "run":
            harness = make_harness(args, args.workspace)
            run_id = args.run_id or Harness.new_run_id()
            print(f"run {run_id}  (workspace {harness.sandbox.root}, state {store.state_dir})", file=sys.stderr)
            return drive(harness, run_id, args.task)
        if args.command == "resume":
            state = store.load(args.run_id)
            harness = make_harness(args, args.workspace or Path(state.workspace))
            return drive(harness, args.run_id, None)
        if args.command == "status":
            return cmd_status(store, args.run_id)
        if args.command == "log":
            return cmd_log(store, args.run_id, args.json)
        if args.command == "verify":
            return cmd_verify(store, args.run_id)
        if args.command == "mcp":
            # stdout is the protocol channel: keep all logging on stderr.
            harness = make_harness(args, args.workspace)
            session = f"mcp-{Harness.new_run_id()}"
            McpServer(harness.mcp_tools(), AuditLog(harness.audit_path(session), session)).serve()
            return EXIT_DONE
    except (CheckpointError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())

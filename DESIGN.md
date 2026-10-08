# Design notes

## Goals

1. **Resume.** A run that is killed at any instant (OOM, deploy, laptop lid, `kill -9`) can be continued
   to the same outcome, with no human repair.
2. **Fan out.** Independent sub-tasks run in parallel. Crash-safety must hold across the fan-out too.
3. **Stay off prod.** The agent is treated as untrusted. It can only act through tools, the tools are
   jailed, and every attempt is recorded where the agent can't rewrite it.

Non-goals: being a general agent framework, multi-machine distribution, a kernel-level sandbox
(delegated to `--launcher`).

## Why own the loop instead of using the SDK tool runner

The SDK's tool runner is the right default for most agents, but it owns the loop. Resume semantics need
a durable write *between* "the model asked for tool X" and "tool X ran", and again between "tool X ran"
and "the result is in the conversation". The loop in `agent.py` is a small state machine over
`RunState` that makes those transitions explicit:

```
pending == []  ──model turn──▶  pending = [calls], results = {}
pending != []  ──each call──▶   results[call_id] = tool_result   (one checkpoint per call)
all answered   ──────────────▶  append ONE user message with all results, pending = []
```

Everything the loop needs is in `RunState`, which is plain JSON. The model's content blocks are stored
verbatim (including `thinking` blocks and their signatures), so a resumed request is byte-identical to the
one that would have been sent.

## Crash windows

Each tool call does four durable writes: audit `tool.start`, execute, checkpoint the result, audit
`tool.end`. A model turn does three: audit `model.request`, the call to Claude, then checkpoint (and audit
`model.response`). The table below covers every gap between those writes.

| Process dies… | On disk | On resume |
|---|---|---|
| during the model call | checkpoint has no new assistant message | the model is asked again. Nothing ran, so this is safe; costs one repeated request |
| after the model reply, before the checkpoint | same as above | same as above |
| after `tool.start`, during execution | checkpoint has no result; audit has an unclosed `tool.start` | **idempotent tool** (`read_file`, `list_dir`, `recall`, `spawn_subagents`): re-run. **Side-effecting tool**: not re-run; the model gets `interrupted: … side effects are unknown. Inspect the workspace before retrying.` |
| after the result checkpoint, before `tool.end` | result saved, `tool.start` unclosed | result found in the checkpoint, so the call is skipped. The checkpoint is consulted *before* the audit log, and that ordering is what makes this window safe |
| between calls in a batch | some results saved | only unanswered calls run |
| after the last result, before the batch message | all results saved | batch message assembled from saved results |
| mid-`checkpoint.save` | temp file is partial, the old `checkpoint.json` is intact (`os.replace` is atomic) | old checkpoint loads; stray `.tmp` ignored |
| mid-audit-append | last JSONL line torn | `AuditLog` truncates to the last full line on open; the chain still verifies |

The deliberate choice is **at-most-once for side effects**. Replaying `git commit` or a DB migration after a
crash is worse than asking the model to check. "Exactly-once" would need tools to be transactional or to
take idempotency keys. The `ToolContext.call_id` passed to every handler is the hook for that.

`tests/test_agent.py::CrashResumeTests` injects a crash at every model step and in the middle of tools,
then asserts the transcript and the count of real side effects match a clean run.
`tests/test_kill_resume.py` does the same with a real `SIGKILL`.

## Fan-out

`spawn_subagents` derives child run ids from `sha256(parent tool_use_id)`, so they are deterministic. The
tool is marked idempotent: if the parent dies while children are working, the resumed parent re-executes
the same call, which loads each child's checkpoint, returns finished children immediately and resumes
the rest. Children run in a `ThreadPoolExecutor`, so the shared components are thread-safe:
`AuditLog.append` and `MemoryStore` are locked, `CheckpointStore` writes are per-run-directory, and the
subprocess resource limits are applied by a tiny exec trampoline rather than `preexec_fn` (which is unsafe
in a threaded parent).

Subagents get every tool except `spawn_subagents` (depth 1), a smaller step budget, and a system prompt
that tells them other agents share the workspace. **Known limitation:** two children told to edit the same
file can race. The parent is told to split work by file. Per-child git worktrees would remove the race.

## The jail

* Paths: `(root / p).resolve()` follows every symlink, then `is_relative_to(root)`. Reads open with
  `O_NOFOLLOW` so a final-component symlink swapped in after the check is refused. Writes go to a temp file
  in the target directory and `os.replace`, which replaces a symlink rather than following it.
* Commands: no shell, argv prefix allow-list (`("git", "status")` allows `git status --short` but not
  `git push`), and path-looking arguments go through the same jail. The environment is rebuilt from scratch:
  `PATH`, `HOME=root`, locale, and nothing else, so `ANTHROPIC_API_KEY` and `AWS_*` never reach the child.
  Output goes to a temp file bounded by `RLIMIT_FSIZE` instead of a pipe, so a process printing forever
  can't exhaust the harness's memory. A timeout `killpg`s the whole session, including grandchildren.
* State (checkpoints, audit, memory) must live outside the jail; `Harness` refuses to start otherwise.

## Audit log

JSON Lines, one record per event, each with `prev` (the previous record's hash) and `hash =
sha256(canonical_json(record without hash))`. Editing a record breaks its own hash. Recomputing that hash
breaks the next record's `prev`. Deleting a record breaks `seq`. `longrun verify` reports the first bad
sequence number. Large tool inputs (file contents) are stored as prefix + length + sha256 to keep the log
bounded while still being checkable against the workspace.

## Talking to Claude

* `claude-opus-5`, `thinking: {type: "adaptive"}`, `output_config.effort` (`high` for the main agent,
  `medium` for subagents), streamed with `get_final_message()` so long turns don't hit HTTP timeouts.
* `eager_input_streaming` on tools, so large `write_file` contents stream as they are generated. The
  registry validates every input against its JSON Schema before running it, which also covers truncated
  inputs. A tool call in a `max_tokens` turn is never executed.
* Prompt caching through top-level `cache_control`. The prefix is kept byte-stable: tools sorted by name,
  the system prompt (including the memory snapshot) frozen into `RunState.system` at start, and history
  append-only.
* Server-side refusal fallbacks (`fallbacks: "default"`). A final `refusal` ends the run as `failed` with
  the reason recorded.
* `anthropic.Anthropic(max_retries=6)` for transient errors. Anything that still fails leaves the run
  `running` with a valid checkpoint, and the CLI prints the exact `longrun resume` command.

## MCP

`mcp_server.py` implements the stdio transport directly (newline-delimited JSON-RPC 2.0: `initialize`
with version negotiation, `ping`, `tools/list`, `tools/call`, notifications). That's roughly 100 lines and
needs no dependency. Tool failures are returned as `isError: true` results rather than protocol errors, so
the client's model can read them and adapt. The calls go through the same registry, so the same jail and
audit apply.

## Roadmap

* **Context management for multi-hour runs.** Enable server-side compaction (or client-side
  summarisation checkpoints) once a run's history approaches the context window.
* **Per-subagent git worktrees** to remove same-file races, merged back by the parent.
* **Budgets:** stop or ask once a run's accumulated `usage` crosses a cost ceiling.
* **Signed audit heads:** periodically publish the latest hash to an append-only store.
* **Network policy** as a first-class option (`unshare -n` / container) instead of via `--launcher`.

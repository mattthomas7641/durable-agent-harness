"""Model backends.

The agent loop only depends on the ``Model`` protocol: given the system
prompt, the conversation, and tool specs, return the next assistant turn as
plain JSON-serialisable dicts (so it can go straight into a checkpoint).

* ``AnthropicModel`` calls Claude through the official SDK.
* ``ScriptedModel`` replays a fixed script. It is deterministic and keyed on
  conversation length rather than an internal counter, so it behaves
  identically after a crash/resume. Used by the tests and the offline demo.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

DEFAULT_MODEL = "claude-opus-5"


@dataclass(frozen=True)
class ModelTurn:
    content: list[dict[str, Any]]
    stop_reason: str
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return [b for b in self.content if b.get("type") == "tool_use"]

    @property
    def text(self) -> str:
        return "\n".join(b["text"] for b in self.content if b.get("type") == "text").strip()


class Model(Protocol):
    name: str

    def next_turn(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn: ...


class AnthropicModel:
    """Claude via the Messages API, streamed so long turns don't hit HTTP timeouts."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        effort: str = "high",
        max_tokens: int = 64_000,
        fallbacks: bool = True,
        client: Any = None,
    ) -> None:
        import anthropic  # imported lazily so the offline path has no hard dependency

        self.name = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.fallbacks = fallbacks
        # Long unattended runs: retry transient 429/5xx/connection errors harder than the default.
        self.client = client or anthropic.Anthropic(max_retries=6)

    def next_turn(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn:
        kwargs: dict[str, Any] = {
            "model": self.name,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": messages,
            # Stream tool inputs as they're generated; the registry validates them before use.
            "tools": [{**spec, "eager_input_streaming": True} for spec in tools],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort},
            # Auto-caching: the breakpoint follows the growing conversation, so each
            # step re-reads the previous steps from cache instead of re-billing them.
            "cache_control": {"type": "ephemeral"},
        }
        if self.fallbacks:
            # If a safety classifier declines, the API retries on a fallback model in-call.
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"

        with self.client.beta.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()

        usage = message.usage
        return ModelTurn(
            # Keep every block (thinking, fallback markers, tool_use) verbatim: they
            # must be echoed back unchanged on the next request.
            content=[b.model_dump(mode="json", by_alias=True, exclude_none=True) for b in message.content],
            stop_reason=message.stop_reason or "end_turn",
            usage={
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
                "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
            },
        )


class ScriptedModel:
    """Replays ``turns``; each turn is ``{"text": ..., "tool_calls": [{"name", "input"}], "delay_s": ...}``."""

    def __init__(self, turns: list[dict[str, Any]], name: str = "scripted") -> None:
        self.turns = turns
        self.name = name

    @classmethod
    def from_file(cls, path: Path, key: str = "turns") -> ScriptedModel:
        data = json.loads(Path(path).read_text())
        return cls(data.get(key, []), name=f"scripted:{Path(path).name}")

    def next_turn(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn:
        index = sum(1 for m in messages if m["role"] == "assistant")
        if index >= len(self.turns):
            return ModelTurn([{"type": "text", "text": "Script exhausted."}], "end_turn")
        spec = self.turns[index]
        if spec.get("delay_s"):
            time.sleep(float(spec["delay_s"]))
        content: list[dict[str, Any]] = []
        if spec.get("text"):
            content.append({"type": "text", "text": spec["text"]})
        for i, call in enumerate(spec.get("tool_calls", [])):
            content.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_{index:03d}_{i}",
                    "name": call["name"],
                    "input": call.get("input", {}),
                }
            )
        stop = spec.get("stop_reason") or ("tool_use" if spec.get("tool_calls") else "end_turn")
        return ModelTurn(content, stop, {"input_tokens": 0, "output_tokens": 0})

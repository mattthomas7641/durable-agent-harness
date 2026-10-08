"""Tool registry: the *only* things the model can do.

Each tool declares a JSON Schema, and inputs are validated before the handler
runs (the model's arguments are untrusted). Handlers never touch the
filesystem or spawn processes directly; they go through the ``Sandbox``.

``idempotent`` marks tools that are safe to re-run if the harness was killed
while they were executing. Side-effecting tools are never silently replayed.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .errors import Denied, ToolInputError
from .isolation import Sandbox
from .memory import MemoryStore


@dataclass(frozen=True)
class ToolContext:
    """Who is calling: lets handlers derive stable ids (e.g. for subagents) from the call."""

    run_id: str
    call_id: str


Handler = Callable[[dict[str, Any], ToolContext], str]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Handler
    idempotent: bool = False

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


@dataclass(frozen=True)
class ToolResult:
    content: str
    is_error: bool = False
    denied: bool = False


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        # Sorted so the request prefix is byte-stable across steps (prompt caching).
        return [self._tools[name].spec() for name in self.names()]

    def without(self, *names: str) -> ToolRegistry:
        return ToolRegistry(t for n, t in self._tools.items() if n not in names)

    def execute(self, name: str, args: Any, ctx: ToolContext) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(f"unknown tool: {name}", is_error=True)
        try:
            validate(args, tool.input_schema)
            return ToolResult(tool.handler(args, ctx))
        except Denied as exc:
            return ToolResult(f"denied: {exc}", is_error=True, denied=True)
        except (ToolInputError, ValueError) as exc:
            return ToolResult(f"invalid input: {exc}", is_error=True)
        except OSError as exc:
            return ToolResult(f"{type(exc).__name__}: {exc}", is_error=True)


# ----------------------------------------------------------------- validation

_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
}


def validate(value: Any, schema: dict[str, Any], where: str = "input") -> None:
    """Validate the subset of JSON Schema that tool definitions use."""
    expected = schema.get("type")
    if expected:
        py_type = _JSON_TYPES[expected]
        if not isinstance(value, py_type) or (expected in ("integer", "number") and isinstance(value, bool)):
            raise ToolInputError(f"{where} must be {expected}")
    if expected == "object":
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise ToolInputError(f"{where}.{key} is required")
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(props)
            if extra:
                raise ToolInputError(f"{where} has unexpected keys: {sorted(extra)}")
        for key, sub in props.items():
            if key in value:
                validate(value[key], sub, f"{where}.{key}")
    elif expected == "array":
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", len(value)):
            raise ToolInputError(f"{where} has {len(value)} items, outside allowed range")
        for i, item in enumerate(value):
            validate(item, schema.get("items", {}), f"{where}[{i}]")
    elif expected == "string":
        if len(value) > schema.get("maxLength", len(value)):
            raise ToolInputError(f"{where} is longer than {schema['maxLength']} characters")
    elif expected == "integer":
        if value < schema.get("minimum", value) or value > schema.get("maximum", value):
            raise ToolInputError(f"{where} is outside allowed range")


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


# -------------------------------------------------------------- builtin tools


def builtin_tools(sandbox: Sandbox, memory: MemoryStore) -> list[Tool]:
    def read_file(args: dict[str, Any], ctx: ToolContext) -> str:
        text, truncated = sandbox.read_text(args["path"])
        return text + ("\n[truncated]" if truncated else "")

    def write_file(args: dict[str, Any], ctx: ToolContext) -> str:
        written = sandbox.write_text(args["path"], args["content"])
        return f"wrote {written} bytes to {args['path']}"

    def list_dir(args: dict[str, Any], ctx: ToolContext) -> str:
        entries = sandbox.list_dir(args.get("path", "."))
        return "\n".join(entries) or "(empty)"

    def run_command(args: dict[str, Any], ctx: ToolContext) -> str:
        return sandbox.run(args["argv"]).to_text()

    def remember(args: dict[str, Any], ctx: ToolContext) -> str:
        memory.remember(args["key"], args["value"], ctx.run_id)
        return f"remembered {args['key']!r}"

    def recall(args: dict[str, Any], ctx: ToolContext) -> str:
        found = memory.recall(args.get("query", ""))
        return json.dumps(found, indent=1) if found else "(no matching memories)"

    path = {"type": "string", "maxLength": 1024, "description": "Path relative to the workspace root."}
    return [
        Tool(
            "read_file",
            "Read a UTF-8 text file from the workspace.",
            _object({"path": path}, ["path"]),
            read_file,
            idempotent=True,
        ),
        Tool(
            "write_file",
            "Create or overwrite a text file in the workspace (atomic).",
            _object({"path": path, "content": {"type": "string"}}, ["path", "content"]),
            write_file,
        ),
        Tool(
            "list_dir",
            "List a directory in the workspace. Directories end with '/'.",
            _object({"path": path}, []),
            list_dir,
            idempotent=True,
        ),
        Tool(
            "run_command",
            "Run an allow-listed command in the workspace without a shell. Pass argv as a list, "
            'e.g. ["python", "-m", "unittest"]. Output is combined stdout+stderr.',
            _object({"argv": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 64}}, ["argv"]),
            run_command,
        ),
        Tool(
            "remember",
            "Save a fact to long-term memory, shared with future runs and subagents.",
            _object(
                {"key": {"type": "string", "maxLength": 128}, "value": {"type": "string", "maxLength": 4000}},
                ["key", "value"],
            ),
            remember,
        ),
        Tool(
            "recall",
            "Search long-term memory by substring. Empty query returns everything.",
            _object({"query": {"type": "string", "maxLength": 200}}, []),
            recall,
            idempotent=True,
        ),
    ]

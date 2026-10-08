"""A minimal MCP server (JSON-RPC 2.0 over stdio) exposing the jailed tools.

Any MCP client (Claude Code, Claude Desktop, an IDE) can point at
``longrun mcp --workspace DIR`` and get the same path-jailed, allow-listed,
audited tools the built-in agent uses. Implemented directly on the wire
protocol (newline-delimited JSON-RPC) with no SDK, to keep it auditable.

Supported methods: ``initialize``, ``ping``, ``tools/list``, ``tools/call``,
and the ``notifications/*`` the client sends.
"""

from __future__ import annotations

import hashlib
import json
import sys
from typing import IO, Any

from . import __version__
from .audit import AuditLog, redact
from .tools import ToolContext, ToolRegistry

SUPPORTED_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class McpServer:
    def __init__(self, tools: ToolRegistry, audit: AuditLog, *, name: str = "longrun") -> None:
        self.tools = tools
        self.audit = audit
        self.name = name

    # ----------------------------------------------------------------- transport

    def serve(self, stdin: IO[str] | None = None, stdout: IO[str] | None = None) -> None:
        """Read one JSON-RPC message per line until EOF; write one response per request."""
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        self.audit.append("mcp.start", tools=self.tools.names())
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                response: dict[str, Any] | None = _error(None, PARSE_ERROR, f"parse error: {exc}")
            else:
                response = self.handle(message)
            if response is not None:
                stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
                stdout.flush()
        self.audit.append("mcp.stop")

    # ------------------------------------------------------------------ dispatch

    def handle(self, message: Any) -> dict[str, Any] | None:
        """Handle one decoded message. Returns ``None`` for notifications."""
        if (
            not isinstance(message, dict)
            or message.get("jsonrpc") != "2.0"
            or not isinstance(message.get("method"), str)
        ):
            msg_id = message.get("id") if isinstance(message, dict) else None
            return _error(msg_id, INVALID_REQUEST, "invalid JSON-RPC 2.0 request")

        method, params = message["method"], message.get("params") or {}
        if "id" not in message:
            return None  # notification (initialized, cancelled, ...): no response allowed
        msg_id = message["id"]
        if not isinstance(params, dict):
            return _error(msg_id, INVALID_PARAMS, "params must be an object")

        try:
            if method == "initialize":
                return _result(msg_id, self._initialize(params))
            if method == "ping":
                return _result(msg_id, {})
            if method == "tools/list":
                return _result(msg_id, {"tools": self._list_tools()})
            if method == "tools/call":
                return self._call_tool(msg_id, params)
        except Exception as exc:  # never let one request take the server down
            return _error(msg_id, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
        return _error(msg_id, METHOD_NOT_FOUND, f"method not found: {method}")

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        requested = params.get("protocolVersion")
        version = requested if requested in SUPPORTED_VERSIONS else SUPPORTED_VERSIONS[0]
        client = params.get("clientInfo", {})
        self.audit.append("mcp.initialize", client=client, protocol=version)
        return {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": self.name, "version": __version__},
            "instructions": "Filesystem and command tools confined to one workspace. Every call is audited.",
        }

    def _list_tools(self) -> list[dict[str, Any]]:
        return [
            {"name": s["name"], "description": s["description"], "inputSchema": s["input_schema"]}
            for s in self.tools.specs()
        ]

    def _call_tool(self, msg_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        name, args = params.get("name"), params.get("arguments") or {}
        if not isinstance(name, str) or self.tools.get(name) is None:
            return _error(msg_id, INVALID_PARAMS, f"unknown tool: {name}")
        call_id = f"mcp-{msg_id}"
        self.audit.append("tool.start", call_id=call_id, tool=name, input=redact(args))
        result = self.tools.execute(name, args, ToolContext(run_id=self.audit.run_id, call_id=call_id))
        self.audit.append(
            "tool.denied" if result.denied else "tool.end",
            call_id=call_id,
            tool=name,
            is_error=result.is_error,
            output_sha256=hashlib.sha256(result.content.encode()).hexdigest(),
        )
        # Tool failures are results (isError), not protocol errors, so the model can see them.
        return _result(msg_id, {"content": [{"type": "text", "text": result.content}], "isError": result.is_error})


def _result(msg_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}

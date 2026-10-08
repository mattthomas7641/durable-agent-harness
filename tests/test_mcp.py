"""MCP server: JSON-RPC 2.0 handling, in-process and over a real stdio subprocess."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import unittest

from longrun.audit import AuditLog, read_records, verify
from longrun.isolation import Sandbox
from longrun.mcp_server import INVALID_PARAMS, INVALID_REQUEST, METHOD_NOT_FOUND, PARSE_ERROR, McpServer
from longrun.memory import MemoryStore
from longrun.tools import ToolRegistry, builtin_tools

from .helpers import TempDirs


class McpHandleTests(TempDirs):
    def setUp(self) -> None:
        super().setUp()
        tools = ToolRegistry(builtin_tools(Sandbox(self.workspace), MemoryStore(self.state_dir / "m.json")))
        self.audit_path = self.state_dir / "mcp.jsonl"
        self.server = McpServer(tools, AuditLog(self.audit_path, "mcp-test"))

    def rpc(self, method: str, params: dict | None = None, msg_id: int = 1) -> dict:
        response = self.server.handle({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params or {}})
        assert response is not None
        return response

    def test_initialize_negotiates_version(self) -> None:
        result = self.rpc("initialize", {"protocolVersion": "2025-03-26", "clientInfo": {"name": "t"}})["result"]
        self.assertEqual(result["protocolVersion"], "2025-03-26")
        self.assertIn("tools", result["capabilities"])
        unknown = self.rpc("initialize", {"protocolVersion": "1999-01-01"})["result"]
        self.assertEqual(unknown["protocolVersion"], "2025-06-18")

    def test_tools_list_uses_mcp_field_names(self) -> None:
        tools = self.rpc("tools/list")["result"]["tools"]
        names = [t["name"] for t in tools]
        self.assertIn("read_file", names)
        self.assertNotIn("spawn_subagents", names)
        self.assertTrue(all("inputSchema" in t for t in tools))

    def test_tools_call_success_and_audit(self) -> None:
        (self.workspace / "hello.txt").write_text("hi there")
        result = self.rpc("tools/call", {"name": "read_file", "arguments": {"path": "hello.txt"}})["result"]
        self.assertEqual(result, {"content": [{"type": "text", "text": "hi there"}], "isError": False})
        events = [r["event"] for r in read_records(self.audit_path)]
        self.assertEqual(events, ["tool.start", "tool.end"])
        self.assertTrue(verify(self.audit_path).ok)

    def test_jail_applies_over_mcp(self) -> None:
        result = self.rpc("tools/call", {"name": "read_file", "arguments": {"path": "../state/m.json"}})["result"]
        self.assertTrue(result["isError"])
        self.assertIn("denied", result["content"][0]["text"])
        self.assertIn("tool.denied", [r["event"] for r in read_records(self.audit_path)])

    def test_protocol_errors(self) -> None:
        self.assertEqual(self.rpc("nope")["error"]["code"], METHOD_NOT_FOUND)
        self.assertEqual(self.rpc("tools/call", {"name": "rm_rf"})["error"]["code"], INVALID_PARAMS)
        bad = self.server.handle({"id": 7, "method": "ping"})  # missing jsonrpc
        assert bad is not None
        self.assertEqual((bad["id"], bad["error"]["code"]), (7, INVALID_REQUEST))
        self.assertEqual(self.server.handle([1, 2])["error"]["code"], INVALID_REQUEST)  # type: ignore[index]

    def test_notifications_get_no_response(self) -> None:
        self.assertIsNone(self.server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))

    def test_serve_loop_handles_parse_errors(self) -> None:
        stdin = io.StringIO('{"jsonrpc":"2.0","id":1,"method":"ping"}\n\nnot json\n')
        stdout = io.StringIO()
        self.server.serve(stdin, stdout)
        lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual(lines[0], {"jsonrpc": "2.0", "id": 1, "result": {}})
        self.assertEqual(lines[1]["error"]["code"], PARSE_ERROR)


class McpStdioTests(TempDirs):
    def test_real_client_session_over_stdio(self) -> None:
        (self.workspace / "notes.md").write_text("# notes")
        messages = [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test"}},
            },
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "list_dir", "arguments": {}}},
        ]
        env = {**os.environ, "LONGRUN_STATE_DIR": str(self.state_dir)}
        proc = subprocess.run(
            [sys.executable, "-m", "longrun", "mcp", "--workspace", str(self.workspace)],
            input="".join(json.dumps(m) + "\n" for m in messages),
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            check=True,
        )
        responses = [json.loads(line) for line in proc.stdout.splitlines()]
        self.assertEqual([r["id"] for r in responses], [1, 2])  # nothing for the notification
        self.assertEqual(responses[0]["result"]["serverInfo"]["name"], "longrun")
        self.assertEqual(responses[1]["result"]["content"][0]["text"], "notes.md")


if __name__ == "__main__":
    unittest.main()

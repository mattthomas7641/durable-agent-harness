"""AnthropicModel request shape and response conversion, against a fake client (no network)."""

from __future__ import annotations

import json
import unittest
from contextlib import contextmanager
from typing import Any

from anthropic.types.beta import BetaMessage

from longrun.model import AnthropicModel

RESPONSE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5",
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "content": [
        {"type": "thinking", "thinking": "", "signature": "sig-abc"},
        {"type": "text", "text": "Reading the file."},
        {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.py"}},
    ],
    "usage": {
        "input_tokens": 120,
        "output_tokens": 30,
        "cache_read_input_tokens": 100,
        "cache_creation_input_tokens": 0,
    },
}


class FakeStream:
    def __init__(self, message: BetaMessage) -> None:
        self.message = message

    def get_final_message(self) -> BetaMessage:
        return self.message


class FakeMessages:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    @contextmanager
    def stream(self, **kwargs: Any):  # type: ignore[no-untyped-def]
        self.requests.append(kwargs)
        yield FakeStream(BetaMessage.model_validate(RESPONSE))


class FakeClient:
    def __init__(self) -> None:
        self.beta = type("Beta", (), {})()
        self.beta.messages = FakeMessages()


class AnthropicModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = FakeClient()
        self.model = AnthropicModel(effort="xhigh", client=self.client)
        tools = [{"name": "read_file", "description": "d", "input_schema": {"type": "object"}}]
        self.turn = self.model.next_turn("system prompt", [{"role": "user", "content": "go"}], tools)
        self.request = self.client.beta.messages.requests[0]

    def test_request_uses_current_api_shape(self) -> None:
        self.assertEqual(self.request["model"], "claude-opus-5")
        self.assertEqual(self.request["thinking"], {"type": "adaptive"})
        self.assertEqual(self.request["output_config"], {"effort": "xhigh"})
        self.assertEqual(self.request["cache_control"], {"type": "ephemeral"})
        self.assertEqual(self.request["fallbacks"], "default")
        self.assertEqual(self.request["betas"], ["server-side-fallback-2026-07-01"])
        self.assertTrue(self.request["tools"][0]["eager_input_streaming"])
        self.assertNotIn("budget_tokens", json.dumps(self.request))

    def test_response_becomes_checkpointable_json(self) -> None:
        json.dumps(self.turn.content)  # must be serialisable as-is
        self.assertEqual(self.turn.stop_reason, "tool_use")
        self.assertEqual(self.turn.tool_calls[0]["input"], {"path": "a.py"})
        self.assertEqual(self.turn.text, "Reading the file.")
        self.assertEqual(self.turn.usage["cache_read_input_tokens"], 100)

    def test_thinking_blocks_are_preserved_for_echo(self) -> None:
        thinking = self.turn.content[0]
        self.assertEqual(thinking["type"], "thinking")
        self.assertEqual(thinking["signature"], "sig-abc")

    def test_fallbacks_can_be_disabled(self) -> None:
        client = FakeClient()
        AnthropicModel(client=client, fallbacks=False).next_turn("s", [{"role": "user", "content": "x"}], [])
        self.assertNotIn("fallbacks", client.beta.messages.requests[0])


if __name__ == "__main__":
    unittest.main()

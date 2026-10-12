"""Внешние MCP-серверы (`MCP_SERVERS_FILE`): файл описания, свои заголовки,
рукопожатие `initialize` с сессией, ответы потоком SSE. Последний класс
проверяет клиент против настоящего `FastMCP` в режиме со сессиями и SSE
(как у большинства внешних серверов) — через Starlette TestClient, без сети."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agents_core.mcp_client import (
    MCPClient, MCPClientError, McpServerSpec, MultiMCPClient, build_mcp_client, load_mcp_servers_file,
)
from tests.test_mcp_client import FakeResponse, FakeSession


def _write(data) -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return path


def _ok(result, headers=None):
    return FakeResponse(200, {"jsonrpc": "2.0", "id": "x", "result": result}, headers=headers)


class ServersFileTests(unittest.TestCase):
    def test_env_substitution_and_fields(self):
        path = _write({
            "tools": {"url": "http://localhost:8001/mcp"},
            "github": {
                "url": "https://api.githubcopilot.com/mcp/", "title": "GitHub",
                "headers": {"Authorization": "Bearer ${GH_TOKEN_TEST}", "X-MCP-Readonly": "true"},
            },
        })
        try:
            with patch.dict(os.environ, {"GH_TOKEN_TEST": "ghp_secret"}):
                specs = load_mcp_servers_file(path)
        finally:
            os.unlink(path)
        self.assertEqual([s.name for s in specs], ["tools", "github"])
        github = specs[1]
        self.assertEqual(github.url, "https://api.githubcopilot.com/mcp/")
        self.assertEqual(github.headers["Authorization"], "Bearer ghp_secret")
        self.assertEqual(github.title, "GitHub")
        self.assertTrue(github.initialize)

    def test_errors(self):
        for data, fragment in (
            ({"git hub": {"url": "http://x"}}, "имя сервера"),
            ({"a": {"headers": {}}}, "нет url"),
            ({"a": {"url": "http://x", "headers": {"A": "${NOPE_NOT_SET_VAR}"}}}, "NOPE_NOT_SET_VAR"),
            ({}, "непустой объект"),
        ):
            path = _write(data)
            try:
                with self.assertRaises(ValueError) as ctx:
                    load_mcp_servers_file(path)
                self.assertIn(fragment, str(ctx.exception))
            finally:
                os.unlink(path)

    def test_build_uses_url_as_is_and_no_shared_api_key(self):
        client = build_mcp_client(
            [("tools", "http://localhost:8001"),
             McpServerSpec("github", "https://api.githubcopilot.com/mcp/readonly", {"Authorization": "Bearer t"}, "GitHub")],
            api_key="shared-key",
        )
        self.assertIsInstance(client, MultiMCPClient)
        ours, github = client.clients["tools"], client.clients["github"]
        self.assertEqual(ours.base_url, "http://localhost:8001/mcp")
        self.assertEqual(ours._headers()["Authorization"], "Bearer shared-key")
        self.assertEqual(github.base_url, "https://api.githubcopilot.com/mcp/readonly")
        self.assertEqual(github._headers()["Authorization"], "Bearer t")
        self.assertTrue(github.handshake)
        self.assertFalse(ours.handshake)


class HandshakeTests(unittest.TestCase):
    def client(self, responses):
        session = FakeSession(responses)
        return MCPClient(
            "https://ext.test/mcp", session=session, max_retries=0, name="github", title="GitHub",
            headers={"X-MCP-Toolsets": "repos"}, normalize_url=False, handshake=True,
        ), session

    def test_initialize_session_then_request(self):
        client, session = self.client([
            _ok({"protocolVersion": "2025-03-26", "capabilities": {}}, headers={"Mcp-Session-Id": "s-1"}),
            FakeResponse(202, None, headers={}),
            _ok({"tools": [{"name": "list_commits", "inputSchema": {"type": "object"}}]}),
        ])
        self.assertEqual([t["function"]["name"] for t in client.list_tools()], ["list_commits"])
        methods = [c["json"].get("method") for c in session.calls]
        self.assertEqual(methods, ["initialize", "notifications/initialized", "tools/list"])
        self.assertNotIn("Mcp-Session-Id", session.calls[0]["headers"])
        last = session.calls[2]["headers"]
        self.assertEqual(last["Mcp-Session-Id"], "s-1")
        self.assertEqual(last["MCP-Protocol-Version"], "2025-03-26")
        self.assertEqual(last["X-MCP-Toolsets"], "repos")
        # Подпись инструментов — title сервера.
        self.assertEqual(client.title, "GitHub")

    def test_group_is_server_title(self):
        client, _ = self.client([
            _ok({"protocolVersion": "2025-06-18"}), FakeResponse(202, None, headers={}),
            _ok({"tools": [{"name": "get_me", "inputSchema": {"type": "object"}}]}),
        ])
        self.assertEqual(client.list_tool_details()[0]["group"], "GitHub")

    def test_expired_session_reinitializes_once(self):
        client, session = self.client([
            _ok({"protocolVersion": "2025-06-18"}, headers={"Mcp-Session-Id": "old"}),
            FakeResponse(202, None, headers={}),
            FakeResponse(404, text="session not found"),
            _ok({"protocolVersion": "2025-06-18"}, headers={"Mcp-Session-Id": "new"}),
            FakeResponse(202, None, headers={}),
            _ok({"content": [{"type": "text", "text": "{\"login\": \"octo\"}"}], "isError": False}),
        ])
        self.assertEqual(client.call_tool("get_me", {}), {"login": "octo"})
        self.assertEqual(session.calls[-1]["headers"]["Mcp-Session-Id"], "new")

    def test_sse_response_parsed(self):
        sse = (
            "event: message\n"
            'data: {"jsonrpc": "2.0", "method": "notifications/progress", "params": {}}\n\n'
            "event: message\n"
            'data: {"jsonrpc": "2.0", "id": "ID", "result": {"tools": [{"name": "t1"}]}}\n\n'
        )

        class SseSession(FakeSession):
            def request(self, method, url, headers=None, json=None, timeout=None):
                resp = super().request(method, url, headers=headers, json=json, timeout=timeout)
                if resp.text.startswith("event:"):
                    resp.text = resp.text.replace('"ID"', f'"{json["id"]}"')
                    resp.content = resp.text.encode("utf-8")
                return resp

        session = SseSession([FakeResponse(200, None, text=sse, headers={"Content-Type": "text/event-stream"})])
        client = MCPClient("http://x/mcp", session=session, max_retries=0)
        self.assertEqual([t["function"]["name"] for t in client.list_tools()], ["t1"])

    def test_sse_cyrillic_decoded_as_utf8(self):
        # Регрессия: requests декодирует text/event-stream без charset как
        # ISO-8859-1; байт 0x85 из «х» превращался в NEL и разрывал строку.
        body = json.dumps({"jsonrpc": "2.0", "id": "1", "result": {
            "content": [{"type": "text", "text": "Исправлен хвост\u2028и перенос"}], "isError": False,
        }}, ensure_ascii=False)
        raw = ("event: message\r\ndata:" + body + "\r\n\r\n").encode("utf-8")
        resp = FakeResponse(200, None, headers={"Content-Type": "text/event-stream"})
        resp.text = raw.decode("iso-8859-1")
        resp.content = raw
        client = MCPClient("http://x/mcp", session=FakeSession([resp]), max_retries=0)
        self.assertEqual(client.call_tool("t", {}), {"result": "Исправлен хвост\u2028и перенос"})

    def test_sse_batch_and_comments(self):
        body = json.dumps([{"jsonrpc": "2.0", "id": "1", "result": {"tools": [{"name": "b"}]}}])
        text = ": keep-alive\n\ndata: " + body + "\n\n"
        resp = FakeResponse(200, None, text=text, headers={"Content-Type": "text/event-stream"})
        client = MCPClient("http://x/mcp", session=FakeSession([resp]), max_retries=0)
        self.assertEqual([t["function"]["name"] for t in client.list_tools()], ["b"])

    def test_sse_without_answer_is_error(self):
        session = FakeSession([FakeResponse(200, None, text="event: ping\ndata: {}\n\n",
                                            headers={"Content-Type": "text/event-stream"})])
        client = MCPClient("http://x/mcp", session=session, max_retries=0)
        with self.assertRaises(MCPClientError):
            client.call_tool("t", {})


try:
    from mcp.server.fastmcp import FastMCP
    from starlette.testclient import TestClient

    REAL_SDK = True
except ImportError:  # pragma: no cover
    REAL_SDK = False


@unittest.skipUnless(REAL_SDK, "нет mcp/starlette")
class RealStatefulServerTests(unittest.TestCase):
    """Сервер со сессиями и SSE-ответами — как у большинства внешних MCP."""

    def setUp(self):
        server = FastMCP("external")  # по умолчанию: сессии и SSE

        @server.tool()
        def echo(text: str) -> dict:
            return {"echo": text}

        self.http = TestClient(server.streamable_http_app(), base_url="http://localhost:8123")
        self.http.__enter__()

    def tearDown(self):
        self.http.__exit__(None, None, None)

    def test_list_and_call_with_session(self):
        client = MCPClient(
            "http://localhost:8123/mcp", session=self.http, max_retries=0, name="ext", title="Внешний",
            normalize_url=False, handshake=True,
        )
        names = [t["function"]["name"] for t in client.list_tools()]
        self.assertEqual(names, ["echo"])
        self.assertTrue(client._session_id)
        self.assertEqual(client.call_tool("echo", {"text": "привет"}), {"echo": "привет"})

    def test_without_handshake_server_rejects(self):
        client = MCPClient("http://localhost:8123/mcp", session=self.http, max_retries=0, normalize_url=False)
        with self.assertRaises(MCPClientError):
            client.call_tool("echo", {"text": "x"})


if __name__ == "__main__":
    unittest.main()

"""Тесты `agents_core.mcp_client.MCPClient` — без реального HTTP: сессия
`requests` подменяется фейком (тот же приём тестируемости, что и
конструктор `DeepSeekClient(session=...)`), т.к. поднять настоящий сервер
Streamable HTTP в этой песочнице нельзя (`fastapi`/`uvicorn` не
устанавливаются — см. `mcp_server/README.md`, "Ограничения окружения
разработки"). Формат JSON-RPC ответов ниже (`structuredContent.result`,
`isError`) подтверждён прямым экспериментом с `mcp.server.fastmcp.FastMCP`
в процессе (см. историю разработки) — это НЕ выдумка, а зафиксированное
поведение реального SDK при `stateless_http=True, json_response=True`."""

from __future__ import annotations

import unittest

from agents_core.mcp_client import MCPClient, MCPClientError, normalize_mcp_base_url


class FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str = "", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (str(payload) if payload is not None else "")
        self.headers = headers if headers is not None else {"Content-Type": "application/json"}
        self.content = self.text.encode("utf-8")

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


class FakeSession:
    """`responses` — список `FakeResponse`, отдаваемых по очереди (каждый
    вызов `.request()` съедает следующий) либо исключение `requests.RequestException`."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def request(self, method, url, headers=None, json=None, timeout=None):
        self.calls.append({"method": method, "url": url, "headers": headers, "json": json})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _rpc_result(result: dict, id_: str = "1"):
    return FakeResponse(200, {"jsonrpc": "2.0", "id": id_, "result": result})


class ListToolsTests(unittest.TestCase):
    def test_converts_to_openai_function_tool_format(self):
        session = FakeSession([_rpc_result({"tools": [
            {"name": "git_pull", "description": "Обновить репозиторий", "inputSchema": {"type": "object", "properties": {}}},
        ]})])
        client = MCPClient("http://mcp:8001/mcp", session=session)
        tools = client.list_tools()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["function"]["name"], "git_pull")
        self.assertEqual(tools[0]["function"]["description"], "Обновить репозиторий")

    def test_skips_tools_without_name(self):
        session = FakeSession([_rpc_result({"tools": [{"description": "no name"}]})])
        client = MCPClient("http://mcp:8001/mcp", session=session)
        self.assertEqual(client.list_tools(), [])

    def test_returns_cached_list_on_network_failure(self):
        import requests

        session = FakeSession([
            _rpc_result({"tools": [{"name": "git_pull", "description": "", "inputSchema": {}}]}),
            requests.ConnectionError("boom"),
            requests.ConnectionError("boom"),
            requests.ConnectionError("boom"),
            requests.ConnectionError("boom"),
        ])
        client = MCPClient("http://mcp:8001/mcp", session=session, max_retries=3, retry_backoff_seconds=0)
        first = client.list_tools()
        self.assertEqual(len(first), 1)
        second = client.list_tools()  # все retry сгорели на сетевой ошибке
        self.assertEqual(second, first)

    def test_returns_empty_list_when_never_succeeded(self):
        import requests

        session = FakeSession([requests.ConnectionError("boom")] * 4)
        client = MCPClient("http://mcp:8001/mcp", session=session, max_retries=3, retry_backoff_seconds=0)
        self.assertEqual(client.list_tools(), [])

    def test_has_cached_tool_reflects_last_successful_list(self):
        session = FakeSession([_rpc_result({"tools": [{"name": "git_pull", "description": "", "inputSchema": {}}]})])
        client = MCPClient("http://mcp:8001/mcp", session=session)
        self.assertFalse(client.has_cached_tool("git_pull"))
        client.list_tools()
        self.assertTrue(client.has_cached_tool("git_pull"))
        self.assertFalse(client.has_cached_tool("unknown"))


class CallToolTests(unittest.TestCase):
    def test_unwraps_structured_result_dict(self):
        session = FakeSession([_rpc_result({
            "content": [{"type": "text", "text": "ignored"}],
            "structuredContent": {"result": {"id": "abc", "enabled": True}},
            "isError": False,
        })])
        client = MCPClient("http://mcp:8001/mcp", session=session)
        result = client.call_tool("register_scheduled_tool", {"name": "x"})
        self.assertEqual(result, {"id": "abc", "enabled": True})

    def test_domain_error_passed_through_as_error_dict(self):
        # isError=False на уровне протокола, но сам инструмент вернул {"error": ...}
        session = FakeSession([_rpc_result({
            "content": [{"type": "text", "text": '{"error": "not found"}'}],
            "structuredContent": {"result": {"error": "not found"}},
            "isError": False,
        })])
        client = MCPClient("http://mcp:8001/mcp", session=session)
        result = client.call_tool("cancel_scheduled_tool", {"task_id": "nope"})
        self.assertEqual(result, {"error": "not found"})

    def test_protocol_level_error_becomes_error_dict(self):
        session = FakeSession([_rpc_result({
            "content": [{"type": "text", "text": "Unknown tool: bogus"}],
            "isError": True,
        })])
        client = MCPClient("http://mcp:8001/mcp", session=session)
        result = client.call_tool("bogus", {})
        self.assertIn("error", result)
        self.assertIn("Unknown tool", result["error"])

    def test_list_result_wrapped_under_result_key(self):
        session = FakeSession([_rpc_result({
            "content": [{"type": "text", "text": "[]"}],
            "structuredContent": {"result": [{"sha": "abc123"}]},
            "isError": False,
        })])
        client = MCPClient("http://mcp:8001/mcp", session=session)
        result = client.call_tool("git_host_list_commits", {"host": "github", "owner": "a", "repo": "b"})
        self.assertEqual(result, {"result": [{"sha": "abc123"}]})

    def test_network_failure_raises_mcp_client_error(self):
        import requests

        session = FakeSession([requests.ConnectionError("boom")] * 4)
        client = MCPClient("http://mcp:8001/mcp", session=session, max_retries=3, retry_backoff_seconds=0)
        with self.assertRaises(MCPClientError):
            client.call_tool("git_pull", {})

    def test_read_timeout_on_tool_call_is_not_retried(self):
        # Сервер мог уже выполнить вызов (например, записать файл) — повтор
        # выполнил бы его второй раз.
        import requests

        session = FakeSession([requests.ReadTimeout("slow")] * 4)
        client = MCPClient("http://mcp:8001/mcp", session=session, max_retries=3, retry_backoff_seconds=0)
        with self.assertRaises(MCPClientError):
            client.call_tool("save_to_text_file", {"filename": "a.md", "content": "x"})
        self.assertEqual(len(session.calls), 1)

    def test_read_timeout_on_list_tools_is_retried(self):
        import requests

        session = FakeSession([
            requests.ReadTimeout("slow"),
            _rpc_result({"tools": [{"name": "t", "inputSchema": {"type": "object"}}]}),
        ])
        client = MCPClient("http://mcp:8001/mcp", session=session, max_retries=3, retry_backoff_seconds=0)
        self.assertEqual(len(client.list_tools()), 1)
        self.assertEqual(len(session.calls), 2)

    def test_retries_on_5xx_then_succeeds(self):
        session = FakeSession([
            FakeResponse(503, text="service unavailable"),
            _rpc_result({"content": [], "structuredContent": {"result": {"ok": True}}, "isError": False}),
        ])
        client = MCPClient("http://mcp:8001/mcp", session=session, max_retries=3, retry_backoff_seconds=0)
        result = client.call_tool("save_result", {"task_id": "x", "data": {}})
        self.assertEqual(result, {"ok": True})

    def test_authorization_header_sent_when_api_key_set(self):
        session = FakeSession([_rpc_result({"content": [], "structuredContent": {"result": {}}, "isError": False})])
        client = MCPClient("http://mcp:8001/mcp", api_key="secret", session=session)
        client.call_tool("save_result", {"task_id": "x", "data": {}})
        self.assertEqual(session.calls[0]["headers"]["Authorization"], "Bearer secret")

    def test_no_authorization_header_when_api_key_empty(self):
        session = FakeSession([_rpc_result({"content": [], "structuredContent": {"result": {}}, "isError": False})])
        client = MCPClient("http://mcp:8001/mcp", session=session)
        client.call_tool("save_result", {"task_id": "x", "data": {}})
        self.assertNotIn("Authorization", session.calls[0]["headers"])


class ConstructorTests(unittest.TestCase):
    def test_rejects_empty_base_url(self):
        with self.assertRaises(ValueError):
            MCPClient("")

    def test_appends_missing_mcp_suffix(self):
        # Регрессия: настроенный без /mcp базовый адрес (перепутан с REST-адресом
        # для AgentsApp) раньше уходил в JSON-RPC как есть и MCP-сервер отвечал
        # 404 — см. README.md, "Частая ошибка конфигурации".
        client = MCPClient("http://localhost:8001/")
        self.assertEqual(client.base_url, "http://localhost:8001/mcp")

    def test_keeps_url_already_ending_with_mcp(self):
        client = MCPClient("http://localhost:8001/mcp")
        self.assertEqual(client.base_url, "http://localhost:8001/mcp")

    def test_keeps_url_already_ending_with_mcp_no_trailing_slash_variants(self):
        client = MCPClient("http://localhost:8001/mcp/")
        self.assertEqual(client.base_url, "http://localhost:8001/mcp")


class NormalizeMcpBaseUrlTests(unittest.TestCase):
    def test_appends_mcp_when_missing(self):
        self.assertEqual(normalize_mcp_base_url("http://mcp:8001"), "http://mcp:8001/mcp")
        self.assertEqual(normalize_mcp_base_url("http://mcp:8001/"), "http://mcp:8001/mcp")

    def test_leaves_url_ending_with_mcp_untouched(self):
        self.assertEqual(normalize_mcp_base_url("http://mcp:8001/mcp"), "http://mcp:8001/mcp")
        self.assertEqual(normalize_mcp_base_url("http://mcp:8001/MCP"), "http://mcp:8001/MCP")

    def test_empty_string_stays_empty(self):
        self.assertEqual(normalize_mcp_base_url(""), "")


if __name__ == "__main__":
    unittest.main()

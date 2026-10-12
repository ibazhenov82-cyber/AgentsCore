"""
Тесты интеграции `Repository` с MCP-сервером (новое ТЗ, третий компонент):
слияние инструментов (`_settings_with_merged_tools`), диспетчеризация
неизвестных локально вызовов в `mcp_client.call_tool`
(`_execute_tool_call`), накопление `Message.tool_events` (source="mcp") и
`SettingsOut.tools_sources` (через `Repository._tools_sources_for`).

Полностью офлайн: `MCPClient` подменяется фейком (`FakeMCPClient`) — ни
реальный MCP-сервер, ни сеть здесь не нужны, как и в
`test_memory_and_profiles.py` для tool-calling в целом."""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List, Optional

from agents_core.mcp_client import MCPClientError
from agents_core.repository import Repository
from tests.test_memory_and_profiles import ToolCallingFakeProvider, _tool_call, make_tool_repo
from tests.test_repository import FakeRegistry, TEST_MODEL_ID, make_repository


class FakeMCPClient:
    """Заменяет `agents_core.mcp_client.MCPClient` в тестах — та же
    публичная поверхность (`list_tools`, `has_cached_tool`, `call_tool`),
    без какого-либо реального HTTP (см. `MCPClient` — там это уже отдельно
    протестировано в `test_mcp_client.py`; здесь важно только то, как
    `Repository` его ИСПОЛЬЗУЕТ)."""

    def __init__(self, tools: Optional[List[Dict[str, Any]]] = None):
        self._tools = tools or []
        self._names = {t["function"]["name"] for t in self._tools}
        #: очередь: следующий call_tool либо вернёт значение отсюда, либо
        #: поднимет исключение, если элемент — Exception.
        self.call_queue: List[Any] = []
        self.calls: List[tuple] = []
        self.metas: List[Optional[Dict[str, Any]]] = []

    def list_tools(self) -> List[Dict[str, Any]]:
        return list(self._tools)

    def has_cached_tool(self, name: str) -> bool:
        return name in self._names

    def tool_group(self, name: str) -> Optional[str]:
        return "GIT API" if name in self._names else None

    def call_tool(self, name: str, arguments: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.calls.append((name, arguments))
        self.metas.append(meta)
        if self.call_queue:
            item = self.call_queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return {"ok": True}


def _mcp_tool(name: str, description: str = "") -> Dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": {}}}}


def make_tool_repo_with_mcp(mcp_client: FakeMCPClient) -> Repository:
    repo = make_tool_repo()
    repo._mcp_client = mcp_client  # type: ignore[attr-defined]
    return repo


class MergedToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mcp = FakeMCPClient(tools=[_mcp_tool("git_pull", "Обновить репозиторий")])
        self.repo = make_repository()
        self.repo._mcp_client = self.mcp  # type: ignore[attr-defined]
        self.agent = self.repo.create_agent("Агент", model=TEST_MODEL_ID)
        self.chat = self.repo.create_chat(self.agent.id, "Чат")

    def tearDown(self) -> None:
        import os
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def _select(self, *names: str):
        self.repo.update_chat_settings(self.chat.id, {"tools_json": json.dumps([_mcp_tool(n) for n in names])})
        return self.repo.get_chat(self.chat.id)

    def test_unselected_mcp_tools_are_not_offered(self):
        # Строгое правило (по замечанию пользователя): ничего не выбрано в
        # настройках — модели не предлагается ни один MCP-инструмент, даже
        # если MCP-клиент подключён.
        merged = self.repo._settings_with_merged_tools(self.chat)
        names = {t["function"]["name"] for t in json.loads(merged.tools_json or "[]")}
        self.assertNotIn("git_pull", names)

    def test_selected_mcp_tool_is_offered(self):
        chat = self._select("git_pull")
        merged = self.repo._settings_with_merged_tools(chat)
        names = {t["function"]["name"] for t in json.loads(merged.tools_json)}
        self.assertEqual(names, {"git_pull"})

    def test_no_mcp_client_means_no_tools_added(self):
        self.repo._mcp_client = None  # type: ignore[attr-defined]
        merged = self.repo._settings_with_merged_tools(self.chat)
        self.assertEqual(merged.tools_json, self.chat.settings.tools_json)

    def test_tools_sources_includes_mcp_when_mcp_tool_selected(self):
        chat = self._select("git_pull")
        self.assertIn("mcp", self.repo._tools_sources_for(chat.settings, chat))

    def test_tools_sources_excludes_mcp_when_nothing_selected(self):
        self.assertNotIn("mcp", self.repo._tools_sources_for(self.chat.settings, self.chat))

    def test_tools_sources_excludes_mcp_when_client_absent(self):
        chat = self._select("git_pull")
        self.repo._mcp_client = None  # type: ignore[attr-defined]
        self.assertNotIn("mcp", self.repo._tools_sources_for(chat.settings, chat))

    def test_selected_tool_appears_once_with_saved_description(self):
        # Выбранный инструмент попадает в запрос ровно один раз и в той
        # версии описания, что сохранена в tools_json.
        self.repo.update_chat_settings(self.chat.id, {
            "tools_json": json.dumps([_mcp_tool("git_pull", "своя копия описания")]),
        })
        chat = self.repo.get_chat(self.chat.id)
        merged = self.repo._settings_with_merged_tools(chat)
        tools = json.loads(merged.tools_json)
        git_pull_entries = [t for t in tools if t["function"]["name"] == "git_pull"]
        self.assertEqual(len(git_pull_entries), 1)
        # tools_json идёт первым в порядке слияния — его версия описания и должна победить.
        self.assertEqual(git_pull_entries[0]["function"]["description"], "своя копия описания")

    def test_configured_tools_json_restricts_mcp_tools(self):
        # Регрессия (реальный баг): модель вызвала execute_git_command, не
        # выбранный в настройках чата — раньше живой MCP-список подмешивался
        # целиком ПОВЕРХ выбранного в «Выбрать из доступных».
        self.mcp = FakeMCPClient(tools=[
            _mcp_tool("git_host_get_repo"), _mcp_tool("execute_git_command"),
        ])
        self.repo._mcp_client = self.mcp  # type: ignore[attr-defined]
        self.repo.update_chat_settings(self.chat.id, {
            "tools_json": json.dumps([_mcp_tool("git_host_get_repo")]),
        })
        chat = self.repo.get_chat(self.chat.id)
        names = {t["function"]["name"] for t in json.loads(self.repo._settings_with_merged_tools(chat).tools_json)}
        self.assertEqual(names, {"git_host_get_repo"})

    def test_explicit_empty_selection_disables_all_mcp_tools(self):
        # "[]" — так пикер кодирует "явно не выбрано ни одного инструмента".
        self.repo.update_chat_settings(self.chat.id, {"tools_json": "[]"})
        chat = self.repo.get_chat(self.chat.id)
        merged = self.repo._settings_with_merged_tools(chat)
        self.assertEqual(json.loads(merged.tools_json or "[]"), [])


class ExecuteToolCallDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mcp = FakeMCPClient(tools=[_mcp_tool("git_pull")])
        self.repo = make_tool_repo_with_mcp(self.mcp)
        self.agent = self.repo.create_agent("Агент", model=TEST_MODEL_ID)
        self.chat = self.repo.create_chat(self.agent.id, "Чат")
        # MCP-инструменты доступны модели только если выбраны в настройках.
        self.repo.update_chat_settings(self.chat.id, {"tools_json": json.dumps([_mcp_tool("git_pull")])})

    def tearDown(self) -> None:
        import os
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def test_local_handler_takes_priority_over_mcp(self):
        # save_working_memory существует локально — MCP не должен трогаться,
        # даже если бы имя случайно совпало (здесь оно просто не в кэше MCP).
        provider: ToolCallingFakeProvider = self.repo._tool_provider  # type: ignore[attr-defined]
        provider.tool_calls_queue = [[_tool_call("call1", "save_working_memory", {"key": "k", "value": "v"})]]
        self.repo.update_chat_settings(self.chat.id, {"memory_tools_enabled": True, "working_memory_enabled": True})
        self.repo.send_message_blocking(self.chat.id, "Привет")
        self.assertEqual(self.mcp.calls, [])

    def test_unknown_local_name_dispatches_to_mcp_when_cached(self):
        provider: ToolCallingFakeProvider = self.repo._tool_provider  # type: ignore[attr-defined]
        self.mcp.call_queue = [{"pulled": True}]
        provider.tool_calls_queue = [[_tool_call("call1", "git_pull", {"repo_path": "/tmp/x"})]]
        user_msg, assistant_msg = self.repo.send_message_blocking(self.chat.id, "Обнови репозиторий")
        self.assertEqual(self.mcp.calls, [("git_pull", {"repo_path": "/tmp/x"})])
        # Сервер получает, из какого чата и агента пришёл вызов (нужно планировщику).
        # и какие MCP-инструменты разрешены в чате (сервер ограничивает ими run_pipeline).
        self.assertEqual(self.mcp.metas[0], {
            "agentscore/chat_id": self.chat.id, "agentscore/agent_id": self.agent.id,
            "agentscore/allowed_tools": ["git_pull"],
        })
        self.assertEqual(assistant_msg.content, provider.final_text)
        events = _mcp_events(assistant_msg)
        self.assertEqual([e["status"] for e in events], ["started", "finished"])
        self.assertTrue(events[1]["ok"])
        self.assertEqual(events[0]["name"], "git_pull")
        # Подпись источника MCP-инструмента — для строки «Использую инструмент …».
        self.assertEqual(events[0]["group"], "GIT API")

    def test_truly_unknown_name_not_in_mcp_cache_returns_error(self):
        provider: ToolCallingFakeProvider = self.repo._tool_provider  # type: ignore[attr-defined]
        provider.tool_calls_queue = [[_tool_call("call1", "does_not_exist_anywhere", {})]]
        user_msg, assistant_msg = self.repo.send_message_blocking(self.chat.id, "Привет")
        self.assertEqual(_mcp_events(assistant_msg), [])
        tool_message = next(m for m in provider.calls_log[1] if m.role == "tool")
        self.assertIn("unknown tool", tool_message.content)

    def test_mcp_tool_not_offered_to_model_is_not_dispatched(self):
        # Вторая линия защиты: даже если модель "придумает" вызов MCP-
        # инструмента, которого не было в предложенном ей списке (чат
        # ограничен выбором в настройках), он не должен уйти на MCP-сервер.
        self.mcp._tools.append(_mcp_tool("execute_git_command"))
        self.mcp._names.add("execute_git_command")
        self.repo.update_chat_settings(self.chat.id, {"tools_json": json.dumps([_mcp_tool("git_pull")])})
        provider: ToolCallingFakeProvider = self.repo._tool_provider  # type: ignore[attr-defined]
        provider.tool_calls_queue = [[_tool_call("call1", "execute_git_command", {"repo_path": "/", "command": "status"})]]
        _, assistant_msg = self.repo.send_message_blocking(self.chat.id, "Выполни git status")
        self.assertEqual(self.mcp.calls, [])
        self.assertEqual(_mcp_events(assistant_msg), [])
        tool_message = next(m for m in provider.calls_log[1] if m.role == "tool")
        self.assertIn("unknown tool", tool_message.content)

    def test_mcp_domain_error_recorded_as_not_ok(self):
        provider: ToolCallingFakeProvider = self.repo._tool_provider  # type: ignore[attr-defined]
        self.mcp.call_queue = [{"error": "репозиторий не найден"}]
        provider.tool_calls_queue = [[_tool_call("call1", "git_pull", {"repo_path": "/nope"})]]
        _, assistant_msg = self.repo.send_message_blocking(self.chat.id, "Обнови")
        events = _mcp_events(assistant_msg)
        finished = events[1]
        self.assertFalse(finished["ok"])
        self.assertEqual(finished["error"], "репозиторий не найден")

    def test_mcp_network_failure_becomes_error_result_and_finished_event(self):
        provider: ToolCallingFakeProvider = self.repo._tool_provider  # type: ignore[attr-defined]
        self.mcp.call_queue = [MCPClientError("сервер недоступен")]
        provider.tool_calls_queue = [[_tool_call("call1", "git_pull", {"repo_path": "/x"})]]
        _, assistant_msg = self.repo.send_message_blocking(self.chat.id, "Обнови")
        tool_message = next(m for m in provider.calls_log[1] if m.role == "tool")
        self.assertIn("сервер недоступен", tool_message.content)
        events = _mcp_events(assistant_msg)
        self.assertFalse(events[1]["ok"])
        self.assertIn("сервер недоступен", events[1]["error"])

    def test_streaming_yields_mcp_tool_call_events_inline(self):
        provider: ToolCallingFakeProvider = self.repo._tool_provider  # type: ignore[attr-defined]
        self.mcp.call_queue = [{"pulled": True}]
        provider.tool_calls_queue = [[_tool_call("call1", "git_pull", {"repo_path": "/tmp/x"})]]
        events = list(self.repo.stream_message(self.chat.id, "Обнови репозиторий"))
        mcp_call_events = [e for e in events if e.get("type") == "tool_call" and e.get("source") == "mcp"]
        self.assertEqual([e["status"] for e in mcp_call_events], ["started", "finished"])
        done = next(e for e in events if e["type"] == "done")
        self.assertEqual(len(_mcp_events(done["message"])), 2)


def _mcp_events(message) -> list:
    """События вызовов MCP-инструментов из `Message.tool_events`."""
    if not message.tool_events:
        return []
    return [e for e in json.loads(message.tool_events) if e.get("source") == "mcp"]


if __name__ == "__main__":
    unittest.main()

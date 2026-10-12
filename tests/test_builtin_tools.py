"""
Встроенные инструменты «Доступ к agents core»: `read_chat` («Чтение чата») —
список чатов, поиск чата по id и названию, переписка, источники RAG и
ссылки; доступ модели только при выборе инструмента в `tools_json`;
показ в общем списке инструментов вместе с MCP.
"""

from __future__ import annotations

import json
import os
import time
import unittest

from agents_core import builtin_tools
from agents_core.models import Message
from tests.test_memory_and_profiles import ToolCallingFakeProvider, _tool_call
from tests.test_repository import FakeRegistry, TEST_MODEL_ID, make_repository

RAG = {
    "status": "ok", "query": "сроки ревью",
    "sources": [
        {"n": 1, "title": "Регламент ревью", "section": "Сроки", "page": 2, "source": "/kb/review.pdf",
         "collection_name": "Регламенты", "doc_date": "2026-01-10", "text": "…"},
        {"n": 2, "title": "Wiki", "source": "https://wiki.example.com/review", "collection_name": "Wiki"},
    ],
    "cited": [1], "confirmed": True,
}


def tools_json(*names: str) -> str:
    return json.dumps([{"type": "function", "function": {
        "name": n, "description": builtin_tools.SPECS[n]["description"],
        "parameters": builtin_tools.SPECS[n]["parameters"]}} for n in names], ensure_ascii=False)


class ReadChatTests(unittest.TestCase):
    def setUp(self):
        self.repo = make_repository()
        self.agent = self.repo.create_agent("Аналитик", model=TEST_MODEL_ID)
        self.other_agent = self.repo.create_agent("Разработчик", model=TEST_MODEL_ID)
        self.current = self.repo.create_chat(self.agent.id, "Текущий")
        self.source = self.repo.create_chat(self.agent.id, "Регламенты ревью")
        self.second = self.repo.create_chat(self.other_agent.id, "Регламенты выкладки")
        now = int(time.time())
        db = self.repo._db
        db.add_message(Message(id=0, chat_id=self.source.id, role="user", content="Сколько дней на ревью?",
                               created_at=now))
        db.add_message(Message(
            id=0, chat_id=self.source.id, role="assistant", created_at=now + 1,
            content="Не позднее двух рабочих дней [1]. Подробнее: https://wiki.example.com/review.",
            rag=json.dumps(RAG, ensure_ascii=False),
            tool_events=json.dumps([{"type": "tool_call", "name": "duckduckgo_search", "status": "started"},
                                    {"type": "tool_call", "name": "duckduckgo_search", "status": "finished"}]),
        ))
        db.add_message(Message(id=0, chat_id=self.source.id, role="user", content="А для срочных?",
                               created_at=now + 2, source="test_dialog"))

    def tearDown(self):
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def call(self, **arguments):
        return builtin_tools.read_chat(self.repo, self.current, arguments)

    def test_list_without_chat(self):
        result = self.call()
        self.assertEqual(result["total"], 3)
        titles = {c["title"]: c for c in result["chats"]}
        self.assertEqual(titles["Регламенты выкладки"]["agent"], "Разработчик")
        self.assertTrue(titles["Текущий"]["current"])

    def test_stats_off_by_default(self):
        result = self.call(chat="регламенты ревью")
        self.assertNotIn("stats", result)
        self.assertTrue(all("stats" not in m for m in result["messages"]))

    def test_stats_on_request(self):
        db = self.repo._db
        db.add_message(Message(
            id=0, chat_id=self.source.id, role="assistant", content="Один день.", created_at=int(time.time()) + 3,
            prompt_tokens=1105, completion_tokens=156, total_tokens=1261, duration_ms=47000,
            metrics=json.dumps({"tokens_per_sec": 8.8, "ttft_ms": 1200, "load_ms": 2000, "memory_mb": 6144,
                                "vram_mb": 6144, "gpu_percent": 100, "num_ctx": 8192}),
        ))
        self.repo.update_chat_settings(self.source.id, {"temperature": 0.2, "num_ctx": 8192})
        result = self.call(chat=self.source.id, include_stats=True)
        last = result["messages"][-1]["stats"]
        self.assertEqual(last["tokens_in"], 1105)
        self.assertEqual(last["duration_s"], 47.0)
        self.assertEqual(last["first_token_s"], 1.2)
        self.assertEqual(last["memory_gb"], 6.0)
        self.assertEqual(last["num_ctx"], 8192)
        self.assertNotIn("stats", result["messages"][0])  # у вопросов статистики нет
        chat_stats = result["stats"]
        self.assertEqual(chat_stats["model"], TEST_MODEL_ID)
        self.assertEqual(chat_stats["settings"]["temperature"], 0.2)
        self.assertEqual(chat_stats["settings"]["num_ctx"], 8192)
        self.assertEqual(chat_stats["summary"]["avg_tokens_per_sec"], 8.8)
        self.assertEqual(chat_stats["summary"]["avg_duration_s"], 45.0)  # без загрузки модели
        self.assertEqual(chat_stats["summary"]["max_memory_gb"], 6.0)

    def test_read_by_title_with_sources_and_links(self):
        result = self.call(chat="регламенты ревью")
        self.assertEqual(result["chat"]["id"], self.source.id)
        self.assertEqual(result["total_messages"], 3)
        user, answer, follow = result["messages"]
        self.assertEqual((user["role"], user["text"]), ("user", "Сколько дней на ревью?"))
        self.assertEqual(answer["sources"][0], {
            "n": 1, "collection": "Регламенты", "document": "Регламент ревью", "section": "Сроки", "page": 2,
            "source": "/kb/review.pdf", "date": "2026-01-10", "cited": True,
        })
        self.assertFalse(answer["sources"][1]["cited"])
        self.assertEqual(answer["links"], ["https://wiki.example.com/review"])
        self.assertEqual(answer["tools"], ["duckduckgo_search"])
        self.assertEqual(follow["from"], "test_dialog")

    def test_read_by_id_last_messages_without_sources(self):
        result = self.call(chat=self.source.id, last_messages=2, include_sources=False)
        self.assertEqual(result["returned"], 2)
        self.assertEqual(result["earlier_messages_omitted"], 1)
        self.assertNotIn("sources", result["messages"][0])
        self.assertNotIn("links", result["messages"][0])

    def test_ambiguous_and_missing(self):
        ambiguous = self.call(chat="Регламенты")
        self.assertIn("несколько чатов", ambiguous["error"])
        self.assertEqual(len(ambiguous["chats"]), 2)
        narrowed = self.call(chat="Регламенты", agent="разработ")
        self.assertEqual(narrowed["chat"]["id"], self.second.id)
        missing = self.call(chat="Отпуска")
        self.assertIn("не найден", missing["error"])
        self.assertEqual(missing["total"], 3)

    def test_long_text_is_truncated(self):
        self.repo._db.add_message(Message(id=0, chat_id=self.second.id, role="assistant",
                                          content="я" * (builtin_tools.MAX_MESSAGE_CHARS + 10),
                                          created_at=int(time.time())))
        message = self.call(chat=self.second.id)["messages"][0]
        self.assertTrue(message["text_truncated"])
        self.assertEqual(len(message["text"]), builtin_tools.MAX_MESSAGE_CHARS + 1)


class ToolAccessTests(unittest.TestCase):
    def setUp(self):
        self.repo = make_repository()
        self.provider = ToolCallingFakeProvider()
        self.repo._registry = FakeRegistry(self.provider)  # type: ignore[attr-defined]
        self.agent = self.repo.create_agent("Агент", model=TEST_MODEL_ID)
        self.chat = self.repo.create_chat(self.agent.id, "Текущий")
        self.other = self.repo.create_chat(self.agent.id, "Исследование")
        self.repo._db.add_message(Message(id=0, chat_id=self.other.id, role="user", content="Найди статьи про RAG",
                                          created_at=int(time.time())))

    def tearDown(self):
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def tool_result(self) -> dict:
        tool_messages = [m for m in self.provider.calls_log[-1] if m.role == "tool"]
        return json.loads(tool_messages[-1].content)

    def test_listed_with_mcp_tools_under_group(self):
        tools = self.repo.list_mcp_tools()
        self.assertEqual([(t["name"], t["group"], t["title"]) for t in tools],
                         [("read_chat", "Доступ к agents core", "Чтение чата")])

    def test_model_reads_chat_when_tool_selected(self):
        self.repo.update_chat_settings(self.chat.id, {"tools_json": tools_json("read_chat")})
        self.provider.tool_calls_queue = [[_tool_call("c1", "read_chat", {"chat": "Исследование"})]]
        _, reply = self.repo.send_message_blocking(self.chat.id, "Что спрашивали в чате «Исследование»?")
        result = self.tool_result()
        self.assertEqual(result["messages"][0]["text"], "Найди статьи про RAG")
        events = json.loads(reply.tool_events)
        self.assertEqual({(e["source"], e["group"]) for e in events}, {("agents_core", "Доступ к agents core")})
        self.assertIn("agents_core", self.repo._tools_sources_for(self.repo.get_chat(self.chat.id).settings))

    def test_not_available_when_not_selected(self):
        self.repo.update_chat_settings(self.chat.id, {"tools_json": json.dumps([{"type": "function", "function": {
            "name": "other_tool", "parameters": {"type": "object", "properties": {}}}}])})
        self.provider.tool_calls_queue = [[_tool_call("c1", "read_chat", {"chat": "Исследование"})]]
        self.repo.send_message_blocking(self.chat.id, "Прочитай чат")
        self.assertIn("unknown tool", self.tool_result()["error"])


if __name__ == "__main__":
    unittest.main()

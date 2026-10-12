"""«Мини-чат с RAG + памятью»: задача поиска (kind=search), память задачи
(цель, уточнения, ограничения, термины, источники), «БЕЗ_ПОИСКА», переходы
по итогам ответа, настройка «Останавливаться на каждом этапе»."""

from __future__ import annotations

import json
import os
import unittest

from agents_core.providers import ChatResult
from agents_core.providers.base import ChatUsage
from agents_core.repository import ValidationError
from agents_core.task_state_machine import (
    format_search_states_for_prompt, kind_allowed, kind_display_name, kind_initial_state, kind_states,
)
from tests.test_memory_and_profiles import ToolCallingFakeProvider, make_tool_repo
from tests.test_rag import FakeKnowledge, result
from tests.test_repository import TEST_MODEL_ID

try:
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    FASTAPI_AVAILABLE = False

if FASTAPI_AVAILABLE:
    from agents_core.api.app import create_app_with_repository


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}


class SearchStateMachineModuleTests(unittest.TestCase):
    def test_search_machine(self):
        self.assertEqual(kind_states("search"), ["clarifying", "answering", "achieved"])
        self.assertEqual(kind_initial_state("search"), "answering")
        self.assertEqual(kind_allowed("search", "clarifying"), ["answering"])
        self.assertEqual(kind_allowed("search", "answering"), ["clarifying", "achieved"])
        self.assertEqual(kind_allowed("search", "achieved"), ["answering"])
        self.assertEqual(kind_display_name("search", "achieved"), "Цель достигнута")
        self.assertIn("2 - ANSWERING (TRANSITIONS: CLARIFYING, ACHIEVED)", format_search_states_for_prompt())

    def test_work_machine_unchanged(self):
        self.assertEqual(kind_states("work"), ["planning", "execution", "validation", "done"])
        self.assertEqual(kind_display_name("work", "execution"), "Выполнение")


class _SearchProvider(ToolCallingFakeProvider):
    """Фейк с моделью переписывания: запросы с инструкцией переписывания
    получают `rewritten`, остальные — как у `ToolCallingFakeProvider`."""

    def __init__(self):
        super().__init__()
        self.rewritten = "поисковый запрос"
        self.rewrite_requests = []

    def _is_rewrite(self, messages) -> bool:
        return bool(messages) and messages[0].role == "system" and messages[0].content.startswith("Ты переписываешь")

    def chat(self, model_id, messages, settings):
        if self._is_rewrite(messages):
            self.rewrite_requests.append(list(messages))
            return ChatResult(content=self.rewritten, usage=ChatUsage(total_tokens=20))
        return super().chat(model_id, messages, settings)


class SearchTaskTests(unittest.TestCase):
    def setUp(self):
        self.repo = make_tool_repo()
        self.provider = _SearchProvider()
        self.repo._registry._provider = self.provider  # type: ignore[attr-defined]
        self.knowledge = FakeKnowledge([
            result(1, "Сервер MCP подключается через MCP_SERVERS_FILE."),
            result(2, "Токену GitHub достаточно прав только на чтение."),
        ])
        self.repo._knowledge = self.knowledge
        agent = self.repo.create_agent("A", TEST_MODEL_ID)
        self.chat = self.repo.create_chat(agent.id, "Чат")
        self.repo.update_chat_settings(self.chat.id, {
            "rag_enabled": True, "collection_ids": ["col_1"], "task_tracking_enabled": True,
            "rag_query_rewrite": "llm",
            "tools_json": json.dumps([{"type": "function", "function": {
                "name": "git_commit", "description": "commit", "parameters": {"type": "object", "properties": {}}}}]),
        })

    def tearDown(self):
        try:
            os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]
        except OSError:
            pass

    # ---- helpers -------------------------------------------------------------

    def send(self, text: str, answer: str = "Через MCP_SERVERS_FILE [1].", calls=None):
        self.provider.final_text = answer
        self.provider.tool_calls_queue = [calls] if calls else []
        return self.repo.send_message_blocking(self.chat.id, text)

    def start_search(self, **extra) -> str:
        _, reply = self.send("Как подключить GitHub MCP?", calls=[
            _tool_call("c1", "start_task", {"kind": "search", "title": "GitHub MCP",
                                            "goal": "Инструкция по подключению GitHub MCP", **extra}),
        ])
        events = json.loads(reply.task_events)
        return events[0]["task_id"]

    def offered_tools(self) -> set:
        chat = self.repo._require_chat(self.chat.id)
        return self.repo._offered_tool_names(self.repo._settings_with_merged_tools(chat))

    # ---- tests ---------------------------------------------------------------

    def test_start_search_task_not_paused_by_default(self):
        task_id = self.start_search()
        detail = self.repo.get_task(task_id)
        task = detail["task"]
        self.assertEqual((task.kind, task.state, task.paused), ("search", "answering", False))
        self.assertEqual(task.goal, "Инструкция по подключению GitHub MCP")
        self.assertEqual(detail["kind_display_name"], "Задача поиска")
        self.assertEqual([s["state"] for s in detail["stages"]], ["clarifying", "answering", "achieved"])
        # Ответ процитировал [1] — фрагмент накоплен в задаче.
        self.assertEqual([s["chunk_id"] for s in task.sources], ["doc_1-1"])

    def test_needs_clarification_starts_in_clarifying(self):
        task_id = self.start_search(needs_clarification=True)
        self.assertEqual(self.repo.get_task(task_id)["task"].state, "clarifying")

    def test_search_task_requires_rag(self):
        self.repo.update_chat_settings(self.chat.id, {"rag_enabled": False})
        chat = self.repo._require_chat(self.chat.id)
        agent = self.repo._require_agent(chat.agent_id)
        res = self.repo._handle_start_task(chat, agent, {"kind": "search", "title": "x"})
        self.assertIn("error", res)
        tool = next(t for t in json.loads(self.repo._settings_with_merged_tools(chat).tools_json)
                    if t["function"]["name"] == "start_task")
        self.assertNotIn("kind", tool["function"]["parameters"]["properties"])

    def test_acting_tools_not_offered_in_search_task(self):
        self.assertIn("git_commit", self.offered_tools())
        task_id = self.start_search()
        offered = self.offered_tools()
        self.assertNotIn("git_commit", offered)
        self.assertTrue({"start_task", "update_task_memory", "apply_task_action"} <= offered)
        # Цель достигнута — ограничение снимается (пользователь может попросить сделать).
        self.repo.set_task_state_manually(task_id, "achieved")
        self.assertIn("git_commit", self.offered_tools())

    def test_prompt_block_with_memory(self):
        task_id = self.start_search()
        self.send("Нужен только просмотр, без записи.", answer="Понял, только чтение [2].", calls=[
            _tool_call("m1", "update_task_memory", {
                "task_id": task_id, "add_constraints": ["только чтение"],
                "add_terms": [{"term": "отчёт", "definition": "список коммитов с авторами"}],
            }),
        ])
        self.send("Какие права нужны токену?", answer="Только чтение [2].")
        system_texts = "\n".join(m.content for m in self.provider.last_messages if m.role == "system")
        self.assertIn(f"[ЗАДАЧА ПОИСКА] id: {task_id}", system_texts)
        self.assertIn("Цель: Инструкция по подключению GitHub MCP", system_texts)
        self.assertIn("- только чтение", system_texts)
        self.assertIn("- отчёт — список коммитов с авторами", system_texts)
        self.assertIn("Ничего не выполняй сам", system_texts)
        # Переписывание запроса видит память задачи.
        self.assertIn("только чтение", self.provider.rewrite_requests[-1][0].content)

    def test_update_task_memory_event_and_remove(self):
        task_id = self.start_search()
        _, reply = self.send("Порт 8081 занят.", answer="Учту [1].", calls=[
            _tool_call("m1", "update_task_memory", {"task_id": task_id, "add_constraints": ["порт 8081 занят"]}),
        ])
        events = json.loads(reply.task_events)
        memory_event = next(e for e in events if e["kind"] == "memory")
        self.assertEqual(memory_event["changes"], [{"op": "+", "field": "constraint", "text": "порт 8081 занят"}])
        task = self.repo.get_task(task_id)["task"]
        self.assertEqual(task.constraints[0]["text"], "порт 8081 занят")
        self.assertIsNotNone(task.constraints[0]["message_id"])
        self.send("Порт освободился.", answer="Хорошо [1].", calls=[
            _tool_call("m2", "update_task_memory", {"task_id": task_id, "remove": ["Порт 8081 занят"]}),
        ])
        self.assertEqual(self.repo.get_task(task_id)["task"].constraints, [])

    def test_no_search_reuses_previous_sources(self):
        self.start_search()
        calls_before = len(self.knowledge.calls)
        self.provider.rewritten = "БЕЗ_ПОИСКА"
        _, reply = self.send("Повтори коротко", answer="Коротко: MCP_SERVERS_FILE [1].")
        self.assertEqual(len(self.knowledge.calls), calls_before)
        rag = json.loads(reply.rag)
        self.assertTrue(rag["rewrite"]["no_search"])
        self.assertTrue(rag["context_reused"]["message_id"])
        self.assertEqual([s["chunk_id"] for s in rag["sources"]], ["doc_1-1", "doc_1-2"])
        self.assertEqual(rag["cited"], [1])
        self.assertFalse(rag["no_sources"])

    def test_no_search_without_previous_sources_searches(self):
        self.provider.rewritten = "БЕЗ_ПОИСКА"
        _, reply = self.send("Повтори")
        self.assertEqual(len(self.knowledge.calls), 1)
        rag = json.loads(reply.rag)
        self.assertEqual(rag["context_reused"], {})
        self.assertIn("no_search_note", rag["rewrite"])

    def test_fallback_to_task_sources_when_nothing_found(self):
        self.start_search()
        self.knowledge.results = []
        _, reply = self.send("А что дальше?", answer="Дальше — MCP_SERVERS_FILE [1].")
        rag = json.loads(reply.rag)
        self.assertEqual(rag["status"], "ok")
        self.assertTrue(rag["task_sources_fallback"])
        self.assertEqual(rag["sources"][0]["chunk_id"], "doc_1-1")
        self.assertTrue(rag["sources"][0]["from_task"])

    def test_task_sources_added_to_new_results(self):
        self.start_search()
        self.knowledge.results = [result(3, "Планировщик пишет в отдельный чат.")]
        _, reply = self.send("А планировщик?", answer="В отдельный чат [1].")
        rag = json.loads(reply.rag)
        self.assertEqual([s["chunk_id"] for s in rag["sources"]], ["doc_1-3", "doc_1-1"])
        self.assertEqual(rag["task_sources_added"], 1)

    def test_auto_transitions_dont_know_and_back(self):
        task_id = self.start_search()
        self.knowledge.results = []
        self.repo.update_task_memory_manually(task_id, clear_sources=True)
        _, reply = self.send("Как настроить winget?", answer="Не знаю. Уточните, какая у вас ОС?")
        rag = json.loads(reply.rag)
        self.assertTrue(rag["dont_know"])
        self.assertTrue(rag["no_sources"])
        self.assertEqual(self.repo.get_task(task_id)["task"].state, "clarifying")
        state_event = next(e for e in json.loads(reply.task_events) if e["kind"] == "state")
        self.assertEqual(state_event["to_state_display_name"], "Уточнение")
        self.knowledge.results = [result(1, "Сервер MCP подключается через MCP_SERVERS_FILE.")]
        self.send("Windows 11", answer="Через MCP_SERVERS_FILE [1].")
        self.assertEqual(self.repo.get_task(task_id)["task"].state, "answering")

    def test_model_sets_achieved_and_new_question_reopens(self):
        task_id = self.start_search()
        self.send("Собери итоговую инструкцию", answer="1. MCP_SERVERS_FILE [1].", calls=[
            _tool_call("a1", "apply_task_action", {"task_id": task_id, "target_state": "achieved"}),
        ])
        detail = self.repo.get_task(task_id)
        self.assertEqual((detail["task"].state, detail["status"], detail["status_display"]),
                         ("achieved", "done", "цель достигнута"))
        self.send("А токен где хранить?", answer="В .env [2].")
        self.assertEqual(self.repo.get_task(task_id)["task"].state, "answering")

    def test_manual_state_and_memory(self):
        task_id = self.start_search()
        with self.assertRaises(ValidationError):
            self.repo.set_task_state_manually(task_id, "planning")
        detail = self.repo.set_task_state_manually(task_id, "achieved")
        self.assertEqual(detail["history"][-1]["applied_by"], "manual")
        detail = self.repo.update_task_memory_manually(
            task_id, goal="Новая цель", constraints=["только CPU"],
            terms=[{"term": "база", "definition": "коллекция «Документация продукта»"}],
        )
        task = detail["task"]
        self.assertEqual((task.goal, task.constraints[0]["text"], task.terms[0]["term"]),
                         ("Новая цель", "только CPU", "база"))
        self.assertEqual(detail["history"][-1]["kind"], "memory")

    def test_task_manager_not_applicable_to_search_task(self):
        task_id = self.start_search()
        with self.assertRaises(ValidationError):
            self.repo.validate_task_step(self.chat.id, task_id)

    def test_paused_search_task_resumed_by_user_message(self):
        self.repo.update_chat_settings(self.chat.id, {"task_pause_each_stage": True})
        task_id = self.start_search()
        self.assertTrue(self.repo.get_task(task_id)["task"].paused)
        self.send("Дальше", answer="Ответ [1].")
        self.assertFalse(self.repo.get_task(task_id)["task"].paused)


class WorkTaskPauseSettingTests(unittest.TestCase):
    def setUp(self):
        self.repo = make_tool_repo()
        agent = self.repo.create_agent("A", TEST_MODEL_ID)
        self.chat = self.repo.create_chat(agent.id, "Чат")
        self.repo.update_chat_settings(self.chat.id, {"task_tracking_enabled": True})
        self.provider = self.repo._tool_provider  # type: ignore[attr-defined]

    def tearDown(self):
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def test_default_off_no_pause_and_continuation(self):
        self.assertFalse(self.repo._require_chat(self.chat.id).settings.task_pause_each_stage)
        self.provider.tool_calls_queue = [[_tool_call("c1", "start_task", {"title": "Сделать отчёт"})]]
        _, reply = self.repo.send_message_blocking(self.chat.id, "Сделай отчёт")
        task_id = json.loads(reply.task_events)[0]["task_id"]
        detail = self.repo.get_task(task_id)
        self.assertEqual((detail["task"].kind, detail["status"]), ("work", "active"))
        self.assertEqual(self.repo.work_task_to_continue(self.chat.id, {"type": "done", "message": reply}), task_id)
        # Подсказка в промпте: не запрашивать подтверждение этапов.
        chat = self.repo._require_chat(self.chat.id)
        agent = self.repo._require_agent(chat.agent_id)
        texts = "\n".join(b.content for b in self.repo._build_task_and_invariant_context(chat, agent))
        self.assertIn("«Останавливаться на каждом этапе» выключен", texts)

    def test_advance_without_pause_allows_several_steps(self):
        self.provider.tool_calls_queue = [[_tool_call("c1", "start_task", {"title": "Отчёт"})]]
        _, reply = self.repo.send_message_blocking(self.chat.id, "Сделай отчёт")
        task_id = json.loads(reply.task_events)[0]["task_id"]
        self.provider.tool_calls_queue = [[
            _tool_call("a1", "apply_task_action", {"task_id": task_id, "target_state": "execution"}),
            _tool_call("a2", "apply_task_action", {"task_id": task_id, "target_state": "validation"}),
        ]]
        self.repo.send_message_blocking(self.chat.id, "Давай")
        detail = self.repo.get_task(task_id)
        self.assertEqual((detail["task"].state, detail["task"].paused), ("validation", False))

    def test_with_setting_on_task_pauses(self):
        self.repo.update_chat_settings(self.chat.id, {"task_pause_each_stage": True})
        self.provider.tool_calls_queue = [[_tool_call("c1", "start_task", {"title": "Отчёт"})]]
        _, reply = self.repo.send_message_blocking(self.chat.id, "Сделай отчёт")
        task_id = json.loads(reply.task_events)[0]["task_id"]
        self.assertEqual(self.repo.get_task(task_id)["status"], "paused")
        self.assertIsNone(self.repo.work_task_to_continue(self.chat.id, {"type": "done", "message": reply}))


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi не установлен")
class SearchTaskApiTests(unittest.TestCase):
    def setUp(self):
        self.repo = make_tool_repo()
        self.repo._knowledge = FakeKnowledge([result(1, "Текст.")])
        self.client = TestClient(create_app_with_repository(self.repo))
        agent = self.repo.create_agent("A", TEST_MODEL_ID)
        self.chat = self.repo.create_chat(agent.id, "Чат")
        self.repo.update_chat_settings(self.chat.id, {
            "rag_enabled": True, "collection_ids": ["col_1"], "task_tracking_enabled": True,
        })

    def tearDown(self):
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def test_create_patch_state_and_machines(self):
        resp = self.client.post(f"/chats/{self.chat.id}/tasks", json={"title": "Как подключить MCP?", "kind": "search"})
        self.assertEqual(resp.status_code, 201, resp.text)
        task = resp.json()["task"]
        self.assertEqual((task["kind"], task["state"], task["goal"]), ("search", "answering", "Как подключить MCP?"))
        task_id = task["id"]

        resp = self.client.patch(f"/tasks/{task_id}/memory", json={
            "constraints": ["только CPU"], "terms": [{"term": "база", "definition": "коллекция"}]})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["constraints"], [{"text": "только CPU", "message_id": None}])
        self.assertEqual(body["terms"], [{"term": "база", "definition": "коллекция"}])

        resp = self.client.post(f"/tasks/{task_id}/state", json={"state": "achieved"})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual((resp.json()["state"], resp.json()["state_display_name"]), ("achieved", "Цель достигнута"))
        self.assertEqual(self.client.post(f"/tasks/{task_id}/state", json={"state": "clarifying"}).status_code, 400)

        machines = self.client.get("/task-state-machine").json()
        self.assertEqual([s["state"] for s in machines["search_states"]], ["clarifying", "answering", "achieved"])
        self.assertEqual(len(machines["states"]), 4)

        settings = self.client.get(f"/chats/{self.chat.id}").json()["settings"]
        self.assertFalse(settings["task_pause_each_stage"])

    def test_search_task_without_rag_rejected(self):
        self.repo.update_chat_settings(self.chat.id, {"rag_enabled": False})
        resp = self.client.post(f"/chats/{self.chat.id}/tasks", json={"title": "x", "kind": "search"})
        self.assertEqual(resp.status_code, 409, resp.text)


if __name__ == "__main__":
    unittest.main()

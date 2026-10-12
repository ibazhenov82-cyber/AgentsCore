"""RAG: «Использовать RAG» в настройках, поиск в базе знаний перед запросом
к модели, сборка контекста, источники в ответе, клиент knowledge_service."""

from __future__ import annotations

import json
import os
import unittest
from typing import List
from unittest.mock import MagicMock

import requests

from agents_core.knowledge import (
    KnowledgeClient, KnowledgeServiceError, citations, search_query, select_sources,
)
from agents_core.providers import ChatResult
from agents_core.providers.base import ChatUsage
from agents_core.repository import ValidationError
from tests.test_repository import TEST_MODEL_ID, make_repository

try:
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    FASTAPI_AVAILABLE = False

if FASTAPI_AVAILABLE:
    from agents_core.api.app import create_app_with_repository


def result(n, text, title="Регламент ревью", section="Сроки", score=0.8, doc_date="2026-05-12", version="2.1"):
    return {
        "chunk_id": f"doc_1-{n}", "score": score, "text": text, "section": section, "page": None,
        "document": {"id": "doc_1", "title": title, "source": "/data/review.md", "source_type": "path",
                     "doc_date": doc_date, "doc_version": version},
        "collection": {"id": "col_1", "name": "Регламенты"},
    }


class FakeKnowledge:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.calls: List[dict] = []

    def retrieve(self, query, collection_ids, top_k, score_threshold, candidate_k=None, rerank="none",
                 rerank_model=None, rerank_threshold=None):
        self.calls.append({"query": query, "collection_ids": collection_ids, "top_k": top_k,
                           "score_threshold": score_threshold, "candidate_k": candidate_k, "rerank": rerank,
                           "rerank_model": rerank_model, "rerank_threshold": rerank_threshold})
        if self.error:
            raise KnowledgeServiceError(self.error)
        return {"query": query, "results": self.results,
                "stages": {"candidates": 20, "after_threshold": 12, "after_dedup": 11, "after_rerank": 4,
                           "returned": len(self.results)},
                "rerank": {"method": rerank, "model": rerank_model, "fallback": False, "error": None},
                "missing_collections": list(getattr(self, "missing", []))}


class RagRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.repo = make_repository(max_input_tokens=100000)
        self.provider = self.repo._fake_provider
        self.knowledge = FakeKnowledge([result(1, "Ревью выполняется не позднее 2 рабочих дней.")])
        self.repo._knowledge = self.knowledge
        agent = self.repo.create_agent("A", TEST_MODEL_ID)
        self.chat = self.repo.create_chat(agent.id, "Чат")
        self.answer = "Не позднее 2 рабочих дней [1]."
        self.rewritten = "сроки ревью кода для срочных исправлений"
        self.rewrite_calls = []
        original = self.provider.chat

        def chat(model_id, messages, settings):
            if messages[0].role == "system" and messages[0].content.startswith("Ты переписываешь"):
                self.rewrite_calls.append((model_id, list(messages), settings))
                if isinstance(self.rewritten, Exception):
                    raise self.rewritten
                return ChatResult(content=self.rewritten, usage=ChatUsage(prompt_tokens=30, completion_tokens=8,
                                                                          total_tokens=38))
            self.provider.last_messages = list(messages)
            res = original(model_id, messages, settings)
            return ChatResult(content=self.answer, usage=res.usage or ChatUsage())

        self.provider.chat = chat

    def tearDown(self):
        try:
            os.unlink(self.repo._db_path_for_cleanup)
        except OSError:
            pass

    def enable(self, **extra):
        self.repo.update_chat_settings(self.chat.id, {"rag_enabled": True, "collection_ids": ["col_1"], **extra})

    def test_disabled_by_default_no_search(self):
        _, reply = self.repo.send_message_blocking(self.chat.id, "Сколько дней на ревью?")
        self.assertEqual(self.knowledge.calls, [])
        self.assertIsNone(reply.rag)

    def test_context_sources_and_citations(self):
        self.enable(rag_top_k=3, rag_score_threshold=0.4)
        user, reply = self.repo.send_message_blocking(self.chat.id, "Сколько дней даётся на ревью кода?")
        self.assertEqual(self.knowledge.calls[0], {"query": "Сколько дней даётся на ревью кода?",
                                                   "collection_ids": ["col_1"], "top_k": 3, "score_threshold": 0.4,
                                                   "candidate_k": 20, "rerank": "none", "rerank_model": None,
                                                   "rerank_threshold": 0.3})
        messages = self.provider.last_messages
        self.assertEqual(messages[-2].role, "system")
        self.assertIn("Отвечай только на основе этих фрагментов", messages[-2].content)
        self.assertEqual(messages[-1].role, "user")
        self.assertIn("Контекст:", messages[-1].content)
        self.assertIn("[1] Регламенты › Регламент ревью › Сроки (дата 2026-05-12, версия 2.1)", messages[-1].content)
        self.assertTrue(messages[-1].content.endswith("Вопрос: Сколько дней даётся на ревью кода?"))
        # В истории — исходный вопрос, без контекста.
        self.assertEqual(user.content, "Сколько дней даётся на ревью кода?")
        rag = json.loads(reply.rag)
        self.assertEqual((rag["status"], rag["cited"], rag["confirmed"]), ("ok", [1], True))
        self.assertEqual(rag["sources"][0]["title"], "Регламент ревью")
        self.assertEqual(rag["sources"][0]["collection_name"], "Регламенты")
        # Фрагменты контекста не попадают в следующий запрос.
        self.repo.send_message_blocking(self.chat.id, "Спасибо, а как оформлять пулл-реквесты в команде?")
        history_texts = [m.content for m in self.provider.last_messages[:-2]]
        self.assertFalse(any("Контекст:" in t for t in history_texts))

    def test_answer_without_citations_is_unconfirmed(self):
        self.enable()
        self.answer = "Думаю, неделя."
        _, reply = self.repo.send_message_blocking(self.chat.id, "Сколько дней на ревью?")
        rag = json.loads(reply.rag)
        self.assertEqual((rag["cited"], rag["confirmed"]), ([], False))

    def test_follow_up_question_uses_previous_question(self):
        self.enable()
        self.repo.send_message_blocking(self.chat.id, "Какие сроки ревью кода в команде?")
        self.repo.send_message_blocking(self.chat.id, "А для срочных?")
        self.assertEqual(self.knowledge.calls[1]["query"], "Какие сроки ревью кода в команде?\nА для срочных?")

    def test_nothing_found(self):
        self.enable(rag_quotes=False)
        self.knowledge.results = []
        _, reply = self.repo.send_message_blocking(self.chat.id, "Какой бюджет на корпоратив?")
        instruction = self.provider.last_messages[-2].content
        self.assertIn("ничего не найдено", instruction)
        self.assertIn("не придумывай", instruction)
        self.assertEqual(self.provider.last_messages[-1].content, "Какой бюджет на корпоратив?")
        self.assertEqual(json.loads(reply.rag)["status"], "not_found")

    def test_not_only_from_kb(self):
        self.enable(rag_only_from_kb=False, rag_quotes=False)
        self.repo.send_message_blocking(self.chat.id, "Сроки ревью?")
        self.assertIn("можешь ответить на основе общих знаний", self.provider.last_messages[-2].content)

    def test_service_unavailable_answers_with_warning(self):
        self.enable()
        self.knowledge.error = "сервис баз знаний недоступен: connection refused"
        _, reply = self.repo.send_message_blocking(self.chat.id, "Сроки ревью?")
        self.assertEqual(reply.status, "complete")
        rag = json.loads(reply.rag)
        self.assertEqual(rag["status"], "unavailable")
        self.assertIn("connection refused", rag["error"])
        self.assertIn("База знаний сейчас недоступна", self.provider.last_messages[-2].content)

    def test_missing_collections_are_reported(self):
        self.enable()
        self.knowledge.missing = ["col_old"]
        with self.assertLogs("agents_core.repository", level="WARNING") as logs:
            _, reply = self.repo.send_message_blocking(self.chat.id, "Сроки ревью?")
        rag = json.loads(reply.rag)
        self.assertEqual(rag["status"], "ok")
        self.assertEqual(rag["missing_collections"], ["col_old"])
        self.assertIn("col_old", logs.output[0])

    def test_service_error_is_logged(self):
        self.enable()
        self.knowledge.error = "сервис баз знаний: HTTP 404: базы знаний не найдены: col_old"
        with self.assertLogs("agents_core.repository", level="WARNING") as logs:
            _, reply = self.repo.send_message_blocking(self.chat.id, "Сроки ревью?")
        self.assertIn("HTTP 404", json.loads(reply.rag)["error"])
        self.assertIn("HTTP 404", logs.output[0])

    def test_context_budget(self):
        self.enable(rag_context_tokens=200)
        self.knowledge.results = [result(i, "слово " * 100) for i in range(1, 5)]
        _, reply = self.repo.send_message_blocking(self.chat.id, "Сроки ревью?")
        self.assertEqual(len(json.loads(reply.rag)["sources"]), 1)

    def test_validation(self):
        self.repo.update_chat_settings(self.chat.id, {"rag_enabled": True})
        with self.assertRaises(ValidationError) as ctx:
            self.repo.send_message_blocking(self.chat.id, "x")
        self.assertIn("не выбраны базы знаний", str(ctx.exception))
        self.enable()
        self.repo._knowledge = None
        with self.assertRaises(ValidationError) as ctx:
            self.repo.send_message_blocking(self.chat.id, "x")
        self.assertIn("KNOWLEDGE_SERVICE_URL", str(ctx.exception))

    def test_settings_persist_and_new_chat_inherits(self):
        agent = self.repo.create_agent("B", TEST_MODEL_ID)
        self.repo.update_agent_settings(agent.id, {"rag_enabled": True, "collection_ids": ["a", "b"], "rag_top_k": 7,
                                                   "rag_score_threshold": 0.5, "rag_only_from_kb": False,
                                                   "rag_context_tokens": 1500})
        chat = self.repo.create_chat(agent.id, "Новый")
        s = self.repo.get_chat(chat.id).settings
        self.assertEqual((s.rag_enabled, s.collection_ids, s.rag_top_k, s.rag_score_threshold, s.rag_only_from_kb,
                          s.rag_context_tokens), (True, ["a", "b"], 7, 0.5, False, 1500))


class RerankAndRewriteTests(RagRepositoryTests.__bases__[0]):
    setUp = RagRepositoryTests.setUp
    tearDown = RagRepositoryTests.tearDown
    enable = RagRepositoryTests.enable

    def test_rerank_params_and_trace(self):
        self.enable(rag_candidate_k=30, rag_top_k=4, rag_rerank="model", rag_rerank_model="local/bge",
                    rag_rerank_threshold=0.6)
        _, reply = self.repo.send_message_blocking(self.chat.id, "Какие сроки ревью кода в команде?")
        call = self.knowledge.calls[0]
        self.assertEqual((call["candidate_k"], call["top_k"], call["rerank"], call["rerank_model"],
                          call["rerank_threshold"]), (30, 4, "model", "local/bge", 0.6))
        rag = json.loads(reply.rag)
        self.assertEqual(rag["stages"]["after_rerank"], 4)
        self.assertEqual(rag["rerank"]["method"], "model")
        self.assertEqual(rag["params"], {"candidate_k": 30, "top_k": 4, "score_threshold": 0.3,
                                         "rerank": "model", "rerank_threshold": 0.6})

    def test_candidate_k_not_less_than_top_k(self):
        self.enable(rag_candidate_k=3, rag_top_k=8)
        self.repo.send_message_blocking(self.chat.id, "Какие сроки ревью кода в команде?")
        self.assertEqual(self.knowledge.calls[0]["candidate_k"], 8)

    def test_rewrite_off_and_follow_up(self):
        self.enable(rag_query_rewrite="off")
        self.repo.send_message_blocking(self.chat.id, "Какие сроки ревью кода в команде?")
        self.repo.send_message_blocking(self.chat.id, "А для срочных?")
        self.assertEqual(self.knowledge.calls[1]["query"], "А для срочных?")
        self.repo.update_chat_settings(self.chat.id, {"rag_query_rewrite": "follow_up"})
        _, reply = self.repo.send_message_blocking(self.chat.id, "А для срочных?")
        rag = json.loads(reply.rag)
        self.assertEqual(rag["query"], "А для срочных?\nА для срочных?")
        self.assertEqual(rag["rewrite"]["mode"], "follow_up")
        self.assertEqual(self.rewrite_calls, [])

    def test_rewrite_by_model(self):
        self.enable(rag_query_rewrite="llm")
        self.repo.send_message_blocking(self.chat.id, "Какие сроки ревью кода в команде?")
        self.rewritten = "Поисковый запрос: «сроки ревью кода для срочных исправлений»"
        _, reply = self.repo.send_message_blocking(self.chat.id, "А для срочных?")
        self.assertEqual(self.knowledge.calls[-1]["query"], "сроки ревью кода для срочных исправлений")
        model_id, messages, settings = self.rewrite_calls[-1]
        self.assertEqual(model_id, "test-model")
        self.assertIn("Пользователь: Какие сроки ревью кода в команде?", messages[1].content)
        self.assertIn("Ассистент: Не позднее 2 рабочих дней [1].", messages[1].content)
        self.assertTrue(messages[1].content.endswith("Вопрос: А для срочных?\n\nПоисковый запрос:"))
        self.assertEqual((settings.temperature, settings.thinking_enabled, settings.tools_json), (0.0, False, ""))
        # Модель отвечает на ИСХОДНЫЙ вопрос.
        self.assertTrue(self.provider.last_messages[-1].content.endswith("Вопрос: А для срочных?"))
        rag = json.loads(reply.rag)
        self.assertEqual((rag["original_query"], rag["query"]), ("А для срочных?", "сроки ревью кода для срочных исправлений"))
        self.assertEqual(rag["rewrite"], {"mode": "llm", "rewritten": True, "model": TEST_MODEL_ID, "error": None,
                                          "tokens": 38, "no_search": False, "task_memory": False})

    def test_rewrite_failure_falls_back_to_question(self):
        from agents_core.providers import ProviderError

        self.enable(rag_query_rewrite="llm")
        self.rewritten = ProviderError("timeout")
        _, reply = self.repo.send_message_blocking(self.chat.id, "Сроки ревью?")
        self.assertEqual(self.knowledge.calls[0]["query"], "Сроки ревью?")
        rag = json.loads(reply.rag)
        self.assertIn("переписать не удалось", rag["rewrite"]["error"])
        self.assertEqual(reply.status, "complete")

    def test_rewrite_unknown_model(self):
        self.enable(rag_query_rewrite="llm", rag_rewrite_model="fake:nope")
        _, reply = self.repo.send_message_blocking(self.chat.id, "Сроки ревью?")
        self.assertIn("не найдена", json.loads(reply.rag)["rewrite"]["error"])
        self.assertEqual(self.rewrite_calls, [])

    def test_settings_validation(self):
        with self.assertRaises(ValidationError):
            self.repo.update_chat_settings(self.chat.id, {"rag_rerank": "llm"})
        with self.assertRaises(ValidationError):
            self.repo.update_chat_settings(self.chat.id, {"rag_query_rewrite": "smart"})
        with self.assertRaises(ValidationError):
            self.repo.update_chat_settings(self.chat.id, {"rag_rewrite_model": "no-provider"})


class QuotesTests(RagRepositoryTests.__bases__[0]):
    """«Цитаты и источники обязательны» (ТЗ «Цитаты, источники и анти-галлюцинации»)."""

    setUp = RagRepositoryTests.setUp
    tearDown = RagRepositoryTests.tearDown
    enable = RagRepositoryTests.enable

    def setUp(self):  # noqa: F811
        RagRepositoryTests.setUp(self)
        self.knowledge.results = [
            result(1, "Ревью кода выполняется не позднее двух рабочих дней после запроса."),
            result(2, "Срочное исправление (hotfix) ревьюится в течение 4 часов в рабочее время."),
        ]

    def test_instruction_requires_quotes_and_dont_know(self):
        self.enable()
        self.repo.send_message_blocking(self.chat.id, "Сроки ревью?")
        instruction = self.provider.last_messages[-2].content
        self.assertIn("«Цитаты:»", instruction)
        self.assertIn("слово в слово", instruction)
        self.assertIn("Не знаю", instruction)
        self.assertIn("уточняющий вопрос", instruction)

    def test_quotes_are_verified_and_block_is_cut(self):
        self.enable()
        self.answer = ("Обычное ревью — не позднее двух рабочих дней [1], срочное — 4 часа [2].\n\n"
                       "**Цитаты:**\n"
                       "- «Ревью кода выполняется не позднее двух рабочих дней» [1]\n"
                       "- «hotfix ревьюится в течение 4 часов» [2]\n"
                       "- «Ревью выполняется за один день» [1]")
        _, reply = self.repo.send_message_blocking(self.chat.id, "Сроки ревью?")
        self.assertEqual(reply.content, "Обычное ревью — не позднее двух рабочих дней [1], срочное — 4 часа [2].")
        rag = json.loads(reply.rag)
        self.assertEqual([(q["n"], q["verified"]) for q in rag["quotes"]], [(1, True), (2, True), (1, False)])
        self.assertEqual(rag["checks"], {"sources": True, "quotes": 3, "quotes_verified": 2, "dont_know": False})
        self.assertEqual(rag["cited"], [1, 2])

    def test_answer_without_quotes_is_marked(self):
        self.enable()
        self.answer = "Не позднее двух рабочих дней."
        _, reply = self.repo.send_message_blocking(self.chat.id, "Сроки ревью?")
        rag = json.loads(reply.rag)
        self.assertEqual(rag["checks"], {"sources": False, "quotes": 0, "quotes_verified": 0, "dont_know": False})
        self.assertFalse(rag["confirmed"])

    def test_not_found_forces_dont_know(self):
        self.enable()
        self.knowledge.results = []
        self.answer = "В базе знаний нет данных о бюджете. Уточните, о каком бюджете речь?"
        _, reply = self.repo.send_message_blocking(self.chat.id, "Какой бюджет на обучение?")
        instruction = self.provider.last_messages[-2].content
        self.assertIn("Начни ответ словами «Не знаю»", instruction)
        self.assertTrue(reply.content.startswith("Не знаю. В базе знаний нет данных"))
        rag = json.loads(reply.rag)
        self.assertTrue(rag["dont_know"])
        self.assertEqual(rag["status"], "not_found")

    def test_model_says_dont_know_with_weak_context(self):
        self.enable()
        self.answer = "**Не знаю**: во фрагментах нет сведений о бюджете. Уточните, пожалуйста, вопрос."
        _, reply = self.repo.send_message_blocking(self.chat.id, "Какой бюджет на обучение?")
        rag = json.loads(reply.rag)
        self.assertTrue(rag["checks"]["dont_know"])
        self.assertEqual(rag["quotes"], [])

    def test_disabled_keeps_previous_behaviour(self):
        self.enable(rag_quotes=False)
        self.answer = "Ответ [1].\n\nЦитаты:\n- «Ревью кода выполняется» [1]"
        _, reply = self.repo.send_message_blocking(self.chat.id, "Сроки ревью?")
        self.assertIn("Цитаты:", reply.content)
        rag = json.loads(reply.rag)
        self.assertEqual((rag["quotes_required"], rag["quotes"], rag["checks"]), (False, [], {}))


class QuoteMatchingTests(unittest.TestCase):
    def test_quote_in_fragment(self):
        from agents_core.knowledge import quote_in_fragment

        fragment = "## Сроки ревью\n\nОбычный pull request должен получить **первое** ревью не позднее двух рабочих дней."
        self.assertTrue(quote_in_fragment("должен получить первое ревью", fragment))
        self.assertTrue(quote_in_fragment("Обычный pull request … не позднее двух рабочих дней.", fragment))
        self.assertTrue(quote_in_fragment("ОБЫЧНЫЙ  pull   request", fragment))
        self.assertFalse(quote_in_fragment("не позднее трёх рабочих дней", fragment))
        self.assertFalse(quote_in_fragment("не позднее двух … Обычный pull", fragment))  # порядок частей
        self.assertFalse(quote_in_fragment("..", fragment))


class KnowledgeClientTimeoutTests(unittest.TestCase):
    def test_timeout_message(self):
        import requests

        from agents_core.knowledge import KnowledgeClient

        class Session:
            def post(self, *args, **kwargs):
                raise requests.ReadTimeout("read timed out")

        client = KnowledgeClient("http://kb", timeout=60, session=Session())
        with self.assertRaises(KnowledgeServiceError) as ctx:
            client.retrieve("q", ["c"], 4, 0.3, candidate_k=30, rerank="model")
        self.assertIn("не ответил за 60 с", str(ctx.exception))
        self.assertIn("Кандидатов до фильтрации", str(ctx.exception))
        with self.assertRaises(KnowledgeServiceError) as ctx:
            client.retrieve("q", ["c"], 4, 0.3)
        self.assertNotIn("реранкер", str(ctx.exception))


class HelpersTests(unittest.TestCase):
    def test_citations(self):
        self.assertEqual(citations("См. [2], а также [1, 3] и [3; 9] и [x]", 3), [1, 2, 3])
        self.assertEqual(citations("без ссылок", 3), [])

    def test_search_query(self):
        self.assertEqual(search_query("А для срочных?", "Сроки ревью?"), "Сроки ревью?\nА для срочных?")
        long = "Какие сроки ревью действуют для срочных исправлений в продакшене?"
        self.assertEqual(search_query(long, "Сроки ревью?"), long)
        self.assertEqual(search_query("Коротко?", None), "Коротко?")

    def test_select_sources_numbering(self):
        sources = select_sources([result(1, "a"), result(2, "b")], 1000)
        self.assertEqual([s["n"] for s in sources], [1, 2])
        self.assertEqual(sources[1]["chunk_id"], "doc_1-2")


class ClientTests(unittest.TestCase):
    def test_retrieve_ok_and_errors(self):
        session = MagicMock()
        ok = MagicMock(status_code=200)
        ok.json.return_value = {"results": [result(1, "x")]}
        session.post.return_value = ok
        client = KnowledgeClient("http://kb:8003/", api_key="k", session=session)
        self.assertEqual(len(client.retrieve("q", ["c"], 5, 0.3)["results"]), 1)
        args, kwargs = session.post.call_args
        self.assertEqual(args[0], "http://kb:8003/api/v1/retrieval")
        self.assertEqual(kwargs["headers"], {"Authorization": "Bearer k"})
        self.assertEqual(kwargs["json"]["collection_ids"], ["c"])

        bad = MagicMock(status_code=404, text="")
        bad.json.return_value = {"error": "коллекция c не найдена"}
        session.post.return_value = bad
        with self.assertRaises(KnowledgeServiceError) as ctx:
            client.retrieve("q", ["c"], 5, 0.3)
        self.assertIn("коллекция c не найдена", str(ctx.exception))

        session.post.side_effect = requests.ConnectionError("refused")
        with self.assertRaises(KnowledgeServiceError):
            client.retrieve("q", ["c"], 5, 0.3)


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi не установлен в этом окружении")
class RagApiTests(unittest.TestCase):
    def setUp(self):
        self.repo = make_repository(max_input_tokens=100000)
        self.repo._knowledge = FakeKnowledge([result(1, "Ревью — 2 рабочих дня.")])
        self.app = create_app_with_repository(self.repo)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.state.run_manager.shutdown(wait=True)
        try:
            os.unlink(self.repo._db_path_for_cleanup)
        except OSError:
            pass

    def test_settings_and_message_rag_via_runs(self):
        agent = self.client.post("/agent", json={"name": "A", "model": TEST_MODEL_ID}).json()
        chat = self.client.post(f"/agents/{agent['id']}/chat", json={"title": "c"}).json()
        resp = self.client.put(f"/chats/{chat['id']}/settings", json={"rag_enabled": True, "collection_ids": ["col_1"]})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["collection_ids"], ["col_1"])
        bad = self.client.put(f"/chats/{chat['id']}/settings", json={"rag_top_k": 0})
        self.assertEqual(bad.status_code, 422)
        run = self.client.post(f"/chats/{chat['id']}/runs", json={"text": "Сроки ревью?"})
        self.assertEqual(run.status_code, 202, run.text)
        self.app.state.run_manager.wait(run.json()["run"]["id"], 5)
        messages = self.client.get(f"/chats/{chat['id']}/messages").json()
        reply = messages[-1] if isinstance(messages, list) else messages["messages"][-1]
        rag = json.loads(reply["rag"])
        self.assertEqual(rag["status"], "ok")
        self.assertEqual(rag["sources"][0]["text"], "Ревью — 2 рабочих дня.")


if __name__ == "__main__":
    unittest.main()

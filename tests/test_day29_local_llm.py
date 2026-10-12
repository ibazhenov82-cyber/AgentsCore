"""
«День 29. Оптимизация локальной LLM» — размер контекста (`num_ctx`) для
локальных моделей и метрики скорости/памяти ответа (`Message.metrics`,
средние в статистике чата). Полностью офлайн: HTTP Ollama подменяется.
"""

from __future__ import annotations

import json
import os
import unittest
from typing import Iterator, List
from unittest.mock import MagicMock, patch

from agents_core.models import Settings
from agents_core.providers.base import ChatResult, ChatUsage, ProviderMessage, StreamDelta
from agents_core.providers.ollama import OllamaProvider
from agents_core.repository import ValidationError
from tests.test_repository import FakeRegistry, TEST_MODEL_ID, make_repository


def _resp(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.ok = True
    resp.status_code = 200
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


class OllamaOptionsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = OllamaProvider(base_url="http://localhost:11434")

    def test_num_ctx_and_num_predict_are_separate_options(self):
        options = self.provider._build_options(Settings(max_tokens=512, num_ctx=8192))
        self.assertEqual(options["num_predict"], 512)
        self.assertEqual(options["num_ctx"], 8192)

    def test_num_ctx_not_sent_when_unset(self):
        options = self.provider._build_options(Settings())
        self.assertNotIn("num_ctx", options)
        self.assertNotIn("num_predict", options)

    def test_chat_reads_timings_in_milliseconds(self):
        final = {
            "message": {"content": "ok"}, "done": True,
            "prompt_eval_count": 1000, "eval_count": 200,
            "prompt_eval_duration": 500_000_000, "eval_duration": 8_000_000_000, "load_duration": 3_000_000_000,
        }
        with patch.object(self.provider, "discover_model") as discover, patch.object(self.provider, "_session") as session:
            discover.return_value = MagicMock(supports_thinking=False)
            session.post.return_value = _resp(final)
            result = self.provider.chat("qwen3:8b", [ProviderMessage("user", "hi")], Settings())
        self.assertEqual(result.usage.completion_tokens, 200)
        self.assertAlmostEqual(result.usage.eval_duration_ms, 8000.0)
        self.assertAlmostEqual(result.usage.prompt_eval_duration_ms, 500.0)
        self.assertAlmostEqual(result.usage.load_duration_ms, 3000.0)

    def test_runtime_stats_from_api_ps(self):
        ps = {"models": [
            {"name": "qwen3:4b", "model": "qwen3:4b", "size": 4 * 1024 ** 3, "size_vram": 0},
            {"name": "qwen3:8b-q8_0", "model": "qwen3:8b-q8_0", "size": 10 * 1024 ** 3,
             "size_vram": 5 * 1024 ** 3, "context_length": 8192},
        ]}
        with patch.object(self.provider, "_session") as session:
            session.get.return_value = _resp(ps)
            stats = self.provider.runtime_stats("qwen3:8b-q8_0")
        self.assertEqual(stats, {"memory_mb": 10240, "vram_mb": 5120, "gpu_percent": 50, "num_ctx": 8192})

    def test_runtime_stats_none_when_model_not_loaded(self):
        with patch.object(self.provider, "_session") as session:
            session.get.return_value = _resp({"models": []})
            self.assertIsNone(self.provider.runtime_stats("qwen3:8b"))


class OllamaManagementHttpTests(unittest.TestCase):
    """Тела запросов запуска/остановки и разбор возможностей модели."""

    def setUp(self) -> None:
        self.provider = OllamaProvider(base_url="http://localhost:11434")

    def test_load_and_unload_bodies(self):
        with patch.object(self.provider, "_session") as session:
            ok = _resp({"load_duration": 1})
            ok.content = b"{}"
            session.post.return_value = ok
            self.provider.load("qwen3:8b", 8192)
            body = session.post.call_args.kwargs["json"]
            self.assertEqual(body, {"model": "qwen3:8b", "keep_alive": -1, "options": {"num_ctx": 8192}})
            self.provider.unload("qwen3:8b")
            self.assertEqual(session.post.call_args.kwargs["json"], {"model": "qwen3:8b", "keep_alive": 0})

    def test_chat_payload_keeps_model_loaded(self):
        with patch.object(self.provider, "discover_model") as discover:
            discover.return_value = MagicMock(supports_thinking=False)
            payload = self.provider._build_payload("qwen3:8b", [ProviderMessage("user", "hi")], Settings(), stream=False)
        self.assertEqual(payload["keep_alive"], -1)

    def test_embedding_model_is_not_chat_model(self):
        with patch.object(self.provider, "_session") as session:
            session.post.return_value = _resp({"capabilities": ["embedding"]})
            self.assertFalse(self.provider.is_chat_model("qwen3-embedding:0.6b"))
            session.post.return_value = _resp({"capabilities": ["completion", "tools"]})
            self.assertTrue(self.provider.is_chat_model("qwen3:8b"))

    def test_show_cache_invalidated_by_new_digest(self):
        with patch.object(self.provider, "_session") as session:
            session.post.return_value = _resp({"capabilities": ["completion"]})
            self.provider.show("qwen3:8b", "d1")
            self.provider.show("qwen3:8b", "d1")
            self.assertEqual(session.post.call_count, 1)
            session.get.return_value = _resp({"models": [{"model": "qwen3:8b", "digest": "d2"}]})
            self.provider.list_tags()
            self.provider.show("qwen3:8b", "d2")
            self.assertEqual(session.post.call_count, 2)


class TimedProvider:
    """Фейк «как Ollama»: отдаёт тайминги в usage и сведения о памяти."""

    name = "timed"

    def __init__(self) -> None:
        self.last_settings = None

    def health(self) -> bool:
        return True

    def discover_model(self, model_id):
        raise NotImplementedError

    def _result(self) -> ChatResult:
        return ChatResult(content="Ответ", usage=ChatUsage(
            prompt_tokens=300, completion_tokens=100, total_tokens=400,
            eval_duration_ms=4000.0, prompt_eval_duration_ms=250.0, load_duration_ms=1500.0,
        ))

    def chat(self, model_id, messages: List[ProviderMessage], settings) -> ChatResult:
        self.last_settings = settings
        return self._result()

    def stream_chat(self, model_id, messages, settings) -> Iterator[StreamDelta]:
        self.last_settings = settings
        yield StreamDelta(reasoning_content="думаю")
        yield StreamDelta(content="Ответ")
        yield StreamDelta(done=True, result=self._result())

    def runtime_stats(self, model_id):
        return {"memory_mb": 6000, "vram_mb": 6000, "gpu_percent": 100, "num_ctx": 8192}


class MetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = make_repository(max_input_tokens=38000, context_window=40960)
        self.provider = TimedProvider()
        self.repo._registry = FakeRegistry(self.provider)  # type: ignore[attr-defined]
        self.agent = self.repo.create_agent("Агент", model=TEST_MODEL_ID)
        self.chat = self.repo.create_chat(self.agent.id, "Чат")

    def tearDown(self) -> None:
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def test_answer_gets_metrics(self):
        _user, reply = self.repo.send_message_blocking(self.chat.id, "Вопрос")
        metrics = json.loads(reply.metrics)
        self.assertEqual(metrics["tokens_per_sec"], 25.0)       # 100 токенов / 4 с
        self.assertEqual(metrics["prompt_tokens_per_sec"], 1200.0)  # 300 / 0,25 с
        self.assertEqual(metrics["load_ms"], 1500)
        self.assertEqual(metrics["memory_mb"], 6000)
        self.assertEqual(metrics["gpu_percent"], 100)
        self.assertEqual(metrics["num_ctx"], 8192)
        self.assertNotIn("ttft_ms", metrics)  # не потоковый ответ

    def test_streamed_answer_has_time_to_first_token(self):
        events = list(self.repo.stream_message(self.chat.id, "Вопрос"))
        done = next(e for e in events if e["type"] == "done")
        metrics = json.loads(done["message"].metrics)
        self.assertIn("ttft_ms", metrics)
        self.assertGreaterEqual(metrics["ttft_ms"], 0)

    def test_chat_stats_averages(self):
        self.repo.send_message_blocking(self.chat.id, "Первый")
        self.repo.send_message_blocking(self.chat.id, "Второй")
        stats = self.repo.chat_stats(self.repo._require_chat(self.chat.id))
        self.assertEqual(stats["measured_answers"], 2)
        self.assertEqual(stats["avg_tokens_per_sec"], 25.0)
        self.assertEqual(stats["max_memory_mb"], 6000)
        self.assertIsNotNone(stats["avg_duration_ms"])

    def test_metrics_without_provider_timings_or_memory(self):
        # Провайдер без таймингов и без runtime_stats (как DeepSeek) —
        # ответ всё равно сохраняется, метрики — только то, что измерено.
        repo = make_repository()
        try:
            agent = repo.create_agent("Агент", model=TEST_MODEL_ID)
            chat = repo.create_chat(agent.id, "Чат")
            _user, reply = repo.send_message_blocking(chat.id, "Вопрос")
            self.assertEqual(reply.status, "complete")
            if reply.metrics:
                self.assertNotIn("memory_mb", json.loads(reply.metrics))
        finally:
            os.unlink(repo._db_path_for_cleanup)  # type: ignore[attr-defined]


class NumCtxSettingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = make_repository(max_input_tokens=38000, context_window=40960)
        self.agent = self.repo.create_agent("Агент", model=TEST_MODEL_ID)

    def tearDown(self) -> None:
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def test_num_ctx_is_stored_and_copied_to_new_chat(self):
        self.repo.update_agent_settings(self.agent.id, {"num_ctx": 8192, "max_tokens": 1024})
        chat = self.repo.create_chat(self.agent.id, "Чат")
        self.assertEqual(self.repo._require_chat(chat.id).settings.num_ctx, 8192)

    def test_context_window_follows_num_ctx_for_local_model(self):
        self.repo.update_agent_settings(self.agent.id, {"num_ctx": 8192, "max_tokens": 1024})
        chat = self.repo._require_chat(self.repo.create_chat(self.agent.id, "Чат").id)
        stats = self.repo.chat_stats(chat)
        self.assertEqual(stats["context_window"], 8192)
        self.assertEqual(stats["max_input_tokens"], 8192 - 1024)

    def test_max_tokens_is_not_chat_budget_when_num_ctx_set(self):
        # С num_ctx бюджет чата — окно num_ctx, max_tokens — только длина ответа.
        self.repo.update_agent_settings(self.agent.id, {"num_ctx": 8192, "max_tokens": 10})
        chat_id = self.repo.create_chat(self.agent.id, "Чат").id
        self.repo.send_message_blocking(chat_id, "Привет")
        stats = self.repo.chat_stats(self.repo._require_chat(chat_id))
        self.assertAlmostEqual(stats["context_fill_ratio"], stats["total_tokens"] / 8192)

    def test_native_window_without_num_ctx(self):
        chat = self.repo._require_chat(self.repo.create_chat(self.agent.id, "Чат").id)
        stats = self.repo.chat_stats(chat)
        self.assertEqual(stats["context_window"], 40960)

    def test_too_small_num_ctx_rejected(self):
        with self.assertRaises(ValidationError):
            self.repo.update_agent_settings(self.agent.id, {"num_ctx": 0})


if __name__ == "__main__":
    unittest.main()

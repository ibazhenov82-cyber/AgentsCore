"""
Тесты логирования (см. `agents_core.logging_setup`,
`agents_core.api.logging_middleware`, а также точки логирования, добавленные
в `agents_core.providers.deepseek_client`/`agents_core.providers.ollama`).

Требование ТЗ: должно быть видно, какие REST API сервиса были вызваны (с
параметрами) и какие внешние API (LLM-провайдеры) вызывает сам сервис (тоже
с параметрами). Тесты ниже проверяют это на уровне логов, а не только "код
не падает".

Middleware проверяется на ГОЛОМ Starlette-приложении (без FastAPI) — в этом
песочничном окружении пакет `fastapi` недоступен (см. `test_api.py`), но сам
middleware написан поверх Starlette-примитивов (`BaseHTTPMiddleware`,
`Request`), которые FastAPI не переопределяет, так что это честная проверка
реального поведения, а не заглушка.
"""

from __future__ import annotations

import json
import logging
import unittest
from unittest.mock import MagicMock, patch

from agents_core import logging_setup
from agents_core.logging_setup import configure_logging, redact, safe_json_dumps, truncate


class TruncateRedactTests(unittest.TestCase):
    def test_truncate_short_text_is_unchanged(self):
        self.assertEqual(truncate("привет", limit=100), "привет")

    def test_truncate_long_text_is_cut_with_marker(self):
        text = "x" * 50
        result = truncate(text, limit=10)
        self.assertTrue(result.startswith("x" * 10))
        self.assertIn("обрезано", result)
        self.assertIn("50", result)  # исходная длина упомянута

    def test_redact_masks_secret_like_keys_case_insensitively(self):
        payload = {"DEEPSEEK_API_KEY": "sk-12345", "Authorization": "Bearer xyz", "ok_field": "visible"}
        result = redact(payload)
        self.assertEqual(result["DEEPSEEK_API_KEY"], "***")
        self.assertEqual(result["Authorization"], "***")
        self.assertEqual(result["ok_field"], "visible")

    def test_redact_recurses_into_nested_structures(self):
        payload = {"outer": {"nested_token": "abc", "list": [{"password": "p"}, {"fine": 1}]}}
        result = redact(payload)
        self.assertEqual(result["outer"]["nested_token"], "***")
        self.assertEqual(result["outer"]["list"][0]["password"], "***")
        self.assertEqual(result["outer"]["list"][1]["fine"], 1)

    def test_safe_json_dumps_never_raises_on_unserializable_value(self):
        class Weird:
            def __repr__(self):
                return "<weird>"

        text = safe_json_dumps({"x": Weird()})
        self.assertIsInstance(text, str)

    def test_safe_json_dumps_applies_redact_and_truncate(self):
        text = safe_json_dumps({"api_key": "sk-secret", "msg": "y" * 5000}, limit=200)
        self.assertNotIn("sk-secret", text)
        self.assertIn("обрезано", text)


class ConfigureLoggingIdempotencyTests(unittest.TestCase):
    def test_repeated_configure_does_not_duplicate_handlers(self):
        logger = logging.getLogger(logging_setup.ROOT_LOGGER_NAME)
        before = len(logger.handlers)
        configure_logging(level="DEBUG")
        configure_logging(level="INFO")
        configure_logging(level="INFO")
        # Первый вызов мог добавить один StreamHandler (если ещё не было
        # настроено раньше в процессе) — повторные вызовы не добавляют новых.
        after_first = len(logger.handlers)
        configure_logging(level="WARNING")
        after_more = len(logger.handlers)
        self.assertEqual(after_first, after_more)
        self.assertGreaterEqual(after_first, before)
        self.assertEqual(logger.level, logging.WARNING)
        # Возвращаем уровень к дефолтному, чтобы не влиять на другие тесты,
        # проверяющие вывод через assertLogs (который временно поднимает
        # уровень сам, но не помешает быть явным здесь).
        configure_logging(level="INFO")


class OllamaProviderLoggingTests(unittest.TestCase):
    """Проверяет, что каждый исходящий HTTP-вызов к Ollama логируется с
    методом, URL и (где применимо) телом запроса — через логгер
    `agents_core.llm.ollama`."""

    def setUp(self) -> None:
        from agents_core.providers.ollama import OllamaProvider

        self.provider = OllamaProvider(base_url="http://localhost:11434")

    def test_health_logs_request_and_response(self):
        resp = MagicMock()
        resp.ok = True
        resp.status_code = 200
        with patch.object(self.provider, "_session") as session:
            session.get.return_value = resp
            with self.assertLogs("agents_core.llm.ollama", level="INFO") as cm:
                self.provider.health()
        joined = "\n".join(cm.output)
        self.assertIn("GET", joined)
        self.assertIn("/api/tags", joined)
        self.assertIn("200", joined)

    def test_health_network_error_is_logged_as_error(self):
        import requests

        with patch.object(self.provider, "_session") as session:
            session.get.side_effect = requests.ConnectionError("boom")
            with self.assertLogs("agents_core.llm.ollama", level="ERROR") as cm:
                ok = self.provider.health()
        self.assertFalse(ok)
        self.assertIn("boom", "\n".join(cm.output))

    def test_chat_logs_request_body_and_status(self):
        from agents_core.models import Settings
        from agents_core.providers.base import ProviderMessage

        resp = MagicMock()
        resp.ok = True
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"message": {"content": "привет"}}
        settings = Settings()
        with patch.object(self.provider, "_session") as session:
            session.post.return_value = resp
            with self.assertLogs("agents_core.llm.ollama", level="INFO") as cm:
                self.provider.chat("qwen3:0.6b", [ProviderMessage(role="user", content="hi")], settings)
        joined = "\n".join(cm.output)
        self.assertIn("/api/chat", joined)
        self.assertIn("qwen3:0.6b", joined)  # модель — часть тела запроса


class DeepSeekClientLoggingTests(unittest.TestCase):
    """Проверяет, что `DeepSeekClient._request` логирует вызов (метод, URL,
    JSON-тело) и итоговый статус ответа через `agents_core.llm.deepseek`, а
    заголовок Authorization (несущий API-ключ) в лог не попадает."""

    def setUp(self) -> None:
        from agents_core.providers.deepseek_client import DeepSeekClient

        self.client = DeepSeekClient(api_key="sk-test-secret-key")

    def test_chat_call_is_logged_without_leaking_api_key(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.ok = True
        resp.json.return_value = {
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        with patch.object(self.client.session, "request", return_value=resp) as mocked:
            with self.assertLogs("agents_core.llm.deepseek", level="INFO") as cm:
                self.client.chat(messages=[{"role": "user", "content": "hi"}], model="deepseek-v4-flash")
        joined = "\n".join(cm.output)
        self.assertIn("chat/completions", joined)
        self.assertIn("deepseek-v4-flash", joined)
        self.assertIn("200", joined)
        self.assertNotIn("sk-test-secret-key", joined)
        mocked.assert_called_once()

    def test_api_error_response_is_logged_at_error_level(self):
        resp = MagicMock()
        resp.status_code = 400
        resp.ok = False
        resp.json.return_value = {"error": {"message": "bad request"}}
        with patch.object(self.client.session, "request", return_value=resp):
            with self.assertLogs("agents_core.llm.deepseek", level="ERROR") as cm:
                with self.assertRaises(Exception):
                    self.client.chat(messages=[{"role": "user", "content": "hi"}], model="deepseek-v4-flash")
        self.assertIn("bad request", "\n".join(cm.output))


class AccessLogMiddlewareTests(unittest.TestCase):
    """Проверяет `AccessLogMiddleware` на голом Starlette-приложении (см.
    докстринг модуля про недоступность `fastapi` в этой песочнице) —
    реальный ASGI-цикл через `starlette.testclient.TestClient`, а не
    заглушка: тело запроса должно и попасть в лог, и остаться читаемым
    обработчиком маршрута, path-параметры должны попасть в лог, а итоговый
    статус в логе должен совпадать с тем, что реально вернул сервер."""

    def _build_app(self):
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        from agents_core.api.logging_middleware import AccessLogMiddleware

        async def echo(request):
            body = await request.json()
            return JSONResponse({"received": body, "path_param": request.path_params.get("item_id")})

        async def not_found(request):
            return JSONResponse({"error": "nope"}, status_code=404)

        app = Starlette(
            routes=[
                Route("/items/{item_id}", echo, methods=["POST"]),
                Route("/missing", not_found, methods=["GET"]),
            ],
            middleware=[],
        )
        app.add_middleware(AccessLogMiddleware)
        return app

    def test_body_is_logged_and_still_readable_by_the_handler(self):
        from starlette.testclient import TestClient

        app = self._build_app()
        client = TestClient(app)
        with self.assertLogs("agents_core.access", level="INFO") as cm:
            resp = client.post("/items/42", json={"api_key": "sk-should-be-masked", "title": "тест"})

        self.assertEqual(resp.status_code, 200)
        # Обработчик ниже по стеку реально получил тело (не пустое из-за
        # того, что middleware его уже "съел").
        data = resp.json()
        self.assertEqual(data["received"]["title"], "тест")
        self.assertEqual(data["path_param"], "42")

        joined = "\n".join(cm.output)
        self.assertIn("POST", joined)
        self.assertIn("/items/42", joined)
        self.assertIn("item_id", joined)  # path_params попали в лог
        self.assertIn("тест", joined)
        self.assertIn("200", joined)
        self.assertNotIn("sk-should-be-masked", joined)  # секрет замаскирован

    def test_logged_status_matches_actual_response_status(self):
        from starlette.testclient import TestClient

        app = self._build_app()
        client = TestClient(app)
        with self.assertLogs("agents_core.access", level="INFO") as cm:
            resp = client.get("/missing")
        self.assertEqual(resp.status_code, 404)
        joined = "\n".join(cm.output)
        self.assertIn("404", joined)


if __name__ == "__main__":
    unittest.main()

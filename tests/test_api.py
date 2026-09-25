"""
Интеграционные тесты HTTP-слоя (FastAPI + TestClient). Требуют пакет
`fastapi` (см. requirements.txt) — если он не установлен, весь класс
пропускается, а не падает, чтобы `python -m unittest discover` работал
и в окружениях без веб-зависимостей (например, где нужно проверить только
бизнес-логику `Repository`).

Провайдер LLM подменяется тем же `FakeProvider`, что и в
`test_repository.py` — реальные DeepSeek/Ollama не вызываются.
"""

from __future__ import annotations

import os
import unittest

try:
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from tests.test_repository import TEST_MODEL_ID, make_repository

if FASTAPI_AVAILABLE:
    from agents_core.api.app import create_app_with_repository


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi не установлен в этом окружении")
class ApiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = make_repository()
        self.app = create_app_with_repository(self.repo)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.app.state.run_manager.shutdown(wait=True)
        try:
            os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]
        except OSError:
            pass

    def _send(self, chat_id: str, text: str, **extra):
        """Отправить сообщение запуском и дождаться его завершения."""
        resp = self.client.post(f"/chats/{chat_id}/runs", json={"text": text, **extra})
        if resp.status_code == 202:
            self.app.state.run_manager.wait(resp.json()["run"]["id"], 5)
        return resp

    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body, {"status": "ok"})

    def test_validation_error_returns_error_key_not_detail(self):
        # Без обработчика RequestValidationError FastAPI по умолчанию
        # отдаёт {"detail": [...]}, а клиент (Android-приложение) умеет
        # показывать только {"error": "..."} — без этого обработчика
        # пользователь видел голое "Ошибка API сервера (422)" без единой
        # подробности.
        resp = self.client.post("/agent", json={"name": 123})  # неверный тип
        self.assertEqual(resp.status_code, 422)
        body = resp.json()
        self.assertIn("error", body)
        self.assertNotIn("detail", body)
        self.assertTrue(body["error"])

    def test_models_and_model_health(self):
        resp = self.client.get("/models")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()[0]["id"], TEST_MODEL_ID)

        resp = self.client.get(f"/model/{TEST_MODEL_ID}/health")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["healthy"])

    def test_default_settings_get_put_reset(self):
        resp = self.client.get("/settings/default")
        self.assertEqual(resp.status_code, 200)

        resp = self.client.put("/settings/default", json={"temperature": 0.3, "model": TEST_MODEL_ID})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["temperature"], 0.3)

        resp = self.client.post("/settings/default/reset")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["temperature"], 1.0)

    def test_agent_and_chat_lifecycle(self):
        resp = self.client.post("/agent", json={"name": "Мой агент", "model": TEST_MODEL_ID})
        self.assertEqual(resp.status_code, 201)
        agent = resp.json()

        resp = self.client.get(f"/agents/{agent['id']}")
        self.assertEqual(resp.status_code, 200)

        resp = self.client.post(f"/agents/{agent['id']}/chat", json={"title": "Чат 1"})
        self.assertEqual(resp.status_code, 201)
        chat = resp.json()
        self.assertEqual(chat["settings"]["model"], TEST_MODEL_ID)

        # модель в настройках чата менять нельзя
        resp = self.client.put(f"/chats/{chat['id']}/settings", json={"model": "ollama:qwen3:0.6b"})
        self.assertEqual(resp.status_code, 400)

        resp = self._send(chat["id"], "Привет")
        self.assertEqual(resp.status_code, 202)
        messages = self.client.get(f"/chats/{chat['id']}/messages").json()
        self.assertEqual(messages[-1]["content"], "echo: Привет")

        resp = self.client.get("/agents")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()[0]["chats"]), 1)

        resp = self.client.post(f"/chats/{chat['id']}/copy", json={"title": "Чат 1 (копия)"})
        self.assertEqual(resp.status_code, 201)

        resp = self.client.delete(f"/agents/{agent['id']}")
        self.assertEqual(resp.status_code, 204)

        resp = self.client.get(f"/agents/{agent['id']}")
        self.assertEqual(resp.status_code, 404)

    def test_message_bulk_delete_and_summarize(self):
        agent = self.client.post("/agent", json={"name": "A", "model": TEST_MODEL_ID}).json()
        chat = self.client.post(f"/agents/{agent['id']}/chat", json={"title": "C1"}).json()
        self._send(chat["id"], "1")
        self._send(chat["id"], "2")

        messages = self.client.get(f"/chats/{chat['id']}/messages").json()
        ids = [m["id"] for m in messages[:2]]
        resp = self.client.post(f"/chats/{chat['id']}/messages/bulk-delete", json={"ids": ids})
        self.assertEqual(resp.status_code, 204)
        self.assertEqual(len(self.client.get(f"/chats/{chat['id']}/messages").json()), 2)

        resp = self.client.post(f"/chats/{chat['id']}/summarize")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["is_summary"])

    def test_new_agent_settings_endpoint(self):
        self.client.put("/settings/default", json={"temperature": 0.3, "model": TEST_MODEL_ID})
        resp = self.client.get("/settings/default/agent")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["temperature"], 0.3)
        self.assertEqual(body["autosummary"], "off")

    def test_unknown_settings_field_is_400(self):
        agent = self.client.post("/agent", json={"name": "A", "model": TEST_MODEL_ID}).json()
        resp = self.client.put(f"/agents/{agent['id']}/settings", json={"not_a_real_field": 1})
        self.assertEqual(resp.status_code, 400)

    def test_chat_not_found_is_404(self):
        resp = self.client.get("/chats/no-such-chat")
        self.assertEqual(resp.status_code, 404)

    def test_branches_create_and_list(self):
        agent = self.client.post("/agent", json={"name": "A", "model": TEST_MODEL_ID}).json()
        chat = self.client.post(f"/agents/{agent['id']}/chat", json={"title": "C1"}).json()
        resp = self.client.post(f"/chats/{chat['id']}/branches", json={"name": "Альтернатива"})
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(resp.json()["number"], 1)
        resp = self.client.get(f"/chats/{chat['id']}/branches")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()), 1)

    def test_branches_create_without_name_auto_names_and_delete(self):
        agent = self.client.post("/agent", json={"name": "A", "model": TEST_MODEL_ID}).json()
        chat = self.client.post(f"/agents/{agent['id']}/chat", json={"title": "C1"}).json()
        resp = self.client.post(f"/chats/{chat['id']}/branches", json={})
        self.assertEqual(resp.status_code, 201)
        branch = resp.json()
        self.assertEqual(branch["number"], 1)
        self.assertEqual(branch["name"], "Ветка 1")

        del_resp = self.client.delete(f"/chats/{chat['id']}/branches/{branch['number']}")
        self.assertEqual(del_resp.status_code, 204)
        self.assertEqual(self.client.get(f"/chats/{chat['id']}/branches").json(), [])

        # Основную ветку (0) удалить нельзя.
        main_del = self.client.delete(f"/chats/{chat['id']}/branches/0")
        self.assertEqual(main_del.status_code, 400)

    def test_agent_and_chat_created_without_name_use_numbered_template(self):
        agent = self.client.post("/agent", json={"model": TEST_MODEL_ID}).json()
        self.assertEqual(agent["name"], "Агент 1")
        chat = self.client.post(f"/agents/{agent['id']}/chat", json={}).json()
        self.assertEqual(chat["title"], "Чат 1")

    def test_chat_detail_branch_query_param_rescopes_stats(self):
        agent = self.client.post("/agent", json={"name": "A", "model": TEST_MODEL_ID}).json()
        chat = self.client.post(f"/agents/{agent['id']}/chat", json={"title": "C1"}).json()
        self._send(chat["id"], "root")
        branch = self.client.post(f"/chats/{chat['id']}/branches", json={"name": "Alt"}).json()
        self._send(chat["id"], "branch-msg", branch=branch["number"])

        main_only = self.client.get(f"/chats/{chat['id']}", params={"branch": 0}).json()
        with_branch = self.client.get(f"/chats/{chat['id']}", params={"branch": branch["number"]}).json()
        self.assertGreater(with_branch["stats"]["total_tokens"], main_only["stats"]["total_tokens"])

    def test_get_facts_without_sticky_facts_strategy_is_400(self):
        agent = self.client.post("/agent", json={"name": "A", "model": TEST_MODEL_ID}).json()
        chat = self.client.post(f"/agents/{agent['id']}/chat", json={"title": "C1"}).json()
        resp = self._send(chat["id"], "Привет", get_facts=True)
        self.assertEqual(resp.status_code, 400)

    def test_memory_snapshot_endpoint_returns_expected_shape(self):
        # Регрессионный тест на баг: get_memory_snapshot() (эндпоинт)
        # собирал MemorySnapshotOut() без поля enabled_memory_types,
        # которое схема требует — pydantic падал с 500 ValidationError.
        # Repository-тесты (test_memory_and_profiles.py) этого не ловили,
        # т.к. обращаются к repo.get_memory_snapshot() напрямую, минуя
        # сборку Pydantic-модели в HTTP-слое.
        agent = self.client.post("/agent", json={"name": "A", "model": TEST_MODEL_ID}).json()
        chat = self.client.post(f"/agents/{agent['id']}/chat", json={"title": "C1"}).json()
        resp = self.client.get(f"/chats/{chat['id']}/memory-snapshot")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        for key in (
            "short_term", "working_memory", "long_term_memory",
            "active_profile", "available_tools", "memory_tools_enabled",
            "enabled_memory_types",
        ):
            self.assertIn(key, body)
        # По умолчанию для нового чата все типы памяти выключены —
        # пользователь включает нужные явно в настройках агента/чата.
        self.assertEqual(body["enabled_memory_types"], [])

        resp = self.client.put(
            f"/chats/{chat['id']}/settings",
            json={"working_memory_enabled": True, "long_term_memory_enabled": True},
        )
        self.assertEqual(resp.status_code, 200)
        resp = self.client.get(f"/chats/{chat['id']}/memory-snapshot")
        self.assertEqual(resp.status_code, 200)
        enabled = resp.json()["enabled_memory_types"]
        self.assertIn("working", enabled)
        self.assertIn("profile", enabled)

    def test_list_registered_skills_endpoint(self):
        resp = self.client.get("/skills")
        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(resp.json(), list)

    def test_profiles_are_a_global_catalog_not_scoped_to_agent(self):
        # Профили теперь общий справочник: POST/GET /profiles без agent_id
        # в пути, и созданный профиль виден и подключаем для ЛЮБОГО агента.
        agent_a = self.client.post("/agent", json={"name": "A", "model": TEST_MODEL_ID}).json()
        agent_b = self.client.post("/agent", json={"name": "B", "model": TEST_MODEL_ID}).json()
        chat_b = self.client.post(f"/agents/{agent_b['id']}/chat", json={"title": "C"}).json()

        resp = self.client.post("/profiles", json={"name": "Общий профиль"})
        self.assertEqual(resp.status_code, 201)
        profile = resp.json()
        self.assertNotIn("agent_id", profile)

        resp = self.client.get("/profiles")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(profile["id"], {p["id"] for p in resp.json()})

        # Подключаем к чату другого агента ("agent_a" тут вообще не при чём).
        resp = self.client.put(
            f"/chats/{chat_b['id']}/active-profile", json={"profile_id": profile["id"]},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["active_profile_id"], profile["id"])
        del agent_a  # использован только для проверки отсутствия привязки

    def test_set_agent_default_profile_endpoint_and_new_chat_inheritance(self):
        agent = self.client.post("/agent", json={"name": "A", "model": TEST_MODEL_ID}).json()
        profile = self.client.post("/profiles", json={"name": "По умолчанию"}).json()

        resp = self.client.put(
            f"/agents/{agent['id']}/default-profile", json={"profile_id": profile["id"]},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["default_profile_id"], profile["id"])

        # Новый чат наследует default_profile_id как свой активный профиль...
        chat = self.client.post(f"/agents/{agent['id']}/chat", json={"title": "C"}).json()
        self.assertEqual(chat["active_profile_id"], profile["id"])

        # ...а уже существующие чаты не меняются задним числом.
        resp = self.client.put(f"/agents/{agent['id']}/default-profile", json={"profile_id": None})
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(resp.json()["default_profile_id"])
        chat_after = self.client.get(f"/chats/{chat['id']}").json()
        self.assertEqual(chat_after["active_profile_id"], profile["id"])


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi не установлен в этом окружении")
class RunsApiTests(unittest.TestCase):
    """Асинхронные запуски, непрочитанные и общая лента по HTTP (ТЗ
    «асинхронные ответы», этап 1)."""

    def setUp(self) -> None:
        import json as _json

        self.json = _json
        self.repo = make_repository()
        self.app = create_app_with_repository(self.repo)
        self.client = TestClient(self.app)
        agent = self.client.post("/agent", json={"name": "A", "model": TEST_MODEL_ID}).json()
        self.chat = self.client.post(f"/agents/{agent['id']}/chat", json={"title": "C"}).json()

    def tearDown(self) -> None:
        self.app.state.run_manager.shutdown(wait=True)
        try:
            os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]
        except OSError:
            pass

    def _events(self, resp) -> list:
        return [self.json.loads(line[len("data:"):].strip()) for line in resp.text.splitlines() if line.startswith("data:")]

    def test_create_run_returns_draft_and_events_stream_completes(self):
        resp = self.client.post(f"/chats/{self.chat['id']}/runs", json={"text": "Привет", "client_request_id": "x1"})
        self.assertEqual(resp.status_code, 202, resp.text)
        body = resp.json()
        self.assertEqual(body["assistant_message"]["status"], "streaming")
        run_id = body["run"]["id"]
        self.app.state.run_manager.wait(run_id, 5)
        events = self._events(self.client.get(f"/runs/{run_id}/events?after=0"))
        self.assertEqual(events[0]["type"], "run_started")
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["message"]["content"], "echo: Привет")
        snapshot = self.client.get(f"/runs/{run_id}").json()
        self.assertEqual(snapshot["run"]["status"], "succeeded")
        self.assertEqual(snapshot["snapshot_seq"], events[-1]["seq"])
        # Повтор с тем же client_request_id — тот же запуск.
        again = self.client.post(f"/chats/{self.chat['id']}/runs", json={"text": "Привет", "client_request_id": "x1"})
        self.assertEqual(again.json()["run"]["id"], run_id)

    def test_unread_count_and_mark_read(self):
        run_id = self.client.post(f"/chats/{self.chat['id']}/runs", json={"text": "Привет"}).json()["run"]["id"]
        self.app.state.run_manager.wait(run_id, 5)
        chat = self.client.get(f"/chats/{self.chat['id']}").json()
        self.assertEqual(chat["unread_count"], 1)
        self.assertEqual(chat["preview"], "echo: Привет")
        resp = self.client.post(f"/chats/{self.chat['id']}/read", json={"message_id": chat["first_unread_message_id"]})
        self.assertEqual(resp.json()["unread_count"], 0)

    def test_unknown_run_is_404_and_stale_cursor_is_410(self):
        self.assertEqual(self.client.get("/runs/nope").status_code, 404)
        cursor = self.client.get("/events/cursor").json()["cursor"]
        self.assertEqual(self.client.get(f"/events?after={cursor + 1000}").status_code, 410)

    def test_create_task_requires_tracking_then_runs(self):
        resp = self.client.post(f"/chats/{self.chat['id']}/tasks", json={"title": "Задача"})
        self.assertEqual(resp.status_code, 409)
        self.client.put(f"/chats/{self.chat['id']}/settings", json={"task_tracking_enabled": True})
        resp = self.client.post(
            f"/chats/{self.chat['id']}/tasks",
            json={"title": "Задача", "description": "Описание", "run": {"auto_pause": True}},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        body = resp.json()
        self.assertEqual(body["message"]["content"], "Задача: Задача\n\nОписание")
        self.assertEqual(body["run"]["kind"], "task_step")
        self.assertIn(self.app.state.run_manager.wait(body["run"]["id"], 5).status, ("succeeded", "failed"))


if __name__ == "__main__":
    unittest.main()

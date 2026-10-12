"""
Тестовые диалоги: разбор файла с вопросами и дубликаты, справочник,
запуск-цепочка в чате (вопрос 1 → ответ 1 → … → ответ N), остановка на
ошибке, отмена, блокировка отправки сообщений, HTTP API.
"""

from __future__ import annotations

import os
import threading
import unittest

from agents_core.repository import NotFoundError, ValidationError
from agents_core.runs import RunConflictError, RunManager
from agents_core.test_dialogs import clean_question, normalize_question, parse_questions
from tests.test_repository import TEST_MODEL_ID, make_repository
from tests.test_runs import WAIT, GatedProvider, _all_events, _gated_repo

try:
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    FASTAPI_AVAILABLE = False

QUESTIONS = ["Сколько дней на ревью?", "А для срочных?", "Кто назначает ревьюера?"]


class ParseTests(unittest.TestCase):
    def test_line_is_question_and_list_markers_are_removed(self):
        text = "﻿1. Первый вопрос?\n\n2) Второй вопрос\n- Третий\n• Четвёртый\n   \n(5) Пятый\n10: Десятый\n"
        result = parse_questions(text)
        self.assertEqual(result["added"], ["Первый вопрос?", "Второй вопрос", "Третий", "Четвёртый", "Пятый",
                                           "Десятый"])
        self.assertEqual(result["lines"], 6)
        self.assertEqual(result["duplicates"], [])

    def test_numbers_inside_text_are_kept(self):
        self.assertEqual(clean_question("2024 год — итоги?"), "2024 год — итоги?")
        self.assertEqual(clean_question("-1"), "-1")

    def test_duplicates_in_file_and_existing(self):
        text = "Что такое RAG?\nчто  такое rag\nСроки ревью\nНовый вопрос\nСРОКИ РЕВЬЮ!"
        result = parse_questions(text, existing=["Сроки ревью?"])
        self.assertEqual(result["added"], ["Что такое RAG?", "Новый вопрос"])
        self.assertEqual(
            [(d["line"], d["reason"], d["duplicate_of"]) for d in result["duplicates"]],
            [(2, "file", "Что такое RAG?"), (3, "existing", "Сроки ревью?"), (5, "existing", "Сроки ревью?")],
        )

    def test_normalization(self):
        self.assertEqual(normalize_question("  Ещё  вопрос?! "), normalize_question("еще вопрос"))
        self.assertNotEqual(normalize_question("вопрос 1"), normalize_question("вопрос 2"))

    def test_too_long_lines_are_skipped(self):
        result = parse_questions("Короткий\n" + "я" * 4001)
        self.assertEqual(result["added"], ["Короткий"])
        self.assertEqual(result["skipped_too_long"], [2])


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.repo = make_repository()

    def tearDown(self):
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def test_crud(self):
        dialog = self.repo.create_test_dialog("  Регламенты ", QUESTIONS)
        self.assertEqual(dialog.name, "Регламенты")
        self.assertEqual(self.repo.get_test_dialog(dialog.id).questions, QUESTIONS)
        updated = self.repo.update_test_dialog(dialog.id, {"questions": QUESTIONS[:2], "name": "Ревью"})
        self.assertEqual((updated.name, updated.questions), ("Ревью", QUESTIONS[:2]))
        self.assertEqual([d.id for d in self.repo.list_test_dialogs()], [dialog.id])
        self.repo.delete_test_dialog(dialog.id)
        with self.assertRaises(NotFoundError):
            self.repo.get_test_dialog(dialog.id)

    def test_validation(self):
        self.repo.create_test_dialog("Первый", QUESTIONS)
        with self.assertRaises(ValidationError) as ctx:
            self.repo.create_test_dialog("первый", ["Вопрос"])
        self.assertIn("уже есть", str(ctx.exception))
        with self.assertRaises(ValidationError):
            self.repo.create_test_dialog("Пустой", [])
        with self.assertRaises(ValidationError):
            self.repo.create_test_dialog("С пустым", ["Вопрос", "  "])
        with self.assertRaises(ValidationError) as ctx:
            self.repo.create_test_dialog("Дубли", ["Сроки ревью?", "Вопрос", "сроки ревью"])
        self.assertIn("№ 1, 3", str(ctx.exception))
        with self.assertRaises(ValidationError):
            self.repo.parse_test_dialog_questions("  \n ")


class RunTests(unittest.TestCase):
    def setUp(self):
        self.repo = make_repository()
        self.manager = RunManager(self.repo, workers=2, timeout=30, events_ttl=60)
        self.agent = self.repo.create_agent("Агент", model=TEST_MODEL_ID)
        self.chat = self.repo.create_chat(self.agent.id, "Чат")
        self.dialog = self.repo.create_test_dialog("Регламенты", QUESTIONS)

    def tearDown(self):
        self.manager.shutdown(wait=True)
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def test_chain_question_answer_in_order(self):
        run, first_user, first_draft = self.manager.start_test_dialog_run(self.chat.id, self.dialog.id)
        self.assertEqual(run.kind, "test_dialog")
        self.assertEqual(first_user.content, QUESTIONS[0])
        self.assertEqual(first_user.source, "test_dialog")
        finished = self.manager.wait(run.id, WAIT)
        self.assertEqual(finished.status, "succeeded", finished.error)
        messages = self.repo.list_messages(self.chat.id)
        self.assertEqual([(m.role, m.content) for m in messages], [
            ("user", QUESTIONS[0]), ("assistant", f"echo: {QUESTIONS[0]}"),
            ("user", QUESTIONS[1]), ("assistant", f"echo: {QUESTIONS[1]}"),
            ("user", QUESTIONS[2]), ("assistant", f"echo: {QUESTIONS[2]}"),
        ])
        self.assertTrue(all(m.status == "complete" for m in messages))
        self.assertEqual(messages[1].id, first_draft.id)
        events = _all_events(self.manager, run.id)
        kinds = [e["type"] for e in events]
        self.assertEqual(kinds.count("message_started"), 2)
        self.assertEqual(kinds.count("test_dialog_step_done"), 3)
        self.assertEqual(kinds[-1], "done")
        self.assertEqual(events[-1]["test_dialog"], {"id": self.dialog.id, "name": "Регламенты", "step": 3,
                                                    "total": 3})
        started = [e for e in events if e["type"] == "message_started"]
        self.assertEqual(started[0]["user_message"].content, QUESTIONS[1])
        self.assertEqual(started[1]["test_dialog"]["step"], 3)
        # Прогресс и последние сообщения сохраняются в запуске.
        self.assertEqual(finished.user_message_id, messages[4].id)
        self.assertEqual(finished.assistant_message_id, messages[5].id)
        self.assertIn('"step": 3', finished.request_json)

    def test_error_stops_chain(self):
        provider = self.repo._fake_provider  # type: ignore[attr-defined]
        original = provider.chat

        def chat(model_id, messages, settings):
            if messages[-1].content == QUESTIONS[1]:
                provider.fail_next = True
            return original(model_id, messages, settings)

        provider.chat = chat
        run, _, _ = self.manager.start_test_dialog_run(self.chat.id, self.dialog.id, use_stream=False)
        finished = self.manager.wait(run.id, WAIT)
        self.assertEqual(finished.status, "failed")
        self.assertIn("вопросе 2 из 3", finished.error)
        messages = self.repo.list_messages(self.chat.id)
        self.assertEqual([m.content for m in messages if m.role == "user"], QUESTIONS[:2])
        self.assertEqual(messages[-1].status, "failed")
        events = _all_events(self.manager, run.id)
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["test_dialog"]["step"], 2)

    def test_empty_or_unknown_dialog(self):
        with self.assertRaises(NotFoundError):
            self.manager.start_test_dialog_run(self.chat.id, "nope")
        self.assertEqual(self.repo.list_messages(self.chat.id), [])


class BlockingAndCancelTests(unittest.TestCase):
    def setUp(self):
        self.provider = GatedProvider()
        self.repo = _gated_repo(self.provider)
        self.manager = RunManager(self.repo, workers=2, timeout=30, events_ttl=60)
        self.agent = self.repo.create_agent("Агент", model=TEST_MODEL_ID)
        self.chat = self.repo.create_chat(self.agent.id, "Чат")
        self.dialog = self.repo.create_test_dialog("Регламенты", QUESTIONS)

    def tearDown(self):
        self.provider.release()
        self.manager.shutdown(wait=True)
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def test_sending_is_blocked_until_last_answer(self):
        run, _, _ = self.manager.start_test_dialog_run(self.chat.id, self.dialog.id)
        self.assertTrue(self.provider.started.wait(WAIT))
        with self.assertRaises(RunConflictError) as ctx:
            self.manager.start_message_run(self.chat.id, "Можно вклиниться?")
        self.assertEqual(ctx.exception.active_run_id, run.id)
        active = self.manager.active_for_chat(self.chat.id)
        self.assertEqual(active.kind, "test_dialog")
        self.provider.release()
        self.assertEqual(self.manager.wait(run.id, WAIT).status, "succeeded")
        self.assertEqual(self.provider.calls, 3)
        # После последнего ответа отправка снова доступна.
        follow, _, _ = self.manager.start_message_run(self.chat.id, "Теперь можно")
        self.assertEqual(self.manager.wait(follow.id, WAIT).status, "succeeded")

    def test_cancel_stops_remaining_questions(self):
        run, _, _ = self.manager.start_test_dialog_run(self.chat.id, self.dialog.id)
        self.assertTrue(self.provider.started.wait(WAIT))
        self.manager.cancel(run.id)
        self.provider.release()
        finished = self.manager.wait(run.id, WAIT)
        self.assertEqual(finished.status, "cancelled")
        users = [m for m in self.repo.list_messages(self.chat.id) if m.role == "user"]
        self.assertEqual(len(users), 1)


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi не установлен в этом окружении")
class ApiTests(unittest.TestCase):
    def setUp(self):
        from agents_core.api.app import create_app_with_repository

        self.repo = make_repository()
        self.app = create_app_with_repository(self.repo)
        self.client = TestClient(self.app)
        self.agent = self.repo.create_agent("Агент", model=TEST_MODEL_ID)
        self.chat = self.repo.create_chat(self.agent.id, "Чат")

    def tearDown(self):
        self.app.state.run_manager.shutdown(wait=True)
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def test_catalog_import_and_run(self):
        created = self.client.post("/test-dialogs", json={"name": "Регламенты", "questions": QUESTIONS[:1]})
        self.assertEqual(created.status_code, 201, created.text)
        dialog_id = created.json()["id"]
        imported = self.client.post("/test-dialogs/import", json={
            "text": "1. Сколько дней на ревью\n2. А для срочных?\n3. а для срочных", "existing": QUESTIONS[:1],
        }).json()
        self.assertEqual(imported["added"], ["А для срочных?"])
        self.assertEqual([d["reason"] for d in imported["duplicates"]], ["existing", "file"])
        updated = self.client.put(f"/test-dialogs/{dialog_id}",
                                  json={"questions": QUESTIONS[:1] + imported["added"]}).json()
        self.assertEqual(updated["question_count"], 2)
        bad = self.client.put(f"/test-dialogs/{dialog_id}", json={"questions": ["А", "а"]})
        self.assertEqual(bad.status_code, 400)
        self.assertIn("повторяющиеся", bad.json()["error"])

        started = self.client.post(f"/chats/{self.chat.id}/test-dialogs/{dialog_id}/runs", json={})
        self.assertEqual(started.status_code, 202, started.text)
        body = started.json()
        self.assertEqual(body["run"]["kind"], "test_dialog")
        self.assertEqual(body["run"]["test_dialog"]["total"], 2)
        self.assertEqual(body["user_message"]["source"], "test_dialog")
        self.app.state.run_manager.wait(body["run"]["id"], WAIT)
        run = self.client.get(f"/runs/{body['run']['id']}").json()
        self.assertEqual(run["run"]["test_dialog"]["step"], 2)
        self.assertEqual(len(self.client.get(f"/chats/{self.chat.id}/messages").json()), 4)
        self.assertEqual(self.client.delete(f"/test-dialogs/{dialog_id}").status_code, 204)
        self.assertEqual(self.client.get(f"/test-dialogs/{dialog_id}").status_code, 404)


if __name__ == "__main__":
    unittest.main()

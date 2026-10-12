"""
Тесты асинхронных запусков (ТЗ «асинхронные ответы», этап 1):
`runs.RunManager`, черновики ответов, отмена, тайм-аут, идемпотентность,
один запуск на чат, восстановление после перезапуска, непрочитанные и общая
лента изменений, создание задачи через API и серверный цикл Менеджера задач.

Полностью офлайн — фиктивные провайдеры из соседних тестов; «медленный»
провайдер (`GatedProvider`) отдаёт куски ответа только по сигналу теста,
чтобы детерминированно проверять параллельность и отмену.
"""

from __future__ import annotations

import json
import os
import threading
import time
import unittest
from typing import Iterator, List

from agents_core.events import EventLog, EventsGone
from agents_core.models import Message, Run
from agents_core.providers.base import ChatResult, ChatUsage, ProviderMessage, StreamDelta
from agents_core.repository import PreconditionFailedError, ValidationError
from agents_core.runs import RunConflictError, RunManager
from tests.test_memory_and_profiles import ToolCallingFakeProvider, _tool_call
from tests.test_repository import FakeRegistry, TEST_MODEL_ID, make_repository

WAIT = 5.0


class GatedProvider:
    """Отдаёт первый кусок ответа сразу, остальные — только после
    `release()`; `started` сигналит, что поток ответа начался."""

    name = "gated"

    def __init__(self, pieces: List[str] = None):
        self.pieces = pieces or ["Раз", " два", " три"]
        self.gate = threading.Event()
        self.started = threading.Event()
        self.calls = 0
        self.fail = False

    def release(self) -> None:
        self.gate.set()

    def health(self) -> bool:
        return True

    def discover_model(self, model_id):
        raise NotImplementedError

    def chat(self, model_id, messages, settings) -> ChatResult:
        self.calls += 1
        self.started.set()
        self.gate.wait(WAIT)
        return ChatResult(content="".join(self.pieces), usage=ChatUsage(prompt_tokens=10, completion_tokens=3, total_tokens=13))

    def stream_chat(self, model_id, messages, settings) -> Iterator[StreamDelta]:
        self.calls += 1
        self.last_messages = list(messages)
        if self.fail:
            from agents_core.providers import ProviderError
            raise ProviderError("simulated failure")
        yield StreamDelta(content=self.pieces[0])
        self.started.set()
        self.gate.wait(WAIT)
        for piece in self.pieces[1:]:
            yield StreamDelta(content=piece)
        yield StreamDelta(done=True, result=ChatResult(
            content="".join(self.pieces), usage=ChatUsage(prompt_tokens=10, completion_tokens=3, total_tokens=13),
        ))


def _gated_repo(provider: GatedProvider):
    repo = make_repository()
    repo._registry = FakeRegistry(provider)  # type: ignore[attr-defined]
    return repo


def _all_events(manager: RunManager, run_id: str) -> List[dict]:
    manager.wait(run_id, WAIT)
    events, closed = manager.read_events(run_id, 0, 0.1)
    assert closed
    return events


class RunBasicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = make_repository()
        self.manager = RunManager(self.repo, workers=2, timeout=30, events_ttl=60)
        self.agent = self.repo.create_agent("Агент", model=TEST_MODEL_ID)
        self.chat = self.repo.create_chat(self.agent.id, "Чат")

    def tearDown(self) -> None:
        self.manager.shutdown(wait=True)
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def test_start_returns_draft_and_run_completes_in_background(self):
        run, user_msg, draft = self.manager.start_message_run(self.chat.id, "Привет")
        self.assertEqual(user_msg.content, "Привет")
        self.assertEqual(draft.role, "assistant")
        self.assertEqual(draft.run_id, run.id)
        finished = self.manager.wait(run.id, WAIT)
        self.assertEqual(finished.status, "succeeded")
        saved = self.repo._db.get_message(draft.id)
        self.assertEqual(saved.status, "complete")
        self.assertEqual(saved.content, "echo: Привет")
        # Сообщений ровно два: запрос и ответ (черновик превратился в ответ).
        self.assertEqual([m.role for m in self.repo.list_messages(self.chat.id)], ["user", "assistant"])

    def test_event_sequence_and_numbering(self):
        run, _, _ = self.manager.start_message_run(self.chat.id, "Привет")
        events = _all_events(self.manager, run.id)
        self.assertEqual([e["seq"] for e in events], list(range(1, len(events) + 1)))
        kinds = [e["type"] for e in events]
        self.assertEqual(kinds[0], "run_started")
        self.assertIn("delta", kinds)
        self.assertIn("message_saved", kinds)
        self.assertEqual(kinds[-1], "done")
        self.assertTrue(all(e["run_id"] == run.id and e["chat_id"] == self.chat.id for e in events))

    def test_read_after_returns_only_newer_events(self):
        run, _, _ = self.manager.start_message_run(self.chat.id, "Привет")
        events = _all_events(self.manager, run.id)
        later, _ = self.manager.read_events(run.id, 2, 0.1)
        self.assertEqual([e["seq"] for e in later], [e["seq"] for e in events if e["seq"] > 2])

    def test_snapshot_is_consistent_after_completion(self):
        run, _, draft = self.manager.start_message_run(self.chat.id, "Привет")
        events = _all_events(self.manager, run.id)
        snap_run, message, seq = self.manager.snapshot(run.id)
        self.assertEqual(seq, events[-1]["seq"])
        self.assertEqual(message.id, draft.id)
        self.assertEqual(message.status, "complete")
        self.assertEqual(snap_run.status, "succeeded")

    def test_idempotent_client_request_id(self):
        run1, _, _ = self.manager.start_message_run(self.chat.id, "Привет", client_request_id="req-1")
        self.manager.wait(run1.id, WAIT)
        run2, _, _ = self.manager.start_message_run(self.chat.id, "Привет", client_request_id="req-1")
        self.assertEqual(run1.id, run2.id)
        self.assertEqual(len(self.repo.list_messages(self.chat.id)), 2)

    def test_validation_error_before_run_is_created(self):
        with self.assertRaises(ValidationError):
            self.manager.start_message_run(self.chat.id, "Привет", get_facts=True)
        self.assertEqual(self.repo.list_messages(self.chat.id), [])
        self.assertEqual(self.manager.list_active(), [])

    def test_provider_failure_marks_run_and_draft_failed(self):
        self.repo._fake_provider.fail_next = True  # type: ignore[attr-defined]
        run, _, draft = self.manager.start_message_run(self.chat.id, "Привет", use_stream=False)
        finished = self.manager.wait(run.id, WAIT)
        self.assertEqual(finished.status, "failed")
        saved = self.repo._db.get_message(draft.id)
        self.assertEqual(saved.status, "failed")
        self.assertIn("simulated", saved.error)

    def test_unknown_run_raises_not_found(self):
        from agents_core.repository import NotFoundError
        with self.assertRaises(NotFoundError):
            self.manager.read_events("nope", 0, 0)


class RunConcurrencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = GatedProvider()
        self.repo = _gated_repo(self.provider)
        self.manager = RunManager(self.repo, workers=4, timeout=30, events_ttl=60)
        self.agent = self.repo.create_agent("Агент", model=TEST_MODEL_ID)
        self.chat1 = self.repo.create_chat(self.agent.id, "Чат 1")
        self.chat2 = self.repo.create_chat(self.agent.id, "Чат 2")

    def tearDown(self) -> None:
        self.provider.release()
        self.manager.shutdown(wait=True)
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def test_second_run_in_same_chat_conflicts(self):
        run, _, _ = self.manager.start_message_run(self.chat1.id, "Первый")
        self.assertTrue(self.provider.started.wait(WAIT))
        with self.assertRaises(RunConflictError) as ctx:
            self.manager.start_message_run(self.chat1.id, "Второй")
        self.assertEqual(ctx.exception.active_run_id, run.id)
        self.provider.release()
        self.assertEqual(self.manager.wait(run.id, WAIT).status, "succeeded")

    def test_runs_in_different_chats_go_in_parallel(self):
        run1, _, _ = self.manager.start_message_run(self.chat1.id, "Первый")
        run2, _, _ = self.manager.start_message_run(self.chat2.id, "Второй")
        deadline = time.monotonic() + WAIT
        while self.provider.calls < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.provider.calls, 2)  # оба ответа начались до «отпускания»
        self.assertEqual(len(self.manager.list_active()), 2)
        self.provider.release()
        self.assertEqual(self.manager.wait(run1.id, WAIT).status, "succeeded")
        self.assertEqual(self.manager.wait(run2.id, WAIT).status, "succeeded")

    def test_cancel_keeps_partial_text_and_excludes_it_from_context(self):
        run, _, draft = self.manager.start_message_run(self.chat1.id, "Первый")
        self.assertTrue(self.provider.started.wait(WAIT))
        self.manager.cancel(run.id)
        self.provider.release()
        finished = self.manager.wait(run.id, WAIT)
        self.assertEqual(finished.status, "cancelled")
        saved = self.repo._db.get_message(draft.id)
        self.assertEqual(saved.status, "cancelled")
        self.assertEqual(saved.content, "Раз")
        events = _all_events(self.manager, run.id)
        self.assertEqual(events[-1]["type"], "cancelled")

        # Следующий запрос: остановленный ответ модели не передаётся.
        self.provider.gate.set()
        run2, _, _ = self.manager.start_message_run(self.chat1.id, "Второй")
        self.manager.wait(run2.id, WAIT)
        sent = [(m.role, m.content) for m in self.provider.last_messages]
        self.assertNotIn(("assistant", "Раз"), sent)

    def test_timeout_fails_run(self):
        manager = RunManager(self.repo, workers=1, timeout=0.2, events_ttl=60)
        try:
            run, _, draft = manager.start_message_run(self.chat2.id, "Долгий")
            # Отмена кооперативная: срабатывает на следующем куске от модели.
            threading.Timer(0.6, self.provider.release).start()
            finished = manager.wait(run.id, WAIT)
            self.assertEqual(finished.status, "failed")
            saved = self.repo._db.get_message(draft.id)
            self.assertEqual(saved.status, "failed")
            self.assertIn("время", saved.error)
        finally:
            self.provider.release()
            manager.shutdown(wait=True)

    def test_draft_is_persisted_while_streaming(self):
        run, _, draft = self.manager.start_message_run(self.chat1.id, "Первый")
        self.assertTrue(self.provider.started.wait(WAIT))
        # Статус «Выполняется запрос к модели» сохраняется сразу; первый кусок
        # текста — при следующем сохранении черновика.
        deadline = time.monotonic() + WAIT
        while time.monotonic() < deadline:
            current = self.repo._db.get_run(run.id)
            if current.current_status:
                break
            time.sleep(0.01)
        self.assertEqual(current.current_status, "Выполняется запрос к модели")
        self.assertEqual(self.repo._db.get_message(draft.id).status, "streaming")
        self.provider.release()
        self.manager.wait(run.id, WAIT)


class RecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = make_repository()
        self.agent = self.repo.create_agent("Агент", model=TEST_MODEL_ID)
        self.chat = self.repo.create_chat(self.agent.id, "Чат")

    def tearDown(self) -> None:
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def test_unfinished_runs_become_interrupted_on_start(self):
        draft = self.repo._db.add_message(Message(
            id=0, chat_id=self.chat.id, role="assistant", content="Частичный", created_at=int(time.time()),
            status="streaming",
        ))
        self.repo._db.create_run(Run(
            id="r1", chat_id=self.chat.id, kind="message", status="running",
            assistant_message_id=draft.id, created_at=int(time.time()),
        ))
        manager = RunManager(self.repo, workers=1)
        try:
            self.assertEqual(manager.recover(), 1)
            self.assertEqual(self.repo._db.get_run("r1").status, "interrupted")
            saved = self.repo._db.get_message(draft.id)
            self.assertEqual(saved.status, "interrupted")
            self.assertEqual(saved.content, "Частичный")
            # Прерванный ответ виден как непрочитанный.
            self.assertEqual(self.repo.chat_activity([self.chat.id])[self.chat.id]["unread_count"], 1)
        finally:
            manager.shutdown(wait=True)


class UnreadAndEventsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = make_repository()
        self.manager = RunManager(self.repo, workers=1, timeout=30, events_ttl=60)
        self.agent = self.repo.create_agent("Агент", model=TEST_MODEL_ID)
        self.cursor = self.repo.events.last_seq
        self.chat = self.repo.create_chat(self.agent.id, "Чат")

    def tearDown(self) -> None:
        self.manager.shutdown(wait=True)
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def _global(self) -> List[dict]:
        events, _ = self.repo.events.read(self.cursor, 0)
        return events

    def test_chat_created_event(self):
        kinds = [(e["type"], e.get("chat_id")) for e in self._global()]
        self.assertIn(("chat_created", self.chat.id), kinds)

    def test_finished_answer_is_unread_until_marked_read(self):
        run, _, draft = self.manager.start_message_run(self.chat.id, "Привет")
        self.manager.wait(run.id, WAIT)
        info = self.repo.chat_activity([self.chat.id])[self.chat.id]
        self.assertEqual(info["unread_count"], 1)
        self.assertEqual(info["first_unread_message_id"], draft.id)
        self.assertEqual(info["preview"], "echo: Привет")
        kinds = [e["type"] for e in self._global()]
        self.assertIn("run_started", kinds)
        self.assertIn("run_finished", kinds)
        self.assertIn("unread_changed", kinds)

        after = self.repo.mark_chat_read(self.chat.id, draft.id)
        self.assertEqual(after["unread_count"], 0)
        # Отметка назад не двигается.
        self.repo.mark_chat_read(self.chat.id, 1)
        self.assertEqual(self.repo.get_chat(self.chat.id).last_read_message_id, draft.id)

    def test_user_messages_from_app_are_not_unread_but_scheduler_requests_are(self):
        self.repo._db.add_message(Message(id=0, chat_id=self.chat.id, role="user", content="сам", created_at=1))
        self.assertEqual(self.repo.chat_activity([self.chat.id]).get(self.chat.id, {}).get("unread_count", 0), 0)
        self.repo._db.add_message(Message(
            id=0, chat_id=self.chat.id, role="user", content="по расписанию", created_at=2, source="scheduler",
        ))
        self.assertEqual(self.repo.chat_activity([self.chat.id])[self.chat.id]["unread_count"], 1)

    def test_scheduler_chat_source_is_stored(self):
        chat = self.repo.create_chat(self.agent.id, "Сводка", source="scheduler")
        self.assertEqual(self.repo.get_chat(chat.id).source, "scheduler")
        self.assertEqual(self.repo.find_latest_chat_by_title(self.agent.id, "Сводка").id, chat.id)

    def test_delete_chat_publishes_event(self):
        self.repo.delete_chat(self.chat.id)
        self.assertIn(("chat_deleted", self.chat.id), [(e["type"], e.get("chat_id")) for e in self._global()])


class EventLogTests(unittest.TestCase):
    def test_trimmed_events_raise_gone(self):
        log = EventLog(max_events=2)
        for i in range(5):
            log.publish({"type": "x", "i": i})
        with self.assertRaises(EventsGone):
            log.read(0, 0)
        events, _ = log.read(3, 0)
        self.assertEqual([e["seq"] for e in events], [4, 5])

    def test_cursor_from_previous_lifetime_raises_gone(self):
        log = EventLog()
        log.publish({"type": "x"})
        with self.assertRaises(EventsGone):
            log.read(10, 0)

    def test_read_waits_for_new_event(self):
        log = EventLog()
        threading.Timer(0.05, lambda: log.publish({"type": "late"})).start()
        events, closed = log.read(0, 2.0)
        self.assertEqual([e["type"] for e in events], ["late"])
        self.assertFalse(closed)


class TaskRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = make_repository()
        self.provider = ToolCallingFakeProvider()
        self.repo._registry = FakeRegistry(self.provider)  # type: ignore[attr-defined]
        self.manager = RunManager(self.repo, workers=1, timeout=30, events_ttl=60)
        self.agent = self.repo.create_agent("Агент", model=TEST_MODEL_ID)
        self.chat = self.repo.create_chat(self.agent.id, "Чат")

    def tearDown(self) -> None:
        self.manager.shutdown(wait=True)
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def test_create_task_requires_task_tracking(self):
        with self.assertRaises(PreconditionFailedError):
            self.repo.create_task_direct(self.chat.id, "Разобрать issues")

    def test_create_task_adds_request_message(self):
        self.repo.update_chat_settings(self.chat.id, {"task_tracking_enabled": True})
        summary, message = self.repo.create_task_direct(
            self.chat.id, "Разобрать issues", "Составь план на неделю", source="scheduler",
        )
        self.assertEqual(summary["task"].description, "Составь план на неделю")
        self.assertEqual(message.content, "Задача: Разобрать issues\n\nСоставь план на неделю")
        self.assertEqual(message.source, "scheduler")
        self.assertEqual(self.repo.chat_activity([self.chat.id])[self.chat.id]["unread_count"], 1)

    def test_task_run_executes_steps_until_model_stops_advancing(self):
        self.repo.update_chat_settings(self.chat.id, {"task_tracking_enabled": True})
        summary, _ = self.repo.create_task_direct(self.chat.id, "Задача", "")
        task_id = summary["task"].id
        # Шаг 1: модель продвигает задачу planning -> execution; шаг 2: просто
        # отвечает текстом (не продвигает) — цикл останавливается.
        self.provider.tool_calls_queue = [[_tool_call("c1", "apply_task_action", {
            "task_id": task_id, "target_state": "execution", "current_step": "делаю",
        })]]
        run, first_draft = self.manager.start_task_run(self.chat.id, task_id, auto_pause=False)
        self.assertEqual(run.kind, "task_run")
        finished = self.manager.wait(run.id, WAIT)
        self.assertEqual(finished.status, "succeeded", finished.error)
        events = _all_events(self.manager, run.id)
        kinds = [e["type"] for e in events]
        self.assertEqual(kinds.count("task_step_done"), 2)
        self.assertEqual(kinds.count("message_started"), 1)
        self.assertIn("task_event", kinds)
        steps = [m for m in self.repo.list_messages(self.chat.id) if m.is_task_manager_step]
        self.assertEqual(len(steps), 2)
        self.assertTrue(all(m.status == "complete" for m in steps))
        self.assertEqual(steps[0].id, first_draft.id)

    def test_task_step_run_is_single_step(self):
        self.repo.update_chat_settings(self.chat.id, {"task_tracking_enabled": True})
        summary, _ = self.repo.create_task_direct(self.chat.id, "Задача", "")
        run, _ = self.manager.start_task_run(self.chat.id, summary["task"].id, auto_pause=True)
        self.assertEqual(run.kind, "task_step")
        self.assertEqual(self.manager.wait(run.id, WAIT).status, "succeeded")
        events = _all_events(self.manager, run.id)
        self.assertEqual([e["type"] for e in events].count("task_step_done"), 1)
        self.assertIn("task_status", events[-1])


    def test_message_run_continues_work_task_without_pauses(self):
        """«Останавливаться на каждом этапе» выключено (по умолчанию): задача,
        начатая ответом на сообщение, сразу ведётся дальше шагами Менеджера
        задач в том же запуске — пока модель продвигает этапы."""
        self.repo.update_chat_settings(self.chat.id, {"task_tracking_enabled": True})
        self.provider.tool_calls_queue = [
            [_tool_call("c1", "start_task", {"title": "Отчёт"})],
        ]
        run, _, _ = self.manager.start_message_run(self.chat.id, "Сделай отчёт")
        finished = self.manager.wait(run.id, WAIT)
        self.assertEqual(finished.status, "succeeded", finished.error)
        kinds = [e["type"] for e in _all_events(self.manager, run.id)]
        self.assertEqual(kinds.count("message_started"), 1)
        self.assertEqual(kinds[-1], "done")
        steps = [m for m in self.repo.list_messages(self.chat.id) if m.is_task_manager_step]
        self.assertEqual(len(steps), 1)

    def test_message_run_stops_with_pause_setting(self):
        self.repo.update_chat_settings(self.chat.id, {"task_tracking_enabled": True, "task_pause_each_stage": True})
        self.provider.tool_calls_queue = [
            [_tool_call("c1", "start_task", {"title": "Отчёт"})],
        ]
        run, _, _ = self.manager.start_message_run(self.chat.id, "Сделай отчёт")
        self.assertEqual(self.manager.wait(run.id, WAIT).status, "succeeded")
        kinds = [e["type"] for e in _all_events(self.manager, run.id)]
        self.assertNotIn("message_started", kinds)


class ToolCallEventsTests(unittest.TestCase):
    def test_all_tools_reported_with_source(self):
        repo = make_repository()
        provider = ToolCallingFakeProvider()
        repo._registry = FakeRegistry(provider)  # type: ignore[attr-defined]
        manager = RunManager(repo, workers=1, timeout=30)
        try:
            agent = repo.create_agent("Агент", model=TEST_MODEL_ID)
            chat = repo.create_chat(agent.id, "Чат")
            repo.update_chat_settings(chat.id, {"memory_tools_enabled": True, "working_memory_enabled": True})
            provider.tool_calls_queue = [[_tool_call("c1", "save_working_memory", {"key": "k", "value": "v"})]]
            run, _, draft = manager.start_message_run(chat.id, "Запомни")
            manager.wait(run.id, WAIT)
            tool_events = [e for e in _all_events(manager, run.id) if e["type"] == "tool_call"]
            self.assertEqual([(e["name"], e["source"], e["status"]) for e in tool_events], [
                ("save_working_memory", "memory", "started"), ("save_working_memory", "memory", "finished"),
            ])
            saved = repo._db.get_message(draft.id)
            self.assertEqual(len(json.loads(saved.tool_events)), 2)
        finally:
            manager.shutdown(wait=True)
            os.unlink(repo._db_path_for_cleanup)  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()

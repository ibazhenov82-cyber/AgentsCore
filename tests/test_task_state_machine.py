"""
Тесты "Работу с задачами требуется переделать" — машина состояний задачи,
теперь заданная в КОДЕ (`agents_core.task_state_machine`, прямой перенос
примера ТЗ на Kotlin: `TaskState`/`transitions`/`transition()`), и её
интеграция в tool-calling/"Менеджер задач". Полностью офлайн, как и
test_memory_and_profiles.py: использует `ToolCallingFakeProvider` вместо
настоящих провайдеров, поэтому не требует установки FastAPI/сетевого
доступа.

Ключевые отличия от прежней (DB-driven) версии этих тестов:
  - явного действия "Отклонить" больше нет вообще (ТЗ: "действия отклонить
    не требуется") — задача либо доходит до "done" по графу переходов, либо
    остаётся в работе/на паузе;
  - `apply_task_action` принимает `target_state` (новый этап), а не
    `action` с "advance"/"pause"/"reject" — доступные `target_state`
    вычисляются из `task_state_machine.allowed_transitions`, а не из
    справочника;
  - "Пауза" — ортогональный состоянию флаг `Task.paused`, не часть графа
    переходов; единственное оставшееся РУЧНОЕ действие (без обращения к
    модели) — `apply_manual_task_action(task_id, "pause")`;
  - каталог состояний/переходов read-only (`get_task_state_machine_info`),
    настраивается только список привязанных инвариантов категории
    "Правило стейт-машины" (`set_task_machine_invariants`).
"""

from __future__ import annotations

import json
import unittest

from agents_core.repository import NotFoundError, Repository, ValidationError
from agents_core.task_state_machine import (
    TASK_STATE_ORDER,
    TaskState,
    allowed_transitions,
    can_transition,
    display_name,
    is_final,
    parse_task_state,
    transition,
)
from tests.test_memory_and_profiles import ToolCallingFakeProvider, _tool_call, make_tool_repo
from tests.test_repository import TEST_MODEL_ID, make_repository


def _enable_task_tracking(repo: Repository, chat_id: str, pause_each_stage: bool = True) -> None:
    """Тесты этого модуля проверяют режим «Останавливаться на каждом этапе»
    (пауза после создания и каждого этапа); режим без пауз (по умолчанию) —
    см. test_search_tasks.py."""
    repo.update_chat_settings(chat_id, {"task_tracking_enabled": True, "task_pause_each_stage": pause_each_stage})


class TaskStateMachineModuleTests(unittest.TestCase):
    """Чистые функции `task_state_machine.py`, без обращения к БД — прямой
    перенос примера ТЗ (`transitions`, `fun transition(ctx, target)`)."""

    def test_order_and_display_names(self):
        self.assertEqual(
            TASK_STATE_ORDER, [TaskState.PLANNING, TaskState.EXECUTION, TaskState.VALIDATION, TaskState.DONE]
        )
        self.assertEqual(display_name(TaskState.PLANNING), "Планирование")
        self.assertEqual(display_name(TaskState.EXECUTION), "Выполнение")
        self.assertEqual(display_name(TaskState.VALIDATION), "Проверка")
        self.assertEqual(display_name(TaskState.DONE), "Готово")

    def test_transitions_match_tz_example(self):
        self.assertEqual(allowed_transitions(TaskState.PLANNING), [TaskState.EXECUTION])
        self.assertEqual(allowed_transitions(TaskState.EXECUTION), [TaskState.VALIDATION, TaskState.PLANNING])
        self.assertEqual(allowed_transitions(TaskState.VALIDATION), [TaskState.DONE, TaskState.EXECUTION])
        self.assertEqual(allowed_transitions(TaskState.DONE), [])

    def test_can_transition_and_transition_helper(self):
        self.assertTrue(can_transition(TaskState.PLANNING, TaskState.EXECUTION))
        self.assertFalse(can_transition(TaskState.PLANNING, TaskState.VALIDATION))
        self.assertFalse(can_transition(TaskState.PLANNING, TaskState.DONE))
        self.assertEqual(transition(TaskState.PLANNING, TaskState.EXECUTION), TaskState.EXECUTION)
        with self.assertRaises(ValueError):
            transition(TaskState.PLANNING, TaskState.DONE)

    def test_is_final_only_for_done(self):
        self.assertFalse(is_final(TaskState.PLANNING))
        self.assertFalse(is_final(TaskState.VALIDATION))
        self.assertTrue(is_final(TaskState.DONE))

    def test_parse_task_state_raises_readable_error_on_unknown_value(self):
        with self.assertRaises(ValueError):
            parse_task_state("nonexistent")


class TaskStateMachineInfoTests(unittest.TestCase):
    """Read-only описание машины (замена формы-редактора справочников) +
    настройка привязанных инвариантов категории "Правило стейт-машины"."""

    def setUp(self) -> None:
        self.repo = make_repository()

    def tearDown(self) -> None:
        import os
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def test_info_lists_four_states_in_order_with_target_states_from_code(self):
        info = self.repo.get_task_state_machine_info()
        states = info["states"]
        self.assertEqual([s["state"] for s in states], ["planning", "execution", "validation", "done"])
        self.assertEqual([s["position"] for s in states], [1, 2, 3, 4])
        by_state = {s["state"]: s for s in states}
        self.assertEqual(by_state["planning"]["target_states"], ["execution"])
        self.assertEqual(by_state["planning"]["target_state_display_names"], ["Выполнение"])
        self.assertEqual(by_state["execution"]["target_states"], ["validation", "planning"])
        self.assertEqual(by_state["validation"]["target_states"], ["done", "execution"])
        self.assertEqual(by_state["done"]["target_states"], [])

    def test_five_default_state_machine_rule_invariants_seeded_but_not_attached(self):
        """ТЗ, п.5 — пять дефолтных инвариантов категории "Правило
        стейт-машины" сидируются в общий справочник, но НЕ привязываются к
        машине автоматически (см. `Database._seed_task_state_machine_rule_invariants`) —
        пользователь подключает нужные вручную, как любой другой инвариант."""
        rule_invariants = [i for i in self.repo.list_invariants() if i.kind == "state_machine_rule"]
        self.assertEqual(len(rule_invariants), 5)
        titles = {i.title for i in rule_invariants}
        self.assertEqual(titles, {
            "Работай только в рамках current step",
            "Не перепрыгивай этапы",
            "Если step завершён — верни next step",
            "Нельзя делать реализацию до утверждённого плана",
            "Нельзя делать финал без валидации",
        })
        info = self.repo.get_task_state_machine_info()
        self.assertEqual(info["invariants"], [])  # ничего не привязано по умолчанию

    def test_set_task_machine_invariants_rejects_wrong_kind(self):
        other = self.repo.create_invariant("Другое", "текст", kind="business_rule")
        with self.assertRaises(ValidationError):
            self.repo.set_task_machine_invariants([other.id])

    def test_set_task_machine_invariants_rejects_unknown_id(self):
        with self.assertRaises(NotFoundError):
            self.repo.set_task_machine_invariants(["nope"])

    def test_set_task_machine_invariants_replaces_whole_list_and_dedupes(self):
        rule = next(i for i in self.repo.list_invariants() if i.kind == "state_machine_rule")
        custom = self.repo.create_invariant("Своё правило", "Всегда объясняй план", kind="state_machine_rule")
        info = self.repo.set_task_machine_invariants([rule.id, custom.id, rule.id])
        self.assertEqual({i.id for i in info["invariants"]}, {rule.id, custom.id})
        # Полная замена — второй вызов с одним id убирает первый.
        info2 = self.repo.set_task_machine_invariants([custom.id])
        self.assertEqual([i.id for i in info2["invariants"]], [custom.id])

    def test_rule_texts_feed_into_task_context_block_rules_section(self):
        """"Rules" в блоке [ЗАДАЧА] — тексты ИМЕННО привязанных инвариантов
        (ТЗ, п.6: "не 3 захардкоженные строки из примера, а динамический
        список"), а не весь каталог."""
        attached = self.repo.create_invariant("Привязанное правило", "Всегда фиксируй CURRENT", kind="state_machine_rule")
        self.repo.create_invariant("Непривязанное правило", "Это не должно попасть в Rules", kind="state_machine_rule")
        self.repo.set_task_machine_invariants([attached.id])
        self.assertEqual(self.repo._state_machine_rule_texts(), ["Всегда фиксируй CURRENT"])


class TaskTrackingGatingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = make_repository()
        self.agent = self.repo.create_agent("Агент задач", model=TEST_MODEL_ID)
        self.chat = self.repo.create_chat(self.agent.id, "Чат")

    def tearDown(self) -> None:
        import os
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def test_task_tracking_disabled_by_default_and_tools_not_offered(self):
        chat = self.repo._require_chat(self.chat.id)
        self.assertFalse(chat.settings.task_tracking_enabled)
        merged = self.repo._settings_with_merged_tools(chat)
        tools = json.loads(merged.tools_json) if merged.tools_json.strip() else []
        names = {t["function"]["name"] for t in tools}
        self.assertNotIn("start_task", names)
        self.assertNotIn("apply_task_action", names)

    def test_start_task_tool_offered_once_enabled(self):
        _enable_task_tracking(self.repo, self.chat.id)
        chat = self.repo._require_chat(self.chat.id)
        merged = self.repo._settings_with_merged_tools(chat)
        tools = json.loads(merged.tools_json)
        names = {t["function"]["name"] for t in tools}
        self.assertIn("start_task", names)
        # apply_task_action ещё не нужен — в чате пока нет ни одной задачи.
        self.assertNotIn("apply_task_action", names)

    def test_apply_task_action_target_state_enum_reflects_current_stage(self):
        _enable_task_tracking(self.repo, self.chat.id)
        chat = self.repo._require_chat(self.chat.id)
        agent = self.repo._require_agent(chat.agent_id)
        self.repo._handle_start_task(chat, agent, {"title": "Т"})
        chat = self.repo._require_chat(self.chat.id)  # перечитать после создания задачи
        merged = self.repo._settings_with_merged_tools(chat)
        tools = json.loads(merged.tools_json)
        apply_tool = next(t for t in tools if t["function"]["name"] == "apply_task_action")
        enum = set(apply_tool["function"]["parameters"]["properties"]["target_state"]["enum"])
        # Из "planning" (единственный неконечный этап с одним выходом)
        # доступен только "execution" — в отличие от старой модели, где
        # были доступны сразу все три действия из любого неконечного
        # состояния, доступность target_state теперь строго по графу
        # переходов из ТЗ.
        self.assertEqual(enum, {"execution"})


class TaskLifecycleToolCallingTests(unittest.TestCase):
    """Сквозной сценарий через настоящий tool-calling цикл (ToolCallingFakeProvider),
    как в test_memory_and_profiles.py — не напрямую через _handle_*, а через
    send_message_blocking, чтобы проверить И интеграцию с _run_tool_loop_blocking/
    Message.task_events, а не только сами обработчики."""

    def setUp(self) -> None:
        self.repo = make_tool_repo()
        self.agent = self.repo.create_agent("Агент задач", model=TEST_MODEL_ID)
        self.chat = self.repo.create_chat(self.agent.id, "Чат")
        _enable_task_tracking(self.repo, self.chat.id)
        self.provider: ToolCallingFakeProvider = self.repo._tool_provider  # type: ignore[attr-defined]

    def tearDown(self) -> None:
        import os
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def _task_id_after_start(self) -> str:
        self.provider.tool_calls_queue = [
            [_tool_call("c1", "start_task", {"title": "Отчёт по продажам Q3"})],
        ]
        _user, assistant = self.repo.send_message_blocking(self.chat.id, "Подготовь отчёт по продажам Q3")
        events = json.loads(assistant.task_events)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["task_title"], "Отчёт по продажам Q3")
        self.assertEqual(events[0]["to_state_display_name"], "Планирование")
        self.assertEqual(events[0]["kind"], "advance")
        tasks = self.repo.list_tasks_for_chat(self.chat.id)
        self.assertEqual(len(tasks), 1)
        return tasks[0]["task"].id

    def test_start_task_creates_task_in_initial_state_paused_and_records_event(self):
        task_id = self._task_id_after_start()
        detail = self.repo.get_task(task_id)
        self.assertEqual(detail["task"].state, "planning")
        self.assertEqual(detail["task"].current_step, None)
        # Задача ВСЕГДА ставится на паузу сразу после start_task, до
        # планирования/выполнения/проверки — пользователь должен увидеть её
        # и подтвердить продолжение.
        self.assertEqual(detail["status"], "paused")
        self.assertEqual(detail["next_state_display_name"], "Выполнение")
        history = detail["history"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["applied_by"], "system")
        self.assertEqual(history[0]["kind"], "pause")
        self.assertEqual(history[0]["from_state"], "planning")
        self.assertEqual(history[0]["to_state"], "planning")

    def test_advance_resumes_from_pause_and_repauses_after_one_step(self):
        task_id = self._task_id_after_start()

        # planning (на паузе) -> execution: ОДНО действие апдейта target_state
        # снимает паузу И продвигает этап (доступность не зависит от
        # статуса) — и тут же сама снова ставит задачу на паузу
        # (auto_pause=true, режим обычного чата) — показать пользователю
        # каждый следующий этап перед продолжением.
        self.provider.tool_calls_queue = [
            [_tool_call("c2", "apply_task_action", {"task_id": task_id, "target_state": "execution"})],
        ]
        _u, a2 = self.repo.send_message_blocking(self.chat.id, "Начинай")
        detail = self.repo.get_task(task_id)
        self.assertEqual(detail["task"].state, "execution")
        self.assertEqual(detail["status"], "paused")
        events2 = json.loads(a2.task_events)
        self.assertEqual(len(events2), 1)  # автопауза не попадает в task_events, только сам advance
        self.assertEqual(events2[0]["kind"], "advance")

        history_after_advance = detail["history"]
        self.assertEqual(len(history_after_advance), 3)  # старт-пауза, advance, автопауза
        self.assertEqual([h["applied_by"] for h in history_after_advance], ["system", "agent", "system"])
        self.assertEqual([h["kind"] for h in history_after_advance], ["pause", "advance", "pause"])

        # execution -> execution (без смены этапа) — модель явно фиксирует
        # current_step, не продвигая этап; в task_events это "update", а не
        # "advance", и автопаузу не порождает (этап не менялся).
        self.provider.tool_calls_queue = [
            [_tool_call("c3", "apply_task_action", {
                "task_id": task_id, "current_step": "сбор данных по региону EMEA",
            })],
        ]
        _u, a3 = self.repo.send_message_blocking(self.chat.id, "Отложи это, вернусь позже")
        detail = self.repo.get_task(task_id)
        self.assertEqual(detail["status"], "paused")  # пауза не снималась этим вызовом
        self.assertEqual(detail["task"].current_step, "сбор данных по региону EMEA")
        events3 = json.loads(a3.task_events)
        self.assertEqual(events3[0]["kind"], "update")
        self.assertEqual(events3[0]["task_id"], task_id)

        # execution -> validation (advance) — снова снимает паузу и
        # продвигает; current_step сохраняется, т.к. модель его не передала
        # повторно (продолжение БЕЗ повторного объяснения).
        self.provider.tool_calls_queue = [
            [_tool_call("c4", "apply_task_action", {"task_id": task_id, "target_state": "validation"})],
        ]
        self.repo.send_message_blocking(self.chat.id, "Продолжай")
        detail = self.repo.get_task(task_id)
        self.assertEqual(detail["status"], "paused")  # снова автопауза
        self.assertEqual(detail["task"].current_step, "сбор данных по региону EMEA")
        self.assertEqual(detail["task"].state, "validation")

    def test_second_advance_of_same_task_in_one_turn_is_blocked_while_auto_pause(self):
        """Задачу нужно показать пользователю ДО того, как сформировался
        ответ на комплексную просьбу — модель не может продвинуть ОДНУ и ту
        же задачу больше одного раза за один свой ответ, пока
        auto_pause=true (обычный чат)."""
        task_id = self._task_id_after_start()
        self.provider.tool_calls_queue = [
            [
                _tool_call("c2", "apply_task_action", {"task_id": task_id, "target_state": "execution"}),
                _tool_call("c3", "apply_task_action", {"task_id": task_id, "target_state": "validation"}),
            ],
        ]
        self.repo.send_message_blocking(self.chat.id, "Сделай сразу всё")
        detail = self.repo.get_task(task_id)
        # Продвинулась только на ОДИН этап (до execution), а не до validation.
        self.assertEqual(detail["task"].state, "execution")
        self.assertEqual(detail["status"], "paused")

    def test_invalid_target_state_for_current_stage_returns_error_and_is_not_applied(self):
        task_id = self._task_id_after_start()
        self.provider.tool_calls_queue = [
            # "planning" не может перейти напрямую в "done".
            [_tool_call("c2", "apply_task_action", {"task_id": task_id, "target_state": "done"})],
        ]
        _u, assistant = self.repo.send_message_blocking(self.chat.id, "Сразу заверши")
        self.assertIsNone(assistant.task_events)
        detail = self.repo.get_task(task_id)
        self.assertEqual(detail["task"].state, "planning")

    def test_unknown_target_state_value_returns_error(self):
        task_id = self._task_id_after_start()
        self.provider.tool_calls_queue = [
            [_tool_call("c2", "apply_task_action", {"task_id": task_id, "target_state": "made_up_state"})],
        ]
        _u, assistant = self.repo.send_message_blocking(self.chat.id, "Сделай что-то несуществующее")
        self.assertIsNone(assistant.task_events)
        detail = self.repo.get_task(task_id)
        self.assertEqual(detail["task"].state, "planning")

    def test_task_reaches_done_and_disappears_from_default_task_list(self):
        task_id = self._task_id_after_start()
        for target in ("execution", "validation", "done"):
            self.provider.tool_calls_queue = [
                [_tool_call(f"c-{target}", "apply_task_action", {"task_id": task_id, "target_state": target})],
            ]
            self.repo.send_message_blocking(self.chat.id, "Продолжай")
        detail = self.repo.get_task(task_id)
        self.assertEqual(detail["task"].state, "done")
        self.assertEqual(detail["status"], "done")
        self.assertEqual(detail["available_actions"], [])
        # done — задача больше не в списке открытых по умолчанию.
        self.assertEqual(self.repo.list_tasks_for_chat(self.chat.id), [])
        summaries = self.repo.list_tasks_for_chat(self.chat.id, include_completed=True)
        self.assertEqual(summaries[0]["status"], "done")

    def test_task_context_block_lists_id_stage_plan_and_rules(self):
        """Новый шаблон (ТЗ "заменить system prompt" / поля task, state,
        step, total, plan, done, current): глобальный prompt-шаблон
        (injected[0], один на весь чат — несёт и текст инвариантов машины
        состояний в <task_state_machine_invariants>) + отдельный блок на
        каждую открытую задачу (injected[1], новая словарность вместо
        старых [STATE]/[PLAN]/[DONE])."""
        rule = self.repo.create_invariant("Веди по шагам", "Не пропускай шаги плана", kind="state_machine_rule")
        self.repo.set_task_machine_invariants([rule.id])
        task_id = self._task_id_after_start()
        chat = self.repo._require_chat(self.chat.id)
        agent = self.repo._require_agent(chat.agent_id)
        injected = self.repo._build_task_and_invariant_context(chat, agent)
        # Один глобальный блок-шаблон + один блок на задачу.
        self.assertEqual(len(injected), 2)
        template_text = injected[0].content
        self.assertIn("TaskState:", template_text)
        self.assertIn("Не пропускай шаги плана", template_text)
        task_text = injected[1].content
        self.assertIn(task_id, task_text)
        self.assertIn("task:", task_text)
        self.assertIn("state: planning", task_text)
        self.assertIn("step: 1", task_text)
        self.assertIn("total: 4", task_text)
        self.assertIn("plan:", task_text)
        self.assertIn("done:", task_text)
        self.assertIn("current:", task_text)

    def test_proactivity_hint_injected_even_without_open_tasks(self):
        """Глобальный шаблон подмешивается сам по себе, даже когда в чате
        ещё нет ни одной задачи — модели не нужно ждать открытой задачи,
        чтобы узнать формат/правила отслеживания."""
        chat = self.repo._require_chat(self.chat.id)
        agent = self.repo._require_agent(chat.agent_id)
        injected = self.repo._build_task_and_invariant_context(chat, agent)
        self.assertEqual(len(injected), 1)
        self.assertIn("возвращай со следующими параметрами", injected[0].content)

    def test_no_injection_when_tracking_and_invariants_both_disabled(self):
        self._task_id_after_start()
        self.repo.update_chat_settings(self.chat.id, {"task_tracking_enabled": False})
        chat = self.repo._require_chat(self.chat.id)
        agent = self.repo._require_agent(chat.agent_id)
        self.assertEqual(self.repo._build_task_and_invariant_context(chat, agent), [])


class TaskManualActionTests(unittest.TestCase):
    """Ручное вмешательство человека без обращения к модели — теперь
    ЕДИНСТВЕННОЕ поддерживаемое действие: "pause" ("Продолжить"/"Выполнить"
    человека ВСЕГДА идёт через run_task_manager_step, см. TaskManagerTests;
    явного "Отклонить" в новой модели нет вовсе)."""

    def setUp(self) -> None:
        self.repo = make_repository()
        self.agent = self.repo.create_agent("Агент задач", model=TEST_MODEL_ID)
        self.chat = self.repo.create_chat(self.agent.id, "Чат")
        _enable_task_tracking(self.repo, self.chat.id)
        chat = self.repo._require_chat(self.chat.id)
        agent = self.repo._require_agent(chat.agent_id)
        started = self.repo._handle_start_task(chat, agent, {"title": "Ручная задача"})
        self.task_id = started["task_id"]
        # Продвигаем её вручную (через обработчик напрямую, минуя ограничение
        # "не больше одного advance за turn", которое завязано на список
        # events из tool-calling цикла) в "execution", прежде чем проверять
        # ручную паузу из этого этапа; auto_pause=False, чтобы не пришлось
        # снимать автопаузу перед проверкой.
        self.repo._handle_apply_task_action(
            chat, agent, {"task_id": self.task_id, "target_state": "execution"}, auto_pause=False,
        )

    def tearDown(self) -> None:
        import os
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def test_manual_pause_is_recorded_with_applied_by_manual(self):
        self.assertEqual(self.repo.get_task(self.task_id)["status"], "active")
        self.repo.apply_manual_task_action(self.task_id, "pause", note="ушёл на встречу")
        detail = self.repo.get_task(self.task_id)
        self.assertEqual(detail["status"], "paused")
        self.assertEqual(detail["history"][-1]["applied_by"], "manual")
        self.assertEqual(detail["history"][-1]["kind"], "pause")
        self.assertEqual(detail["history"][-1]["note"], "ушёл на встречу")

    def test_manual_pause_rejects_already_paused_task(self):
        self.repo.apply_manual_task_action(self.task_id, "pause")
        with self.assertRaises(ValidationError):
            self.repo.apply_manual_task_action(self.task_id, "pause")

    def test_manual_pause_rejects_done_task(self):
        chat = self.repo._require_chat(self.chat.id)
        agent = self.repo._require_agent(chat.agent_id)
        self.repo._handle_apply_task_action(chat, agent, {"task_id": self.task_id, "target_state": "validation"}, auto_pause=False)
        self.repo._handle_apply_task_action(chat, agent, {"task_id": self.task_id, "target_state": "done"}, auto_pause=False)
        self.assertEqual(self.repo.get_task(self.task_id)["status"], "done")
        with self.assertRaises(ValidationError):
            self.repo.apply_manual_task_action(self.task_id, "pause")

    def test_manual_action_rejects_unsupported_kind(self):
        """Явного "Отклонить" в новой модели нет вовсе (ТЗ: "действия
        отклонить не требуется")."""
        with self.assertRaises(ValidationError):
            self.repo.apply_manual_task_action(self.task_id, "advance")
        with self.assertRaises(ValidationError):
            self.repo.apply_manual_task_action(self.task_id, "reject")

    def test_manual_action_rejects_unknown_task(self):
        with self.assertRaises(NotFoundError):
            self.repo.apply_manual_task_action("does-not-exist", "pause", note=None)


class TaskManagerTests(unittest.TestCase):
    """"Менеджер задач": кнопка «Продолжить» — run_task_manager_step(auto_pause=True),
    один шаг и снова пауза; кнопка «Выполнить» — run_task_manager_step(auto_pause=False),
    без остановок до done, клиент вызывает в цикле. Использует
    `ToolCallingFakeProvider.stream_chat`, т.к. `run_task_manager_step`
    всегда работает в потоковом режиме."""

    def setUp(self) -> None:
        self.repo = make_tool_repo()
        self.agent = self.repo.create_agent("Агент задач", model=TEST_MODEL_ID)
        self.chat = self.repo.create_chat(self.agent.id, "Чат")
        _enable_task_tracking(self.repo, self.chat.id)
        self.provider: ToolCallingFakeProvider = self.repo._tool_provider  # type: ignore[attr-defined]
        chat = self.repo._require_chat(self.chat.id)
        agent = self.repo._require_agent(chat.agent_id)
        started = self.repo._handle_start_task(chat, agent, {"title": "Отчёт по продажам Q3"})
        self.task_id = started["task_id"]
        # start_task уже поставил задачу на паузу — это ожидаемое исходное
        # состояние для всех тестов этого класса.

    def tearDown(self) -> None:
        import os
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def test_continue_button_resumes_from_pause_advances_once_and_repauses(self):
        self.provider.tool_calls_queue = [
            [_tool_call("c1", "apply_task_action", {"task_id": self.task_id, "target_state": "execution"})],
        ]
        events = list(self.repo.run_task_manager_step(self.chat.id, self.task_id))  # auto_pause=True по умолчанию
        done = events[-1]
        self.assertEqual(done["type"], "done")
        # Один шаг — и снова пауза: клиенту нужно новое нажатие "Продолжить".
        self.assertFalse(done["should_continue"])
        self.assertEqual(done["task_status"], "paused")
        self.assertTrue(done["message"].is_task_manager_step)
        # Ни одного сообщения с ролью user — шаг НЕ создаёт фальшивую реплику
        # пользователя, эфемерная реплика "продолжай" уходит только провайдеру.
        messages = self.repo.list_messages(self.chat.id)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, "assistant")

    def test_step_stops_when_model_does_not_advance(self):
        # Очередь пуста -> провайдер отвечает обычным текстом, без tool_calls;
        # статус остаётся таким, каким был (задача создана уже на паузе).
        events = list(self.repo.run_task_manager_step(self.chat.id, self.task_id))
        done = events[-1]
        self.assertFalse(done["should_continue"])
        self.assertEqual(done["task_status"], "paused")

    def test_execute_button_auto_pause_false_reaches_final_state_within_one_call_each(self):
        self.provider.tool_calls_queue = [
            [_tool_call("c1", "apply_task_action", {"task_id": self.task_id, "target_state": "execution"})],
        ]
        events = list(self.repo.run_task_manager_step(self.chat.id, self.task_id, auto_pause=False))
        self.assertEqual(events[-1]["task_status"], "active")
        self.assertTrue(events[-1]["should_continue"])

        self.provider.tool_calls_queue = [
            [_tool_call("c2", "apply_task_action", {"task_id": self.task_id, "target_state": "validation"})],
        ]
        events = list(self.repo.run_task_manager_step(self.chat.id, self.task_id, auto_pause=False))
        self.assertEqual(events[-1]["task_status"], "active")

        self.provider.tool_calls_queue = [
            [_tool_call("c3", "apply_task_action", {"task_id": self.task_id, "target_state": "done"})],
        ]
        events = list(self.repo.run_task_manager_step(self.chat.id, self.task_id, auto_pause=False))
        done = events[-1]
        self.assertEqual(done["task_status"], "done")
        self.assertFalse(done["should_continue"])  # завершена — продолжать нечего

    def test_step_raises_when_tracking_disabled(self):
        self.repo.update_chat_settings(self.chat.id, {"task_tracking_enabled": False})
        with self.assertRaises(ValidationError):
            list(self.repo.run_task_manager_step(self.chat.id, self.task_id))

    def test_step_allowed_when_task_is_paused_resumes_and_advances(self):
        """В отличие от простого запрета, «Продолжить» РАЗРЕШЕНО вызывать,
        даже когда задача уже на паузе — это и есть штатный сценарий
        (снимает паузу и продвигает этап за один вызов)."""
        chat = self.repo._require_chat(self.chat.id)
        agent = self.repo._require_agent(chat.agent_id)
        self.repo._handle_apply_task_action(chat, agent, {"task_id": self.task_id, "target_state": "execution"}, auto_pause=False)
        self.repo.apply_manual_task_action(self.task_id, "pause")
        self.assertEqual(self.repo.get_task(self.task_id)["status"], "paused")
        self.provider.tool_calls_queue = [
            [_tool_call("c1", "apply_task_action", {"task_id": self.task_id, "target_state": "validation"})],
        ]
        events = list(self.repo.run_task_manager_step(self.chat.id, self.task_id))
        done = events[-1]
        self.assertEqual(done["type"], "done")
        detail = self.repo.get_task(self.task_id)
        self.assertEqual(detail["task"].state, "validation")

    def test_step_raises_when_task_is_done(self):
        chat = self.repo._require_chat(self.chat.id)
        agent = self.repo._require_agent(chat.agent_id)
        for target in ("execution", "validation", "done"):
            self.repo._handle_apply_task_action(chat, agent, {"task_id": self.task_id, "target_state": target}, auto_pause=False)
        self.assertEqual(self.repo.get_task(self.task_id)["status"], "done")
        with self.assertRaises(ValidationError):
            list(self.repo.run_task_manager_step(self.chat.id, self.task_id))

    def test_max_steps_limit_stops_without_calling_the_model(self):
        # Бюджет продвижений default-машины без ухода в "done" — 2 (planning->
        # execution->validation); лимит выставлен так, чтобы уложиться в него
        # с "Выполнить" (auto_pause=false, статус остаётся "active").
        self.repo.update_chat_settings(self.chat.id, {"task_manager_max_steps": 2})
        for i, target in enumerate(("execution", "validation")):
            self.provider.tool_calls_queue = [
                [_tool_call(f"c{i}", "apply_task_action", {"task_id": self.task_id, "target_state": target})],
            ]
            events = list(self.repo.run_task_manager_step(self.chat.id, self.task_id, auto_pause=False))
            done = events[-1]
            self.assertTrue(done["should_continue"])
            self.assertEqual(done["task_status"], "active")

        calls_before = len(self.provider.calls_log)
        events = list(self.repo.run_task_manager_step(self.chat.id, self.task_id, auto_pause=False))
        done = events[-1]
        self.assertFalse(done["should_continue"])
        self.assertIn("лимит", done["message"].content.lower())
        self.assertTrue(done["message"].is_task_manager_step)
        # Провайдер не вызывался вообще для этого (лимитного) шага.
        self.assertEqual(len(self.provider.calls_log), calls_before)

    def test_consecutive_steps_reset_by_real_user_message(self):
        chat = self.repo._require_chat(self.chat.id)
        agent = self.repo._require_agent(chat.agent_id)
        # planning -> execution, БЕЗ автопаузы (auto_pause=False), статус active.
        self.repo._handle_apply_task_action(chat, agent, {"task_id": self.task_id, "target_state": "execution"}, auto_pause=False)
        self.provider.tool_calls_queue = [
            [_tool_call("c1", "apply_task_action", {"task_id": self.task_id, "target_state": "validation"})],
        ]
        list(self.repo.run_task_manager_step(self.chat.id, self.task_id, auto_pause=False))
        self.assertEqual(self.repo._consecutive_task_manager_steps(self.chat.id), 1)

        self.repo.send_message_blocking(self.chat.id, "Обычное сообщение пользователя")
        self.assertEqual(self.repo._consecutive_task_manager_steps(self.chat.id), 0)


if __name__ == "__main__":
    unittest.main()

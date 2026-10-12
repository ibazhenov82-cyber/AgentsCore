"""
Тесты "Инварианты и ограничения состояния": общий справочник (по аналогии
со справочником профилей), множественный выбор для агента/чата (объединение
на уровне чата), инъекция в системный промпт структурированным блоком
`[INVARIANTS]` (отдельного тумблера-настройки нет — инварианты считаются
разрешёнными сами по себе, как только для агента/чата выбран хотя бы один,
см. `Repository._active_invariants`), и
(новое ТЗ "Работу с задачами требуется переделать", раздел "Инварианты",
п.2-3) ДОПОЛНИТЕЛЬНАЯ код-уровневая проверка ответа модели для категорий с
чекером + один автоматический переспрос при нарушении, с текстом
предупреждения ПЕРЕД ответом (см. `InvariantCodeLevelValidationTests`).
Полностью офлайн, как и test_repository.py/test_memory_and_profiles.py.
"""

from __future__ import annotations

import unittest

from agents_core.repository import NotFoundError, Repository, ValidationError
from tests.test_memory_and_profiles import ToolCallingFakeProvider, make_tool_repo
from tests.test_repository import TEST_MODEL_ID, make_repository


class InvariantCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo: Repository = make_repository()

    def test_create_invariant_requires_non_empty_title_and_rule_text(self):
        with self.assertRaises(ValidationError):
            self.repo.create_invariant("", "текст правила")
        with self.assertRaises(ValidationError):
            self.repo.create_invariant("Название", "  ")

    def test_create_invariant_rejects_unknown_kind(self):
        with self.assertRaises(ValidationError):
            self.repo.create_invariant("Название", "текст", kind="unknown_kind")

    def test_create_invariant_defaults_to_active_and_no_kind(self):
        inv = self.repo.create_invariant("Название", "текст правила")
        self.assertTrue(inv.is_active)
        self.assertIsNone(inv.kind)

    def test_catalog_is_shared_list_ordered_by_creation(self):
        # make_repository() сидирует 5 дефолтных инвариантов категории
        # "Правило стейт-машины" (см. Database._seed_task_state_machine_rule_invariants)
        # — они уже в справочнике до создания этих двух, поэтому проверяем
        # относительный порядок ХВОСТА списка, а не список целиком.
        before = {i.id for i in self.repo.list_invariants()}
        first = self.repo.create_invariant("Первый", "текст 1")
        second = self.repo.create_invariant("Второй", "текст 2")
        listed = self.repo.list_invariants()
        new_ids = [i.id for i in listed if i.id not in before]
        self.assertEqual(new_ids, [first.id, second.id])

    def test_update_invariant_rejects_unknown_field(self):
        inv = self.repo.create_invariant("Название", "текст")
        with self.assertRaises(ValidationError):
            self.repo.update_invariant(inv.id, {"scope_type": "agent"})

    def test_update_invariant_partial_patch(self):
        inv = self.repo.create_invariant("Название", "текст", kind="business_rule")
        updated = self.repo.update_invariant(inv.id, {"is_active": False})
        self.assertFalse(updated.is_active)
        self.assertEqual(updated.title, "Название")  # не тронуто

    def test_get_and_delete_unknown_invariant_raise_not_found(self):
        with self.assertRaises(NotFoundError):
            self.repo.get_invariant("nope")
        with self.assertRaises(NotFoundError):
            self.repo.delete_invariant("nope")


class InvariantSelectionTests(unittest.TestCase):
    """Множественный выбор для агента/чата — по аналогии с профилями, но
    список, а не одиночная ссылка (см. итоговую концепцию, замечание
    пользователя: "общий справочник ... применительно к чату или агенту
    можно выбрать эти инварианты в настройках, множественный выбор")."""

    def setUp(self) -> None:
        self.repo: Repository = make_repository()
        self.agent = self.repo.create_agent("Агент")
        self.chat = self.repo.create_chat(self.agent.id, "Чат")
        self.inv1 = self.repo.create_invariant("Первый", "правило 1")
        self.inv2 = self.repo.create_invariant("Второй", "правило 2")

    def test_set_agent_invariants_rejects_unknown_id(self):
        with self.assertRaises(NotFoundError):
            self.repo.set_agent_invariants(self.agent.id, ["nope"])

    def test_set_chat_invariants_rejects_unknown_id(self):
        with self.assertRaises(NotFoundError):
            self.repo.set_chat_invariants(self.chat.id, ["nope"])

    def test_set_agent_invariants_persists_and_dedupes(self):
        updated = self.repo.set_agent_invariants(self.agent.id, [self.inv1.id, self.inv1.id, self.inv2.id])
        self.assertEqual(updated.invariant_ids, [self.inv1.id, self.inv2.id])
        reloaded = self.repo.get_agent(self.agent.id)
        self.assertEqual(reloaded.invariant_ids, [self.inv1.id, self.inv2.id])

    def test_set_chat_invariants_independent_of_agent(self):
        self.repo.set_agent_invariants(self.agent.id, [self.inv1.id])
        self.repo.set_chat_invariants(self.chat.id, [self.inv2.id])
        agent = self.repo.get_agent(self.agent.id)
        chat = self.repo.get_chat(self.chat.id)
        self.assertEqual(agent.invariant_ids, [self.inv1.id])
        self.assertEqual(chat.invariant_ids, [self.inv2.id])

    def test_new_chat_does_not_inherit_agent_invariants_as_snapshot(self):
        """В отличие от default_profile_id (копируется в новый чат), выбор
        инвариантов агента — живая настройка: не копируется построчно в
        Chat.invariant_ids нового чата (объединение считается на лету в
        _effective_invariant_ids), поэтому у нового чата список пуст."""
        self.repo.set_agent_invariants(self.agent.id, [self.inv1.id])
        new_chat = self.repo.create_chat(self.agent.id, "Другой чат")
        self.assertEqual(new_chat.invariant_ids, [])

    def test_deleting_invariant_clears_it_from_agent_and_chat_selection(self):
        self.repo.set_agent_invariants(self.agent.id, [self.inv1.id, self.inv2.id])
        self.repo.set_chat_invariants(self.chat.id, [self.inv1.id])
        self.repo.delete_invariant(self.inv1.id)
        agent = self.repo.get_agent(self.agent.id)
        chat = self.repo.get_chat(self.chat.id)
        self.assertEqual(agent.invariant_ids, [self.inv2.id])
        self.assertEqual(chat.invariant_ids, [])


class InvariantInjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo: Repository = make_repository()
        self.agent = self.repo.create_agent("Агент")
        self.chat = self.repo.create_chat(self.agent.id, "Чат")
        self.inv_agent = self.repo.create_invariant("Глобальный", "Не транслитерировать термины", kind="business_rule")
        self.inv_chat = self.repo.create_invariant("Чатовый", "Только Clean Architecture", kind="tech_decision")
        self.inv_inactive = self.repo.create_invariant("Неактивный", "не должен попадать в промпт", is_active=False)
        self.repo.set_agent_invariants(self.agent.id, [self.inv_agent.id])
        self.repo.set_chat_invariants(self.chat.id, [self.inv_chat.id, self.inv_inactive.id])

    def _reload(self):
        return self.repo.get_chat(self.chat.id), self.repo.get_agent(self.agent.id)

    def test_no_injection_when_chat_and_agent_have_no_invariants_selected(self):
        # task_tracking_enabled тоже false по умолчанию, поэтому единый метод
        # `_build_task_and_invariant_context` (заменивший
        # `_build_invariant_injection_messages`/`_build_task_injection_messages`,
        # см. новое ТЗ "Работу с задачами требуется переделать") не добавляет
        # вообще ничего, если ни задачи не отслеживаются, ни выбран хотя бы
        # один инвариант. Используем СВЕЖИЕ агента/чат — у self.agent/self.chat
        # из setUp уже есть выбор, а значит инъекция для них уже должна идти.
        agent = self.repo.create_agent("Другой агент")
        chat = self.repo.create_chat(agent.id, "Другой чат")
        self.assertEqual(self.repo._build_task_and_invariant_context(chat, agent), [])

    def test_no_injection_when_nothing_selected_even_if_enabled(self):
        self.repo.set_agent_invariants(self.agent.id, [])
        self.repo.set_chat_invariants(self.chat.id, [])
        chat, agent = self._reload()
        self.assertEqual(self.repo._build_task_and_invariant_context(chat, agent), [])

    def test_injection_unions_agent_and_chat_selection_and_skips_inactive(self):
        chat, agent = self._reload()
        messages = self.repo._build_task_and_invariant_context(chat, agent)
        self.assertEqual(len(messages), 1)
        text = messages[0].content
        self.assertEqual(messages[0].role, "system")
        self.assertIn("[INVARIANTS]", text)
        self.assertIn("Не транслитерировать термины", text)
        self.assertIn("Только Clean Architecture", text)
        self.assertNotIn("не должен попадать в промпт", text)
        self.assertIn("ЗАПРЕЩЕНО", text)

    def test_injection_included_in_main_request_context_when_enabled(self):
        """Инъекция реально попадает в запрос к модели через тот же путь,
        что и память/задачи (см. `_build_request_context`) — используем
        внутренний метод напрямую, чтобы не тянуть провайдера."""
        chat, agent = self._reload()
        context = self.repo._build_request_context(chat, agent, [], "Привет")
        system_texts = [m.content for m in context if m.role == "system"]
        self.assertTrue(any("Только Clean Architecture" in t for t in system_texts))

    def test_clearing_selection_after_being_set_removes_injection_again(self):
        chat, agent = self._reload()
        self.assertEqual(len(self.repo._build_task_and_invariant_context(chat, agent)), 1)
        self.repo.set_agent_invariants(self.agent.id, [])
        self.repo.set_chat_invariants(self.chat.id, [])
        chat, agent = self._reload()
        self.assertEqual(self.repo._build_task_and_invariant_context(chat, agent), [])


class InvariantCodeLevelValidationTests(unittest.TestCase):
    """"Инварианты", п.2-3 нового ТЗ: ДОПОЛНИТЕЛЬНО к обычной инъекции в
    промпт, ответ модели проверяется программно для категорий с чекером
    (см. `agents_core.invariant_checks.CHECKERS`) — при нарушении модель
    переспрашивается ОДИН раз, а в итоговом сообщении перед ответом
    появляется предупреждение (см. `Repository._validate_and_maybe_retry`).
    Использует `ToolCallingFakeProvider`, как test_task_state_machine.py."""

    def setUp(self) -> None:
        self.repo = make_tool_repo()
        self.agent = self.repo.create_agent("Агент", model=TEST_MODEL_ID)
        self.chat = self.repo.create_chat(self.agent.id, "Чат")
        self.provider: ToolCallingFakeProvider = self.repo._tool_provider  # type: ignore[attr-defined]

    def tearDown(self) -> None:
        import os
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def test_stack_constraint_violation_gets_warning_prefix_and_triggers_one_retry(self):
        inv = self.repo.create_invariant("StackOnly", "Разрешено: Kotlin, Ktor", kind="stack_constraint")
        self.repo.set_chat_invariants(self.chat.id, [inv.id])
        self.provider.final_text = "Предлагаю решение на Spring Boot и Java."

        _user, assistant = self.repo.send_message_blocking(self.chat.id, "Помоги с бэкендом")

        self.assertTrue(assistant.content.startswith("⚠ Нарушение инварианта"))
        self.assertIn("StackOnly", assistant.content)
        self.assertIn("Spring Boot", assistant.content)
        self.assertIn("Java", assistant.content)
        self.assertIn("Разрешено: Kotlin, Ktor", assistant.content)
        # Исходный вызов + один переспрос при нарушении.
        self.assertEqual(len(self.provider.calls_log), 2)

    def test_compliant_response_has_no_warning_and_is_not_retried(self):
        inv = self.repo.create_invariant("StackOnly", "Разрешено: Kotlin, Ktor", kind="stack_constraint")
        self.repo.set_chat_invariants(self.chat.id, [inv.id])
        self.provider.final_text = "Предлагаю решение на Ktor."

        _user, assistant = self.repo.send_message_blocking(self.chat.id, "Помоги с бэкендом")

        self.assertFalse(assistant.content.startswith("⚠"))
        self.assertEqual(assistant.content, "Предлагаю решение на Ktor.")
        self.assertEqual(len(self.provider.calls_log), 1)

    def test_unrecognized_category_is_never_code_validated(self):
        """business_rule нет в invariant_checks.CHECKERS — соблюдение
        полностью доверяется модели, программная проверка не применяется."""
        inv = self.repo.create_invariant("Скидочная политика", "Разрешено: только праздничные скидки", kind="business_rule")
        self.repo.set_chat_invariants(self.chat.id, [inv.id])
        self.provider.final_text = "Скидка 90% без повода."

        _user, assistant = self.repo.send_message_blocking(self.chat.id, "Сделай скидку")

        self.assertFalse(assistant.content.startswith("⚠"))
        self.assertEqual(len(self.provider.calls_log), 1)

    def test_no_check_when_no_invariant_selected_even_with_violating_rule_text(self):
        # Инвариант создан, но НЕ привязан ни к агенту, ни к чату — эффективный
        # набор пуст, значит инъекции/проверки быть не должно (новая логика:
        # инварианты "разрешены", только если хотя бы один реально выбран).
        self.repo.create_invariant("StackOnly", "Разрешено: Kotlin, Ktor", kind="stack_constraint")
        self.provider.final_text = "Предлагаю решение на Spring Boot и Java."

        _user, assistant = self.repo.send_message_blocking(self.chat.id, "Помоги с бэкендом")

        self.assertFalse(assistant.content.startswith("⚠"))
        self.assertEqual(len(self.provider.calls_log), 1)


if __name__ == "__main__":
    unittest.main()

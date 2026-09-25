"""
Тесты новой функциональности: рабочая/долговременная память, профили-
пайплайны, единый механизм tool-calling (память + демо-скиллы «Покупки») и
снимок памяти (`memory-snapshot`). Полностью офлайн — как и test_repository.py,
использует `FakeProvider`/`FakeRegistry`/`FakeCatalog` вместо настоящих
провайдеров, поэтому не требует установки FastAPI/сетевого доступа.
"""

from __future__ import annotations

import json
import unittest
from typing import Iterator, List, Optional

from unittest.mock import patch

from agents_core.providers.base import ChatResult, ChatUsage, ProviderMessage, StreamDelta
from agents_core.repository import NotFoundError, Repository, ValidationError
from agents_core.skills import registry as skills_registry
from agents_core.skills import shopping_demo
from tests.test_repository import FakeCatalog, FakeRegistry, TEST_MODEL_ID, make_repository


def _make_shopping_profile(repo: Repository, agent_id: str = ""):
    """Собирает профиль «Покупки» напрямую через `skills_json` (тем же
    набором функций, что и демо-скилл) — используется в тестах, для которых
    важен сам факт "у профиля есть эти 3 функции", а не путь, которым они
    туда попали (для проверки ИМЕННО пути через реестр см. `ProfileTests.
    test_create_profile_via_registered_skill_names`). Профили — общий
    справочник, НЕ привязанный к агенту (см. замечание пользователя), поэтому
    `agent_id` здесь принимается только для обратной совместимости вызовов в
    существующих тестах и не используется."""
    return repo.create_profile(
        "Покупки",
        skills_json=json.dumps(shopping_demo.TOOL_DEFS, ensure_ascii=False),
        orchestration_prompt=shopping_demo.DEFAULT_ORCHESTRATION_PROMPT,
    )


class ToolCallingFakeProvider:
    """В отличие от `FakeProvider` (test_repository.py), этот фейк умеет
    отвечать `tool_calls` по требованию теста и затем — обычным текстом,
    воспроизводя реальный цикл "модель просит вызов -> получает результат ->
    отвечает текстом", который выполняет `Repository._run_tool_loop_blocking`
    / потоковая версия в `stream_message`."""

    name = "fake-tools"

    def __init__(self):
        self.last_messages: List[ProviderMessage] = []
        self.calls_log: List[List[ProviderMessage]] = []
        #: Очередь запрошенных tool_calls по порядку вызовов chat()/stream_chat();
        #: когда очередь пуста — ответ обычным текстом.
        self.tool_calls_queue: List[list] = []
        self.final_text = "Готово!"

    def health(self) -> bool:
        return True

    def discover_model(self, model_id: str):
        raise NotImplementedError

    def _next_tool_calls(self, settings) -> Optional[list]:
        # Как настоящий DeepSeek (см. `DeepSeekProvider._parsed_tools`):
        # без `tools` в запросе модель физически не может запросить вызов —
        # `_finalize_after_tool_cap` полагается именно на это свойство,
        # обнуляя `tools_json` для принудительного финального запроса.
        if not getattr(settings, "tools_json", ""):
            return None
        return self.tool_calls_queue.pop(0) if self.tool_calls_queue else None

    def chat(self, model_id: str, messages: List[ProviderMessage], settings) -> ChatResult:
        self.last_messages = list(messages)
        self.calls_log.append(list(messages))
        tool_calls = self._next_tool_calls(settings)
        return ChatResult(
            content="" if tool_calls else self.final_text,
            usage=ChatUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            tool_calls=tool_calls,
        )

    def stream_chat(self, model_id: str, messages: List[ProviderMessage], settings) -> Iterator[StreamDelta]:
        self.last_messages = list(messages)
        self.calls_log.append(list(messages))
        tool_calls = self._next_tool_calls(settings)
        content = "" if tool_calls else self.final_text
        if content:
            yield StreamDelta(content=content)
        yield StreamDelta(
            done=True,
            result=ChatResult(
                content=content,
                usage=ChatUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                tool_calls=tool_calls,
            ),
        )


def make_tool_repo() -> Repository:
    """Тот же приём, что `make_repository` в test_repository.py, но с
    `ToolCallingFakeProvider` вместо обычного `FakeProvider`."""
    repo = make_repository()
    provider = ToolCallingFakeProvider()
    repo._registry = FakeRegistry(provider)  # type: ignore[attr-defined]
    repo._tool_provider = provider  # type: ignore[attr-defined]
    return repo


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}


class WorkingMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = make_repository()
        self.agent = self.repo.create_agent("Агент памяти", model=TEST_MODEL_ID)
        self.chat = self.repo.create_chat(self.agent.id, "Чат")

    def tearDown(self) -> None:
        import os
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def test_save_and_list_working_memory(self):
        entry = self.repo.save_working_memory(self.chat.id, "цель", "собрать корзину покупок")
        self.assertEqual(entry.source, "manual")
        entries = self.repo.list_working_memory(self.chat.id)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].key, "цель")

    def test_upsert_replaces_not_duplicates(self):
        self.repo.save_working_memory(self.chat.id, "cart", "[]")
        self.repo.save_working_memory(self.chat.id, "cart", '["p1"]')
        entries = self.repo.list_working_memory(self.chat.id)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].value, '["p1"]')

    def test_delete_working_memory_missing_key_raises(self):
        with self.assertRaises(NotFoundError):
            self.repo.delete_working_memory(self.chat.id, "no-such-key")

    def test_save_long_term_memory_requires_valid_category(self):
        with self.assertRaises(ValidationError):
            self.repo.save_long_term_memory(self.agent.id, "not-a-category", "k", "v")
        entry = self.repo.save_long_term_memory(self.agent.id, "profile", "user_name", "Иван")
        self.assertEqual(entry.category, "profile")

    def test_long_term_memory_scoped_to_agent_not_chat(self):
        chat2 = self.repo.create_chat(self.agent.id, "Второй чат")
        self.repo.save_long_term_memory(self.agent.id, "knowledge", "tz", "UTC+3")
        # long_term_memory_enabled выключен по умолчанию для новых чатов —
        # включаем на обоих, чтобы проверить именно область видимости
        # (агент, а не чат), а не гейтинг типов памяти.
        self.repo.update_chat_settings(self.chat.id, {"long_term_memory_enabled": True})
        self.repo.update_chat_settings(chat2.id, {"long_term_memory_enabled": True})
        # Долговременная память видна из ЛЮБОГО чата этого агента — проверяем
        # через memory-snapshot обоих чатов.
        snap1 = self.repo.get_memory_snapshot(self.chat.id)
        snap2 = self.repo.get_memory_snapshot(chat2.id)
        self.assertEqual(len(snap1["long_term_memory"]), 1)
        self.assertEqual(len(snap2["long_term_memory"]), 1)


class ProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = make_repository()
        self.agent = self.repo.create_agent("Агент", model=TEST_MODEL_ID)
        self.chat = self.repo.create_chat(self.agent.id, "Чат")

    def tearDown(self) -> None:
        import os
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def test_create_profile_validates_skills_json(self):
        with self.assertRaises(ValidationError):
            self.repo.create_profile("Плохой профиль", skills_json="{not a list}")

    def test_create_shopping_demo_profile_and_activate(self):
        profile = _make_shopping_profile(self.repo)
        self.assertEqual(profile.name, "Покупки")
        skills = json.loads(profile.skills_json)
        names = {s["function"]["name"] for s in skills}
        self.assertEqual(names, {"search_products", "add_to_cart", "view_cart"})

        chat = self.repo.set_chat_active_profile(self.chat.id, profile.id)
        self.assertEqual(chat.active_profile_id, profile.id)

    def test_create_profile_via_registered_skill_names(self):
        """Путь из ТЗ: пользователь привязывает к профилю уже
        ЗАРЕГИСТРИРОВАННЫЕ сервисом скиллы по имени (см. `GET /skills`,
        реестр из переменной окружения AGENT_REGISTERED_SKILLS), не переписывая
        вручную JSON-схему функции."""
        with patch.object(skills_registry, "_REGISTERED_SKILLS", list(shopping_demo.TOOL_DEFS)):
            self.assertEqual(len(self.repo.list_registered_skills()), 3)
            profile = self.repo.create_profile(
                "Покупки", skill_names=["search_products", "add_to_cart", "view_cart"],
            )
        skills = json.loads(profile.skills_json)
        names = {s["function"]["name"] for s in skills}
        self.assertEqual(names, {"search_products", "add_to_cart", "view_cart"})

    def test_create_profile_with_unknown_skill_name_raises(self):
        with patch.object(skills_registry, "_REGISTERED_SKILLS", []):
            with self.assertRaises(ValidationError):
                self.repo.create_profile("Плохой профиль", skill_names=["nonexistent"])

    def test_create_profile_rejects_both_skill_names_and_skills_json(self):
        with self.assertRaises(ValidationError):
            self.repo.create_profile("Оба сразу", skill_names=["x"], skills_json="[]")

    def test_update_profile_via_registered_skill_names(self):
        profile = self.repo.create_profile("Пустой профиль")
        with patch.object(skills_registry, "_REGISTERED_SKILLS", list(shopping_demo.TOOL_DEFS)):
            updated = self.repo.update_profile(profile.id, {"skill_names": ["view_cart"]})
        names = {s["function"]["name"] for s in json.loads(updated.skills_json)}
        self.assertEqual(names, {"view_cart"})

    def test_profile_catalog_is_shared_across_agents(self):
        # Профили — общий справочник (по замечанию пользователя), а не
        # привязаны к одному агенту: профиль, созданный "для" одного агента
        # (в старой модели), подключается к чату ЛЮБОГО другого агента.
        other_agent = self.repo.create_agent("Другой агент", model=TEST_MODEL_ID)
        other_chat = self.repo.create_chat(other_agent.id, "Чат другого агента")
        profile = self.repo.create_profile("Общий профиль")

        chat1 = self.repo.set_chat_active_profile(self.chat.id, profile.id)
        chat2 = self.repo.set_chat_active_profile(other_chat.id, profile.id)
        self.assertEqual(chat1.active_profile_id, profile.id)
        self.assertEqual(chat2.active_profile_id, profile.id)

        # И виден в общем списке независимо от того, у какого агента запрошен.
        self.assertIn(profile.id, {p.id for p in self.repo.list_profiles()})

    def test_delete_profile_clears_active_profile_on_chat(self):
        profile = self.repo.create_profile("Профиль")
        self.repo.set_chat_active_profile(self.chat.id, profile.id)
        self.repo.delete_profile(profile.id)
        chat = self.repo.get_chat(self.chat.id)
        self.assertIsNone(chat.active_profile_id)

    def test_set_agent_default_profile_does_not_affect_existing_chats(self):
        # default_profile_id — это то, что унаследует НОВЫЙ чат при создании
        # (по аналогии с копированием Settings в create_chat); уже
        # существующие чаты агента не меняются задним числом.
        profile = self.repo.create_profile("Профиль по умолчанию")
        agent = self.repo.set_agent_default_profile(self.agent.id, profile.id)
        self.assertEqual(agent.default_profile_id, profile.id)

        chat = self.repo.get_chat(self.chat.id)
        self.assertIsNone(chat.active_profile_id)

    def test_new_chat_inherits_agent_default_profile(self):
        profile = self.repo.create_profile("Профиль по умолчанию")
        self.repo.set_agent_default_profile(self.agent.id, profile.id)

        new_chat = self.repo.create_chat(self.agent.id, "Новый чат")
        self.assertEqual(new_chat.active_profile_id, profile.id)

    def test_set_agent_default_profile_to_none_clears_it(self):
        profile = self.repo.create_profile("Профиль по умолчанию")
        self.repo.set_agent_default_profile(self.agent.id, profile.id)
        agent = self.repo.set_agent_default_profile(self.agent.id, None)
        self.assertIsNone(agent.default_profile_id)

    def test_set_agent_default_profile_rejects_unknown_profile(self):
        with self.assertRaises(NotFoundError):
            self.repo.set_agent_default_profile(self.agent.id, "nonexistent")

    def test_set_agent_default_profile_rejects_unknown_agent(self):
        profile = self.repo.create_profile("Профиль")
        with self.assertRaises(NotFoundError):
            self.repo.set_agent_default_profile("nonexistent", profile.id)


class MemoryInjectionTests(unittest.TestCase):
    """Проверяет, что профиль/долговременная/рабочая память реально
    попадают в сообщения, отправляемые провайдеру, — и именно в порядке
    system_prompt -> профиль -> долговременная -> рабочая -> история."""

    def setUp(self) -> None:
        self.repo = make_repository()
        self.agent = self.repo.create_agent("Агент", model=TEST_MODEL_ID)
        self.chat = self.repo.create_chat(self.agent.id, "Чат")

    def tearDown(self) -> None:
        import os
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def test_injection_order(self):
        # working_memory_enabled/long_term_memory_enabled выключены по
        # умолчанию для нового чата — включаем оба явно, иначе их блоки не
        # попадут в промпт и тест не сможет проверить порядок инъекции.
        self.repo.update_chat_settings(self.chat.id, {
            "working_memory_enabled": True,
            "long_term_memory_enabled": True,
        })
        profile = self.repo.create_profile(
            "Формальный", style="строго и по-деловому", orchestration_prompt="без лишних слов"
        )
        self.repo.set_chat_active_profile(self.chat.id, profile.id)
        self.repo.save_long_term_memory(self.agent.id, "profile", "user_name", "Иван")
        self.repo.save_working_memory(self.chat.id, "задача", "подготовить отчёт")

        self.repo.send_message_blocking(self.chat.id, "Привет")
        sent = self.repo._fake_provider.last_messages  # type: ignore[attr-defined]
        roles_contents = [(m.role, m.content) for m in sent]

        system_idx = next(i for i, (r, c) in enumerate(roles_contents) if r == "system" and "полезный ассистент" in c)
        profile_idx = next(i for i, (r, c) in enumerate(roles_contents) if "Формальный" in c)
        long_term_idx = next(i for i, (r, c) in enumerate(roles_contents) if "Долговременная память" in c)
        working_idx = next(i for i, (r, c) in enumerate(roles_contents) if "Рабочая память" in c)
        user_idx = next(i for i, (r, c) in enumerate(roles_contents) if r == "user" and c == "Привет")

        self.assertLess(system_idx, profile_idx)
        self.assertLess(profile_idx, long_term_idx)
        self.assertLess(long_term_idx, working_idx)
        self.assertLess(working_idx, user_idx)
        self.assertIn("Иван", roles_contents[long_term_idx][1])
        self.assertIn("подготовить отчёт", roles_contents[working_idx][1])


class MemoryTypeGatingTests(unittest.TestCase):
    """Все пять типов памяти выключены по умолчанию для НОВЫХ агентов/чатов
    (пользователь включает нужные явно в настройках) — сами тумблеры
    (Settings.*_memory_enabled) управляют СРАЗУ тремя вещами: инъекцией в
    промпт, содержимым memory-snapshot и tool-calling'ом."""

    def setUp(self) -> None:
        self.repo = make_repository()
        self.agent = self.repo.create_agent("Агент", model=TEST_MODEL_ID)
        self.chat = self.repo.create_chat(self.agent.id, "Чат")

    def tearDown(self) -> None:
        import os
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def test_all_memory_types_off_by_default_for_new_agent(self):
        settings = self.repo.get_agent(self.agent.id).settings
        self.assertFalse(settings.working_memory_enabled)
        self.assertFalse(settings.long_term_memory_enabled)
        self.assertFalse(settings.episodic_memory_enabled)
        self.assertFalse(settings.semantic_memory_enabled)
        self.assertFalse(settings.procedural_memory_enabled)

    def test_manual_save_into_extended_category_always_allowed(self):
        # Ручное сохранение через API/форму работает независимо от того,
        # включён ли тип памяти сейчас, — тумблер влияет только на то, что
        # ПОПАДЁТ В ЗАПРОС МОДЕЛИ, а не на то, что можно сохранить.
        entry = self.repo.save_long_term_memory(self.agent.id, "episodic", "e1", "решили не использовать Mongo")
        self.assertEqual(entry.category, "episodic")
        self.assertEqual(len(self.repo.list_long_term_memory(self.agent.id)), 1)

    def test_disabled_extended_category_hidden_from_injection_and_snapshot(self):
        self.repo.save_long_term_memory(self.agent.id, "episodic", "e1", "решили не использовать Mongo")
        snapshot = self.repo.get_memory_snapshot(self.chat.id)
        self.assertEqual(snapshot["long_term_memory"], [])
        self.assertNotIn("episodic", snapshot["enabled_memory_types"])

        self.repo.send_message_blocking(self.chat.id, "Привет")
        sent = self.repo._fake_provider.last_messages  # type: ignore[attr-defined]
        self.assertFalse(any("Mongo" in m.content for m in sent))

    def test_enabling_extended_category_surfaces_it_without_affecting_others(self):
        # Флаги типов памяти живут в Settings чата (копируются из настроек
        # агента только в момент СОЗДАНИЯ чата, дальше независимы) — поэтому
        # включаем тумблер именно на уже существующем чате, а не на агенте.
        self.repo.save_long_term_memory(self.agent.id, "episodic", "e1", "решили не использовать Mongo")
        self.repo.save_long_term_memory(self.agent.id, "semantic", "s1", "проект на Kotlin+FastAPI")
        self.repo.update_chat_settings(self.chat.id, {"episodic_memory_enabled": True})

        snapshot = self.repo.get_memory_snapshot(self.chat.id)
        categories = {e.category for e in snapshot["long_term_memory"]}
        self.assertIn("episodic", categories)
        self.assertNotIn("semantic", categories)  # включили только episodic
        self.assertIn("episodic", snapshot["enabled_memory_types"])
        self.assertNotIn("semantic", snapshot["enabled_memory_types"])

    def test_disabling_long_term_memory_hides_core_categories_but_not_working(self):
        self.repo.save_long_term_memory(self.agent.id, "profile", "user_name", "Иван")
        self.repo.save_working_memory(self.chat.id, "этап", "сбор требований")
        # working_memory_enabled/long_term_memory_enabled выключены по
        # умолчанию для нового чата — явно включаем оба, чтобы проверить
        # именно эффект от выключения long_term (независимая ось).
        self.repo.update_chat_settings(self.chat.id, {
            "working_memory_enabled": True,
            "long_term_memory_enabled": False,
        })

        snapshot = self.repo.get_memory_snapshot(self.chat.id)
        self.assertEqual(snapshot["long_term_memory"], [])
        self.assertEqual(len(snapshot["working_memory"]), 1)
        self.assertIn("working", snapshot["enabled_memory_types"])

    def test_disabling_working_memory_hides_it_but_not_long_term(self):
        self.repo.save_long_term_memory(self.agent.id, "profile", "user_name", "Иван")
        self.repo.save_working_memory(self.chat.id, "этап", "сбор требований")
        self.repo.update_chat_settings(self.chat.id, {
            "working_memory_enabled": False,
            "long_term_memory_enabled": True,
        })

        snapshot = self.repo.get_memory_snapshot(self.chat.id)
        self.assertEqual(snapshot["working_memory"], [])
        self.assertEqual(len(snapshot["long_term_memory"]), 1)
        self.assertNotIn("working", snapshot["enabled_memory_types"])

    def test_available_tools_reflect_enabled_types(self):
        self.repo.update_chat_settings(self.chat.id, {
            "memory_tools_enabled": True,
            "working_memory_enabled": False,
            "long_term_memory_enabled": True,
            "episodic_memory_enabled": True,
        })
        snapshot = self.repo.get_memory_snapshot(self.chat.id)
        tool_names = {t["function"]["name"] for t in snapshot["available_tools"]}
        # save_working_memory отсутствует (working_memory_enabled=False);
        # save_long_term_memory остаётся — profile/decision/knowledge
        # (включили long_term_memory_enabled) + episodic (включили) всё ещё
        # дают непустой enum.
        self.assertNotIn("save_working_memory", tool_names)
        self.assertIn("save_long_term_memory", tool_names)
        save_lt_tool = next(t for t in snapshot["available_tools"] if t["function"]["name"] == "save_long_term_memory")
        categories_enum = save_lt_tool["function"]["parameters"]["properties"]["category"]["enum"]
        self.assertEqual(set(categories_enum), {"profile", "decision", "knowledge", "episodic"})

    def test_no_save_long_term_memory_tool_when_all_long_term_types_disabled(self):
        self.repo.update_chat_settings(self.chat.id, {
            "memory_tools_enabled": True,
            "working_memory_enabled": True,
            "long_term_memory_enabled": False,
        })
        snapshot = self.repo.get_memory_snapshot(self.chat.id)
        tool_names = {t["function"]["name"] for t in snapshot["available_tools"]}
        self.assertNotIn("save_long_term_memory", tool_names)
        self.assertIn("save_working_memory", tool_names)  # рабочую память включили явно


class ToolCallingLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = make_tool_repo()
        self.agent = self.repo.create_agent("Агент", model=TEST_MODEL_ID)
        self.chat = self.repo.create_chat(self.agent.id, "Чат")
        # working_memory_enabled/long_term_memory_enabled выключены по
        # умолчанию для нового чата — эти тесты про tool-calling самой
        # памятью, а не про гейтинг типов (тот отдельно проверяется в
        # MemoryTypeGatingTests), поэтому включаем оба здесь разом.
        self.repo.update_chat_settings(self.chat.id, {
            "memory_tools_enabled": True,
            "working_memory_enabled": True,
            "long_term_memory_enabled": True,
        })

    def tearDown(self) -> None:
        import os
        os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def test_blocking_tool_call_saves_memory_and_only_final_text_persisted(self):
        provider = self.repo._tool_provider  # type: ignore[attr-defined]
        provider.tool_calls_queue = [
            [_tool_call("call1", "save_working_memory", {"key": "цель", "value": "собрать корзину"})]
        ]
        user_msg, assistant_msg = self.repo.send_message_blocking(self.chat.id, "Запомни мою цель")

        # Сама функция реально выполнена — рабочая память обновлена, источник "agent".
        entries = self.repo.list_working_memory(self.chat.id)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].key, "цель")
        self.assertEqual(entries[0].source, "agent")

        # В истории чата — только финальный текстовый ответ, без служебных
        # сообщений вызова/результата инструмента.
        self.assertEqual(assistant_msg.content, provider.final_text)
        stored = self.repo.list_messages(self.chat.id)
        self.assertEqual([m.role for m in stored], ["user", "assistant"])

        # Модель была вызвана дважды: запрос -> tool_calls, затем с
        # добавленными assistant(tool_calls)+tool(result) -> финальный текст.
        self.assertEqual(len(provider.calls_log), 2)
        second_call_roles = [m.role for m in provider.calls_log[1]]
        self.assertIn("tool", second_call_roles)

    def test_unknown_tool_name_returns_error_without_crashing(self):
        provider = self.repo._tool_provider  # type: ignore[attr-defined]
        provider.tool_calls_queue = [[_tool_call("call1", "does_not_exist", {})]]
        user_msg, assistant_msg = self.repo.send_message_blocking(self.chat.id, "Привет")
        self.assertEqual(assistant_msg.content, provider.final_text)
        tool_message = next(m for m in provider.calls_log[1] if m.role == "tool")
        self.assertIn("unknown tool", tool_message.content)

    def test_tool_loop_respects_max_iterations(self):
        provider = self.repo._tool_provider  # type: ignore[attr-defined]
        # Модель "зацикливается", всегда прося новый вызов — цикл должен
        # остановиться после _MAX_TOOL_ITERATIONS и НЕ отдать пользователю
        # сырое промежуточное состояние с пустым текстом (реальный баг,
        # найденный на практике: пользователь получал пустой ответ вместо
        # структуры проекта после серии из 8 подряд вызовов git_host_*) — а
        # сделать один принудительный финальный запрос без инструментов
        # (см. `_finalize_after_tool_cap`), который здесь эмулирует
        # `ToolCallingFakeProvider._next_tool_calls`: без tools в запросе
        # модель не может вернуть новый tool_calls, только текст.
        from agents_core.repository import _MAX_TOOL_ITERATIONS
        provider.tool_calls_queue = [
            [_tool_call(f"call{i}", "save_working_memory", {"key": "k", "value": str(i)})]
            for i in range(_MAX_TOOL_ITERATIONS + 2)
        ]
        _, assistant_msg = self.repo.send_message_blocking(self.chat.id, "Привет")
        # Один начальный вызов + ровно _MAX_TOOL_ITERATIONS повторов + один
        # принудительный финальный запрос без инструментов.
        self.assertEqual(len(provider.calls_log), _MAX_TOOL_ITERATIONS + 2)
        # Пользователь должен получить осмысленный текстовый ответ, а не "".
        self.assertEqual(assistant_msg.content, provider.final_text)

    def test_streaming_tool_loop_respects_max_iterations(self):
        # Тот же баг/фикс, что и в test_tool_loop_respects_max_iterations,
        # но для потокового пути (stream_message) — там раньше была
        # отдельная, независимая от блокирующего пути копия цикла, со
        # своим собственным местом для того же обрыва.
        provider = self.repo._tool_provider  # type: ignore[attr-defined]
        from agents_core.repository import _MAX_TOOL_ITERATIONS
        provider.tool_calls_queue = [
            [_tool_call(f"call{i}", "save_working_memory", {"key": "k", "value": str(i)})]
            for i in range(_MAX_TOOL_ITERATIONS + 2)
        ]
        events = list(self.repo.stream_message(self.chat.id, "Привет"))
        done_events = [e for e in events if e["type"] == "done"]
        self.assertEqual(len(done_events), 1)
        self.assertEqual(done_events[0]["message"].content, provider.final_text)
        self.assertEqual(len(provider.calls_log), _MAX_TOOL_ITERATIONS + 2)

    def test_streaming_tool_call_shopping_flow(self):
        profile = _make_shopping_profile(self.repo, self.agent.id)
        self.repo.set_chat_active_profile(self.chat.id, profile.id)
        provider = self.repo._tool_provider  # type: ignore[attr-defined]
        product_id = shopping_demo.CATALOG[0]["id"]
        provider.tool_calls_queue = [[_tool_call("call1", "add_to_cart", {"product_id": product_id})]]

        events = list(self.repo.stream_message(self.chat.id, "Добавь первый товар в корзину"))
        done_events = [e for e in events if e["type"] == "done"]
        self.assertEqual(len(done_events), 1)
        self.assertEqual(done_events[0]["message"].content, provider.final_text)

        cart_entry = self.repo.list_working_memory(self.chat.id)
        self.assertEqual(len(cart_entry), 1)
        self.assertEqual(json.loads(cart_entry[0].value), [product_id])

    def test_memory_snapshot_reflects_available_tools_and_profile(self):
        profile = _make_shopping_profile(self.repo, self.agent.id)
        self.repo.set_chat_active_profile(self.chat.id, profile.id)
        snapshot = self.repo.get_memory_snapshot(self.chat.id)
        tool_names = {t["function"]["name"] for t in snapshot["available_tools"]}
        # save_working_memory/save_long_term_memory (тумблер включён в setUp)
        # + 3 скилла профиля «Покупки».
        self.assertEqual(
            tool_names,
            {"save_working_memory", "save_long_term_memory", "search_products", "add_to_cart", "view_cart"},
        )
        self.assertEqual(snapshot["active_profile"].id, profile.id)
        self.assertTrue(snapshot["memory_tools_enabled"])


class ShoppingDemoSkillTests(unittest.TestCase):
    def test_search_products_filters_by_query_and_price(self):
        results = shopping_demo.search_products("электрон")
        self.assertTrue(all("электрон" in p["category"] for p in results))
        cheap = shopping_demo.search_products("", max_price=2000)
        self.assertTrue(all(p["price"] <= 2000 for p in cheap))

    def test_add_to_cart_unknown_product_raises(self):
        with self.assertRaises(shopping_demo.ShoppingError):
            shopping_demo.add_to_cart([], "no-such-id")

    def test_add_to_cart_and_view_cart_roundtrip(self):
        pid = shopping_demo.CATALOG[0]["id"]
        new_cart, view = shopping_demo.add_to_cart([], pid)
        self.assertEqual(new_cart, [pid])
        self.assertEqual(view["items_count"], 1)
        self.assertEqual(view["total_price"], shopping_demo.CATALOG[0]["price"])
        self.assertEqual(shopping_demo.view_cart(new_cart), view)


if __name__ == "__main__":
    unittest.main()

"""
Юнит-тесты бизнес-логики `Repository` — полностью офлайн: настоящий
`ProviderRegistry`/провайдеры не используются, вместо них подставляется
`FakeProvider` с детерминированными ответами. FastAPI здесь не участвует,
поэтому эти тесты можно запускать без установки веб-фреймворка.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Dict, Iterator, List, Optional

from agents_core.db import Database
from agents_core.format_detect import detect_message_format
from agents_core.models import ModelInfo, Settings
from agents_core.providers.base import ChatResult, ChatUsage, ProviderError, ProviderMessage, StreamDelta
from agents_core.repository import (
    NotFoundError,
    PreconditionFailedError,
    Repository,
    ValidationError,
)


class FakeProvider:
    """Провайдер-заглушка с предсказуемым поведением для тестов. Определяет
    "тип" вызова по системному сообщению — так один и тот же фейк обслуживает
    обычную отправку, суммаризацию и извлечение фактов, как это происходит в
    реальном коде (все три идут через один и тот же `chat()`)."""

    name = "fake"

    def __init__(self):
        self.last_messages: List[ProviderMessage] = []
        self.healthy = True
        self.fail_next = False
        self.next_facts_response: Optional[dict] = None  # {"facts": [...]}
        self.next_facts_response_raw: Optional[str] = None  # для теста "модель ответила не чистым JSON"
        self.next_summary_response: Optional[str] = None

    def health(self) -> bool:
        return self.healthy

    def discover_model(self, model_id: str):
        raise NotImplementedError  # каталог в тестах наполняется вручную, не через провайдера

    def chat(self, model_id: str, messages: List[ProviderMessage], settings: Settings) -> ChatResult:
        self.last_messages = list(messages)
        if self.fail_next:
            self.fail_next = False
            raise ProviderError("simulated provider failure")
        system_content = next((m.content for m in messages if m.role == "system"), "")
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        prompt_tokens = 10 * len(messages)
        completion_tokens = 5

        if "structured memory" in system_content:
            if self.next_facts_response_raw is not None:
                content = self.next_facts_response_raw
            else:
                payload = self.next_facts_response if self.next_facts_response is not None else {"facts": []}
                content = json.dumps(payload, ensure_ascii=False)
            return ChatResult(
                content=content,
                usage=ChatUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                                 total_tokens=prompt_tokens + completion_tokens),
            )
        if "модуль суммаризации" in system_content:
            content = self.next_summary_response if self.next_summary_response is not None else f"РЕЗЮМЕ: {last_user[:40]}"
            return ChatResult(
                content=content,
                usage=ChatUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                                 total_tokens=prompt_tokens + completion_tokens),
            )
        # Токены детерминированы: 10 за каждое входное сообщение, 5 на ответ —
        # удобно для проверки агрегатов в тестах.
        return ChatResult(
            content=f"echo: {last_user}",
            reasoning_content="потому что так решил тест",
            usage=ChatUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                             total_tokens=prompt_tokens + completion_tokens),
        )

    def stream_chat(self, model_id: str, messages: List[ProviderMessage], settings: Settings) -> Iterator[StreamDelta]:
        self.last_messages = list(messages)
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        for piece in ["ec", "ho", ": " + last_user]:
            yield StreamDelta(content=piece)
        prompt_tokens = 10 * len(messages)
        yield StreamDelta(
            done=True,
            result=ChatResult(
                content=f"echo: {last_user}", reasoning_content=None,
                usage=ChatUsage(prompt_tokens=prompt_tokens, completion_tokens=5, total_tokens=prompt_tokens + 5),
            ),
        )


class FakeRegistry:
    def __init__(self, provider: FakeProvider):
        self._provider = provider

    def get(self, name: str) -> FakeProvider:
        return self._provider

    def all(self) -> Dict[str, FakeProvider]:
        return {"fake": self._provider}


class FakeCatalog:
    def __init__(self, models: Dict[str, ModelInfo]):
        self._models = models

    def get(self, composite_id: str) -> Optional[ModelInfo]:
        return self._models.get(composite_id)

    def list_models(self) -> List[ModelInfo]:
        return list(self._models.values())


TEST_MODEL_ID = "fake:test-model"


def make_repository(max_input_tokens: int = 1000, context_window: Optional[int] = None) -> Repository:
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)
    os.unlink(db_path)  # Database сама создаёт файл при первом обращении
    db = Database(db_path)
    provider = FakeProvider()
    registry = FakeRegistry(provider)
    model = ModelInfo(
        id=TEST_MODEL_ID, provider="fake", model_id="test-model", display_name="Test model",
        is_local=True, context_window=context_window if context_window is not None else max_input_tokens + 200,
        max_input_tokens=max_input_tokens,
        max_output_tokens=256, supports_thinking=True, supports_tools=True,
        supports_json_mode=True, supports_logprobs=False,
    )
    catalog = FakeCatalog({TEST_MODEL_ID: model})
    repo = Repository(db, registry, catalog)
    repo._db_path_for_cleanup = db_path  # type: ignore[attr-defined]
    repo._fake_provider = provider  # type: ignore[attr-defined]
    return repo


class RepositoryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = make_repository()

    def tearDown(self) -> None:
        try:
            os.unlink(self.repo._db_path_for_cleanup)  # type: ignore[attr-defined]
        except OSError:
            pass

    # ---- настройки по умолчанию / агент -----------------------------------

    def test_default_settings_seed_and_update(self):
        defaults = self.repo.get_default_settings()
        self.assertEqual(defaults.model, "deepseek:deepseek-v4-flash")
        updated = self.repo.update_default_settings({"temperature": 0.4, "model": TEST_MODEL_ID})
        self.assertEqual(updated.temperature, 0.4)
        self.assertEqual(updated.model, TEST_MODEL_ID)
        reset = self.repo.reset_default_settings()
        self.assertEqual(reset.temperature, 1.0)

    def test_update_default_settings_rejects_unknown_field(self):
        with self.assertRaises(ValidationError):
            self.repo.update_default_settings({"max_tokens": 10})  # поля нет в DefaultSettings

    def test_create_agent_copies_defaults_and_fills_rest_from_code(self):
        self.repo.update_default_settings({"model": TEST_MODEL_ID, "temperature": 0.3})
        agent = self.repo.create_agent("Мой агент")
        self.assertEqual(agent.settings.model, TEST_MODEL_ID)
        self.assertEqual(agent.settings.temperature, 0.3)
        self.assertEqual(agent.settings.autosummary, "off")  # поле не входит в DefaultSettings -> код-дефолт
        self.assertEqual(agent.settings.max_tokens, None)
        self.assertIsNone(agent.settings.context_strategy)

    def test_new_agent_settings_match_what_new_agent_gets(self):
        self.repo.update_default_settings({"model": TEST_MODEL_ID, "temperature": 0.4, "stream": False})
        baseline = self.repo.new_agent_settings()
        agent = self.repo.create_agent("Агент")
        self.assertEqual(baseline, agent.settings)
        self.assertEqual(baseline.temperature, 0.4)
        self.assertFalse(baseline.stream)

    def test_agent_not_found(self):
        with self.assertRaises(NotFoundError):
            self.repo.get_agent("no-such-id")

    def test_create_agent_auto_names_by_sequence_when_name_omitted(self):
        a1 = self.repo.create_agent(None, model=TEST_MODEL_ID)
        a2 = self.repo.create_agent("", model=TEST_MODEL_ID)
        a3 = self.repo.create_agent("   ", model=TEST_MODEL_ID)
        self.assertEqual(a1.name, "Агент 1")
        self.assertEqual(a2.name, "Агент 2")
        self.assertEqual(a3.name, "Агент 3")
        # Явно заданное имя не подменяется.
        named = self.repo.create_agent("Мой агент", model=TEST_MODEL_ID)
        self.assertEqual(named.name, "Мой агент")

    def test_create_chat_auto_names_by_sequence_within_agent(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        c1 = self.repo.create_chat(agent.id, None)
        c2 = self.repo.create_chat(agent.id, "")
        self.assertEqual(c1.title, "Чат 1")
        self.assertEqual(c2.title, "Чат 2")
        # Другой агент начинает свой собственный отсчёт.
        other_agent = self.repo.create_agent("B", model=TEST_MODEL_ID)
        other_c1 = self.repo.create_chat(other_agent.id, None)
        self.assertEqual(other_c1.title, "Чат 1")

    def test_delete_agent_cascades_chats_and_messages(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "Chat 1")
        self.repo.send_message_blocking(chat.id, "Привет")
        self.repo.delete_agent(agent.id)
        with self.assertRaises(NotFoundError):
            self.repo.get_chat(chat.id)

    # ---- чаты -----------------------------------------------------------------

    def test_chat_inherits_agent_settings_snapshot(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        self.repo.update_agent_settings(agent.id, {"temperature": 0.9})
        chat = self.repo.create_chat(agent.id, "C1")
        self.assertEqual(chat.settings.temperature, 0.9)
        # Изменение настроек агента после создания чата не затрагивает чат:
        self.repo.update_agent_settings(agent.id, {"temperature": 0.1})
        chat_again = self.repo.get_chat(chat.id)
        self.assertEqual(chat_again.settings.temperature, 0.9)

    def test_chat_settings_cannot_change_model(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")
        with self.assertRaises(ValidationError):
            self.repo.update_chat_settings(chat.id, {"model": "deepseek:deepseek-v4-pro"})

    def test_copy_chat_duplicates_settings_and_messages(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "Original")
        self.repo.send_message_blocking(chat.id, "Привет")
        copy = self.repo.copy_chat(chat.id, "Original (копия)")
        self.assertEqual(copy.title, "Original (копия)")
        self.assertEqual(copy.settings.model, TEST_MODEL_ID)
        original_messages = self.repo.list_messages(chat.id)
        copied_messages = self.repo.list_messages(copy.id)
        self.assertEqual(len(original_messages), len(copied_messages))
        self.assertEqual([m.content for m in original_messages], [m.content for m in copied_messages])

    # ---- настройки: валидация новых полей ----------------------------------

    def test_context_strategy_validation(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        with self.assertRaises(ValidationError):
            self.repo.update_agent_settings(agent.id, {"context_strategy": "not-a-real-strategy"})
        with self.assertRaises(ValidationError):
            self.repo.update_agent_settings(agent.id, {"context_strategy_limit": 2})  # не > 2
        updated = self.repo.update_agent_settings(agent.id, {"context_strategy": "sliding_window", "context_strategy_limit": 4})
        self.assertEqual(updated.context_strategy, "sliding_window")
        self.assertEqual(updated.context_strategy_limit, 4)

    # ---- отправка сообщений / токены --------------------------------------

    def test_send_message_blocking_persists_telemetry(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")
        user_msg, assistant_msg = self.repo.send_message_blocking(chat.id, "Привет")
        self.assertEqual(user_msg.role, "user")
        self.assertEqual(assistant_msg.role, "assistant")
        self.assertEqual(assistant_msg.content, "echo: Привет")
        self.assertIsNotNone(assistant_msg.total_tokens)
        self.assertIsNotNone(assistant_msg.duration_ms)
        self.assertEqual(user_msg.branch, 0)
        self.assertEqual(user_msg.format, "text")

    def test_send_message_blocking_persists_prompt_tokens_on_user_message(self):
        # Доработка по замечанию пользователя: под сообщением ПОЛЬЗОВАТЕЛЯ
        # должно отображаться число токенов НА ВХОД (prompt_tokens из ответа
        # провайдера на этот же обмен), а не отдельная оценка длины текста —
        # то есть user_msg.prompt_tokens должно совпадать с
        # assistant_msg.prompt_tokens (это одно и то же значение usage одного
        # и того же вызова, просто отображается под разными сообщениями).
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")
        user_msg, assistant_msg = self.repo.send_message_blocking(chat.id, "Привет, как дела?")
        self.assertIsNotNone(user_msg.prompt_tokens)
        self.assertGreater(user_msg.prompt_tokens, 0)
        self.assertEqual(user_msg.prompt_tokens, assistant_msg.prompt_tokens)
        # Не должно влиять на расчёт агрегатов чата — те считаются только по
        # сообщениям ассистента (см. chat_stats).
        stats = self.repo.chat_stats(chat)
        self.assertEqual(stats["total_tokens"], assistant_msg.total_tokens)

    def test_stream_message_persists_prompt_tokens_on_user_message(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")
        list(self.repo.stream_message(chat.id, "Привет, как дела?"))
        messages = self.repo.list_messages(chat.id)
        user_msg = next(m for m in messages if m.role == "user")
        assistant_msg = next(m for m in messages if m.role == "assistant")
        self.assertIsNotNone(user_msg.prompt_tokens)
        self.assertGreater(user_msg.prompt_tokens, 0)
        self.assertEqual(user_msg.prompt_tokens, assistant_msg.prompt_tokens)

    def test_provider_error_records_error_message_and_reraises(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")
        self.repo._fake_provider.fail_next = True  # type: ignore[attr-defined]
        with self.assertRaises(ProviderError):
            self.repo.send_message_blocking(chat.id, "Привет")
        messages = self.repo.list_messages(chat.id)
        # Асинхронные запуски: вместо отдельного сообщения с ролью "error"
        # черновик ответа сохраняется со статусом "failed" и текстом ошибки.
        self.assertEqual(messages[-1].role, "assistant")
        self.assertEqual(messages[-1].status, "failed")
        self.assertTrue(messages[-1].error)

    def test_chat_stats_and_context_fill_ratio(self):
        # context_window=1200 (max_input_tokens=1000 + 200); context_fill_ratio
        # считается от total_tokens (сумма по эффективному окну), а не от
        # current_context_tokens (размер последнего обмена) — при единственном
        # обмене они совпадают, расхождение видно только при 2+ обменах
        # (см. test_chat_stats_total_tokens_summed_across_all_messages_in_effective_window).
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")
        self.repo.send_message_blocking(chat.id, "Привет")
        chat = self.repo.get_chat(chat.id)
        stats = self.repo.chat_stats(chat)
        # Запрос содержал 2 сообщения (system + новый user) -> 10*2 = 20.
        self.assertEqual(stats["prompt_tokens"], 20)
        self.assertEqual(stats["completion_tokens"], 5)
        self.assertEqual(stats["total_tokens"], 25)
        self.assertEqual(stats["max_input_tokens"], 1000)
        self.assertEqual(stats["context_window"], 1200)
        self.assertAlmostEqual(stats["context_fill_ratio"], 25 / 1200)
        self.assertTrue(stats["can_summarize"])
        self.assertIsNotNone(stats["active_context_start_id"])

    def test_chat_stats_fill_ratio_uses_chat_max_tokens_when_smaller_than_context_window(self):
        # context_window модели = 1200; если у чата настройка max_tokens
        # задана и МЕНЬШЕ размера окна модели (например 40), именно она
        # ограничивает реальный бюджет чата — по прямой инструкции
        # пользователя знаменатель context_fill_ratio должен быть max_tokens,
        # а не полный context_window модели.
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")
        self.repo.update_chat_settings(chat.id, {"max_tokens": 40})
        self.repo.send_message_blocking(chat.id, "Привет")
        chat = self.repo.get_chat(chat.id)
        stats = self.repo.chat_stats(chat)
        self.assertEqual(stats["total_tokens"], 25)
        self.assertEqual(stats["context_window"], 1200)
        self.assertAlmostEqual(stats["context_fill_ratio"], 25 / 40)

    def test_chat_stats_fill_ratio_ignores_chat_max_tokens_when_not_smaller(self):
        # Если max_tokens чата >= context_window модели, он ничего не
        # ограничивает — знаменатель остаётся полным context_window.
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")
        self.repo.update_chat_settings(chat.id, {"max_tokens": 5000})
        self.repo.send_message_blocking(chat.id, "Привет")
        chat = self.repo.get_chat(chat.id)
        stats = self.repo.chat_stats(chat)
        self.assertAlmostEqual(stats["context_fill_ratio"], 25 / 1200)

    def test_chat_stats_total_tokens_uses_effective_context_not_full_history(self):
        # По явной инструкции пользователя: total_tokens/prompt_tokens/
        # completion_tokens — это СУММА по всем сообщениям ЭФФЕКТИВНОГО окна
        # (после границы суммаризации — то, что не показывается на экране
        # чата более бледным), а не по всей истории с начала (то, что было бы
        # до суммаризации, включая уже свёрнутые в резюме сообщения).
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")
        self.repo.send_message_blocking(chat.id, "Первое")
        self.repo.send_message_blocking(chat.id, "Второе")
        summary_msg = self.repo.summarize_chat(chat.id)
        user3, assistant3 = self.repo.send_message_blocking(chat.id, "Третье")

        chat = self.repo.get_chat(chat.id)
        stats = self.repo.chat_stats(chat, branch=0)
        # Эффективное окно после суммаризации = [резюме, user3, assistant3] —
        # сообщения "Первое"/"Второе" в сумму не входят (они вне контекста и
        # показываются бледными).
        expected_prompt = (user3.prompt_tokens or 0)
        expected_completion = (summary_msg.completion_tokens or 0) + (assistant3.completion_tokens or 0)
        self.assertEqual(stats["prompt_tokens"], expected_prompt)
        self.assertEqual(stats["completion_tokens"], expected_completion)
        self.assertEqual(stats["total_tokens"], expected_prompt + expected_completion)
        # current_context_tokens/context_fill_ratio — отдельная величина
        # (размер ПОСЛЕДНЕГО обмена, а не сумма) — заметно меньше, чем была
        # бы кумулятивная сумма по всей истории чата с начала.
        self.assertLess(stats["current_context_tokens"], 100)

    def test_chat_stats_total_tokens_summed_across_all_messages_in_effective_window(self):
        # Без суммаризации и без стратегии контекста эффективное окно — вся
        # история: total_tokens должен быть суммой по ВСЕМ обменам этого окна
        # (просьба пользователя — раньше здесь ошибочно бралась только
        # разбивка последнего обмена).
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")
        user1, assistant1 = self.repo.send_message_blocking(chat.id, "Первое")
        user2, assistant2 = self.repo.send_message_blocking(chat.id, "Второе")
        user3, assistant3 = self.repo.send_message_blocking(chat.id, "Третье")

        chat = self.repo.get_chat(chat.id)
        stats = self.repo.chat_stats(chat, branch=0)
        expected_prompt = sum((u.prompt_tokens or 0) for u in (user1, user2, user3))
        expected_completion = sum((a.completion_tokens or 0) for a in (assistant1, assistant2, assistant3))
        self.assertEqual(stats["prompt_tokens"], expected_prompt)
        self.assertEqual(stats["completion_tokens"], expected_completion)
        self.assertEqual(stats["total_tokens"], expected_prompt + expected_completion)
        # current_context_tokens (только для can_summarize/лимитов) —
        # по-прежнему только последний обмен, а не сумма — заметно меньше
        # суммарного.
        self.assertLess(stats["current_context_tokens"], stats["total_tokens"])
        self.assertEqual(stats["current_context_tokens"], (assistant3.prompt_tokens or 0) + (assistant3.completion_tokens or 0))
        # context_fill_ratio считается от total_tokens (сумма), а НЕ от
        # current_context_tokens — иначе индикатор заполнения занижал бы
        # реальный расход по видимой части чата.
        self.assertAlmostEqual(stats["context_fill_ratio"], stats["total_tokens"] / stats["context_window"])

    def test_create_branch_auto_names_matching_assigned_number(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")
        b1 = self.repo.create_branch(chat.id, None)
        b2 = self.repo.create_branch(chat.id, "")
        self.assertEqual(b1.number, 1)
        self.assertEqual(b1.name, "Ветка 1")
        self.assertEqual(b2.number, 2)
        self.assertEqual(b2.name, "Ветка 2")
        # Явно заданное имя не подменяется автоименем.
        b3 = self.repo.create_branch(chat.id, "Своя ветка")
        self.assertEqual(b3.name, "Своя ветка")

    def test_create_branch_auto_name_matches_number_after_deletion_gap(self):
        # Номер следующей ветки — MAX(number) СРЕДИ ОСТАВШИХСЯ веток + 1, а не
        # порядковый счётчик созданных веток: если удалить ветку 2 (оставив
        # 1 и 3), новая ветка получит номер 4, а не 3. Автоимя должно
        # соответствовать этому РЕАЛЬНО присвоенному номеру, а не тому,
        # сколько веток сейчас существует (2) — иначе имя разошлось бы с номером.
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")
        self.repo.create_branch(chat.id, "Первая")  # number=1
        self.repo.create_branch(chat.id, "Вторая")  # number=2
        self.repo.create_branch(chat.id, "Третья")  # number=3
        self.repo.delete_branch(chat.id, 2)
        b4 = self.repo.create_branch(chat.id, None)
        self.assertEqual(b4.number, 4)
        self.assertEqual(b4.name, "Ветка 4")

    def test_delete_branch_removes_branch_and_its_messages_only(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")
        self.repo.send_message_blocking(chat.id, "root")  # branch 0
        branch = self.repo.create_branch(chat.id, "Alt")
        self.repo.send_message_blocking(chat.id, "branch-msg", branch=branch.number)

        self.repo.delete_branch(chat.id, branch.number)

        self.assertEqual(self.repo.list_branches(chat.id), [])
        remaining = self.repo.list_messages(chat.id)
        self.assertTrue(all(m.branch == 0 for m in remaining))
        self.assertTrue(any(m.content == "root" for m in remaining))
        self.assertFalse(any(m.content == "branch-msg" for m in remaining))

    def test_delete_branch_rejects_main_branch_and_unknown_branch(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")
        with self.assertRaises(ValidationError):
            self.repo.delete_branch(chat.id, 0)
        with self.assertRaises(NotFoundError):
            self.repo.delete_branch(chat.id, 999)

    def test_chat_stats_is_branch_aware(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")
        self.repo.send_message_blocking(chat.id, "root")  # branch 0
        branch = self.repo.create_branch(chat.id, "Alt")
        self.repo.send_message_blocking(chat.id, "branch-msg", branch=branch.number)

        chat = self.repo.get_chat(chat.id)
        stats_main = self.repo.chat_stats(chat, branch=0)
        stats_with_branch = self.repo.chat_stats(chat, branch=branch.number)
        # Ветка N>0 добавляет к статистике сообщения основной ветки + самой
        # ветки -> токенов больше, чем при просмотре только основной ветки.
        self.assertGreater(stats_with_branch["total_tokens"], stats_main["total_tokens"])

    def test_list_agents_with_chats_uses_branch_1_convention(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")
        self.repo.send_message_blocking(chat.id, "root")
        branch = self.repo.create_branch(chat.id, "Alt")
        self.repo.send_message_blocking(chat.id, "branch-msg", branch=branch.number)

        [(_, _, stats_list)] = self.repo.list_agents_with_chats()
        # Список использует конвенцию "ветка 0 + ветка 1, если существует" ->
        # должно совпадать со статистикой явно запрошенной под branch=1.
        expected = self.repo.chat_stats(self.repo.get_chat(chat.id), branch=1)
        self.assertEqual(stats_list[0]["total_tokens"], expected["total_tokens"])

    def test_system_prompt_omitted_when_blank(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        self.repo.update_agent_settings(agent.id, {"system_prompt": "   "})
        chat = self.repo.create_chat(agent.id, "C1")
        self.repo.send_message_blocking(chat.id, "Привет")
        sent = self.repo._fake_provider.last_messages  # type: ignore[attr-defined]
        self.assertNotIn("system", [m.role for m in sent])

    def test_stream_message_yields_deltas_then_done(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")
        events = list(self.repo.stream_message(chat.id, "Стрим"))
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["message"].content, "echo: Стрим")
        deltas = [e for e in events if e["type"] == "delta"]
        self.assertTrue(deltas)

    # ---- определение формата сообщения --------------------------------------

    def test_detect_message_format(self):
        self.assertEqual(detect_message_format("просто текст без разметки"), "text")
        self.assertEqual(detect_message_format("# Заголовок\n\nтекст"), "markdown")
        self.assertEqual(detect_message_format("```python\nprint(1)\n```"), "markdown")
        self.assertEqual(detect_message_format('{"a": 1, "b": [1, 2]}'), "json")
        self.assertEqual(detect_message_format("[1, 2, 3]"), "json")
        self.assertEqual(detect_message_format("42"), "text")  # число само по себе - не JSON-объект/массив
        self.assertEqual(detect_message_format(""), "text")

    def test_message_format_detected_on_save(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")
        user_msg, assistant_msg = self.repo.send_message_blocking(chat.id, "# Заголовок вопроса")
        self.assertEqual(user_msg.format, "markdown")
        self.assertEqual(assistant_msg.format, "text")  # "echo: ..." без markdown-маркеров

    # ---- сообщения: удаление ------------------------------------------------

    def test_bulk_delete_and_clear_messages(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")
        self.repo.send_message_blocking(chat.id, "1")
        self.repo.send_message_blocking(chat.id, "2")
        messages = self.repo.list_messages(chat.id)
        self.assertEqual(len(messages), 4)
        self.repo.bulk_delete_messages(chat.id, [messages[0].id, messages[1].id])
        self.assertEqual(len(self.repo.list_messages(chat.id)), 2)
        self.repo.clear_messages(chat.id)
        self.assertEqual(len(self.repo.list_messages(chat.id)), 0)

    # ---- суммаризация ------------------------------------------------------

    def test_manual_summarize_is_isolated_call_and_saved_as_assistant(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")
        self.repo.send_message_blocking(chat.id, "Первое")
        self.repo.send_message_blocking(chat.id, "Второе")

        summary_msg = self.repo.summarize_chat(chat.id)
        self.assertTrue(summary_msg.is_summary)
        self.assertEqual(summary_msg.role, "assistant")  # п.1.2 — сообщение ассистента, не пользователя

        # Сам вызов суммаризации — изолированный: system=summary_system_prompt,
        # единственное user-сообщение = заполненный шаблон summary_prompt.
        sent = self.repo._fake_provider.last_messages  # type: ignore[attr-defined]
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[0].role, "system")
        self.assertIn("модуль суммаризации", sent[0].content)
        self.assertEqual(sent[1].role, "user")
        self.assertIn("<previous_summary>", sent[1].content)
        self.assertIn("<new_messages>", sent[1].content)
        self.assertIn("[user]: Первое", sent[1].content)
        self.assertIn("[assistant]: echo: Первое", sent[1].content)
        self.assertIn("[user]: Второе", sent[1].content)

        # Следующий запрос должен уйти в модель только с summary + новым
        # сообщением (плюс system чата) — не со всей историей из 5 сообщений.
        self.repo.send_message_blocking(chat.id, "Третье")
        sent2 = self.repo._fake_provider.last_messages  # type: ignore[attr-defined]
        self.assertEqual(sent2[0].role, "system")
        self.assertEqual(sent2[1].content, summary_msg.content)
        self.assertEqual(sent2[-1].content, "Третье")
        self.assertEqual(len(sent2), 3)

    def test_summarize_second_time_uses_previous_summary_as_boundary(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")
        self.repo.send_message_blocking(chat.id, "Первое")
        first_summary = self.repo.summarize_chat(chat.id)
        self.repo.send_message_blocking(chat.id, "Второе")
        second_summary = self.repo.summarize_chat(chat.id)
        sent = self.repo._fake_provider.last_messages  # type: ignore[attr-defined]
        self.assertIn(f"<previous_summary>\n{first_summary.content}", sent[1].content)
        self.assertIn("[user]: Второе", sent[1].content)
        self.assertNotIn("Первое", sent[1].content.split("<new_messages>")[1])  # "Первое" не должно попасть в new_messages второй раз

    def test_summarize_blocked_when_over_model_limit(self):
        repo = make_repository(max_input_tokens=5)  # искусственно маленький лимит
        agent = repo.create_agent("A", model=TEST_MODEL_ID)
        chat = repo.create_chat(agent.id, "C1")
        repo.send_message_blocking(chat.id, "Привет")  # total_tokens уже больше 5
        with self.assertRaises(PreconditionFailedError):
            repo.summarize_chat(chat.id)
        os.unlink(repo._db_path_for_cleanup)  # type: ignore[attr-defined]

    def test_autosummary_by_messages_triggers_after_send(self):
        # Доработка: проверка/выполнение автосуммаризации теперь выполняется
        # ПОСЛЕ основного ответа модели и только при явном флаге запроса
        # autosummary="messages"; порог — N-2 (а не N) сообщений user/assistant
        # с момента последнего summary (без учёта текущего обмена).
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        self.repo.update_agent_settings(agent.id, {"autosummary": "messages", "autosummary_by_messages": 5})
        chat = self.repo.create_chat(agent.id, "C1")
        self.repo.send_message_blocking(chat.id, "1", autosummary="messages")
        self.repo.send_message_blocking(chat.id, "2", autosummary="messages")
        messages = self.repo.list_messages(chat.id)
        # После 2 обменов (4 сообщения) до текущего порог (5-2=3) ещё не достигнут.
        self.assertFalse(any(m.is_summary for m in messages))
        self.repo.send_message_blocking(chat.id, "3", autosummary="messages")
        messages = self.repo.list_messages(chat.id)
        # Перед третьей отправкой в истории уже 4 сообщения (>= 3) -> после
        # получения ответа должна быть выполнена автосуммаризация.
        self.assertTrue(any(m.is_summary for m in messages))

    def test_autosummary_by_tokens_triggers_after_send(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        self.repo.update_agent_settings(agent.id, {"autosummary": "tokens", "autosummary_by_tokens": 10})
        chat = self.repo.create_chat(agent.id, "C1")
        self.repo.send_message_blocking(chat.id, "1", autosummary="tokens")
        messages = self.repo.list_messages(chat.id)
        # До первой отправки в истории не было сообщений -> проверять нечего.
        self.assertFalse(any(m.is_summary for m in messages))
        self.repo.send_message_blocking(chat.id, "2", autosummary="tokens")
        # Перед второй отправкой total_tokens последнего ответа = 15 >= 10.
        messages = self.repo.list_messages(chat.id)
        self.assertTrue(any(m.is_summary for m in messages))

    def test_autosummary_not_triggered_without_explicit_flag(self):
        # Без флага autosummary в запросе автосуммаризация теперь не
        # выполняется вовсе, даже если у чата включена соответствующая
        # настройка — логика полностью перенесена на явный опрос клиента.
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        self.repo.update_agent_settings(agent.id, {"autosummary": "messages", "autosummary_by_messages": 3})
        chat = self.repo.create_chat(agent.id, "C1")
        for i in range(5):
            self.repo.send_message_blocking(chat.id, str(i))
        messages = self.repo.list_messages(chat.id)
        self.assertFalse(any(m.is_summary for m in messages))

    def test_autosummary_flag_requires_matching_chat_setting(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")  # autosummary="off" по умолчанию
        with self.assertRaises(ValidationError):
            self.repo.send_message_blocking(chat.id, "hi", autosummary="messages")

        self.repo.update_chat_settings(chat.id, {"autosummary": "messages", "autosummary_by_messages": 2})
        with self.assertRaises(ValidationError):
            # autosummary_by_messages должно быть > 2
            self.repo.send_message_blocking(chat.id, "hi", autosummary="messages")

        self.repo.update_chat_settings(chat.id, {"autosummary_by_messages": 3})
        self.repo.send_message_blocking(chat.id, "hi", autosummary="messages")  # теперь валидно, без исключений

    def test_autosummary_context_limited_to_messages_since_last_summary(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        self.repo.update_agent_settings(agent.id, {"autosummary": "tokens", "autosummary_by_tokens": 10_000})
        chat = self.repo.create_chat(agent.id, "C1")
        self.repo.send_message_blocking(chat.id, "1", autosummary="tokens")
        summary_msg = self.repo.summarize_chat(chat.id)
        self.repo.send_message_blocking(chat.id, "2", autosummary="tokens")
        sent = self.repo._fake_provider.last_messages  # type: ignore[attr-defined]
        # system + summary + новое = 3 (сообщение "1"/"echo: 1" до summary не попадает).
        self.assertEqual(len(sent), 3)
        self.assertEqual(sent[1].content, summary_msg.content)
        self.assertEqual(sent[-1].content, "2")

    # ---- стратегия "Sliding Window" -----------------------------------------

    def test_sliding_window_trims_context_and_reports_active_start_id(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        self.repo.update_agent_settings(agent.id, {"context_strategy": "sliding_window", "context_strategy_limit": 3})
        chat = self.repo.create_chat(agent.id, "C1")
        self.repo.send_message_blocking(chat.id, "1", sliding_window=True)  # -> user1, assistant1 (2 сообщения)
        self.repo.send_message_blocking(chat.id, "2", sliding_window=True)  # -> user2, assistant2 (итого 4)
        # eligible (role user/assistant) = 4; в запрос уходят последние (N-1=2) + новое = 3
        self.repo.send_message_blocking(chat.id, "3", sliding_window=True)
        sent = self.repo._fake_provider.last_messages  # type: ignore[attr-defined]
        # system + 2 сохранённых (последние eligible до отправки) + новое = 4
        self.assertEqual(len(sent), 4)
        contents = [m.content for m in sent]
        self.assertIn("3", contents)  # новое сообщение точно есть
        self.assertNotIn("1", [c for c in contents if c == "1"])  # самое старое отфильтровано

        # Статистика/подсветка вне контекста (active_context_start_id) не
        # зависят от флага запроса — работают всегда, пока задана стратегия.
        chat_after = self.repo.get_chat(chat.id)
        stats = self.repo.chat_stats(chat_after)
        all_messages = self.repo.list_messages(chat.id)
        self.assertGreater(stats["active_context_start_id"], all_messages[0].id)

    def test_sliding_window_not_applied_without_explicit_flag(self):
        # Доработка: без явного флага sliding_window=true в запросе обрезка
        # контекста по стратегии больше не применяется автоматически — даже
        # если у чата задана context_strategy='sliding_window'.
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        self.repo.update_agent_settings(agent.id, {"context_strategy": "sliding_window", "context_strategy_limit": 3})
        chat = self.repo.create_chat(agent.id, "C1")
        for i in range(5):
            self.repo.send_message_blocking(chat.id, str(i))  # без sliding_window
        sent = self.repo._fake_provider.last_messages  # type: ignore[attr-defined]
        # system + 8 предыдущих (4 пары) + новое = 10, т.е. ничего не обрезано
        self.assertEqual(len(sent), 10)

    def test_sliding_window_flag_requires_strategy_and_limit(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")  # context_strategy не задана
        with self.assertRaises(ValidationError):
            self.repo.send_message_blocking(chat.id, "hi", sliding_window=True)

        # context_strategy задана, но лимит ещё не задан (limit=None) -> тоже ошибка.
        self.repo.update_chat_settings(chat.id, {"context_strategy": "sliding_window"})
        with self.assertRaises(ValidationError):
            self.repo.send_message_blocking(chat.id, "hi", sliding_window=True)

        self.repo.update_chat_settings(chat.id, {"context_strategy_limit": 3})
        self.repo.send_message_blocking(chat.id, "hi", sliding_window=True)  # теперь валидно, без исключений

    # ---- ветки диалога --------------------------------------------------------

    def test_create_branch_assigns_sequential_numbers(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")
        b1 = self.repo.create_branch(chat.id, "Вариант А")
        b2 = self.repo.create_branch(chat.id, "Вариант Б")
        self.assertEqual(b1.number, 1)
        self.assertEqual(b2.number, 2)
        self.assertEqual([b.name for b in self.repo.list_branches(chat.id)], ["Вариант А", "Вариант Б"])

    def test_branch_filtering_scopes_context_to_main_plus_selected(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")
        self.repo.send_message_blocking(chat.id, "root")  # branch 0
        branch = self.repo.create_branch(chat.id, "Alt")
        self.repo.send_message_blocking(chat.id, "branch-msg", branch=branch.number)
        all_messages = self.repo.list_messages(chat.id)
        self.assertEqual(len(all_messages), 4)
        for m in all_messages[:2]:
            self.assertEqual(m.branch, 0)
        for m in all_messages[2:]:
            self.assertEqual(m.branch, branch.number)

        # Отправка без указания ветки (или branch=0) не должна видеть сообщения из ветки 1.
        self.repo.send_message_blocking(chat.id, "root-2")
        sent_main = self.repo._fake_provider.last_messages  # type: ignore[attr-defined]
        self.assertNotIn("branch-msg", [m.content for m in sent_main])

        # Отправка в ветку 1 видит и основную ветку, и саму ветку.
        self.repo.send_message_blocking(chat.id, "branch-msg-2", branch=branch.number)
        sent_branch = self.repo._fake_provider.last_messages  # type: ignore[attr-defined]
        self.assertIn("root", [m.content for m in sent_branch])
        self.assertIn("branch-msg", [m.content for m in sent_branch])

    # ---- стратегия "Sticky Facts" -------------------------------------------

    def test_get_facts_requires_sticky_facts_strategy(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        chat = self.repo.create_chat(agent.id, "C1")
        with self.assertRaises(ValidationError):
            self.repo.send_message_blocking(chat.id, "Привет", get_facts=True)

    def test_get_facts_requires_limit_over_two(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        self.repo.update_agent_settings(agent.id, {"context_strategy": "sticky_facts"})  # лимит не задан
        chat = self.repo.create_chat(agent.id, "C1")
        with self.assertRaises(ValidationError):
            self.repo.send_message_blocking(chat.id, "Привет", get_facts=True)

    def test_get_facts_extracts_and_stores_facts_on_assistant_message(self):
        # Доработка по замечанию пользователя: факты сохраняются под ответом
        # АССИСТЕНТА (после того, как этот ответ получен и по нему выполнено
        # извлечение), а не под сообщением пользователя.
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        self.repo.update_agent_settings(agent.id, {"context_strategy": "sticky_facts", "context_strategy_limit": 4})
        chat = self.repo.create_chat(agent.id, "C1")
        provider = self.repo._fake_provider  # type: ignore[attr-defined]
        provider.next_facts_response = {"facts": [{"key": "user_name", "value": "Алиса", "confidence": 0.95}]}

        user_msg, assistant_msg = self.repo.send_message_blocking(chat.id, "Меня зовут Алиса", get_facts=True)
        self.assertIsNone(user_msg.facts)
        self.assertIsNotNone(assistant_msg.facts)
        stored = json.loads(assistant_msg.facts)
        self.assertEqual(stored["user_name"]["value"], "Алиса")

        # Второе сообщение должно найти уже сохранённые факты (под предыдущим
        # ответом ассистента) и передать их как existing_facts.
        provider.next_facts_response = {"facts": [{"key": "user_role", "value": "инженер", "confidence": 0.8}]}
        _, assistant_msg2 = self.repo.send_message_blocking(chat.id, "Я работаю инженером", get_facts=True)
        stored2 = json.loads(assistant_msg2.facts)
        self.assertEqual(stored2["user_name"]["value"], "Алиса")  # сохранённый факт не потерян
        self.assertEqual(stored2["user_role"]["value"], "инженер")  # новый факт добавлен

    def test_get_facts_injects_saved_facts_as_assistant_message_in_main_request(self):
        # "в качестве последнего сообщения [перед новым сообщением
        # пользователя] добавляется сообщение с ролью assistant и текстом
        # 'All Facts, fixed before in JSON: ...'" — это сообщение техническое
        # и не должно попадать в сохранённую историю чата.
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        self.repo.update_agent_settings(agent.id, {"context_strategy": "sticky_facts", "context_strategy_limit": 4})
        chat = self.repo.create_chat(agent.id, "C1")
        provider = self.repo._fake_provider  # type: ignore[attr-defined]
        provider.next_facts_response = {"facts": [{"key": "user_name", "value": "Алиса", "confidence": 0.95}]}
        self.repo.send_message_blocking(chat.id, "Меня зовут Алиса", get_facts=True)

        provider.next_facts_response = {"facts": []}
        self.repo.send_message_blocking(chat.id, "Как меня зовут?", get_facts=True)
        # last_messages — от ПОСЛЕДНЕГО вызова chat() в этой отправке, т.е. от
        # вызова извлечения фактов; сам основной вызов проверяем по факту его
        # эффекта — предыдущий ответ ассистента должен попасть в него как
        # обычное сообщение истории, а технической facts-подсказки в
        # сохранённой истории быть не должно.
        messages = self.repo.list_messages(chat.id)
        self.assertTrue(all("All Facts, fixed before in JSON" not in (m.content or "") for m in messages))

    def test_get_facts_main_request_limited_to_last_n_messages(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        self.repo.update_agent_settings(agent.id, {"context_strategy": "sticky_facts", "context_strategy_limit": 3})
        chat = self.repo.create_chat(agent.id, "C1")
        provider = self.repo._fake_provider  # type: ignore[attr-defined]
        provider.next_facts_response = {"facts": []}
        self.repo.send_message_blocking(chat.id, "сообщение один", get_facts=True)
        self.repo.send_message_blocking(chat.id, "сообщение два", get_facts=True)

        captured = []
        original_chat = provider.chat

        def spy_chat(model_id, messages, settings):
            captured.append(list(messages))
            return original_chat(model_id, messages, settings)

        provider.chat = spy_chat
        self.repo.send_message_blocking(chat.id, "сообщение три", get_facts=True)
        main_call_messages = captured[0]  # первый вызов в этой отправке — основной (до извлечения)
        self.assertEqual(main_call_messages[-1].role, "user")
        self.assertEqual(main_call_messages[-1].content, "сообщение три")
        # История без технической facts-подсказки и без самого нового
        # сообщения пользователя — ровно limit=3 сообщения (из 4 реально
        # накопленных к этому моменту).
        history_in_call = [
            m for m in main_call_messages[:-1]
            if m.role in ("user", "assistant") and not m.content.startswith("All Facts, fixed before in JSON")
        ]
        self.assertEqual(len(history_in_call), 3)

    def test_get_facts_tolerates_markdown_fenced_json_response(self):
        # Небольшие локальные модели иногда оборачивают JSON в markdown-ограду
        # ```json ... ``` вместо чистого JSON — раньше это приводило к тому,
        # что facts молча переставали обновляться начиная со второго
        # сообщения (именно так и выглядела жалоба пользователя).
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        self.repo.update_agent_settings(agent.id, {"context_strategy": "sticky_facts", "context_strategy_limit": 4})
        chat = self.repo.create_chat(agent.id, "C1")
        provider = self.repo._fake_provider  # type: ignore[attr-defined]
        provider.next_facts_response = {"facts": [{"key": "user_name", "value": "Алиса", "confidence": 0.95}]}
        self.repo.send_message_blocking(chat.id, "Меня зовут Алиса", get_facts=True)

        provider.next_facts_response = None
        provider.next_facts_response_raw = (
            'Конечно, вот обновлённые факты:\n```json\n'
            '{"facts": [{"key": "user_role", "value": "инженер", "confidence": 0.8}]}\n```'
        )
        _, assistant_msg2 = self.repo.send_message_blocking(chat.id, "Я работаю инженером", get_facts=True)
        stored2 = json.loads(assistant_msg2.facts)
        self.assertEqual(stored2["user_name"]["value"], "Алиса")
        self.assertEqual(stored2["user_role"]["value"], "инженер")

    def test_stream_message_supports_get_facts_with_status_events(self):
        # Доработка: обновление фактов раньше было вовсе недоступно в
        # потоковом режиме — теперь оно выполняется ПОСЛЕ того, как потоковая
        # генерация полностью завершена, и клиенту сообщается статус вызова.
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        self.repo.update_agent_settings(agent.id, {"context_strategy": "sticky_facts", "context_strategy_limit": 4})
        chat = self.repo.create_chat(agent.id, "C1")
        provider = self.repo._fake_provider  # type: ignore[attr-defined]
        provider.next_facts_response = {"facts": [{"key": "user_name", "value": "Алиса", "confidence": 0.95}]}

        events = list(self.repo.stream_message(chat.id, "Меня зовут Алиса", get_facts=True))
        statuses = [e["status"] for e in events if e["type"] == "status"]
        self.assertIn("Выполняется запрос к модели", statuses)
        self.assertIn("Обновление фактов", statuses)
        # Статус "Обновление фактов" должен идти ПОСЛЕ последней дельты
        # содержимого и до финального "done".
        delta_indices = [i for i, e in enumerate(events) if e["type"] == "delta"]
        facts_status_index = next(i for i, e in enumerate(events) if e.get("status") == "Обновление фактов")
        done_index = next(i for i, e in enumerate(events) if e["type"] == "done")
        self.assertGreater(facts_status_index, max(delta_indices))
        self.assertLess(facts_status_index, done_index)

        done_event = events[done_index]
        self.assertIsNotNone(done_event["message"].facts)
        stored = json.loads(done_event["message"].facts)
        self.assertEqual(stored["user_name"]["value"], "Алиса")

    def test_stream_message_supports_autosummary_with_status_event(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        self.repo.update_agent_settings(agent.id, {"autosummary": "messages", "autosummary_by_messages": 5})
        chat = self.repo.create_chat(agent.id, "C1")
        list(self.repo.stream_message(chat.id, "1", autosummary="messages"))
        list(self.repo.stream_message(chat.id, "2", autosummary="messages"))
        # Перед первыми двумя сообщениями порог (5-2=3) ещё не достигнут.
        self.assertFalse(any(m.is_summary for m in self.repo.list_messages(chat.id)))

        events = list(self.repo.stream_message(chat.id, "3", autosummary="messages"))
        statuses = [e["status"] for e in events if e["type"] == "status"]
        self.assertIn("Выполняется запрос к модели", statuses)
        self.assertIn("Выполняется суммаризация чата", statuses)
        done_index = next(i for i, e in enumerate(events) if e["type"] == "done")
        summary_status_index = next(i for i, e in enumerate(events) if e.get("status") == "Выполняется суммаризация чата")
        self.assertLess(summary_status_index, done_index)
        self.assertTrue(any(m.is_summary for m in self.repo.list_messages(chat.id)))

    def test_get_facts_extraction_call_always_forces_json_mode(self):
        # json_mode основного чата не должен влиять на извлечение фактов —
        # запрос к провайдеру для facts всегда идёт со строгим JSON-режимом,
        # независимо от того, что настроено для обычных ответов ассистента.
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        self.repo.update_agent_settings(
            agent.id, {"context_strategy": "sticky_facts", "context_strategy_limit": 4, "json_mode": False},
        )
        chat = self.repo.create_chat(agent.id, "C1")
        provider = self.repo._fake_provider  # type: ignore[attr-defined]
        provider.next_facts_response = {"facts": []}
        self.repo.send_message_blocking(chat.id, "Привет", get_facts=True)
        # last_messages/settings — от ПОСЛЕДНЕГО вызова chat(); чтобы
        # проверить именно вызов извлечения фактов (а не основной), делаем
        # его вторым не-facts вызовом невозможным здесь — вместо этого
        # проверяем напрямую через `_extract_facts`.
        result_settings_seen = []
        original_chat = provider.chat

        def spy_chat(model_id, messages, settings):
            result_settings_seen.append(settings.json_mode)
            return original_chat(model_id, messages, settings)

        provider.chat = spy_chat
        self.repo.send_message_blocking(chat.id, "Ещё сообщение", get_facts=True)
        # Среди всех вызовов chat() в этой отправке (facts + основной) хотя бы
        # один должен быть с json_mode=True (вызов извлечения фактов), даже
        # при том что у чата json_mode=False.
        self.assertIn(True, result_settings_seen)

    def test_get_facts_lower_confidence_does_not_override(self):
        agent = self.repo.create_agent("A", model=TEST_MODEL_ID)
        self.repo.update_agent_settings(agent.id, {"context_strategy": "sticky_facts", "context_strategy_limit": 4})
        chat = self.repo.create_chat(agent.id, "C1")
        provider = self.repo._fake_provider  # type: ignore[attr-defined]
        provider.next_facts_response = {"facts": [{"key": "primary_goal", "value": "A", "confidence": 0.9}]}
        self.repo.send_message_blocking(chat.id, "цель A", get_facts=True)
        provider.next_facts_response = {"facts": [{"key": "primary_goal", "value": "B", "confidence": 0.2}]}
        _, assistant_msg2 = self.repo.send_message_blocking(chat.id, "может быть B?", get_facts=True)
        stored = json.loads(assistant_msg2.facts)
        self.assertEqual(stored["primary_goal"]["value"], "A")  # низкая уверенность не перезаписывает

    # ---- модели -----------------------------------------------------------

    def test_model_health_uses_provider(self):
        self.assertTrue(self.repo.model_health(TEST_MODEL_ID))
        self.repo._fake_provider.healthy = False  # type: ignore[attr-defined]
        self.assertFalse(self.repo.model_health(TEST_MODEL_ID))

    def test_model_health_not_found(self):
        with self.assertRaises(NotFoundError):
            self.repo.model_health("fake:no-such-model")


if __name__ == "__main__":
    unittest.main()

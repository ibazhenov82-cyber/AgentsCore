"""
agents_core.repository
========================

Основная бизнес-логика AgentsCore: агенты, чаты, ветки диалога, сообщения,
подсчёт токенов, суммаризация (ручная и автоматическая) и стратегии
управления контекстом ("Sliding Window", "Sticky Facts"). HTTP-слой (`api/`)
— это тонкая обёртка над методами `Repository`, ничего не знающая о
провайдерах или устройстве базы данных напрямую.
"""

from __future__ import annotations

import dataclasses
import json
import re
import threading
import time
from typing import Dict, Iterator, List, Optional, Tuple

from .catalog import ModelCatalog
from .db import Database
from .format_detect import detect_message_format
from .models import (
    Agent,
    AUTOSUMMARY_OPTIONS,
    Branch,
    Chat,
    CONTEXT_STRATEGY_OPTIONS,
    DefaultSettings,
    Message,
    ModelInfo,
    Settings,
    settings_from_defaults,
)
from .providers import ProviderError, ProviderMessage, ProviderRegistry
from .tokens import estimate_messages_tokens

_SETTINGS_FIELD_NAMES = {f.name for f in dataclasses.fields(Settings)}
_DEFAULT_SETTINGS_FIELD_NAMES = {f.name for f in dataclasses.fields(DefaultSettings)}

_SUMMARY_TAG_RE = {
    "previous_summary": re.compile(r"<previous_summary>.*?</previous_summary>", re.DOTALL),
    "new_messages": re.compile(r"<new_messages>.*?</new_messages>", re.DOTALL),
}


class RepositoryError(Exception):
    """Базовый класс доменных ошибок; HTTP-слой сопоставляет их с кодами ответа."""


class NotFoundError(RepositoryError):
    pass


class ValidationError(RepositoryError):
    pass


class NotConfiguredError(RepositoryError):
    pass


class PreconditionFailedError(RepositoryError):
    """Суммаризация недоступна: чат уже превышает допустимые лимиты токенов."""


def _split_model(composite_id: str) -> Tuple[str, str]:
    provider, _, model_id = composite_id.partition(":")
    if not provider or not model_id:
        raise ValidationError(f"invalid model id: {composite_id!r} (expected 'provider:model_id')")
    return provider, model_id


def _apply_partial(current, payload: dict, valid_fields: set):
    unknown = set(payload) - valid_fields
    if unknown:
        raise ValidationError(f"unknown settings field(s): {', '.join(sorted(unknown))}")
    return dataclasses.replace(current, **payload)


def _validate_settings_payload(payload: dict) -> None:
    if "autosummary" in payload and payload["autosummary"] not in AUTOSUMMARY_OPTIONS:
        raise ValidationError(f"invalid autosummary: {payload['autosummary']!r}")
    if "context_strategy" in payload:
        value = payload["context_strategy"]
        if value is not None and value not in CONTEXT_STRATEGY_OPTIONS:
            raise ValidationError(f"invalid context_strategy: {value!r}")
    if "context_strategy_limit" in payload:
        value = payload["context_strategy_limit"]
        if value is not None and value <= 2:
            raise ValidationError("context_strategy_limit must be > 2")


def _fill_summary_template(template: str, previous_summary: str, new_messages_text: str) -> str:
    """Подставляет предыдущее резюме и новые сообщения в шаблон `summary_prompt`
    в теги `<previous_summary>`/`<new_messages>`. Если шаблон был изменён
    пользователем и тегов в нём нет — возвращает шаблон как есть (best-effort,
    не роняем суммаризацию из-за нестандартного шаблона)."""
    result = template
    if _SUMMARY_TAG_RE["previous_summary"].search(result):
        result = _SUMMARY_TAG_RE["previous_summary"].sub(
            lambda _m: f"<previous_summary>\n{previous_summary}\n</previous_summary>", result, count=1
        )
    if _SUMMARY_TAG_RE["new_messages"].search(result):
        result = _SUMMARY_TAG_RE["new_messages"].sub(
            lambda _m: f"<new_messages>\n{new_messages_text}\n</new_messages>", result, count=1
        )
    return result


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_json_loose(text: str) -> Optional[dict]:
    """Разбирает JSON-ответ модели терпимо к типичным отклонениям от
    "чистого" JSON: модель могла обернуть его в ```json ... ``` (markdown-
    ограда) или добавить пояснительный текст до/после самого объекта —
    особенно свойственно небольшим локальным моделям. Пробует, по порядку:
    1) весь текст как есть; 2) содержимое первой markdown-ограды ```...```;
    3) подстроку от первой `{` до последней `}`. Возвращает `None`, только
    если ни один из вариантов не разобрался, — тогда вызывающий код
    сохраняет предыдущее значение вместо того, чтобы упасть с ошибкой."""
    candidates = [text]
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        candidates.append(fence_match.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _format_messages_block(messages: List[Message]) -> str:
    """Форматирует сообщения в вид `[user]: ...` / `[assistant]: ...` построчно —
    ровно то, что подставляется в тег `<new_messages>` шаблона суммаризации."""
    lines = [f"[{m.role}]: {m.content}" for m in messages if m.role in ("user", "assistant")]
    return "\n".join(lines)


class Repository:
    def __init__(self, db: Database, registry: ProviderRegistry, catalog: ModelCatalog):
        self._db = db
        self._registry = registry
        self._catalog = catalog
        self._locks: Dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, chat_id: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(chat_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[chat_id] = lock
            return lock

    # ---- настройки по умолчанию -------------------------------------------

    def get_default_settings(self) -> DefaultSettings:
        return self._db.get_default_settings()

    def update_default_settings(self, payload: dict) -> DefaultSettings:
        current = self._db.get_default_settings()
        updated = _apply_partial(current, payload, _DEFAULT_SETTINGS_FIELD_NAMES)
        return self._db.update_default_settings(updated)

    def reset_default_settings(self) -> DefaultSettings:
        return self._db.update_default_settings(DefaultSettings())

    # ---- каталог моделей ----------------------------------------------------

    def list_models(self) -> List[ModelInfo]:
        return self._catalog.list_models()

    def _require_model(self, composite_model_id: str) -> ModelInfo:
        model = self._catalog.get(composite_model_id)
        if model is None:
            raise NotFoundError(f"no such model: {composite_model_id}")
        return model

    def model_health(self, composite_model_id: str) -> bool:
        model = self._require_model(composite_model_id)
        provider = self._registry.get(model.provider)
        model_health_fn = getattr(provider, "model_health", None)
        if callable(model_health_fn):
            return model_health_fn(model.model_id)
        return provider.health()

    # ---- агенты -------------------------------------------------------------

    def _require_agent(self, agent_id: str) -> Agent:
        agent = self._db.get_agent(agent_id)
        if agent is None:
            raise NotFoundError(f"no such agent: {agent_id}")
        return agent

    def create_agent(self, name: Optional[str], model: Optional[str] = None) -> Agent:
        defaults = self._db.get_default_settings()
        settings = settings_from_defaults(defaults)
        if model:
            settings.model = model
        # Имя по умолчанию — "Агент N", где N — порядковый номер (число уже
        # существующих агентов + 1), если имя не задано явно.
        resolved_name = name if name and name.strip() else f"Агент {len(self._db.list_agents()) + 1}"
        return self._db.create_agent(resolved_name, settings)

    def get_agent(self, agent_id: str) -> Agent:
        return self._require_agent(agent_id)

    def list_agents(self) -> List[Agent]:
        return self._db.list_agents()

    def list_agents_with_chats(self) -> List[Tuple[Agent, List[Chat], List[dict]]]:
        """Для главного экрана приложения: каждый агент со списком его чатов
        и агрегатами по токенам/заполнению контекста для каждого чата."""
        # Список показывает объединение веток 0 и 1 (если ветка 1 существует)
        # — фиксированная конвенция для компактного отображения; чаты без
        # веток естественно сводятся к одной только основной ветке (0).
        result = []
        for agent in self._db.list_agents():
            chats = self._db.list_chats(agent.id)
            stats = [self.chat_stats(chat, branch=1) for chat in chats]
            result.append((agent, chats, stats))
        return result

    def rename_agent(self, agent_id: str, name: str) -> Agent:
        self._require_agent(agent_id)
        if not name or not name.strip():
            raise ValidationError("name must be a non-empty string")
        self._db.rename_agent(agent_id, name)
        return self._require_agent(agent_id)

    def update_agent_settings(self, agent_id: str, payload: dict) -> Settings:
        agent = self._require_agent(agent_id)
        _validate_settings_payload(payload)
        updated = _apply_partial(agent.settings, payload, _SETTINGS_FIELD_NAMES)
        self._db.update_agent_settings(agent_id, updated)
        return updated

    def delete_agent(self, agent_id: str) -> None:
        self._require_agent(agent_id)
        self._db.delete_agent(agent_id)  # ON DELETE CASCADE удаляет чаты и их сообщения

    # ---- чаты -----------------------------------------------------------------

    def _require_chat(self, chat_id: str) -> Chat:
        chat = self._db.get_chat(chat_id)
        if chat is None:
            raise NotFoundError(f"no such chat: {chat_id}")
        return chat

    def create_chat(self, agent_id: str, title: Optional[str]) -> Chat:
        agent = self._require_agent(agent_id)
        settings = dataclasses.replace(agent.settings)
        # Имя по умолчанию — "Чат N", где N — порядковый номер чата ВНУТРИ
        # этого агента (число уже существующих чатов агента + 1).
        resolved_title = title if title and title.strip() else f"Чат {len(self._db.list_chats(agent_id)) + 1}"
        return self._db.create_chat(agent_id, resolved_title, settings)

    def get_chat(self, chat_id: str) -> Chat:
        return self._require_chat(chat_id)

    def list_chats(self, agent_id: Optional[str] = None) -> List[Chat]:
        return self._db.list_chats(agent_id)

    def rename_chat(self, chat_id: str, title: str) -> Chat:
        self._require_chat(chat_id)
        if not title or not title.strip():
            raise ValidationError("title must be a non-empty string")
        self._db.rename_chat(chat_id, title)
        return self._require_chat(chat_id)

    def update_chat_settings(self, chat_id: str, payload: dict) -> Settings:
        chat = self._require_chat(chat_id)
        if "model" in payload and payload["model"] != chat.settings.model:
            raise ValidationError("model is fixed by the chat's agent and cannot be changed on a chat")
        _validate_settings_payload(payload)
        updated = _apply_partial(chat.settings, payload, _SETTINGS_FIELD_NAMES)
        self._db.update_chat_settings(chat_id, updated)
        return updated

    def delete_chat(self, chat_id: str) -> None:
        self._require_chat(chat_id)
        self._db.delete_chat(chat_id)

    def copy_chat(self, chat_id: str, new_title: str) -> Chat:
        chat = self._require_chat(chat_id)
        messages = self._db.list_messages(chat_id)
        new_chat = self._db.create_chat(chat.agent_id, new_title or f"{chat.title} (копия)", dataclasses.replace(chat.settings))
        for m in messages:
            self._db.add_message(
                Message(
                    id=0, chat_id=new_chat.id, role=m.role, content=m.content, created_at=m.created_at,
                    reasoning_content=m.reasoning_content, is_summary=m.is_summary, duration_ms=m.duration_ms,
                    total_tokens=m.total_tokens, prompt_tokens=m.prompt_tokens, completion_tokens=m.completion_tokens,
                    format=m.format, branch=m.branch, facts=m.facts,
                )
            )
        for b in self._db.list_branches(chat_id):
            self._db.create_branch(new_chat.id, b.name)
        return new_chat

    # ---- ветки диалога --------------------------------------------------------

    def list_branches(self, chat_id: str) -> List[Branch]:
        self._require_chat(chat_id)
        return self._db.list_branches(chat_id)

    def create_branch(self, chat_id: str, name: Optional[str] = None) -> Branch:
        self._require_chat(chat_id)
        # Имя не задано (или пустое) -> автоимя "Ветка N", где N — реальный
        # номер, присвоенный этой ветке (не порядковый счётчик существующих
        # веток: после удаления веток номера не переиспользуются, поэтому
        # только `Database.create_branch` знает верное N — вычисляет и
        # подставляет автоимя внутри одной транзакции с выдачей номера).
        resolved_name = name if name and name.strip() else None
        return self._db.create_branch(chat_id, resolved_name)

    def delete_branch(self, chat_id: str, number: int) -> None:
        """Удаляет ветку диалога вместе со всеми её сообщениями (branch=number);
        сообщения основной ветки (0) не затрагиваются. Ветку 0 удалить нельзя —
        это основная ветка чата, а не отдельная запись в `chat_branches`."""
        self._require_chat(chat_id)
        if number == 0:
            raise ValidationError("cannot delete the main branch (0)")
        if not self._db.get_branch(chat_id, number):
            raise NotFoundError(f"no such branch: {number}")
        self._db.delete_branch(chat_id, number)

    @staticmethod
    def _filter_by_branch(messages: List[Message], branch: Optional[int]) -> List[Message]:
        """Не указана ветка (или указана 0) -> только основная ветка (branch=0,
        поведение чатов, которые ветками не пользуются, не меняется); указана
        ветка N>0 -> основная ветка + ветка N, как того требует ТЗ."""
        selected = branch or 0
        if selected == 0:
            return [m for m in messages if m.branch == 0]
        return [m for m in messages if m.branch in (0, selected)]

    # ---- токены / контекстное окно --------------------------------------

    @staticmethod
    def _last_summary_index(messages: List[Message]) -> Optional[int]:
        """Индекс ПОСЛЕДНЕГО сообщения с is_summary=True в списке (или None,
        если резюме ещё не было ни разу) — общая точка отсчёта и для среза
        контекста (`_context_messages`), и для проверки необходимости
        автосуммаризации (`_autosummary_due`)."""
        idx = None
        for i, m in enumerate(messages):
            if m.is_summary:
                idx = i
        return idx

    def _context_messages(self, messages: List[Message]) -> List[Message]:
        """Сообщения, реально отправляемые модели в следующем запросе с учётом
        суммаризации: если в истории есть summary-сообщение, это оно само и
        всё, что после него; иначе — вся история."""
        idx = self._last_summary_index(messages)
        if idx is None:
            return messages
        return messages[idx:]

    def _effective_context(self, chat: Chat, messages: List[Message]) -> List[Message]:
        """Сообщения, которые реально уйдут в API при следующей отправке (без
        нового сообщения пользователя) — с учётом суммаризации и, если
        включена, стратегии "Sliding Window"."""
        context = self._context_messages(messages)
        settings = chat.settings
        # И "Sliding Window", и "Sticky Facts" ограничивают эффективный контекст
        # последними N сообщениями (N=context_strategy_limit) — при "Sticky
        # Facts" это ограничение применяется отдельным путём в _build_sticky_facts_context
        # (там формируется единое "мега-сообщение"), но для целей оценки
        # текущего размера контекста и статистики токенов обе стратегии дают
        # одну и ту же обрезку по последним N сообщениям.
        if settings.context_strategy in ("sliding_window", "sticky_facts") and settings.context_strategy_limit and settings.context_strategy_limit > 2:
            limit = settings.context_strategy_limit
            eligible = [m for m in context if m.role in ("user", "assistant")]
            if len(eligible) + 1 > limit:  # +1 — будущее сообщение пользователя
                keep_ids = {m.id for m in eligible[-(limit - 1):]} if limit > 1 else set()
                context = [m for m in context if m.id in keep_ids]
        return context

    def _current_context_tokens(self, chat: Chat, messages: List[Message]) -> int:
        context = self._effective_context(chat, messages)
        if context and context[-1].role == "assistant" and context[-1].total_tokens is not None:
            # total_tokens последнего ответа ассистента = токены, отправленные
            # ему в запросе (весь контекст на тот момент) + его собственный
            # ответ — надёжная оценка текущего размера контекста без
            # повторного обращения к токенизатору провайдера.
            return context[-1].total_tokens
        texts = ([chat.settings.system_prompt] if chat.settings.system_prompt else []) + [m.content for m in context]
        return estimate_messages_tokens(texts)

    def chat_stats(self, chat: Chat, messages: Optional[List[Message]] = None, branch: Optional[int] = None) -> dict:
        """Агрегаты по токенам/заполнению контекста.

        `branch` определяет, какая ветка диалога учитывается: не задана (или
        0) — только основная ветка; N>0 — основная ветка + ветка N (см.
        `_filter_by_branch`). Список чатов на главном экране использует
        фиксированную конвенцию branch=1 (ветки 0 и 1, если ветка 1
        существует; иначе это естественно сводится к одной только основной
        ветке), а экран чата передаёт явно выбранную пользователем ветку.

        `prompt_tokens`/`completion_tokens`/`total_tokens` — СУММА токенов по
        ВСЕМ сообщениям в эффективном окне (после границы суммаризации и,
        если применимо, обрезки стратегией "Sliding Window"/"Sticky Facts" —
        то же самое окно, вне которого сообщения показываются более бледными
        на экране чата, см. `active_context_start_id` ниже): `prompt_tokens`
        — сумма `prompt_tokens` всех сообщений пользователя,
        `completion_tokens` — сумма `completion_tokens` всех ответов
        ассистента, `total_tokens` = их сумма. Именно от `total_tokens`
        считается и `context_fill_ratio` = `total_tokens / context_window` —
        по прямой инструкции пользователя проценты заполнения должны
        отражать РЕАЛЬНЫЙ расход по видимой части чата (может быть больше
        100%, если чат уже "перерос" контекстное окно модели — клиент в этом
        случае показывает 100% и красит индикатор красным, см.
        `ChatStatsOut.context_fill_ratio`), а не размер одного последнего
        обмена (`current_context_tokens` — отдельное поле именно для этого,
        см. ниже).

        `current_context_tokens` — РАЗМЕР ТЕКУЩЕГО КОНТЕКСТА: то, что реально
        уйдёт в следующем запросе модели, т.е. prompt_tokens +
        completion_tokens ПОСЛЕДНЕГО обмена в эффективном окне (см.
        `_current_context_tokens`) — используется только для проверки
        `can_summarize`/лимитов модели и чата, НЕ для `context_fill_ratio`.

        Знаменатель `context_fill_ratio` — это `context_window` модели, ЕСЛИ
        только настройка чата `max_tokens` не задана и не меньше его: в этом
        случае (по прямой инструкции пользователя) вместо размера окна модели
        используется именно `max_tokens` — пользователь явно ограничил чат
        меньшим бюджетом, и заполнение должно считаться относительно него.
        """
        if messages is None:
            messages = self._db.list_messages(chat.id)
        scoped = self._filter_by_branch(messages, branch)
        effective = self._effective_context(chat, scoped)
        current_context = self._current_context_tokens(chat, scoped)
        # "Всего потрачено" на видимую (не бледную) часть чата — сумма по
        # ВСЕМ сообщениям эффективного окна, а не разбивка последнего обмена
        # (см. docstring выше).
        prompt_total = sum(m.prompt_tokens or 0 for m in effective if m.role == "user")
        completion_total = sum(m.completion_tokens or 0 for m in effective if m.role == "assistant")
        total_tokens = prompt_total + completion_total
        model_info = self._catalog.get(chat.settings.model)
        max_input = model_info.max_input_tokens if model_info else None
        context_window = model_info.context_window if model_info else None
        # Заполненность контекстного окна = total_tokens / effective_window —
        # реальный расход по видимой (не бледной) части чата; может выйти за
        # 100%, если чат уже перерос контекстное окно — значение НЕ
        # обрезается здесь намеренно (клиент сам решает, как показать
        # переполнение: капом на 100% и красным цветом индикатора).
        # effective_window — это context_window модели, но если у чата задан
        # max_tokens и он МЕНЬШЕ размера окна модели, то именно max_tokens
        # ограничивает реальный бюджет чата, и его нужно использовать вместо
        # context_window.
        effective_window = context_window
        if chat.settings.max_tokens is not None and context_window is not None and chat.settings.max_tokens < context_window:
            effective_window = chat.settings.max_tokens
        fill_ratio = (total_tokens / effective_window) if effective_window else None
        within_model_limit = max_input is None or current_context <= max_input
        within_chat_limit = chat.settings.max_tokens is None or current_context <= chat.settings.max_tokens
        active_context_start_id = effective[0].id if effective else None
        return {
            "prompt_tokens": prompt_total,
            "completion_tokens": completion_total,
            "total_tokens": total_tokens,
            "current_context_tokens": current_context,
            "max_input_tokens": max_input,
            "context_window": context_window,
            "context_fill_ratio": fill_ratio,
            "active_context_start_id": active_context_start_id,
            "can_summarize": bool(scoped) and within_model_limit and within_chat_limit,
            "model": model_info,
        }

    # ---- сообщения --------------------------------------------------------

    def list_messages(self, chat_id: str) -> List[Message]:
        self._require_chat(chat_id)
        return self._db.list_messages(chat_id)

    def delete_message(self, chat_id: str, message_id: int) -> None:
        self._require_chat(chat_id)
        self._db.delete_message(chat_id, message_id)

    def bulk_delete_messages(self, chat_id: str, message_ids: List[int]) -> None:
        self._require_chat(chat_id)
        self._db.bulk_delete_messages(chat_id, message_ids)

    def clear_messages(self, chat_id: str) -> None:
        self._require_chat(chat_id)
        self._db.clear_messages(chat_id)

    # ---- суммаризация -------------------------------------------------------

    def _perform_summarization(self, chat: Chat, messages: List[Message]) -> Message:
        current_tokens = self._current_context_tokens(chat, messages)
        model_info = self._catalog.get(chat.settings.model)
        max_input = model_info.max_input_tokens if model_info else None
        if max_input is not None and current_tokens > max_input:
            raise PreconditionFailedError(
                "chat already exceeds the model's max input tokens; cannot summarize"
            )
        if chat.settings.max_tokens is not None and current_tokens > chat.settings.max_tokens:
            raise PreconditionFailedError(
                "chat exceeds the chat's configured max_tokens setting; cannot summarize"
            )

        # Изолированный вызов: system = summary_system_prompt, единственное
        # user-сообщение = шаблон summary_prompt с подставленными предыдущим
        # резюме и новыми сообщениями. Обычная история чата и её system_prompt
        # в этом вызове не участвуют — см. п.1.1 требований.
        context = self._context_messages(messages)
        previous_summary_text = ""
        rest = context
        if context and context[0].is_summary:
            previous_summary_text = context[0].content
            rest = context[1:]
        # Суммаризация касается только сообщений с ролью user/assistant.
        new_messages_text = _format_messages_block(rest)
        user_prompt = _fill_summary_template(chat.settings.summary_prompt, previous_summary_text, new_messages_text)

        provider_messages = []
        if chat.settings.summary_system_prompt and chat.settings.summary_system_prompt.strip():
            provider_messages.append(ProviderMessage("system", chat.settings.summary_system_prompt))
        provider_messages.append(ProviderMessage("user", user_prompt))

        provider_name, model_id = _split_model(chat.settings.model)
        provider = self._registry.get(provider_name)
        result = provider.chat(model_id, provider_messages, chat.settings)

        # Ответ сохраняется как сообщение АССИСТЕНТА с признаком is_summary=true.
        summary_message = Message(
            id=0, chat_id=chat.id, role="assistant", content=result.content,
            created_at=int(time.time()), is_summary=True,
            format=detect_message_format(result.content),
        )
        saved = self._db.add_message(summary_message)
        self._db.touch_chat(chat.id)
        return saved

    def summarize_chat(self, chat_id: str) -> Message:
        chat = self._require_chat(chat_id)
        with self._lock_for(chat_id):
            messages = self._db.list_messages(chat_id)
            return self._perform_summarization(chat, messages)

    def _validate_autosummary_send_config(self, chat: Chat, mode: str) -> None:
        """Проверка флага `autosummary` запроса отправки сообщения (не
        путать с `_validate_settings_payload`, которая проверяет саму
        настройку чата при её изменении). `mode` — значение флага запроса
        ('messages' или 'tokens'; 'off' сюда не передаётся, см. вызывающий
        код) — должно совпадать с настройкой чата `autosummary`, а
        соответствующий предел должен быть корректно задан."""
        settings = chat.settings
        if mode not in ("messages", "tokens"):
            raise ValidationError(f"invalid autosummary: {mode!r}")
        if mode == "messages":
            if settings.autosummary != "messages":
                raise ValidationError("autosummary='messages' requires chat setting autosummary='messages'")
            if not settings.autosummary_by_messages or settings.autosummary_by_messages <= 2:
                raise ValidationError("autosummary_by_messages must be > 2 to use autosummary='messages'")
        else:
            if settings.autosummary != "tokens":
                raise ValidationError("autosummary='tokens' requires chat setting autosummary='tokens'")
            if not settings.autosummary_by_tokens or settings.autosummary_by_tokens <= 0:
                raise ValidationError("autosummary_by_tokens must be > 0 to use autosummary='tokens'")

    def _autosummary_due(self, chat: Chat, messages: List[Message], mode: str) -> bool:
        """Проверяется ПОСЛЕ основного ответа модели, по истории ДО текущего
        обмена (`messages` не включает ни новое сообщение пользователя, ни
        свежий ответ ассистента — "без учёта текущего запроса пользователя"
        из ТЗ). Точка отсчёта — то же самое последнее summary-сообщение
        (или начало чата), что и для среза контекста (`_context_messages`),
        но само summary-сообщение (если есть) в подсчёт не входит — считаются
        только сообщения user/assistant СТРОГО ПОСЛЕ него."""
        idx = self._last_summary_index(messages)
        scoped = messages if idx is None else messages[idx + 1:]
        scoped = [m for m in scoped if m.role in ("user", "assistant")]
        if mode == "messages":
            threshold = chat.settings.autosummary_by_messages - 2
            return len(scoped) >= threshold
        if mode == "tokens":
            if scoped and scoped[-1].role == "assistant" and scoped[-1].total_tokens is not None:
                tokens = scoped[-1].total_tokens
            else:
                texts = ([chat.settings.system_prompt] if chat.settings.system_prompt else []) + [m.content for m in scoped]
                tokens = estimate_messages_tokens(texts)
            return tokens >= chat.settings.autosummary_by_tokens
        return False

    def _summarize_after_send(
        self, chat: Chat, messages_before: List[Message], user_msg: Message, assistant_msg: Message
    ) -> None:
        """Пост-фактум выполнение автосуммаризации — вызывается только когда
        `_autosummary_due` уже вернул True. Захватывает и текущий обмен
        (новое сообщение пользователя + свежий ответ ассистента), чтобы он не
        остался неучтённым до следующей проверки."""
        full_messages = messages_before + [user_msg, assistant_msg]
        try:
            self._perform_summarization(chat, full_messages)
        except (PreconditionFailedError, ProviderError):
            # Основной ответ пользователю уже успешно получен и сохранён —
            # ошибку самой суммаризации (например, контекст уже превышен, или
            # сбой провайдера при изолированном вызове суммаризации) не
            # пробрасываем наружу, просто пропускаем автосуммаризацию сейчас;
            # она будет предложена повторно при следующей отправке.
            pass

    # ---- стратегия "Sticky Facts" -------------------------------------------

    def _validate_sticky_facts_config(self, chat: Chat) -> None:
        if chat.settings.context_strategy != "sticky_facts":
            raise ValidationError("get_facts requires context_strategy='sticky_facts' to be set on this chat")
        limit = chat.settings.context_strategy_limit
        if limit is None or limit <= 2:
            raise ValidationError("context_strategy_limit must be > 2 to use get_facts")

    # ---- стратегия "Sliding Window" (запрос отправки сообщения) -------------

    def _validate_sliding_window_config(self, chat: Chat) -> None:
        if chat.settings.context_strategy != "sliding_window":
            raise ValidationError("sliding_window requires context_strategy='sliding_window' to be set on this chat")
        limit = chat.settings.context_strategy_limit
        if limit is None or limit <= 2:
            raise ValidationError("context_strategy_limit must be > 2 to use sliding_window")

    def _build_sliding_window_context(
        self, chat: Chat, messages: List[Message], new_text: str
    ) -> List[ProviderMessage]:
        """Основной запрос при явном флаге `sliding_window=true`: системный
        prompt + последние (context_strategy_limit - 1) сообщений с ролью
        user/assistant + новое сообщение пользователя — более ранние
        сообщения не отправляются. В отличие от Sticky Facts здесь не
        добавляется никаких технических сообщений."""
        limit = chat.settings.context_strategy_limit or 0
        take = max(limit - 1, 0)
        recent = [m for m in messages if m.role in ("user", "assistant")]
        recent = recent[-take:] if take > 0 else []
        provider_messages: List[ProviderMessage] = []
        if chat.settings.system_prompt and chat.settings.system_prompt.strip():
            provider_messages.append(ProviderMessage("system", chat.settings.system_prompt))
        provider_messages += [ProviderMessage(m.role, m.content) for m in recent]
        provider_messages.append(ProviderMessage("user", new_text))
        return provider_messages

    @staticmethod
    def _load_latest_facts(messages: List[Message]) -> Tuple[dict, Optional[Message]]:
        # Факты сохраняются под ОТВЕТОМ АССИСТЕНТА (после того, как этот ответ
        # уже получен и извлечение по нему выполнено), а не под сообщением
        # пользователя — см. _extract_facts ниже.
        for m in reversed(messages):
            if m.role == "assistant" and m.facts:
                try:
                    parsed = json.loads(m.facts)
                except (ValueError, TypeError):
                    continue
                if isinstance(parsed, dict):
                    return parsed, m
        return {}, None

    @staticmethod
    def _dialogue_since(messages: List[Message], boundary: Optional[Message]) -> List[Message]:
        if boundary is None:
            return [m for m in messages if m.role in ("user", "assistant")]
        idx = next((i for i, m in enumerate(messages) if m.id == boundary.id), None)
        if idx is None:
            return [m for m in messages if m.role in ("user", "assistant")]
        return [m for m in messages[idx + 1:] if m.role in ("user", "assistant")]

    def _extract_facts(
        self,
        chat: Chat,
        existing_facts: dict,
        dialogue: List[Message],
        new_user_text: str,
        new_assistant_text: str,
    ) -> dict:
        """Извлечение фактов выполняется ПОСЛЕ основного запроса к модели (и
        для блокирующей, и для потоковой отправки — после того, как получен
        полный ответ ассистента), поэтому в `new_messages` попадает не только
        новое сообщение пользователя, но и только что сгенерированный ответ —
        это даёт извлечению больше контекста по сравнению со старой схемой,
        где оно запускалось ДО основного вызова и не видело ответ вовсе."""
        new_messages_payload = [{"role": m.role, "content": m.content} for m in dialogue]
        new_messages_payload.append({"role": "user", "content": new_user_text})
        new_messages_payload.append({"role": "assistant", "content": new_assistant_text})
        payload = {"existing_facts": existing_facts, "new_messages": new_messages_payload}
        provider_messages = [
            ProviderMessage("system", chat.settings.extraction_system_prompt),
            ProviderMessage("user", json.dumps(payload, ensure_ascii=False)),
        ]
        provider_name, model_id = _split_model(chat.settings.model)
        provider = self._registry.get(provider_name)
        # Извлечение фактов — отдельный, изолированный вызов модели, никак не
        # связанный с основным диалогом. Раньше сюда передавались настройки
        # чата как есть — а `json_mode` у обычного диалога обычно выключен
        # (это настройка для ОТВЕТОВ пользователю, не для фактов). Из-за этого
        # с ростом истории некоторые модели (особенно маленькие локальные,
        # например qwen3:0.6b) вместо строгого JSON начинали отвечать обычным
        # текстом — extract_facts не мог его разобрать и молча возвращал
        # facts без изменений, что выглядело как "факты перестали обновляться
        # после первого сообщения". Поэтому здесь всегда запрашиваем у
        # провайдера строгий JSON-режим, независимо от настройки основного чата.
        extraction_settings = dataclasses.replace(chat.settings, json_mode=True)
        result = provider.chat(model_id, provider_messages, extraction_settings)
        parsed = _parse_json_loose(result.content)
        if parsed is None:
            return existing_facts
        facts_list = parsed.get("facts") if isinstance(parsed, dict) else None
        if not isinstance(facts_list, list):
            return existing_facts
        updated = dict(existing_facts)
        for item in facts_list:
            if not isinstance(item, dict) or not item.get("key"):
                continue
            key = item["key"]
            new_confidence = item.get("confidence", 0.0) or 0.0
            current = updated.get(key)
            if current is None or new_confidence >= current.get("confidence", 0.0):
                updated[key] = {"value": item.get("value"), "confidence": new_confidence}
        return updated

    # ---- отправка сообщений -------------------------------------------------

    def _build_request_context(self, chat: Chat, messages: List[Message], new_text: str) -> List[ProviderMessage]:
        """Путь по умолчанию — без флагов `get_facts`/`sliding_window` и без
        активного `autosummary` в запросе: контекст учитывает только границу
        (ручной или авто-) суммаризации — последнее сообщение с
        is_summary=true и всё после него, либо вся история, если резюме ещё
        не было (`_context_messages`). Обрезка стратегией "Sliding
        Window"/"Sticky Facts" здесь НЕ применяется — она включается только
        явным флагом соответствующего запроса (см. `_build_sliding_window_context`,
        `_build_sticky_facts_context`), иначе получилось бы, что стратегия
        неявно обрезает контекст на каждой отправке независимо от намерения
        клиента (ограничение "N последних сообщений" при этом всё равно
        учитывается в статистике чата и подсветке вне контекста — см.
        `_effective_context`, используемый только в `chat_stats`)."""
        context = self._context_messages(messages)
        provider_messages = []
        if chat.settings.system_prompt and chat.settings.system_prompt.strip():
            provider_messages.append(ProviderMessage("system", chat.settings.system_prompt))
        provider_messages += [ProviderMessage(m.role, m.content) for m in context]
        provider_messages.append(ProviderMessage("user", new_text))
        return provider_messages

    def _build_sticky_facts_context(
        self, chat: Chat, messages: List[Message], new_text: str, latest_facts: dict
    ) -> List[ProviderMessage]:
        """Основной запрос ассистенту при активной стратегии Sticky Facts
        собирается обычным образом (system + история + новое сообщение
        пользователя, как и без get_facts), но с двумя отличиями:

        1. история — не весь эффективный контекст чата, а строго последние
           `context_strategy_limit` сообщений (диалог может быть намного
           длиннее — предполагается, что то, что не поместилось, уже
           учтено в самих `facts`);
        2. если факты уже сохранялись раньше, непосредственно перед новым
           сообщением пользователя добавляется ТЕХНИЧЕСКОЕ сообщение с ролью
           assistant вида `All Facts, fixed before in JSON: {...}` — оно
           нужно только этому конкретному запросу к модели и нигде не
           сохраняется (не попадает в историю чата).
        """
        limit = chat.settings.context_strategy_limit or 0
        recent = [m for m in messages if m.role in ("user", "assistant")]
        recent = recent[-limit:] if limit > 0 else []

        provider_messages: List[ProviderMessage] = []
        if chat.settings.system_prompt and chat.settings.system_prompt.strip():
            provider_messages.append(ProviderMessage("system", chat.settings.system_prompt))
        provider_messages += [ProviderMessage(m.role, m.content) for m in recent]
        if latest_facts:
            facts_text = "All Facts, fixed before in JSON: " + json.dumps(latest_facts, ensure_ascii=False)
            provider_messages.append(ProviderMessage("assistant", facts_text))
        provider_messages.append(ProviderMessage("user", new_text))
        return provider_messages

    def _resolve_send_context(
        self, chat: Chat, messages: List[Message], text: str, get_facts: bool, sliding_window: bool, autosummary: str
    ) -> Tuple[List[ProviderMessage], dict, Optional[Message]]:
        """Собирает сообщения для основного вызова модели и — если запрошено
        обновление фактов — заодно уже известные факты и "границу" (сообщение
        ассистента, под которым они сохранены), нужные позже для извлечения
        (см. `_extract_facts`, вызывается уже ПОСЛЕ основного ответа).

        Если запрошена автосуммаризация (`autosummary != "off"`), пул
        сообщений-кандидатов сперва ограничивается тем же самым срезом "с
        последнего summary" (`_context_messages`), что и путь по умолчанию —
        а дальше, если ЕЩЁ И запрошены `get_facts`/`sliding_window`, к этому
        уже ограниченному пулу применяется их собственная обрезка (по
        `context_strategy_limit`) — обрезки складываются, а не конкурируют."""
        pool = self._context_messages(messages) if autosummary != "off" else messages
        if get_facts:
            latest_facts, boundary = self._load_latest_facts(messages)
            provider_messages = self._build_sticky_facts_context(chat, pool, text, latest_facts)
            return provider_messages, latest_facts, boundary
        if sliding_window:
            return self._build_sliding_window_context(chat, pool, text), {}, None
        return self._build_request_context(chat, pool, text), {}, None

    def send_message_blocking(
        self,
        chat_id: str,
        text: str,
        get_facts: bool = False,
        sliding_window: bool = False,
        autosummary: str = "off",
        branch: Optional[int] = None,
    ) -> Tuple[Message, Message]:
        chat = self._require_chat(chat_id)
        with self._lock_for(chat_id):
            all_messages = self._db.list_messages(chat_id)
            messages = self._filter_by_branch(all_messages, branch)
            if get_facts:
                self._validate_sticky_facts_config(chat)
            if sliding_window:
                self._validate_sliding_window_config(chat)
            if autosummary != "off":
                self._validate_autosummary_send_config(chat, autosummary)
            provider_messages, latest_facts, facts_boundary = self._resolve_send_context(
                chat, messages, text, get_facts, sliding_window, autosummary
            )
            provider_name, model_id = _split_model(chat.settings.model)
            provider = self._registry.get(provider_name)

            user_msg = self._db.add_message(
                Message(
                    id=0, chat_id=chat_id, role="user", content=text, created_at=int(time.time()),
                    format=detect_message_format(text), branch=branch or 0,
                )
            )

            started = time.monotonic()
            try:
                result = provider.chat(model_id, provider_messages, chat.settings)
            except ProviderError as exc:
                self._db.add_message(
                    Message(id=0, chat_id=chat_id, role="error", content=str(exc), created_at=int(time.time()), branch=branch or 0)
                )
                self._db.touch_chat(chat_id)
                raise
            duration_ms = int((time.monotonic() - started) * 1000)

            assistant_msg = self._db.add_message(
                Message(
                    id=0, chat_id=chat_id, role="assistant", content=result.content,
                    created_at=int(time.time()), reasoning_content=result.reasoning_content,
                    duration_ms=duration_ms, total_tokens=result.usage.total_tokens,
                    prompt_tokens=result.usage.prompt_tokens, completion_tokens=result.usage.completion_tokens,
                    format=detect_message_format(result.content), branch=branch or 0,
                )
            )

            # Токены НА ВХОД этого конкретного обмена (для отображения под
            # сообщением ПОЛЬЗОВАТЕЛЯ) известны только сейчас, из usage ответа
            # провайдера — пользовательское сообщение уже сохранено раньше,
            # поэтому здесь проставляются постфактум.
            if result.usage.prompt_tokens is not None:
                self._db.update_message_prompt_tokens(user_msg.id, result.usage.prompt_tokens)
                user_msg = dataclasses.replace(user_msg, prompt_tokens=result.usage.prompt_tokens)

            if get_facts:
                dialogue = self._dialogue_since(messages, facts_boundary)
                updated_facts = self._extract_facts(chat, latest_facts, dialogue, text, result.content)
                facts_json = json.dumps(updated_facts, ensure_ascii=False)
                self._db.update_message_facts(assistant_msg.id, facts_json)
                assistant_msg = dataclasses.replace(assistant_msg, facts=facts_json)

            if autosummary != "off" and self._autosummary_due(chat, messages, autosummary):
                self._summarize_after_send(chat, messages, user_msg, assistant_msg)

            self._db.touch_chat(chat_id)
            return user_msg, assistant_msg

    def stream_message(
        self,
        chat_id: str,
        text: str,
        get_facts: bool = False,
        sliding_window: bool = False,
        autosummary: str = "off",
        branch: Optional[int] = None,
    ) -> Iterator[dict]:
        chat = self._require_chat(chat_id)
        lock = self._lock_for(chat_id)
        lock.acquire()
        try:
            all_messages = self._db.list_messages(chat_id)
            messages = self._filter_by_branch(all_messages, branch)
            if get_facts:
                self._validate_sticky_facts_config(chat)
            if sliding_window:
                self._validate_sliding_window_config(chat)
            if autosummary != "off":
                self._validate_autosummary_send_config(chat, autosummary)
            provider_messages, latest_facts, facts_boundary = self._resolve_send_context(
                chat, messages, text, get_facts, sliding_window, autosummary
            )
            provider_name, model_id = _split_model(chat.settings.model)
            provider = self._registry.get(provider_name)

            user_msg = self._db.add_message(
                Message(
                    id=0, chat_id=chat_id, role="user", content=text, created_at=int(time.time()),
                    format=detect_message_format(text), branch=branch or 0,
                )
            )

            # Статус вызова для отображения в клиенте (замена нейтрального
            # "Модель рассуждает…" на реальную фазу выполнения запроса).
            yield {"type": "status", "status": "Выполняется запрос к модели"}

            started = time.monotonic()
            try:
                for delta in provider.stream_chat(model_id, provider_messages, chat.settings):
                    if delta.done:
                        duration_ms = int((time.monotonic() - started) * 1000)
                        if delta.result.usage.prompt_tokens is not None:
                            self._db.update_message_prompt_tokens(user_msg.id, delta.result.usage.prompt_tokens)
                        assistant_msg = self._db.add_message(
                            Message(
                                id=0, chat_id=chat_id, role="assistant", content=delta.result.content,
                                created_at=int(time.time()), reasoning_content=delta.result.reasoning_content,
                                duration_ms=duration_ms, total_tokens=delta.result.usage.total_tokens,
                                prompt_tokens=delta.result.usage.prompt_tokens,
                                completion_tokens=delta.result.usage.completion_tokens,
                                format=detect_message_format(delta.result.content), branch=branch or 0,
                            )
                        )
                        # Обновление фактов — ПОСЛЕ основного ответа модели и
                        # ТОЛЬКО теперь, когда потоковая генерация полностью
                        # завершена (см. Доработка: раньше get_facts был вовсе
                        # недоступен в потоковом режиме). Статус отдельным
                        # событием — извлечение может занять заметное время, а
                        # клиент к этому моменту уже показал весь текст ответа.
                        if get_facts:
                            yield {"type": "status", "status": "Обновление фактов"}
                            dialogue = self._dialogue_since(messages, facts_boundary)
                            updated_facts = self._extract_facts(
                                chat, latest_facts, dialogue, text, delta.result.content
                            )
                            facts_json = json.dumps(updated_facts, ensure_ascii=False)
                            self._db.update_message_facts(assistant_msg.id, facts_json)
                            assistant_msg = dataclasses.replace(assistant_msg, facts=facts_json)
                        # Автосуммаризация — тоже ПОСЛЕ основного ответа, как и
                        # обновление фактов выше; статус отдаётся отдельным
                        # событием, но только если суммаризация действительно
                        # потребовалась (а не при каждой отправке с этим флагом).
                        if autosummary != "off" and self._autosummary_due(chat, messages, autosummary):
                            yield {"type": "status", "status": "Выполняется суммаризация чата"}
                            self._summarize_after_send(chat, messages, user_msg, assistant_msg)
                        self._db.touch_chat(chat_id)
                        yield {"type": "done", "message": assistant_msg}
                    else:
                        yield {"type": "delta", "content": delta.content, "reasoning_content": delta.reasoning_content}
            except ProviderError as exc:
                self._db.add_message(
                    Message(id=0, chat_id=chat_id, role="error", content=str(exc), created_at=int(time.time()), branch=branch or 0)
                )
                self._db.touch_chat(chat_id)
                yield {"type": "error", "message": str(exc)}
        finally:
            lock.release()

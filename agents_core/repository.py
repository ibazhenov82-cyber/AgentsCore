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
import sqlite3
import threading
import time
from typing import Dict, Iterator, List, Optional, Tuple

from .catalog import ModelCatalog
from .config import AgentConfig
from .db import Database
from .format_detect import detect_message_format
from .invariant_checks import format_violation_warning, validate_response
from .mcp_client import MCPClient, MCPClientError
from .models import (
    Agent,
    AUTOSUMMARY_OPTIONS,
    Branch,
    Chat,
    CONTEXT_STRATEGY_OPTIONS,
    DefaultSettings,
    Invariant,
    INVARIANT_KIND_LABELS,
    INVARIANT_KIND_OPTIONS,
    LONG_TERM_MEMORY_CATEGORIES,
    LONG_TERM_MEMORY_CATEGORY_ENABLE_FIELD,
    LONG_TERM_MEMORY_CORE_CATEGORIES,
    LONG_TERM_MEMORY_EXTENDED_CATEGORIES,
    LongTermMemoryEntry,
    Message,
    ModelInfo,
    Profile,
    Settings,
    Task,
    TASK_APPLIED_BY_OPTIONS,
    TaskTransitionLog,
    WorkingMemoryEntry,
    settings_from_defaults,
)
from .providers import ChatResult, ProviderError, ProviderMessage, ProviderRegistry
from .skills import registry as skills_registry
from .skills import shopping_demo
from .task_state_machine import (
    allowed_transitions,
    can_transition,
    format_states_for_prompt,
    TASK_STATE_ORDER,
    TaskState as TaskStateEnum,
    display_name as task_state_display_name,
    parse_task_state,
)
from .tokens import estimate_messages_tokens

_SETTINGS_FIELD_NAMES = {f.name for f in dataclasses.fields(Settings)}
_DEFAULT_SETTINGS_FIELD_NAMES = {f.name for f in dataclasses.fields(DefaultSettings)}

#: Максимум "туда-обратно" вызовов модели в одном обмене tool-calling —
#: защита от зацикливания (модель бесконечно вызывает инструменты вместо
#: финального текстового ответа). После достижения предела последний
#: результат провайдера возвращается как есть, даже если в нём снова
#: запрошены tool_calls.
_MAX_TOOL_ITERATIONS = 4

# ---------------------------------------------------------------------------
# Встроенные функции памяти (доступны всем чатам с memory_tools_enabled=true,
# независимо от активного профиля) — см. итоговый документ, разделы 2 и 3.
# ---------------------------------------------------------------------------

_SAVE_WORKING_MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "save_working_memory",
        "description": (
            "Сохранить факт в РАБОЧУЮ память текущего чата (данные текущей задачи; "
            "видны только внутри этого чата, не переносятся в другие чаты этого агента)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Короткий ключ факта, например 'текущая_цель'"},
                "value": {"type": "string", "description": "Значение факта"},
            },
            "required": ["key", "value"],
        },
    },
}

_LONG_TERM_CATEGORY_HINTS = {
    "profile": "'profile' — факт о пользователе/предпочтение",
    "decision": "'decision' — принятое решение/договорённость",
    "knowledge": "'knowledge' — прочее полезное знание",
    "episodic": "'episodic' — конкретный прошлый эпизод/событие",
    "semantic": "'semantic' — обобщённое устойчивое знание",
    "procedural": "'procedural' — как выполнять задачу/процесс",
}


def _build_save_long_term_memory_tool(categories: List[str]) -> dict:
    """`save_long_term_memory` строится ДИНАМИЧЕСКИ на каждый запрос — `enum`
    категории сужается до реально включённых сейчас типов памяти (см.
    `Repository._enabled_long_term_categories`), чтобы модель не пыталась
    сохранить факт в отключённый пользователем тип."""
    return {
        "type": "function",
        "function": {
            "name": "save_long_term_memory",
            "description": (
                "Сохранить факт в ДОЛГОВРЕМЕННУЮ память агента (переживает текущий чат — "
                "будет виден и в других чатах этого же агента)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": categories,
                        "description": "; ".join(_LONG_TERM_CATEGORY_HINTS[c] for c in categories),
                    },
                    "key": {"type": "string", "description": "Короткий ключ факта"},
                    "value": {"type": "string", "description": "Значение факта"},
                },
                "required": ["category", "key", "value"],
            },
        },
    }

# ---------------------------------------------------------------------------
# "Работу с задачами требуется переделать" (новое ТЗ) — встроенные функции
# start_task/apply_task_action, доступные чатам с task_tracking_enabled=true.
# Машина состояний теперь ЕДИНАЯ и задана в коде (`task_state_machine.py`) —
# в отличие от прежней версии, здесь больше нет выбора модели состояний
# (`machine_system_name`) и явного действия "Отклонить"; вместо системного
# имени действия модель называет ЦЕЛЕВОЙ ЭТАП (`target_state`), а enum
# сужается на каждый запрос до состояний, реально достижимых из текущего
# этапа хотя бы одной из открытых задач чата (см. `Repository._settings_with_merged_tools`).
# ---------------------------------------------------------------------------

def _build_start_task_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "start_task",
            "description": (
                "Начать отслеживание НОВОЙ задачи в этом чате (см. «Состояние задачи»). "
                "Используй, когда в сообщении пользователя выделяется отдельная задача, "
                "которую стоит вести по этапам — не для каждой реплики. НЕ жди прямой просьбы "
                "пользователя отслеживать задачу или явной команды вроде «возьми в работу» — "
                "начинай сам, если задача очевидна по смыслу сообщения. В чате может быть "
                "несколько открытых задач одновременно. Задача всегда начинается с этапа "
                "«Планирование»."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Короткая формулировка задачи"},
                    "plan": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Согласованный план — список шагов до выполнения задачи (PLAN). "
                            "Можно оставить пустым и задать позже через apply_task_action."
                        ),
                    },
                },
                "required": ["title"],
            },
        },
    }


def _build_apply_task_action_tool(target_state_options: List[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "apply_task_action",
            "description": (
                "Обновить прогресс уже начатой задачи этого чата — продвинуть этап "
                "(в т.ч. в рамках автономного продолжения работы «Менеджером задач», без нового "
                "сообщения пользователя — см. системную реплику «Продолжай самостоятельно "
                "работать над задачей»), и/или обновить текущий шаг (CURRENT) и список "
                "выполненных шагов (DONE). Вызывай сам, как только по смыслу переписки (или "
                "собственного предыдущего ответа) этап можно считать пройденным — не жди, чтобы "
                "пользователь явно попросил «отправляй дальше»/«на проверку» и т.п. task_id "
                "ВСЕГДА обязателен (бери из блока [ЗАДАЧА] этого чата — открытых задач может быть "
                "несколько одновременно, неоднозначность недопустима)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Id задачи из блока [ЗАДАЧА] этого чата"},
                    "target_state": {
                        "type": "string",
                        "enum": target_state_options,
                        "description": (
                            "Новый этап (STATE) — должен быть доступен из ТЕКУЩЕГО этапа именно "
                            "этой задачи; можно не указывать, если этап не меняется, а обновляются "
                            "только current_step/completed_step."
                        ),
                    },
                    "current_step": {
                        "type": "string",
                        "description": "Короткое описание текущего шага (CURRENT), например 'сбор данных по региону EMEA'",
                    },
                    "completed_step": {
                        "type": "string",
                        "description": "Если предыдущий шаг завершён — его краткое описание; будет добавлено в список DONE",
                    },
                    "note": {"type": "string", "description": "Необязательный комментарий к переходу"},
                },
                "required": ["task_id"],
            },
        },
    }


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


_TASK_STATES_TAG_RE = re.compile(r"<task_states>.*?</task_states>", re.DOTALL)
_TASK_STATE_MACHINE_INVARIANTS_TAG_RE = re.compile(r"<task_state_machine_invariants>.*?</task_state_machine_invariants>", re.DOTALL)


def _fill_task_tracking_template(template: str, task_states_text: str, invariants_text: str) -> str:
    """Подставляет заполненные значения ВМЕСТО плейсхолдеров
    `<task_states></task_states>`/`<task_state_machine_invariants></task_state_machine_invariants>`
    целиком (тег вместе с содержимым заменяется на голый текст — по примеру
    в ТЗ итоговый prompt содержит "TaskState: 1 - PLANNING (...), ...", а не
    "TaskState: <task_states>1 - PLANNING (...), ...</task_states>"). Если
    шаблон был отредактирован пользователем и каких-то тегов в нём нет —
    возвращает как есть в этой части (тот же принцип, что и
    `_fill_summary_template`)."""
    result = _TASK_STATES_TAG_RE.sub(lambda _m: task_states_text, template, count=1)
    result = _TASK_STATE_MACHINE_INVARIANTS_TAG_RE.sub(lambda _m: invariants_text, result, count=1)
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
    def __init__(
        self, db: Database, registry: ProviderRegistry, catalog: ModelCatalog,
        mcp_client: Optional[MCPClient] = None,
    ):
        self._db = db
        self._registry = registry
        self._catalog = catalog
        self._locks: Dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        #: Клиент отдельного MCP-сервера (новое ТЗ, третий компонент) —
        #: `None` по умолчанию (полностью опционально, обратная
        #: совместимость: существующие вызовы `Repository(db, registry,
        #: catalog)` продолжают работать без изменений и без сетевого
        #: похода куда-либо за инструментами). См. `_settings_with_merged_tools`
        #: и `_execute_tool_call` — единственные два места, где он используется.
        self._mcp_client = mcp_client

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
        # Профиль по умолчанию для агента (см. Agent.default_profile_id)
        # копируется в новый чат ТОЛЬКО в момент создания — точно так же, как
        # Settings копируются из agent.settings, а не как живая ссылка;
        # дальше чат может выбрать другой профиль независимо от агента.
        return self._db.create_chat(agent_id, resolved_title, settings, active_profile_id=agent.default_profile_id)

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
        self, chat: Chat, agent: Agent, messages: List[Message], new_text: str
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
        provider_messages += self._build_memory_injection_messages(chat, agent)
        provider_messages += self._build_task_and_invariant_context(chat, agent)
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

    # ---- память (working_memory / long_term_memory) -------------------------

    def _require_working_memory_entry(self, chat_id: str, key: str) -> WorkingMemoryEntry:
        entry = self._db.get_working_memory(chat_id, key)
        if entry is None:
            raise NotFoundError(f"no such working memory key: {key!r}")
        return entry

    def list_working_memory(self, chat_id: str) -> List[WorkingMemoryEntry]:
        self._require_chat(chat_id)
        return self._db.list_working_memory(chat_id)

    def save_working_memory(self, chat_id: str, key: str, value: str, source: str = "manual") -> WorkingMemoryEntry:
        """Ручное сохранение (из формы/API) всегда доступно, независимо от
        настройки `memory_tools_enabled` — она ограничивает только
        САМОСТОЯТЕЛЬНОЕ сохранение агентом через tool-calling."""
        self._require_chat(chat_id)
        if not key or not key.strip():
            raise ValidationError("key must be a non-empty string")
        return self._db.upsert_working_memory(chat_id, key.strip(), value, source=source)

    def delete_working_memory(self, chat_id: str, key: str) -> None:
        self._require_chat(chat_id)
        self._require_working_memory_entry(chat_id, key)
        self._db.delete_working_memory(chat_id, key)

    def list_long_term_memory(self, agent_id: str, category: Optional[str] = None) -> List[LongTermMemoryEntry]:
        self._require_agent(agent_id)
        if category is not None and category not in LONG_TERM_MEMORY_CATEGORIES:
            raise ValidationError(f"invalid category: {category!r}")
        return self._db.list_long_term_memory(agent_id, category)

    def save_long_term_memory(self, agent_id: str, category: str, key: str, value: str, source: str = "manual") -> LongTermMemoryEntry:
        self._require_agent(agent_id)
        if category not in LONG_TERM_MEMORY_CATEGORIES:
            raise ValidationError(f"invalid category: {category!r}")
        if not key or not key.strip():
            raise ValidationError("key must be a non-empty string")
        return self._db.upsert_long_term_memory(agent_id, category, key.strip(), value, source=source)

    def delete_long_term_memory(self, agent_id: str, category: str, key: str) -> None:
        self._require_agent(agent_id)
        if category not in LONG_TERM_MEMORY_CATEGORIES:
            raise ValidationError(f"invalid category: {category!r}")
        entries = self._db.list_long_term_memory(agent_id, category)
        if not any(e.key == key for e in entries):
            raise NotFoundError(f"no such long-term memory key: {key!r}")
        self._db.delete_long_term_memory(agent_id, category, key)

    # ---- профили-пайплайны ----------------------------------------------------

    def _require_profile(self, profile_id: str) -> Profile:
        profile = self._db.get_profile(profile_id)
        if profile is None:
            raise NotFoundError(f"no such profile: {profile_id}")
        return profile

    def list_profiles(self) -> List[Profile]:
        """Общий справочник профилей — один список для ВСЕХ агентов (по
        замечанию пользователя: профиль описывается один раз и выбирается в
        настройках любого агента/чата, а не создаётся заново под каждого)."""
        return self._db.list_profiles()

    def get_profile(self, profile_id: str) -> Profile:
        return self._require_profile(profile_id)

    @staticmethod
    def _validate_skills_json(skills_json: str) -> None:
        if not skills_json.strip():
            return
        try:
            parsed = json.loads(skills_json)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"skills_json is not valid JSON: {exc}") from exc
        if not isinstance(parsed, list):
            raise ValidationError("skills_json must be a JSON array of OpenAI function-tool descriptions")

    @staticmethod
    def _resolve_skill_names(skill_names: List[str]) -> str:
        """Переводит имена ЗАРЕГИСТРИРОВАННЫХ скиллов (см. `skills.registry`,
        ведётся через переменную окружения AGENT_REGISTERED_SKILLS) в готовый
        `skills_json` — так UI/API может привязать скилл к профилю по имени,
        не заставляя пользователя вручную писать JSON-схему функции."""
        resolved = []
        unknown = []
        for name in skill_names:
            skill = skills_registry.get_registered_skill(name)
            if skill is None:
                unknown.append(name)
            else:
                resolved.append(skill)
        if unknown:
            raise ValidationError(f"unknown registered skill(s): {', '.join(unknown)}")
        return json.dumps(resolved, ensure_ascii=False)

    def list_registered_skills(self) -> List[dict]:
        """Плоский список скиллов, зарегистрированных сервисом (переменная
        окружения AGENT_REGISTERED_SKILLS) — из них пользователь выбирает
        подмножество при создании/редактировании профиля (`skill_names`)."""
        return skills_registry.list_registered_skills()

    def create_profile(
        self, name: str, style: Optional[str] = None, format: Optional[str] = None,
        constraints: Optional[str] = None, skills_json: str = "", orchestration_prompt: Optional[str] = None,
        skill_names: Optional[List[str]] = None,
    ) -> Profile:
        if not name or not name.strip():
            raise ValidationError("name must be a non-empty string")
        if skill_names is not None:
            if skills_json.strip():
                raise ValidationError("provide either skill_names or skills_json, not both")
            skills_json = self._resolve_skill_names(skill_names)
        else:
            self._validate_skills_json(skills_json)
        return self._db.create_profile(name.strip(), style, format, constraints, skills_json, orchestration_prompt)

    def update_profile(self, profile_id: str, payload: dict) -> Profile:
        self._require_profile(profile_id)
        allowed = {"name", "style", "format", "constraints", "skills_json", "orchestration_prompt", "skill_names"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValidationError(f"unknown profile field(s): {', '.join(sorted(unknown))}")
        if "name" in payload and (not payload["name"] or not payload["name"].strip()):
            raise ValidationError("name must be a non-empty string")
        payload = dict(payload)
        if "skill_names" in payload:
            skill_names = payload.pop("skill_names")
            if payload.get("skills_json", "").strip():
                raise ValidationError("provide either skill_names or skills_json, not both")
            payload["skills_json"] = self._resolve_skill_names(skill_names)
        elif "skills_json" in payload:
            self._validate_skills_json(payload["skills_json"])
        self._db.update_profile(profile_id, payload)
        return self._require_profile(profile_id)

    def delete_profile(self, profile_id: str) -> None:
        self._require_profile(profile_id)
        self._db.delete_profile(profile_id)  # у чатов active_profile_id снимается через ON DELETE SET NULL

    def set_chat_active_profile(self, chat_id: str, profile_id: Optional[str]) -> Chat:
        chat = self._require_chat(chat_id)
        if profile_id is not None:
            self._require_profile(profile_id)  # профиль общий — подходит любому агенту/чату
        self._db.set_chat_active_profile(chat_id, profile_id)
        return self._require_chat(chat_id)

    def set_agent_default_profile(self, agent_id: str, profile_id: Optional[str]) -> Agent:
        """Профиль ПО УМОЛЧАНИЮ для агента: не применяется задним числом к
        уже существующим чатам (у каждого своя, независимая настройка через
        `set_chat_active_profile`) — только копируется в НОВЫЕ чаты этого
        агента при создании (см. `create_chat`)."""
        self._require_agent(agent_id)
        if profile_id is not None:
            self._require_profile(profile_id)
        self._db.set_agent_default_profile(agent_id, profile_id)
        return self._require_agent(agent_id)

    # ---- инварианты ("День 14", общий справочник для ВСЕХ агентов) ---------

    def _require_invariant(self, invariant_id: str) -> Invariant:
        invariant = self._db.get_invariant(invariant_id)
        if invariant is None:
            raise NotFoundError(f"no such invariant: {invariant_id}")
        return invariant

    def list_invariants(self) -> List[Invariant]:
        """Общий справочник инвариантов — один список для ВСЕХ агентов, по
        аналогии с `list_profiles` (заводится и редактируется по образцу
        профилей, а не привязывается к одному чату/агенту при создании)."""
        return self._db.list_invariants()

    def get_invariant(self, invariant_id: str) -> Invariant:
        return self._require_invariant(invariant_id)

    @staticmethod
    def _validate_invariant_kind(kind: Optional[str]) -> None:
        if kind is not None and kind not in INVARIANT_KIND_OPTIONS:
            raise ValidationError(f"invalid invariant kind: {kind!r}")

    def create_invariant(
        self, title: str, rule_text: str, kind: Optional[str] = None, is_active: bool = True,
    ) -> Invariant:
        if not title or not title.strip():
            raise ValidationError("title must be a non-empty string")
        if not rule_text or not rule_text.strip():
            raise ValidationError("rule_text must be a non-empty string")
        self._validate_invariant_kind(kind)
        return self._db.create_invariant(title.strip(), rule_text.strip(), kind=kind, is_active=is_active)

    def update_invariant(self, invariant_id: str, payload: dict) -> Invariant:
        self._require_invariant(invariant_id)
        allowed = {"title", "rule_text", "kind", "is_active"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValidationError(f"unknown invariant field(s): {', '.join(sorted(unknown))}")
        if "title" in payload and (not payload["title"] or not payload["title"].strip()):
            raise ValidationError("title must be a non-empty string")
        if "rule_text" in payload and (not payload["rule_text"] or not payload["rule_text"].strip()):
            raise ValidationError("rule_text must be a non-empty string")
        if "kind" in payload:
            self._validate_invariant_kind(payload["kind"])
        self._db.update_invariant(invariant_id, payload)
        return self._require_invariant(invariant_id)

    def delete_invariant(self, invariant_id: str) -> None:
        self._require_invariant(invariant_id)
        # Выбор этого инварианта снимается у ВСЕХ агентов/чатов, где он был
        # выбран — см. `Database.delete_invariant` (в отличие от профилей,
        # это не FK со SET NULL, а обычный JSON-массив, чистим вручную).
        self._db.delete_invariant(invariant_id)

    def set_agent_invariants(self, agent_id: str, invariant_ids: List[str]) -> Agent:
        """Множественный выбор инвариантов для агента — действует во всех
        его чатах (см. `Repository._effective_invariant_ids`), по аналогии с
        тем, как `default_profile_id` выбирается из общего справочника, но
        здесь выбор МНОЖЕСТВЕННЫЙ и это не разовый снимок для новых чатов, а
        живая настройка самого агента (по замечанию пользователя — как у
        профилей: общий справочник + множественный выбор в настройках)."""
        self._require_agent(agent_id)
        deduped = list(dict.fromkeys(invariant_ids))
        for invariant_id in deduped:
            self._require_invariant(invariant_id)
        self._db.set_agent_invariants(agent_id, deduped)
        return self._require_agent(agent_id)

    def set_chat_invariants(self, chat_id: str, invariant_ids: List[str]) -> Chat:
        """То же самое, но уровня чата — итоговый набор для чата является
        объединением этого списка и `Agent.invariant_ids` его агента."""
        self._require_chat(chat_id)
        deduped = list(dict.fromkeys(invariant_ids))
        for invariant_id in deduped:
            self._require_invariant(invariant_id)
        self._db.set_chat_invariants(chat_id, deduped)
        return self._require_chat(chat_id)

    # ---- машина состояний задач (read-only, задана в коде) -------------------
    # "Работу с задачами требуется переделать" (новое ТЗ) — каталог
    # состояний/переходов больше не в БД, см. `task_state_machine.py`; здесь
    # остаётся только read-only описание для Android-экрана (замена формы
    # редактирования состояний/действий/машин) и настройка привязанных к
    # машине инвариантов категории "Правило стейт-машины".

    def get_task_state_machine_info(self) -> dict:
        """Read-only описание единственной (заданной в коде) машины
        состояний — номер, отображаемое и системное имя, список достижимых
        состояний для каждого этапа — плюс текущий список привязанных
        инвариантов категории "Правило стейт-машины" (см.
        `set_task_machine_invariants`)."""
        states = [
            {
                "position": index,
                "state": state.value,
                "display_name": task_state_display_name(state),
                "target_states": [s.value for s in allowed_transitions(state)],
                "target_state_display_names": [task_state_display_name(s) for s in allowed_transitions(state)],
            }
            for index, state in enumerate(TASK_STATE_ORDER, start=1)
        ]
        invariant_ids = set(self._db.get_task_machine_invariant_ids())
        invariants = [inv for inv in self._db.list_invariants() if inv.id in invariant_ids]
        return {"states": states, "invariants": invariants}

    def set_task_machine_invariants(self, invariant_ids: List[str]) -> dict:
        """Заменяет весь список привязанных к машине состояний инвариантов —
        только категории "Правило стейт-машины" (по аналогии с
        `set_agent_invariants`/`set_chat_invariants`, но без объединения:
        здесь ровно один список на всю систему, см.
        `Database.set_task_machine_invariant_ids`)."""
        deduped = list(dict.fromkeys(invariant_ids))
        for invariant_id in deduped:
            invariant = self._require_invariant(invariant_id)
            if invariant.kind != "state_machine_rule":
                raise ValidationError(
                    f"invariant {invariant_id!r} has kind {invariant.kind!r}, expected 'state_machine_rule'"
                )
        self._db.set_task_machine_invariant_ids(deduped)
        return self.get_task_state_machine_info()

    # ---- задачи (чтение/агрегирование, изменение — см. tool-calling ниже) ----

    _TASK_STATUS_LABELS = {"active": "активна", "paused": "на паузе", "done": "завершена"}

    def _require_task(self, task_id: str) -> Task:
        task = self._db.get_task(task_id)
        if task is None:
            raise NotFoundError(f"no such task: {task_id}")
        return task

    def _task_status(self, task: Task) -> str:
        """"active" | "paused" | "done" — см. `TASK_STATUS_OPTIONS`. Заметно
        проще прежней версии: "done" прямо по значению `task.state`, "paused"
        — прямо по флагу `task.paused` (ортогональному состоянию, см.
        докстринг `task_state_machine`), без обращения к истории переходов."""
        if task.state == TaskStateEnum.DONE.value:
            return "done"
        return "paused" if task.paused else "active"

    def _task_next_state_display_name(self, task: Task) -> Optional[str]:
        """Этап, в который ведёт "Продолжить"/"Выполнить" из текущего этапа —
        для карточки-подтверждения в чате и списка задач. Если из текущего
        этапа возможно несколько целей (см. `TASK_TRANSITIONS` — например,
        VALIDATION может вернуться в EXECUTION), берётся ПЕРВАЯ по порядку —
        она всегда прямое продолжение вперёд (порядок задан примером ТЗ:
        `EXECUTION to listOf(VALIDATION, PLANNING)` — VALIDATION первым).
        `None`, если задача уже завершена."""
        targets = allowed_transitions(parse_task_state(task.state))
        return task_state_display_name(targets[0]) if targets else None

    def _task_summary(self, task: Task, chat_title: Optional[str] = None) -> dict:
        status = self._task_status(task)
        summary = {
            "task": task,
            "status": status,
            "status_display": self._TASK_STATUS_LABELS.get(status, status),
            "state_display_name": task_state_display_name(parse_task_state(task.state)),
            "next_state_display_name": self._task_next_state_display_name(task) if status != "done" else None,
        }
        if chat_title is not None:
            summary["chat_title"] = chat_title
        return summary

    def list_tasks_for_chat(self, chat_id: str, include_completed: bool = False) -> List[dict]:
        self._require_chat(chat_id)
        tasks = self._db.list_tasks_for_chat(chat_id)
        summaries = [self._task_summary(t) for t in tasks]
        if not include_completed:
            summaries = [s for s in summaries if s["status"] != "done"]
        return summaries

    def list_tasks_for_agent(self, agent_id: str, include_completed: bool = False) -> List[dict]:
        """Агрегированный список задач по ВСЕМ чатам агента — блок "Задачи"
        на карточке агента, с указанием родительского чата у каждой задачи
        (по замечанию пользователя)."""
        self._require_agent(agent_id)
        tasks = self._db.list_tasks_for_agent(agent_id)
        chat_titles: Dict[str, str] = {}
        summaries = []
        for task in tasks:
            if task.chat_id not in chat_titles:
                chat = self._db.get_chat(task.chat_id)
                chat_titles[task.chat_id] = chat.title if chat is not None else task.chat_id
            summaries.append(self._task_summary(task, chat_title=chat_titles[task.chat_id]))
        if not include_completed:
            summaries = [s for s in summaries if s["status"] != "done"]
        return summaries

    def get_task(self, task_id: str) -> dict:
        """Полные детали задачи для экрана "Задача": вычисляемый статус,
        степпер по фиксированным четырём этапам (иконка: "check" — для уже
        пройденных, "pause" — для текущего, если задача на паузе, "none" —
        для ещё не достигнутых), доступные действия (продвижение в каждое из
        достижимых состояний + пауза, если задача сейчас не на паузе) и
        полная история переходов."""
        task = self._require_task(task_id)
        status = self._task_status(task)
        current_state = parse_task_state(task.state)
        current_index = TASK_STATE_ORDER.index(current_state)

        stages = []
        for index, state in enumerate(TASK_STATE_ORDER):
            is_current = state == current_state
            if is_current and status == "paused":
                icon = "pause"
            elif index <= current_index:
                icon = "check"
            else:
                icon = "none"
            stages.append({
                "state": state.value,
                "display_name": task_state_display_name(state),
                "is_current": is_current,
                "is_final": state == TaskStateEnum.DONE,
                "icon": icon,
            })

        available_actions = []
        if status != "done":
            for target in allowed_transitions(current_state):
                available_actions.append({
                    "kind": "advance",
                    "to_state": target.value,
                    "to_state_display_name": task_state_display_name(target),
                })
            if not task.paused:
                available_actions.append({
                    "kind": "pause", "to_state": current_state.value,
                    "to_state_display_name": task_state_display_name(current_state),
                })

        def _state_display(value: str) -> str:
            try:
                return task_state_display_name(parse_task_state(value))
            except ValueError:
                return value

        history = [
            {
                "id": log.id,
                "from_state": log.from_state,
                "from_state_display_name": _state_display(log.from_state),
                "to_state": log.to_state,
                "to_state_display_name": _state_display(log.to_state),
                "kind": log.kind,
                "applied_by": log.applied_by,
                "note": log.note,
                "created_at": log.created_at,
            }
            for log in self._db.list_task_transition_log(task.id)
        ]

        return {
            "task": task,
            "status": status,
            "status_display": self._TASK_STATUS_LABELS.get(status, status),
            "state_display_name": task_state_display_name(current_state),
            "next_state_display_name": self._task_next_state_display_name(task) if status != "done" else None,
            "step": current_index + 1,
            "total": len(TASK_STATE_ORDER),
            "stages": stages,
            "available_actions": available_actions,
            "history": history,
        }

    def apply_manual_task_action(self, task_id: str, action: str, note: Optional[str] = None) -> dict:
        """Ручное вмешательство человека — теперь только "Пауза" (кнопка
        "Продолжить"/"Выполнить" ВСЕГДА обращается к модели, см.
        `run_task_manager_step`; явного действия "Отклонить" в новой модели
        нет вовсе, см. `task_state_machine.py`)."""
        task = self._require_task(task_id)
        if action != "pause":
            raise ValidationError(f"unsupported manual action: {action!r} (only 'pause' is supported)")
        if task.state == TaskStateEnum.DONE.value:
            raise ValidationError("cannot pause a task that is already done")
        if task.paused:
            raise ValidationError("task is already paused")
        self._db.set_task_paused(task.id, True)
        self._db.add_task_transition_log(task.id, task.state, task.state, kind="pause", applied_by="manual", note=note)
        return self.get_task(task.id)

    def delete_task(self, task_id: str) -> None:
        self._require_task(task_id)
        self._db.delete_task(task_id)

    # ---- единый механизм tool-calling (память + скиллы профиля) -------------
    #
    # Реестр "имя функции -> обработчик" — общий для встроенных функций
    # памяти (доступны всегда, если включён тумблер memory_tools_enabled) и
    # для доменных скиллов активного профиля (доступны, только пока этот
    # профиль подключён к чату). Схема функции (что видит модель в `tools`)
    # и реализация обработчика здесь сознательно разделены: `tools_json`
    # профиля можно отредактировать (например, поменять description), но
    # ЧТО РЕАЛЬНО ДЕЛАЕТ функция — фиксировано в коде (см. "Принятые по
    # умолчанию решения": скиллы фиксированные, не производятся из
    # произвольного пользовательского JSON).

    @staticmethod
    def _enabled_long_term_categories(settings: Settings) -> List[str]:
        """Основные категории (profile/decision/knowledge) включены разом
        флагом `long_term_memory_enabled`; каждая расширенная категория
        (episodic/semantic/procedural) — своим отдельным флагом, независимо
        от основных и друг от друга (см. `models.LONG_TERM_MEMORY_CATEGORY_ENABLE_FIELD`)."""
        enabled = list(LONG_TERM_MEMORY_CORE_CATEGORIES) if settings.long_term_memory_enabled else []
        enabled += [c for c in LONG_TERM_MEMORY_EXTENDED_CATEGORIES if getattr(settings, LONG_TERM_MEMORY_CATEGORY_ENABLE_FIELD[c])]
        return enabled

    def _handle_save_working_memory(self, chat: Chat, agent: Agent, arguments: dict, auto_pause: bool = True, events: Optional[List[dict]] = None) -> dict:
        if not chat.settings.working_memory_enabled:
            return {"error": "working memory is disabled for this chat"}
        key = str(arguments.get("key") or "").strip()
        if not key:
            return {"error": "key is required"}
        value = arguments.get("value")
        entry = self._db.upsert_working_memory(chat.id, key, "" if value is None else str(value), source="agent")
        return {"saved": True, "key": entry.key, "value": entry.value, "source": entry.source}

    def _handle_save_long_term_memory(self, chat: Chat, agent: Agent, arguments: dict, auto_pause: bool = True, events: Optional[List[dict]] = None) -> dict:
        category = arguments.get("category")
        enabled_categories = self._enabled_long_term_categories(chat.settings)
        if category not in enabled_categories:
            return {"error": f"category must be one of the currently enabled types: {enabled_categories}"}
        key = str(arguments.get("key") or "").strip()
        if not key:
            return {"error": "key is required"}
        value = arguments.get("value")
        entry = self._db.upsert_long_term_memory(agent.id, category, key, "" if value is None else str(value), source="agent")
        return {"saved": True, "category": entry.category, "key": entry.key, "value": entry.value, "source": entry.source}

    def _handle_search_products(self, chat: Chat, agent: Agent, arguments: dict, auto_pause: bool = True, events: Optional[List[dict]] = None) -> dict:
        return {"products": shopping_demo.search_products(arguments.get("query") or "", arguments.get("max_price"))}

    def _current_cart(self, chat_id: str) -> List[str]:
        entry = self._db.get_working_memory(chat_id, "cart")
        if entry is None:
            return []
        try:
            parsed = json.loads(entry.value)
        except (ValueError, TypeError):
            return []
        return parsed if isinstance(parsed, list) else []

    def _handle_add_to_cart(self, chat: Chat, agent: Agent, arguments: dict, auto_pause: bool = True, events: Optional[List[dict]] = None) -> dict:
        product_id = arguments.get("product_id")
        if not product_id:
            return {"error": "product_id is required"}
        try:
            new_cart, view = shopping_demo.add_to_cart(self._current_cart(chat.id), product_id)
        except shopping_demo.ShoppingError as exc:
            return {"error": str(exc)}
        self._db.upsert_working_memory(chat.id, "cart", json.dumps(new_cart, ensure_ascii=False), source="agent")
        return view

    def _handle_view_cart(self, chat: Chat, agent: Agent, arguments: dict, auto_pause: bool = True, events: Optional[List[dict]] = None) -> dict:
        return shopping_demo.view_cart(self._current_cart(chat.id))

    # ---- start_task / apply_task_action (tool-calling) -----------------------

    def _open_tasks_for_chat(self, chat: Chat) -> List[Task]:
        return [t for t in self._db.list_tasks_for_chat(chat.id) if self._task_status(t) != "done"]

    def _task_allowed_target_states(self, task: Task) -> List["TaskStateEnum"]:
        return allowed_transitions(parse_task_state(task.state))

    def _handle_start_task(self, chat: Chat, agent: Agent, arguments: dict, auto_pause: bool = True, events: Optional[List[dict]] = None) -> dict:
        if not chat.settings.task_tracking_enabled:
            return {"error": "task tracking is disabled for this chat"}
        title = str(arguments.get("title") or "").strip()
        if not title:
            return {"error": "title is required"}
        plan_raw = arguments.get("plan")
        plan = [str(s).strip() for s in plan_raw if str(s).strip()] if isinstance(plan_raw, list) else []
        task = self._db.create_task(chat.id, title, plan=plan)
        state_name = task_state_display_name(TaskStateEnum.PLANNING)
        # Редизайн "Менеджера задач" (замечание пользователя, перенесено из
        # предыдущей версии): прежде чем приступать к планированию/
        # выполнению/проверке, задача ВСЕГДА показывается пользователю и
        # ставится на паузу — независимо от `auto_pause` (он относится
        # только к ПРОДВИЖЕНИЮ уже начатой задачи, см. `_handle_apply_task_action`).
        self._db.set_task_paused(task.id, True)
        self._db.add_task_transition_log(
            task.id, TaskStateEnum.PLANNING.value, TaskStateEnum.PLANNING.value,
            kind="pause", applied_by="system", note=None,
        )
        return {
            "started": True,
            "task_id": task.id,
            "title": task.title,
            "state": state_name,
            "status": "paused",
            "_event": {
                "task_id": task.id, "task_title": task.title, "from_state_display_name": None,
                "to_state_display_name": state_name, "kind": "advance",
            },
        }

    def _handle_apply_task_action(self, chat: Chat, agent: Agent, arguments: dict, auto_pause: bool = True, events: Optional[List[dict]] = None) -> dict:
        if not chat.settings.task_tracking_enabled:
            return {"error": "task tracking is disabled for this chat"}
        task_id = arguments.get("task_id")
        task = self._db.get_task(task_id) if task_id else None
        if task is None or task.chat_id != chat.id:
            return {"error": "unknown task_id for this chat"}
        current_state = parse_task_state(task.state)
        if current_state == TaskStateEnum.DONE:
            return {"error": "task is already done"}

        target_state_raw = arguments.get("target_state")
        to_state = current_state
        if target_state_raw:
            try:
                to_state = parse_task_state(str(target_state_raw))
            except ValueError:
                return {"error": f"unknown target_state: {target_state_raw!r}"}
            if not can_transition(current_state, to_state):
                options = sorted(s.value for s in allowed_transitions(current_state))
                return {"error": f"transition {current_state.value} -> {to_state.value} is not allowed; available: {options}"}

        # Редизайн "Менеджера задач": пока auto_pause=true (обычный чат и
        # кнопка "Продолжить"), одна и та же задача может быть продвинута
        # (сменить этап) не более ОДНОГО раза за один ответ модели — иначе
        # пользователь не успеет увидеть промежуточный этап и подтвердить
        # продолжение. В режиме "Выполнить" (auto_pause=false) это
        # ограничение не действует.
        if auto_pause and to_state != current_state and events is not None:
            if any(e.get("task_id") == task.id and e.get("kind") == "advance" for e in events):
                return {
                    "error": (
                        "task already advanced once in this turn; the task manager pauses after a "
                        "single step in this mode — stop here and wait for the user to confirm "
                        "before continuing this task further"
                    )
                }

        # Собственно "снимает паузу" из докстринга task_state_machine.py —
        # раньше нигде не было вызова `set_task_paused(..., False)` вообще
        # (пауза только ставилась, никогда не снималась программно). Снимаем
        # её именно здесь, ПЕРЕД применением перехода, а не отдельным логом
        # "resume" — пауза ортогональна графу переходов (см. task_state_machine.py),
        # поэтому её снятие не требует отдельной записи в истории: она и так
        # видна по соседней записи kind="advance" (или по её отсутствию, если
        # автопауза сразу поставит новую). Обновление БЕЗ смены этапа
        # (target_state не передан) статус паузы не трогает — им явно
        # управляет только «Пауза»/продвижение.
        if to_state != current_state:
            self._db.set_task_paused(task.id, False)

        current_step_raw = arguments.get("current_step")
        completed_step = arguments.get("completed_step")
        note = arguments.get("note")
        done_steps = list(task.done_steps)
        if completed_step and str(completed_step).strip():
            done_steps.append(str(completed_step).strip())
        self._db.update_task_progress(
            task.id,
            state=to_state.value if to_state != current_state else None,
            current_step=current_step_raw if current_step_raw is not None else task.current_step,
            done_steps=done_steps if completed_step else None,
        )
        if to_state != current_state:
            self._db.add_task_transition_log(
                task.id, current_state.value, to_state.value, kind="advance", applied_by="agent", note=note,
            )
        # Автопауза сразу после продвижения (auto_pause=true, этап сменился,
        # новый этап не конечный) — тот же приём, что и при создании задачи.
        if auto_pause and to_state != current_state and to_state != TaskStateEnum.DONE:
            self._db.set_task_paused(task.id, True)
            self._db.add_task_transition_log(
                task.id, to_state.value, to_state.value, kind="pause", applied_by="system", note=None,
            )
        updated_task = self._db.get_task(task.id)
        status = self._task_status(updated_task) if updated_task is not None else "active"
        return {
            "applied": True,
            "task_id": task.id,
            "state": task_state_display_name(to_state),
            "status": status,
            "_event": {
                "task_id": task.id, "task_title": task.title,
                "from_state_display_name": task_state_display_name(current_state),
                "to_state_display_name": task_state_display_name(to_state),
                "kind": "advance" if to_state != current_state else "update",
            },
        }

    _TOOL_HANDLERS = {
        "save_working_memory": _handle_save_working_memory,
        "save_long_term_memory": _handle_save_long_term_memory,
        "search_products": _handle_search_products,
        "add_to_cart": _handle_add_to_cart,
        "view_cart": _handle_view_cart,
        "start_task": _handle_start_task,
        "apply_task_action": _handle_apply_task_action,
    }

    def _settings_with_merged_tools(self, chat: Chat) -> Settings:
        """Собирает итоговый `tools` для запроса к провайдеру: собственные
        функции агента/чата (`tools_json` настроек) + встроенные функции
        памяти (если включён тумблер) + скиллы активного профиля. Возвращает
        `chat.settings` без изменений, если добавлять нечего (сохраняет
        прежнее поведение — tools=None — там, где раньше ничего не менялось)."""
        tools: list = []
        if chat.settings.tools_json.strip():
            try:
                parsed = json.loads(chat.settings.tools_json)
                if isinstance(parsed, list):
                    tools.extend(parsed)
            except (ValueError, TypeError):
                pass
        if chat.settings.memory_tools_enabled:
            if chat.settings.working_memory_enabled:
                tools.append(_SAVE_WORKING_MEMORY_TOOL)
            enabled_categories = self._enabled_long_term_categories(chat.settings)
            if enabled_categories:
                tools.append(_build_save_long_term_memory_tool(enabled_categories))
        if chat.settings.task_tracking_enabled:
            tools.append(_build_start_task_tool())
            open_tasks = self._open_tasks_for_chat(chat)
            if open_tasks:
                target_states = sorted({
                    s.value for task in open_tasks for s in self._task_allowed_target_states(task)
                })
                if target_states:
                    tools.append(_build_apply_task_action_tool(target_states))
        if chat.active_profile_id:
            profile = self._db.get_profile(chat.active_profile_id)
            if profile and profile.skills_json.strip():
                try:
                    parsed = json.loads(profile.skills_json)
                    if isinstance(parsed, list):
                        tools.extend(parsed)
                except (ValueError, TypeError):
                    pass
        # Реальный баг, найденный на практике: раньше ПОЛНЫЙ список
        # инструментов живого MCP-сервера подмешивался БЕЗУСЛОВНО, при любом
        # состоянии `tools_json` этого чата — режим «Выбрать из доступных» в
        # настройках AgentsApp (сохраняет туда ТОЛЬКО отмеченные инструменты)
        # на деле НИЧЕГО не ограничивал: модель получала (и вызывала) все
        # инструменты MCP-сервера, включая `execute_git_command` (произвольная
        # git-команда над локальной рабочей копией), не выбранный в чате.
        #
        # Правило теперь (по замечанию пользователя "использование
        # инструментов должно быть ограничено настройкой"): модели доступны
        # РОВНО те MCP-инструменты, что перечислены в `tools_json` чата/агента
        # (выбором в «Выбрать из доступных» или вручную в JSON). Ничего не
        # выбрано — ни одного MCP-инструмента. Встроенные инструменты памяти/
        # задач/скиллов по-прежнему управляются своими тумблерами (выше).
        #
        # `list_tools()` всё равно вызывается (если в чате вообще что-то
        # выбрано) — он обновляет кэш `MCPClient`, по которому
        # `_execute_tool_call` понимает, что имя принадлежит MCP-серверу и
        # вызов нужно отправить туда. Сам не бросает исключение при
        # недоступности сервера (отдаёт кэш последнего успешного ответа).
        if self._mcp_client is not None and chat.settings.tools_json.strip():
            self._mcp_client.list_tools()
        if not tools:
            return chat.settings
        # Дедупликация по имени функции (первое вхождение побеждает) —
        # например, если в `tools_json` вручную вписана функция с тем же
        # именем, что и у скилла профиля: большинство провайдеров отвергают
        # запрос с повторяющимися именами функций. tools_json идёт первым и
        # потому имеет приоритет.
        seen_names: set = set()
        deduped: list = []
        for tool in tools:
            name = None
            if isinstance(tool, dict):
                name = (tool.get("function") or {}).get("name") if isinstance(tool.get("function"), dict) else None
            if name is None or name not in seen_names:
                if name is not None:
                    seen_names.add(name)
                deduped.append(tool)
        return dataclasses.replace(chat.settings, tools_json=json.dumps(deduped, ensure_ascii=False))

    def _tools_sources_for(self, settings: Settings, chat: Optional[Chat] = None) -> List[str]:
        """Список активных ИСТОЧНИКОВ инструментов текущего чата/агента —
        только для интерфейса (новое ТЗ, `SettingsOut.tools_sources`), чтобы
        клиент мог подсветить, что реально даёт эффект (например, не
        показывать раздел «Инструменты MCP» активным, если MCP выключен для
        этого агента/сервиса в целом). НЕ влияет на сам запрос к модели —
        это отдельный, чисто описательный расчёт по тем же условиям, что и
        `_settings_with_merged_tools` выше (нарочно раздельно: тот метод
        решает, что дописать в tools_json, этот — что из этого показать в UI,
        сохранять их синхронными приходится за счёт одинаковых условий, а не
        общего кода, т.к. `_settings_with_merged_tools` возвращает готовый
        `Settings`, а не список источников)."""
        sources: List[str] = []
        if settings.tools_json.strip():
            sources.append("own")
        if settings.memory_tools_enabled and (
            settings.working_memory_enabled or self._enabled_long_term_categories(settings)
        ):
            sources.append("memory")
        if settings.task_tracking_enabled:
            sources.append("task")
        if chat is not None and chat.active_profile_id:
            profile = self._db.get_profile(chat.active_profile_id)
            if profile and profile.skills_json.strip():
                sources.append("skills")
        # MCP-инструменты больше не подмешиваются автоматически (см.
        # `_settings_with_merged_tools`) — источник "mcp" активен, только если
        # среди выбранных в `tools_json` есть хотя бы один инструмент,
        # известный MCP-серверу.
        if self._mcp_client is not None and any(
            self._mcp_client.has_cached_tool(name) for name in self._offered_tool_names(settings)
        ):
            sources.append("mcp")
        return sources

    @staticmethod
    def _offered_tool_names(settings: Settings) -> set:
        """Имена функций, РЕАЛЬНО предложенных модели в этом запросе (т.е.
        итоговый `tools_json` после `_settings_with_merged_tools`) — граница,
        по которой `_execute_tool_call` разрешает диспетчеризацию на
        MCP-сервер (см. там)."""
        if not settings.tools_json.strip():
            return set()
        try:
            parsed = json.loads(settings.tools_json)
        except (ValueError, TypeError):
            return set()
        names: set = set()
        for tool in parsed if isinstance(parsed, list) else []:
            fn = tool.get("function") if isinstance(tool, dict) else None
            if isinstance(fn, dict) and fn.get("name"):
                names.add(fn["name"])
        return names

    def _execute_tool_call(
        self, chat: Chat, agent: Agent, call: dict, events: Optional[List[dict]] = None, auto_pause: bool = True,
        mcp_events: Optional[List[dict]] = None, allowed_mcp_names: Optional[set] = None,
    ) -> str:
        """Выполняет ОДИН запрошенный моделью вызов и возвращает JSON-текст —
        именно он уйдёт обратно провайдеру как содержимое tool-сообщения.
        Неизвестное имя функции или сбой обработчика не поднимают исключение
        наружу — модель получает `{"error": ...}` и может отреагировать сама
        (например, попробовать другой вызов или объяснить пользователю).

        `events` — необязательный список-аккумулятор (тот же на все итерации
        одного обмена send_message_blocking/stream_message/run_task_manager_step):
        обработчики задач (`start_task`/`apply_task_action`) кладут в свой
        результат служебный ключ `_event` — сюда он переносится и снимается с
        ответа, уходящего модели, а после завершения tool-цикла список
        сериализуется в `Message.task_events`, чтобы клиент мог показать
        переход инлайн в ленте чата (см. `Message.task_events`). Он же
        передаётся В обработчик — `_handle_apply_task_action` использует его,
        чтобы не дать модели продвинуть ОДНУ и ту же задачу больше одного
        раза за один вызов, пока `auto_pause=true` (редизайн "Менеджера
        задач", см. `_handle_apply_task_action`).

        `auto_pause` — уникальный для КАЖДОГО обмена параметр (передаётся
        сюда явно, а не хранится изменяемым атрибутом `Repository` — тот
        singleton, общий на все чаты, и хранение флага там было бы гонкой
        между параллельными запросами разных чатов): `true` для обычного
        чата и кнопки "Продолжить" (шаг + пауза), `false` — только для кнопки
        "Выполнить" (см. `run_task_manager_step`).

        `mcp_events` — отдельный от `events` аккумулятор (см.
        `Message.mcp_events`): заполняется ТОЛЬКО когда вызов на самом деле
        уходит на MCP-сервер (имя не найдено среди `_TOOL_HANDLERS`, но есть
        в последнем успешном `list_tools()` клиента) — событиями
        `{"type":"mcp_call","name","status":"started"|"finished","ok","error"}`.
        Собственные инструменты (`_TOOL_HANDLERS`) в него не попадают — они
        уже отражены (при необходимости) в `events`/`task_events`.

        `allowed_mcp_names` — вторая линия защиты для ограничения
        инструментов настройками чата (баг, найденный на практике: модель
        вызвала `execute_git_command`, не выбранный в настройках чата). Кэш
        `MCPClient` хранит ПОЛНЫЙ список инструментов сервера независимо от
        настроек конкретного чата, поэтому одной проверки
        `has_cached_tool` недостаточно: если передан этот набор (все реальные
        вызывающие места передают — см. `_offered_tool_names`), на
        MCP-сервер уходят только те имена, что были реально предложены модели
        в этом запросе; остальные получают тот же ответ, что и заведомо
        неизвестное имя. `None` — без ограничения (прежнее поведение)."""
        fn = call.get("function") or {}
        name = fn.get("name") or ""
        raw_args = fn.get("arguments") or "{}"
        try:
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            if not isinstance(arguments, dict):
                arguments = {}
        except (ValueError, TypeError):
            arguments = {}
        handler = self._TOOL_HANDLERS.get(name)
        if handler is not None:
            try:
                result = handler(self, chat, agent, arguments, auto_pause, events)
            except Exception as exc:  # сбой одного скилла не должен ронять весь запрос
                result = {"error": str(exc)}
        elif (
            self._mcp_client is not None
            and self._mcp_client.has_cached_tool(name)
            and (allowed_mcp_names is None or name in allowed_mcp_names)
        ):
            # Инструмент неизвестен локально, но известен MCP-серверу —
            # диспетчеризация туда (новое ТЗ, интеграция с MCP-сервером).
            if mcp_events is not None:
                mcp_events.append({"type": "mcp_call", "name": name, "status": "started", "ok": None, "error": None})
            try:
                result = self._mcp_client.call_tool(name, arguments)
                ok = not (isinstance(result, dict) and "error" in result)
                if mcp_events is not None:
                    mcp_events.append({
                        "type": "mcp_call", "name": name, "status": "finished", "ok": ok,
                        "error": result.get("error") if isinstance(result, dict) and not ok else None,
                    })
            except MCPClientError as exc:  # сетевой сбой самого MCP-сервера
                result = {"error": str(exc)}
                if mcp_events is not None:
                    mcp_events.append({"type": "mcp_call", "name": name, "status": "finished", "ok": False, "error": str(exc)})
        else:
            result = {"error": f"unknown tool: {name!r}"}
        event = result.pop("_event", None) if isinstance(result, dict) else None
        if event is not None and events is not None:
            events.append(event)
        return json.dumps(result, ensure_ascii=False)

    def _run_tool_loop_blocking(
        self, provider, model_id: str, messages: List[ProviderMessage], settings: Settings,
        chat: Chat, agent: Agent, result: ChatResult, events: Optional[List[dict]] = None,
        mcp_events: Optional[List[dict]] = None,
    ) -> ChatResult:
        iterations = 0
        allowed_mcp_names = self._offered_tool_names(settings)
        while result.tool_calls and iterations < _MAX_TOOL_ITERATIONS:
            messages.append(ProviderMessage("assistant", result.content, tool_calls=result.tool_calls))
            for call in result.tool_calls:
                tool_output = self._execute_tool_call(
                    chat, agent, call, events=events, mcp_events=mcp_events, allowed_mcp_names=allowed_mcp_names,
                )
                fn_name = (call.get("function") or {}).get("name")
                messages.append(ProviderMessage("tool", tool_output, tool_call_id=call.get("id"), name=fn_name))
            iterations += 1
            result = provider.chat(model_id, messages, settings)
        return self._finalize_after_tool_cap(provider, model_id, messages, settings, result)

    def _finalize_after_tool_cap(
        self, provider, model_id: str, messages: List[ProviderMessage], settings: Settings, result: ChatResult,
    ) -> ChatResult:
        """Баг, найденный на практике (реальный лог пользователя): цикл
        tool-calling обрывается по `_MAX_TOOL_ITERATIONS`, а не потому что
        модель сама закончила — `result` в этот момент несёт СЛЕДУЮЩий раунд
        `tool_calls`, который так и не будет выполнен, а `result.content`
        часто вообще пуст (весь бюджет ответа модель потратила на
        `reasoning_content`, планируя очередные вызовы). Раньше это сырое
        промежуточное состояние сохранялось как финальное сообщение
        ассистента — пользователь получал пустой ответ (реальный кейс: 8
        подряд вызовов git_host_*, а на выходе — пустая строка вместо
        структуры проекта).

        Если `result.tool_calls` всё ещё непуст — значит бюджет итераций
        исчерпан, а не модель закончила сама: делаем ОДИН дополнительный
        запрос БЕЗ инструментов (`tools_json=""` — `_parsed_tools` при
        пустой строке возвращает `None`, провайдер не передаст `tools` в
        API и модель физически не сможет запросить ещё один вызов),
        явно попросив подвести итог по уже полученным (и УЖЕ добавленным
        в `messages`) результатам инструментов. Если `tool_calls` пуст —
        модель закончила сама, `result` уже финальный, ничего делать не
        надо."""
        if not result.tool_calls:
            return result
        forced_messages = list(messages)
        forced_messages.append(ProviderMessage(
            "system",
            "Лимит вызовов инструментов на этот ответ исчерпан, инструменты больше "
            "недоступны. Сформулируй окончательный ответ пользователю на основе уже "
            "полученных выше результатов вызовов инструментов, не пытаясь вызвать "
            "ещё один инструмент.",
        ))
        no_tools_settings = dataclasses.replace(settings, tools_json="")
        return provider.chat(model_id, forced_messages, no_tools_settings)

    # ---- "Инварианты" (новое ТЗ): код-уровневая валидация + переспрос -------

    def _validate_and_maybe_retry(
        self, provider, model_id: str, provider_messages: List[ProviderMessage], settings: Settings,
        chat: Chat, agent: Agent, result: ChatResult, events: Optional[List[dict]] = None,
    ) -> Tuple[ChatResult, Optional[str]]:
        """"Инварианты" (новое ТЗ, раздел "Инварианты", п.2-3) — программная
        проверка ответа модели для категорий, у которых есть код-чекер (см.
        `invariant_checks.CHECKERS`); категории без чекера — доверяем
        модели, ничего не проверяем. При нарушении — ОДИН раз переспрашивает
        модель, добавив список нарушений системным сообщением (прямой аналог
        `retry(query, violations)` из примера ТЗ), и возвращает обновлённый
        результат плюс текст предупреждения для интерфейса чата (см.
        `invariant_checks.format_violation_warning`), который вызывающий код
        обязан показать ПЕРЕД текстом ответа. Второй элемент — `None`, если
        нарушений не было (в т.ч. если инварианты выключены/не выбраны)."""
        invariants = self._active_invariants(chat, agent)
        if not invariants:
            return result, None
        violations = validate_response(result.content, invariants)
        if not violations:
            return result, None
        warning_text = format_violation_warning(violations)
        violation_lines = "\n".join(f"- {inv.title}: {reason}" for inv, reason in violations)
        provider_messages.append(ProviderMessage("assistant", result.content))
        provider_messages.append(ProviderMessage(
            "system",
            "Предыдущий ответ нарушил инвариант(ы):\n" + violation_lines +
            "\nПерепиши ответ полностью так, чтобы он соответствовал ВСЕМ инвариантам.",
        ))
        retried = provider.chat(model_id, provider_messages, settings)
        retried = self._run_tool_loop_blocking(provider, model_id, provider_messages, settings, chat, agent, retried, events=events)
        return retried, warning_text

    # ---- сборка сообщений памяти/профиля для запроса -------------------------

    def _effective_invariant_ids(self, chat: Chat, agent: Agent) -> List[str]:
        """Итоговый набор id инвариантов для этого чата — объединение
        выбранных на уровне агента и выбранных на уровне самого чата (см.
        `Agent.invariant_ids`/`Chat.invariant_ids`), без дублей, порядок не
        важен (сортировка по `created_at` наводится уже при построении
        текста, см. `_active_invariants`/`_build_task_and_invariant_context`)."""
        return list(dict.fromkeys(list(agent.invariant_ids) + list(chat.invariant_ids)))

    def _active_invariants(self, chat: Chat, agent: Agent) -> List[Invariant]:
        """"Работу с задачами требуется переделать" (раздел "Инварианты") —
        отдельного тумблера `invariants_enabled` больше нет: инварианты
        считаются разрешёнными сами по себе, как только для этого чата/агента
        выбран хотя бы один (см. `_effective_invariant_ids`); если не выбрано
        ни одного — они просто не подмешиваются, без отдельного выключателя."""
        selected_ids = set(self._effective_invariant_ids(chat, agent))
        if not selected_ids:
            return []
        return [inv for inv in self._db.list_invariants() if inv.id in selected_ids and inv.is_active]

    def _build_memory_injection_messages(self, chat: Chat, agent: Agent) -> List[ProviderMessage]:
        """Порядок вставки (сразу после system_prompt, до истории диалога) —
        профиль -> долговременная память -> рабочая память, см. итоговый
        документ, раздел "Инъекция в запрос". Каждый блок подставляется,
        только если соответствующий тип памяти включён в `chat.settings`
        (working_memory_enabled / long_term_memory_enabled + по отдельности
        episodic_/semantic_/procedural_memory_enabled для расширенных
        категорий) — выключенный тип не просто "пуст", а полностью
        отсутствует в запросе к модели, как и не выключавшиеся данные."""
        blocks: List[ProviderMessage] = []
        profile = self._db.get_profile(chat.active_profile_id) if chat.active_profile_id else None
        if profile is not None:
            parts = [f"Активен профиль «{profile.name}»."]
            if profile.style:
                parts.append(f"Стиль ответа: {profile.style}.")
            if profile.format:
                parts.append(f"Требования к формату ответа: {profile.format}.")
            if profile.constraints:
                parts.append(f"Ограничения: {profile.constraints}.")
            if profile.orchestration_prompt:
                parts.append(f"Как использовать доступные функции этого профиля: {profile.orchestration_prompt}")
            blocks.append(ProviderMessage("system", " ".join(parts)))
        enabled_categories = set(self._enabled_long_term_categories(chat.settings))
        long_term = [e for e in self._db.list_long_term_memory(agent.id) if e.category in enabled_categories]
        if long_term:
            lines = [f"- [{e.category}] {e.key}: {e.value}" for e in long_term]
            blocks.append(ProviderMessage(
                "system",
                "Долговременная память об этом агенте (сохранённые ранее факты, "
                "решения и знания — действуют во всех чатах агента):\n" + "\n".join(lines),
            ))
        if chat.settings.working_memory_enabled:
            working = self._db.list_working_memory(chat.id)
            if working:
                lines = [f"- {e.key}: {e.value}" for e in working]
                blocks.append(ProviderMessage(
                    "system",
                    "Рабочая память текущего чата (данные текущей задачи):\n" + "\n".join(lines),
                ))
        return blocks

    def _build_task_tracking_prompt(self) -> str:
        """Системный prompt "Менеджера задач" (заменяет прежний
        `_TASK_MANAGER_PROACTIVITY_HINT`) — подмешивается ОДИН раз, пока
        `task_tracking_enabled=true`, независимо от того, есть ли уже
        открытые задачи в чате. Заполняется по шаблону
        `AgentConfig.TASK_TRACKING_PROMPT_TEMLATE` (по умолчанию — текст из
        ТЗ, редактируется через .env, см. `config.py`): `<task_states>`
        заполняется описанием состояний машины прямо из кода
        (`task_state_machine.format_states_for_prompt`), а
        `<task_state_machine_invariants>` — текстами инвариантов категории
        "Правило стейт-машины", привязанных к машине состояний (та же
        настройка, что и в блоке "Rules" на экране "Модели состояний
        задач" — см. `_state_machine_rule_texts`), по одной на строке."""
        invariants_text = "\n".join(self._state_machine_rule_texts())
        return _fill_task_tracking_template(
            AgentConfig.TASK_TRACKING_PROMPT_TEMLATE, format_states_for_prompt(), invariants_text,
        )

    def _state_machine_rule_texts(self) -> List[str]:
        """Тексты инвариантов категории "Правило стейт-машины", привязанных
        к машине состояний (см. `set_task_machine_invariants`) — это то, что
        новое ТЗ называет "Rules" в примере `buildPrompt`, только теперь
        настраивается на экране машины состояний, а не хардкодится."""
        ids = set(self._db.get_task_machine_invariant_ids())
        if not ids:
            return []
        return [inv.rule_text for inv in self._db.list_invariants() if inv.id in ids and inv.is_active]

    def _build_task_context_block(self, task: Task) -> str:
        """Скрытый служебный блок с ТЕКУЩИМИ данными ОДНОЙ задачи — те же
        поля, что модель должна "возвращать" по системному prompt'у (см.
        `_build_task_tracking_prompt`): task/state/step/total/plan/done/
        current, плюс id задачи (нужен для `apply_task_action`). Добавляется
        в промпт МОДЕЛИ отдельным `system`-сообщением — исходный текст
        запроса пользователя (`Message.content`) при этом не меняется, см.
        `_build_task_and_invariant_context`.

        `step`/`total` — по шаблону ТЗ это НЕ прогресс по `plan` (как было
        раньше: пройденных/всего пунктов плана), а позиция ТЕКУЩЕГО состояния
        в машине состояний (1..4) и общее число её состояний (всегда 4) —
        см. пример заполнения `task_states` в сопроводительном сообщении к
        доработке. Прогресс по плану по-прежнему виден целиком через
        `plan`/`done` (списки), просто не сведён к отдельным числам."""
        state = parse_task_state(task.state)
        position = TASK_STATE_ORDER.index(state) + 1
        total = len(TASK_STATE_ORDER)
        lines = [
            f"[ЗАДАЧА] id: {task.id}",
            f"task: {task.title}",
            f"state: {state.value}",
            f"step: {position}",
            f"total: {total}",
            f"plan: {json.dumps(task.plan, ensure_ascii=False)}",
            f"done: {json.dumps(task.done_steps, ensure_ascii=False)}",
            f"current: {task.current_step or ''}",
        ]
        return "\n".join(lines)

    def _build_task_and_invariant_context(self, chat: Chat, agent: Agent) -> List[ProviderMessage]:
        """"Работу с задачами требуется переделать" (новое ТЗ) — заменяет
        прежние `_build_task_injection_messages`/`_build_invariant_injection_messages`
        единым служебным блоком, добавляемым в промпт МОДЕЛИ отдельными
        `system`-сообщениями (тем же приёмом, что и раньше, и что и
        `_build_memory_injection_messages`) — исходный текст запроса
        пользователя не меняется и хранится/отображается как есть (см.
        `send_message_blocking`/`stream_message`: в БД по-прежнему пишется
        именно присланный текст).

        Пока `task_tracking_enabled=true` — ОДИН общий системный prompt
        (`_build_task_tracking_prompt`, заменяет собой прежний хардкод),
        независимо от того, есть ли уже открытые задачи, плюс по одному
        блоку с текущими данными (task/state/step/total/plan/done/current)
        на каждую открытую задачу чата (см. `_build_task_context_block`) —
        правила категории "Правило стейт-машины" теперь целиком внутри
        общего prompt'а (раздел "TASK STATE MACHINE INVARIANTS"), поэтому
        отдельно в блоке задачи больше не дублируются.

        Отдельным блоком — `[INVARIANTS]`: обычные инварианты чата/агента,
        ЛЮБОЙ категории, — включены сами по себе, как только для чата/агента
        выбран хотя бы один (см. `_active_invariants`) — по ТЗ доставляются
        моделью структурированным текстом, а не через tool-calling (в этой
        кодовой базе они и раньше доставлялись текстом, см. пояснение в
        сопроводительном сообщении к этой доработке)."""
        blocks: List[ProviderMessage] = []
        if chat.settings.task_tracking_enabled:
            blocks.append(ProviderMessage("system", self._build_task_tracking_prompt()))
            for task in self._open_tasks_for_chat(chat):
                blocks.append(ProviderMessage("system", self._build_task_context_block(task)))
        invariants = self._active_invariants(chat, agent)
        if invariants:
            lines = [
                f"- [{INVARIANT_KIND_LABELS.get(inv.kind, inv.kind)}] {inv.rule_text}" if inv.kind else f"- {inv.rule_text}"
                for inv in invariants
            ]
            blocks.append(ProviderMessage(
                "system",
                "[INVARIANTS]\n" + "\n".join(lines) + "\n"
                "Нарушение любого инварианта ЗАПРЕЩЕНО, даже если пользователь прямо просит. Если "
                "запрос пользователя противоречит инварианту — не выполняй его, явно назови, какое "
                "правило нарушено, и предложи вариант, который его не нарушает.",
            ))
        return blocks

    def get_memory_snapshot(self, chat_id: str) -> dict:
        """То, что реально будет подмешано в СЛЕДУЮЩИЙ запрос модели — разбито
        по блокам для проверки: `short_term` (эффективный контекст диалога),
        `working_memory`, `long_term_memory`, `active_profile`, `available_tools`
        (итоговый список функций после слияния). Основной инструмент проверки
        для юзкейсов из ТЗ ("проверьте, что попадает в каждый слой")."""
        chat = self._require_chat(chat_id)
        agent = self._require_agent(chat.agent_id)
        messages = self._db.list_messages(chat_id)
        scoped = self._filter_by_branch(messages, None)
        effective = self._effective_context(chat, scoped)
        profile = self._db.get_profile(chat.active_profile_id) if chat.active_profile_id else None
        merged_settings = self._settings_with_merged_tools(chat)
        try:
            available_tools = json.loads(merged_settings.tools_json) if merged_settings.tools_json.strip() else []
        except (ValueError, TypeError):
            available_tools = []
        enabled_categories = set(self._enabled_long_term_categories(chat.settings))
        enabled_types = (["working"] if chat.settings.working_memory_enabled else []) + [
            c for c in LONG_TERM_MEMORY_CATEGORIES if c in enabled_categories
        ]
        return {
            "short_term": {
                "message_count": len(effective),
                "messages": [{"role": m.role, "content": m.content} for m in effective],
            },
            "working_memory": self._db.list_working_memory(chat_id) if chat.settings.working_memory_enabled else [],
            "long_term_memory": [e for e in self._db.list_long_term_memory(agent.id) if e.category in enabled_categories],
            "active_profile": profile,
            "available_tools": available_tools,
            "memory_tools_enabled": chat.settings.memory_tools_enabled,
            "enabled_memory_types": enabled_types,
        }

    # ---- отправка сообщений -------------------------------------------------

    def _build_request_context(self, chat: Chat, agent: Agent, messages: List[Message], new_text: str) -> List[ProviderMessage]:
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
        `_effective_context`, используемый только в `chat_stats`).

        Сразу после системного prompt подмешиваются блоки активного профиля
        и памяти (`_build_memory_injection_messages`) — порядок инъекции:
        system_prompt -> профиль -> долговременная память -> рабочая память
        -> история диалога -> новое сообщение пользователя."""
        context = self._context_messages(messages)
        provider_messages = []
        if chat.settings.system_prompt and chat.settings.system_prompt.strip():
            provider_messages.append(ProviderMessage("system", chat.settings.system_prompt))
        provider_messages += self._build_memory_injection_messages(chat, agent)
        provider_messages += self._build_task_and_invariant_context(chat, agent)
        provider_messages += [ProviderMessage(m.role, m.content) for m in context]
        provider_messages.append(ProviderMessage("user", new_text))
        return provider_messages

    def _build_sticky_facts_context(
        self, chat: Chat, agent: Agent, messages: List[Message], new_text: str, latest_facts: dict
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
        provider_messages += self._build_memory_injection_messages(chat, agent)
        provider_messages += self._build_task_and_invariant_context(chat, agent)
        provider_messages += [ProviderMessage(m.role, m.content) for m in recent]
        if latest_facts:
            facts_text = "All Facts, fixed before in JSON: " + json.dumps(latest_facts, ensure_ascii=False)
            provider_messages.append(ProviderMessage("assistant", facts_text))
        provider_messages.append(ProviderMessage("user", new_text))
        return provider_messages

    def _resolve_send_context(
        self, chat: Chat, agent: Agent, messages: List[Message], text: str, get_facts: bool, sliding_window: bool, autosummary: str
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
            provider_messages = self._build_sticky_facts_context(chat, agent, pool, text, latest_facts)
            return provider_messages, latest_facts, boundary
        if sliding_window:
            return self._build_sliding_window_context(chat, agent, pool, text), {}, None
        return self._build_request_context(chat, agent, pool, text), {}, None

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
        agent = self._require_agent(chat.agent_id)
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
                chat, agent, messages, text, get_facts, sliding_window, autosummary
            )
            provider_name, model_id = _split_model(chat.settings.model)
            provider = self._registry.get(provider_name)
            request_settings = self._settings_with_merged_tools(chat)

            user_msg = self._db.add_message(
                Message(
                    id=0, chat_id=chat_id, role="user", content=text, created_at=int(time.time()),
                    format=detect_message_format(text), branch=branch or 0,
                )
            )

            task_events: List[dict] = []
            mcp_events: List[dict] = []
            started = time.monotonic()
            try:
                result = provider.chat(model_id, provider_messages, request_settings)
                # Единый механизм tool-calling: если модель запросила вызов
                # функций (памяти и/или скиллов активного профиля) — выполняем
                # их и повторяем запрос, пока модель не ответит обычным
                # текстом (или не будет достигнут предел итераций). Сам обмен
                # "вызов -> результат" НЕ сохраняется как сообщения чата —
                # в истории остаётся только финальный текстовый ответ (кроме
                # событий задач — см. Message.task_events — и вызовов через
                # MCP-сервер, см. Message.mcp_events).
                result = self._run_tool_loop_blocking(
                    provider, model_id, provider_messages, request_settings, chat, agent, result,
                    events=task_events, mcp_events=mcp_events,
                )
                # "Инварианты" (новое ТЗ) — код-уровневая проверка + один
                # переспрос модели при нарушении; текст предупреждения (если
                # было нарушение) выводится ПЕРЕД ответом в интерфейсе чата —
                # здесь это реализовано как префикс самого сохраняемого
                # сообщения ассистента (см. `_validate_and_maybe_retry`).
                result, violation_warning = self._validate_and_maybe_retry(
                    provider, model_id, provider_messages, request_settings, chat, agent, result,
                    events=task_events,
                )
            except ProviderError as exc:
                self._db.add_message(
                    Message(id=0, chat_id=chat_id, role="error", content=str(exc), created_at=int(time.time()), branch=branch or 0)
                )
                self._db.touch_chat(chat_id)
                raise
            duration_ms = int((time.monotonic() - started) * 1000)
            final_content = f"{violation_warning}\n\n{result.content}" if violation_warning else result.content

            assistant_msg = self._db.add_message(
                Message(
                    id=0, chat_id=chat_id, role="assistant", content=final_content,
                    created_at=int(time.time()), reasoning_content=result.reasoning_content,
                    duration_ms=duration_ms, total_tokens=result.usage.total_tokens,
                    prompt_tokens=result.usage.prompt_tokens, completion_tokens=result.usage.completion_tokens,
                    format=detect_message_format(final_content), branch=branch or 0,
                    task_events=json.dumps(task_events, ensure_ascii=False) if task_events else None,
                    mcp_events=json.dumps(mcp_events, ensure_ascii=False) if mcp_events else None,
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
        agent = self._require_agent(chat.agent_id)
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
                chat, agent, messages, text, get_facts, sliding_window, autosummary
            )
            provider_name, model_id = _split_model(chat.settings.model)
            provider = self._registry.get(provider_name)
            request_settings = self._settings_with_merged_tools(chat)

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
                # Единый механизм tool-calling в потоковом режиме: каждая
                # "итерация" — один полный потоковый вызов провайдера; если
                # финальный результат итерации несёт tool_calls, выполняем их,
                # добавляем сообщения вызова/результата в контекст и запускаем
                # СЛЕДУЮЩУЮ потоковую итерацию — клиент в этот момент видит
                # промежуточный статус, а не разрыв соединения. Итоговое
                # сообщение ассистента, сохраняемое в чат, — это только
                # содержимое ПОСЛЕДНЕЙ итерации (той, что уже не запросила
                # новых вызовов); сам обмен "вызов -> результат" в историю
                # чата не попадает.
                final_result: Optional[ChatResult] = None
                task_events: List[dict] = []
                mcp_events: List[dict] = []
                iterations = 0
                while True:
                    done_result: Optional[ChatResult] = None
                    for delta in provider.stream_chat(model_id, provider_messages, request_settings):
                        if delta.done:
                            done_result = delta.result
                        else:
                            yield {"type": "delta", "content": delta.content, "reasoning_content": delta.reasoning_content}
                    if done_result is not None and done_result.tool_calls and iterations < _MAX_TOOL_ITERATIONS:
                        yield {"type": "status", "status": "Выполняется вызов инструментов"}
                        provider_messages.append(
                            ProviderMessage("assistant", done_result.content, tool_calls=done_result.tool_calls)
                        )
                        for call in done_result.tool_calls:
                            mcp_events_before = len(mcp_events)
                            tool_output = self._execute_tool_call(
                                chat, agent, call, events=task_events, mcp_events=mcp_events,
                                allowed_mcp_names=self._offered_tool_names(request_settings),
                            )
                            # Новые события MCP (см. Message.mcp_events) уходят в поток СРАЗУ
                            # (started/finished этого конкретного вызова), а не только постфактум
                            # в сохранённом сообщении — клиент показывает "Инструмент: <name>" по
                            # ходу генерации, не дожидаясь конца ответа.
                            for mcp_event in mcp_events[mcp_events_before:]:
                                yield mcp_event
                            fn_name = (call.get("function") or {}).get("name")
                            provider_messages.append(
                                ProviderMessage("tool", tool_output, tool_call_id=call.get("id"), name=fn_name)
                            )
                        iterations += 1
                        yield {"type": "status", "status": "Выполняется запрос к модели"}
                        continue
                    final_result = done_result
                    break

                # Реальный баг, найденный на практике: если `tool_calls` у
                # `final_result` всё ещё непуст — цикл выше оборвался по
                # `_MAX_TOOL_ITERATIONS`, а не потому что модель закончила
                # сама, и `final_result.content` в этом случае часто пуст
                # (см. `_finalize_after_tool_cap`) — без этого шага
                # пользователь получал бы пустое сообщение вместо ответа.
                if final_result is not None and final_result.tool_calls:
                    yield {"type": "status", "status": "Формулирую итоговый ответ"}
                    final_result = self._finalize_after_tool_cap(
                        provider, model_id, provider_messages, request_settings, final_result,
                    )

                # "Инварианты" (новое ТЗ) — та же код-уровневая проверка +
                # переспрос, что и в `send_message_blocking` (см.
                # `_validate_and_maybe_retry`); переспрос делается ОДНИМ
                # блокирующим вызовом (не повторным стримом) ради простоты —
                # это редкий путь (только при нарушении), в отличие от
                # основного, всегда потокового, ответа. Предупреждение
                # выводится ПЕРЕД ответом: отдельным `delta`-событием до
                # того, как в поток уйдёт (пере-сгенерированный) текст ответа.
                final_result, violation_warning = self._validate_and_maybe_retry(
                    provider, model_id, provider_messages, request_settings, chat, agent, final_result,
                    events=task_events,
                )
                if violation_warning:
                    yield {"type": "delta", "content": f"{violation_warning}\n\n", "reasoning_content": None}
                    yield {"type": "delta", "content": final_result.content, "reasoning_content": None}
                final_content = f"{violation_warning}\n\n{final_result.content}" if violation_warning else final_result.content

                duration_ms = int((time.monotonic() - started) * 1000)
                if final_result.usage.prompt_tokens is not None:
                    self._db.update_message_prompt_tokens(user_msg.id, final_result.usage.prompt_tokens)
                assistant_msg = self._db.add_message(
                    Message(
                        id=0, chat_id=chat_id, role="assistant", content=final_content,
                        created_at=int(time.time()), reasoning_content=final_result.reasoning_content,
                        duration_ms=duration_ms, total_tokens=final_result.usage.total_tokens,
                        prompt_tokens=final_result.usage.prompt_tokens,
                        completion_tokens=final_result.usage.completion_tokens,
                        format=detect_message_format(final_content), branch=branch or 0,
                        task_events=json.dumps(task_events, ensure_ascii=False) if task_events else None,
                        mcp_events=json.dumps(mcp_events, ensure_ascii=False) if mcp_events else None,
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
                        chat, latest_facts, dialogue, text, final_result.content
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
            except ProviderError as exc:
                self._db.add_message(
                    Message(id=0, chat_id=chat_id, role="error", content=str(exc), created_at=int(time.time()), branch=branch or 0)
                )
                self._db.touch_chat(chat_id)
                yield {"type": "error", "message": str(exc)}
        finally:
            lock.release()

    # ---- "Менеджер задач" (обновление "Дня 13") ------------------------------
    #
    # Пока задача активна (не на паузе, не завершена), пользователь может
    # ничего не писать — система сама, без нового сообщения, просит модель
    # продолжить работу над задачей и стримит ответ; клиент вызывает
    # `run_task_manager_step` ещё раз, пока в событии `done` не придёт
    # `should_continue=False` (задача продвинута этим шагом и осталась
    # активной — единственная причина продолжать цикл). Пауза — это ЧЕЛОВЕК,
    # нажавший кнопку (см. `apply_manual_task_action` с `kind=PAUSE`), а не
    # что-то, что цикл решает сам: цикл просто не запускает следующий шаг,
    # если задача не активна. Один шаг НЕ создаёт сообщение с ролью `user` в
    # истории — эфемерная реплика "продолжай" подмешивается только в ЭТОТ
    # конкретный запрос к провайдеру (тем же приёмом, что уже используется
    # для инъекции памяти/задач/инвариантов), а результат сохраняется как
    # обычное сообщение ассистента с `is_task_manager_step=True`.
    # ---------------------------------------------------------------------------

    def _task_manager_continue_text(self, task: Task) -> str:
        # Этап/шаг/план/done задачи модель уже видит в блоке [ЗАДАЧА]/[STATE]/
        # [CURRENT]/[PLAN]/[DONE] (см. `_build_task_and_invariant_context`,
        # подмешивается в этот же запрос) — здесь только сама команда
        # продолжить, без дублирования состояния.
        return (
            f"Продолжай самостоятельно работать над задачей «{task.title}» "
            "(текущий этап и шаг — в системном блоке [ЗАДАЧА] этого запроса). "
            "Если для продолжения не хватает информации от пользователя — задай вопрос и НЕ вызывай "
            "apply_task_action в этом ответе (менеджер задач сам остановится и дождётся пользователя). "
            "Если этап пройден — продвинь его вызовом apply_task_action, как обычно."
        )

    def _consecutive_task_manager_steps(self, chat_id: str) -> int:
        """Подряд идущие сообщения ассистента с `is_task_manager_step=True`
        считая с конца истории чата, до первого сообщения другого рода
        (обычный ответ на реальное сообщение пользователя, ошибка и т.п.) —
        используется как защита от зацикливания (см.
        `Settings.task_manager_max_steps`). Считается на уровне ЧАТА в
        целом, а не отдельной задачи — упрощение: в рамках одного чата
        обычно ведётся не более одной активной задачи одновременно."""
        count = 0
        for m in reversed(self._db.list_messages(chat_id)):
            if m.role == "assistant" and m.is_task_manager_step:
                count += 1
                continue
            break
        return count

    def run_task_manager_step(self, chat_id: str, task_id: str, auto_pause: bool = True) -> Iterator[dict]:
        """Один автономный шаг "Менеджера задач" — тот же формат событий,
        что и `stream_message` (`status`/`delta`/`done`/`error`), но `done`
        дополнительно несёт `should_continue` (клиент вызывает этот метод
        ещё раз, если true) и `task_status` (актуальный статус задачи после
        шага). Требует `task_tracking_enabled=true` и статус задачи ЛЮБОЙ,
        кроме "done" — иначе `ValidationError`.

        `auto_pause` (редизайн "Менеджера задач", кнопки "Продолжить"/
        "Выполнить" — замечание пользователя): `true` (по умолчанию, кнопка
        "Продолжить") — задача МОЖЕТ быть на паузе прямо сейчас (это
        нормально: "Продолжить" одним действием снимает паузу И продвигает
        этап, т.к. допустимость перехода определяется `can_transition` и не
        зависит от статуса паузы), но после того как модель продвинет её
        вызовом `apply_task_action` (kind="advance"), шаг сам поставит её на
        паузу заново — `should_continue` в этом случае приходит `false`,
        клиент вызывает эндпоинт ещё раз только по новому нажатию
        пользователя. `false` (кнопка "Выполнить") — пауза не вставляется
        автоматически, `should_continue=true` до состояния done (или пока
        модель не остановится сама) — клиент вызывает эндпоинт в цикле;
        пользователь может прервать цикл в любой момент отдельным вызовом
        ручного действия "Пауза" (см. `apply_manual_task_action`)."""
        chat = self._require_chat(chat_id)
        agent = self._require_agent(chat.agent_id)
        task = self._require_task(task_id)
        if task.chat_id != chat_id:
            raise NotFoundError(f"task {task_id!r} does not belong to chat {chat_id!r}")
        if not chat.settings.task_tracking_enabled:
            raise ValidationError("task tracking is disabled for this chat")
        current_status = self._task_status(task)
        if current_status == "done":
            raise ValidationError("task is already done; task manager is not applicable")

        lock = self._lock_for(chat_id)
        lock.acquire()
        try:
            max_steps = chat.settings.task_manager_max_steps
            if max_steps > 0 and self._consecutive_task_manager_steps(chat_id) >= max_steps:
                notice = (
                    "Достигнут лимит автоматических шагов Менеджера задач подряд — нажмите "
                    "«Продолжить», чтобы продолжить вручную."
                )
                assistant_msg = self._db.add_message(Message(
                    id=0, chat_id=chat_id, role="assistant", content=notice,
                    created_at=int(time.time()), format="text", branch=0,
                    is_task_manager_step=True,
                ))
                self._db.touch_chat(chat_id)
                yield {"type": "done", "message": assistant_msg, "should_continue": False, "task_status": current_status}
                return

            all_messages = self._db.list_messages(chat_id)
            messages = self._filter_by_branch(all_messages, None)
            continue_text = self._task_manager_continue_text(task)
            provider_messages = self._build_request_context(chat, agent, messages, continue_text)
            provider_name, model_id = _split_model(chat.settings.model)
            provider = self._registry.get(provider_name)
            request_settings = self._settings_with_merged_tools(chat)

            yield {"type": "status", "status": "Менеджер задач продолжает работу"}

            started = time.monotonic()
            try:
                final_result: Optional[ChatResult] = None
                task_events: List[dict] = []
                mcp_events: List[dict] = []
                iterations = 0
                while True:
                    done_result: Optional[ChatResult] = None
                    for delta in provider.stream_chat(model_id, provider_messages, request_settings):
                        if delta.done:
                            done_result = delta.result
                        else:
                            yield {"type": "delta", "content": delta.content, "reasoning_content": delta.reasoning_content}
                    if done_result is not None and done_result.tool_calls and iterations < _MAX_TOOL_ITERATIONS:
                        yield {"type": "status", "status": "Выполняется вызов инструментов"}
                        provider_messages.append(
                            ProviderMessage("assistant", done_result.content, tool_calls=done_result.tool_calls)
                        )
                        for call in done_result.tool_calls:
                            mcp_events_before = len(mcp_events)
                            tool_output = self._execute_tool_call(
                                chat, agent, call, events=task_events, auto_pause=auto_pause, mcp_events=mcp_events,
                                allowed_mcp_names=self._offered_tool_names(request_settings),
                            )
                            for mcp_event in mcp_events[mcp_events_before:]:
                                yield mcp_event
                            fn_name = (call.get("function") or {}).get("name")
                            provider_messages.append(
                                ProviderMessage("tool", tool_output, tool_call_id=call.get("id"), name=fn_name)
                            )
                        iterations += 1
                        yield {"type": "status", "status": "Менеджер задач продолжает работу"}
                        continue
                    final_result = done_result
                    break

                # См. аналогичное исправление и комментарий в `stream_message`
                # выше и в `_finalize_after_tool_cap` — тот же баг актуален и
                # для потокового продолжения Менеджера задач.
                if final_result is not None and final_result.tool_calls:
                    yield {"type": "status", "status": "Формулирую итоговый ответ"}
                    final_result = self._finalize_after_tool_cap(
                        provider, model_id, provider_messages, request_settings, final_result,
                    )

                # "Инварианты" (новое ТЗ) — код-уровневая проверка + один
                # переспрос модели при нарушении, тот же путь, что и в
                # `stream_message` (см. `_validate_and_maybe_retry`).
                # Предупреждение выводится ПЕРЕД ответом: отдельным
                # `delta`-событием до (пере-сгенерированного) текста ответа.
                final_result, violation_warning = self._validate_and_maybe_retry(
                    provider, model_id, provider_messages, request_settings, chat, agent, final_result,
                    events=task_events,
                )
                if violation_warning:
                    yield {"type": "delta", "content": f"{violation_warning}\n\n", "reasoning_content": None}
                    yield {"type": "delta", "content": final_result.content, "reasoning_content": None}
                final_content = f"{violation_warning}\n\n{final_result.content}" if violation_warning else final_result.content

                duration_ms = int((time.monotonic() - started) * 1000)
                assistant_msg = self._db.add_message(Message(
                    id=0, chat_id=chat_id, role="assistant", content=final_content,
                    created_at=int(time.time()), reasoning_content=final_result.reasoning_content,
                    duration_ms=duration_ms, total_tokens=final_result.usage.total_tokens,
                    prompt_tokens=final_result.usage.prompt_tokens,
                    completion_tokens=final_result.usage.completion_tokens,
                    format=detect_message_format(final_content), branch=0,
                    task_events=json.dumps(task_events, ensure_ascii=False) if task_events else None,
                    mcp_events=json.dumps(mcp_events, ensure_ascii=False) if mcp_events else None,
                    is_task_manager_step=True,
                ))
                self._db.touch_chat(chat_id)

                refreshed_task = self._db.get_task(task_id)
                new_status = self._task_status(refreshed_task) if refreshed_task is not None else current_status
                advanced_this_task = any(
                    e.get("task_id") == task_id and e.get("kind") == "advance" for e in task_events
                )
                should_continue = new_status == "active" and advanced_this_task
                yield {"type": "done", "message": assistant_msg, "should_continue": should_continue, "task_status": new_status}
            except ProviderError as exc:
                self._db.add_message(
                    Message(id=0, chat_id=chat_id, role="error", content=str(exc), created_at=int(time.time()), branch=0)
                )
                self._db.touch_chat(chat_id)
                yield {"type": "error", "message": str(exc)}
        finally:
            lock.release()

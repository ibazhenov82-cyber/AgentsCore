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
import logging
import re
import sqlite3
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .catalog import ModelCatalog
from .config import AgentConfig
from .db import Database
from .format_detect import detect_message_format
from .invariant_checks import format_violation_warning, validate_response
from .events import CancelToken, EventLog, RunCancelled
from .knowledge import (
    RAG_STATUS_NOT_FOUND, RAG_STATUS_OK, RAG_STATUS_UNAVAILABLE, REWRITE_INSTRUCTION, RERANK_MODES, REWRITE_MODES,
    KnowledgeClient, KnowledgeServiceError, RagOutcome, augmented_question, citations, clean_rewritten, finalize_answer,
    rag_instruction, rewrite_instruction, rewrite_request_text, search_query, select_sources, is_no_search,
    starts_with_dont_know,
)
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
    TestDialog,
    TASK_APPLIED_BY_OPTIONS,
    TaskTransitionLog,
    WorkingMemoryEntry,
    settings_from_defaults,
)
from .providers import ChatResult, ProviderError, ProviderMessage, ProviderRegistry
from .skills import registry as skills_registry
from .skills import shopping_demo
from .task_state_machine import (
    KIND_SEARCH,
    KIND_WORK,
    TASK_KIND_LABELS,
    TASK_KINDS,
    format_search_states_for_prompt,
    format_states_for_prompt,
    kind_allowed,
    kind_can_transition,
    kind_display_name,
    kind_final_state,
    kind_initial_state,
    kind_is_final,
    kind_is_valid_state,
    kind_states,
)
from .tokens import estimate_messages_tokens, estimate_tokens
from . import builtin_tools
from . import test_dialogs as test_dialog_rules

_logger = logging.getLogger("agents_core.repository")

_SETTINGS_FIELD_NAMES = {f.name for f in dataclasses.fields(Settings)}
_DEFAULT_SETTINGS_FIELD_NAMES = {f.name for f in dataclasses.fields(DefaultSettings)}

#: Максимум "туда-обратно" вызовов модели в одном обмене tool-calling —
#: защита от зацикливания (модель бесконечно вызывает инструменты вместо
#: финального текстового ответа). После достижения предела последний
#: результат провайдера возвращается как есть, даже если в нём снова
#: запрошены tool_calls.
_MAX_TOOL_ITERATIONS = 6

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

def _build_start_task_tool(kinds: Sequence[str] = ("work",)) -> dict:
    kinds = list(kinds) or ["work"]
    search_hint = (
        " Тип «search» («Задача поиска») — когда пользователь ищет ответ/инструкцию по базе "
        "знаний («как настроить…», «где описано…», «что такое…»): ассистент только ищет и "
        "отвечает со ссылками на источники, ничего не выполняет. Тип «work» («Рабочая "
        "задача») — когда нужно что-то СДЕЛАТЬ (изменить файлы, настроить, запустить) по этапам "
        "«Планирование» → «Выполнение» → «Проверка» → «Завершено»."
        if "search" in kinds else ""
    )
    properties: Dict[str, Any] = {
        "title": {"type": "string", "description": "Короткая формулировка задачи"},
        "goal": {
            "type": "string",
            "description": "Цель пользователя одной-двумя фразами — то, к чему ведёт диалог/работа",
        },
        "plan": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Только для «work»: согласованный план — список шагов до выполнения задачи (PLAN). "
                "Можно оставить пустым и задать позже через apply_task_action."
            ),
        },
    }
    properties["clarifications"] = {
        "type": "array", "items": {"type": "string"},
        "description": "Уточнения пользователя, уже известные на старте (необязательно)",
    }
    properties["constraints"] = {
        "type": "array", "items": {"type": "string"},
        "description": "Ограничения пользователя, уже известные на старте (необязательно)",
    }
    properties["terms"] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"term": {"type": "string"}, "definition": {"type": "string"}},
            "required": ["term", "definition"],
        },
        "description": "Термины, определённые пользователем (необязательно)",
    }
    if "search" in kinds:
        properties["needs_clarification"] = {
            "type": "boolean",
            "description": (
                "Только для «search»: true, если запрос неполон и до поиска нужно уточнить цель или "
                "условия — задача начнётся с этапа «Уточнение»; иначе — с «Поиск ответа»"
            ),
        }
    required = ["title"]
    if len(kinds) > 1:
        properties = {
            "kind": {
                "type": "string", "enum": kinds,
                "description": "Тип задачи: 'search' — задача поиска ответа, 'work' — рабочая задача",
            },
            **properties,
        }
        required = ["kind", "title", "goal"]
    return {
        "type": "function",
        "function": {
            "name": "start_task",
            "description": (
                "Начать отслеживание НОВОЙ задачи в этом чате (см. «Состояние задачи»). "
                "Используй, когда в сообщении пользователя выделяется отдельная задача или цель, "
                "которую стоит вести по этапам — не для каждой реплики. НЕ жди прямой просьбы "
                "пользователя отслеживать задачу или явной команды вроде «возьми в работу» — "
                "начинай сам, если задача очевидна по смыслу сообщения. В чате может быть "
                "несколько открытых задач одновременно." + search_hint
            ),
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def _build_update_task_memory_tool() -> dict:
    """«Память задачи» (ТЗ «Мини-чат с RAG + памятью»): модель сама фиксирует
    цель, уточнения, ограничения и термины пользователя — код хранит их в
    `Task` и подмешивает в каждый следующий промпт и в переписывание запроса."""
    item_list = lambda desc: {"type": "array", "items": {"type": "string"}, "description": desc}  # noqa: E731
    return {
        "type": "function",
        "function": {
            "name": "update_task_memory",
            "description": (
                "Обновить память задачи (блок [ЗАДАЧА ПОИСКА]/[ЗАДАЧА]): цель, уточнения, "
                "ограничения и термины пользователя. Вызывай ДО ответа каждый раз, когда "
                "пользователь сообщил что-то, что должно действовать до конца диалога: уточнил "
                "цель («мне нужно…»), поставил ограничение («только CPU», «порт занят», «без …»), "
                "определил термин («под X я понимаю Y») или отменил что-то сказанное раньше "
                "(тогда передай это в remove). Не дублируй то, что уже есть в памяти."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Id задачи из блока [ЗАДАЧА ПОИСКА]/[ЗАДАЧА]"},
                    "goal": {"type": "string", "description": "Новая формулировка цели, если она изменилась"},
                    "add_clarifications": item_list("Новые уточнения пользователя (кратко, своими словами)"),
                    "add_constraints": item_list("Новые ограничения пользователя (кратко)"),
                    "add_terms": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "term": {"type": "string"},
                                "definition": {"type": "string"},
                            },
                            "required": ["term", "definition"],
                        },
                        "description": "Термины, которые пользователь определил («под X я понимаю Y»)",
                    },
                    "remove": item_list(
                        "Тексты уточнений/ограничений или названия терминов, которые пользователь отменил"
                    ),
                },
                "required": ["task_id"],
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
                "(у задачи поиска: «Уточнение»/«Поиск ответа»/«Цель достигнута») "
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


def _tool_name(tool: Any) -> Optional[str]:
    fn = tool.get("function") if isinstance(tool, dict) else None
    return fn.get("name") if isinstance(fn, dict) else None


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
    if "rag_rerank" in payload and payload["rag_rerank"] not in RERANK_MODES:
        raise ValidationError(f"rag_rerank: ожидается одно из {sorted(RERANK_MODES)}")
    if "rag_query_rewrite" in payload and payload["rag_query_rewrite"] not in REWRITE_MODES:
        raise ValidationError(f"rag_query_rewrite: ожидается одно из {sorted(REWRITE_MODES)}")
    if payload.get("rag_rewrite_model"):
        _split_model(payload["rag_rewrite_model"])
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
        mcp_client: Optional[MCPClient] = None, knowledge_client: Optional[KnowledgeClient] = None,
    ):
        self._db = db
        #: Клиент сервиса баз знаний — `None`, если RAG не настроен
        #: (`KNOWLEDGE_SERVICE_URL` пуст). См. `_prepare_rag`.
        self._knowledge = knowledge_client
        self._registry = registry
        self._catalog = catalog
        self._locks: Dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        #: Клиент MCP-серверов — `None`, если MCP не настроен (тогда
        #: инструменты только локальные). См. `_settings_with_merged_tools`
        #: и `_execute_tool_call` — единственные два места, где он используется.
        self._mcp_client = mcp_client
        #: Общая лента изменений (ТЗ «асинхронные ответы», раздел 2.6): новые/
        #: изменённые/удалённые чаты, запуски, непрочитанные — клиент держит
        #: одно SSE-соединение `GET /events` и обновляет главный экран сразу.
        self.events = EventLog(max_events=1000, max_age_seconds=600)

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

    def new_agent_settings(self) -> Settings:
        """Полные настройки, которые получит новый агент: настройки по
        умолчанию + встроенные значения остальных полей. База сравнения для
        бейджей агента в списке (показываются только отличия)."""
        return settings_from_defaults(self._db.get_default_settings())

    def create_agent(self, name: Optional[str], model: Optional[str] = None) -> Agent:
        settings = self.new_agent_settings()
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
        chats = self._db.list_chats(agent_id)
        self._db.delete_agent(agent_id)  # ON DELETE CASCADE удаляет чаты и их сообщения
        for chat in chats:
            self._publish_chat_event("chat_deleted", chat.id, agent_id)

    # ---- чаты -----------------------------------------------------------------

    def _require_chat(self, chat_id: str) -> Chat:
        chat = self._db.get_chat(chat_id)
        if chat is None:
            raise NotFoundError(f"no such chat: {chat_id}")
        return chat

    def create_chat(self, agent_id: str, title: Optional[str], source: str = "app") -> Chat:
        """`source` — кто создаёт чат: "app" (пользователь) | "scheduler"
        (планировщик: «Запрос агенту»/«Задача агенту» в новый чат)."""
        agent = self._require_agent(agent_id)
        settings = dataclasses.replace(agent.settings)
        # Имя по умолчанию — "Чат N", где N — порядковый номер чата ВНУТРИ
        # этого агента (число уже существующих чатов агента + 1).
        resolved_title = title if title and title.strip() else f"Чат {len(self._db.list_chats(agent_id)) + 1}"
        # Профиль по умолчанию для агента (см. Agent.default_profile_id)
        # копируется в новый чат ТОЛЬКО в момент создания — точно так же, как
        # Settings копируются из agent.settings, а не как живая ссылка;
        # дальше чат может выбрать другой профиль независимо от агента.
        chat = self._db.create_chat(
            agent_id, resolved_title, settings, active_profile_id=agent.default_profile_id, source=source,
        )
        self._publish_chat_event("chat_created", chat.id, agent_id)
        return chat

    def find_latest_chat_by_title(self, agent_id: str, title: str) -> Optional[Chat]:
        self._require_agent(agent_id)
        return self._db.find_latest_chat_by_title(agent_id, title)

    def get_chat(self, chat_id: str) -> Chat:
        return self._require_chat(chat_id)

    def list_chats(self, agent_id: Optional[str] = None) -> List[Chat]:
        return self._db.list_chats(agent_id)

    def rename_chat(self, chat_id: str, title: str) -> Chat:
        self._require_chat(chat_id)
        if not title or not title.strip():
            raise ValidationError("title must be a non-empty string")
        self._db.rename_chat(chat_id, title)
        chat = self._require_chat(chat_id)
        self._publish_chat_event("chat_updated", chat_id, chat.agent_id)
        return chat

    def update_chat_settings(self, chat_id: str, payload: dict) -> Settings:
        chat = self._require_chat(chat_id)
        if "model" in payload and payload["model"] != chat.settings.model:
            raise ValidationError("model is fixed by the chat's agent and cannot be changed on a chat")
        _validate_settings_payload(payload)
        updated = _apply_partial(chat.settings, payload, _SETTINGS_FIELD_NAMES)
        self._db.update_chat_settings(chat_id, updated)
        self._publish_chat_event("chat_updated", chat_id, chat.agent_id)
        return updated

    def delete_chat(self, chat_id: str) -> None:
        chat = self._require_chat(chat_id)
        self._db.delete_chat(chat_id)
        self._publish_chat_event("chat_deleted", chat_id, chat.agent_id)

    def copy_chat(self, chat_id: str, new_title: str) -> Chat:
        chat = self._require_chat(chat_id)
        # Черновики идущего ответа не копируются — только завершённые сообщения.
        messages = [m for m in self._db.list_messages(chat_id) if m.status != "streaming"]
        new_chat = self._db.create_chat(chat.agent_id, new_title or f"{chat.title} (копия)", dataclasses.replace(chat.settings))
        for m in messages:
            self._db.add_message(
                Message(
                    id=0, chat_id=new_chat.id, role=m.role, content=m.content, created_at=m.created_at,
                    reasoning_content=m.reasoning_content, is_summary=m.is_summary, duration_ms=m.duration_ms,
                    total_tokens=m.total_tokens, prompt_tokens=m.prompt_tokens, completion_tokens=m.completion_tokens,
                    format=m.format, branch=m.branch, facts=m.facts, status=m.status, error=m.error,
                    tool_events=m.tool_events, task_events=m.task_events,
                    is_task_manager_step=m.is_task_manager_step, source=m.source,
                )
            )
        for b in self._db.list_branches(chat_id):
            self._db.create_branch(new_chat.id, b.name)
        # Копия считается прочитанной целиком — пользователь сам её создал.
        if messages:
            self._db.set_last_read(new_chat.id, max(m.id for m in self._db.list_messages(new_chat.id)))
        self._publish_chat_event("chat_created", new_chat.id, new_chat.agent_id)
        return self._require_chat(new_chat.id)

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
        self._publish_unread(chat_id)

    def bulk_delete_messages(self, chat_id: str, message_ids: List[int]) -> None:
        self._require_chat(chat_id)
        self._db.bulk_delete_messages(chat_id, message_ids)
        self._publish_unread(chat_id)

    def clear_messages(self, chat_id: str) -> None:
        self._require_chat(chat_id)
        self._db.clear_messages(chat_id)
        self._publish_unread(chat_id)

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
            messages = [m for m in self._db.list_messages(chat_id) if m.status == "complete"]
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

    # ---- тестовые диалоги (общий справочник) --------------------------------

    def _require_test_dialog(self, dialog_id: str) -> TestDialog:
        dialog = self._db.get_test_dialog(dialog_id)
        if dialog is None:
            raise NotFoundError(f"тестовый диалог {dialog_id} не найден")
        return dialog

    def list_test_dialogs(self) -> List[TestDialog]:
        return self._db.list_test_dialogs()

    def get_test_dialog(self, dialog_id: str) -> TestDialog:
        return self._require_test_dialog(dialog_id)

    def _clean_test_dialog_name(self, name: Optional[str], exclude_id: Optional[str] = None) -> str:
        cleaned = (name or "").strip()
        if not cleaned:
            raise ValidationError("name: нужно наименование тестового диалога")
        if len(cleaned) > test_dialog_rules.MAX_NAME_CHARS:
            raise ValidationError(f"name: не длиннее {test_dialog_rules.MAX_NAME_CHARS} символов")
        for other in self._db.list_test_dialogs():
            if other.id != exclude_id and other.name.casefold() == cleaned.casefold():
                raise ValidationError(f"name: тестовый диалог «{cleaned}» уже есть")
        return cleaned

    @staticmethod
    def _clean_test_dialog_questions(questions: List[str]) -> List[str]:
        try:
            return test_dialog_rules.validate_questions(list(questions))
        except ValueError as exc:
            raise ValidationError(str(exc)) from None

    def create_test_dialog(self, name: str, questions: List[str]) -> TestDialog:
        return self._db.create_test_dialog(
            self._clean_test_dialog_name(name), self._clean_test_dialog_questions(questions),
        )

    def update_test_dialog(self, dialog_id: str, payload: dict) -> TestDialog:
        self._require_test_dialog(dialog_id)
        unknown = set(payload) - {"name", "questions"}
        if unknown:
            raise ValidationError(f"unknown test dialog field(s): {', '.join(sorted(unknown))}")
        name = self._clean_test_dialog_name(payload["name"], exclude_id=dialog_id) if "name" in payload else None
        questions = (self._clean_test_dialog_questions(payload["questions"] or [])
                     if "questions" in payload else None)
        self._db.update_test_dialog(dialog_id, name=name, questions=questions)
        return self._require_test_dialog(dialog_id)

    def delete_test_dialog(self, dialog_id: str) -> None:
        self._require_test_dialog(dialog_id)
        self._db.delete_test_dialog(dialog_id)

    @staticmethod
    def parse_test_dialog_questions(text: str, existing: Optional[List[str]] = None) -> dict:
        """Импорт вопросов из текстового файла: строка = вопрос, дубликаты (в
        файле и среди уже записанных вопросов) пропускаются и перечисляются."""
        if text is None or not text.strip():
            raise ValidationError("text: файл пустой")
        return test_dialog_rules.parse_questions(text, existing or [])

    # ---- машина состояний задач (read-only, задана в коде) -------------------
    # "Работу с задачами требуется переделать" (новое ТЗ) — каталог
    # состояний/переходов больше не в БД, см. `task_state_machine.py`; здесь
    # остаётся только read-only описание для Android-экрана (замена формы
    # редактирования состояний/действий/машин) и настройка привязанных к
    # машине инвариантов категории "Правило стейт-машины".

    def get_task_state_machine_info(self) -> dict:
        """Read-only описание машин состояний (заданы в коде) обоих типов
        задач — номер, отображаемое и системное имя, список достижимых
        состояний для каждого этапа — плюс текущий список привязанных
        инвариантов категории "Правило стейт-машины" (см.
        `set_task_machine_invariants`; относятся к рабочей задаче)."""

        def _states(kind: str) -> List[dict]:
            return [
                {
                    "position": index,
                    "state": state,
                    "display_name": kind_display_name(kind, state),
                    "target_states": kind_allowed(kind, state),
                    "target_state_display_names": [kind_display_name(kind, t) for t in kind_allowed(kind, state)],
                }
                for index, state in enumerate(kind_states(kind), start=1)
            ]

        invariant_ids = set(self._db.get_task_machine_invariant_ids())
        invariants = [inv for inv in self._db.list_invariants() if inv.id in invariant_ids]
        return {"states": _states(KIND_WORK), "search_states": _states(KIND_SEARCH), "invariants": invariants}

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
    _SEARCH_STATUS_LABELS = {"active": "активна", "paused": "на паузе", "done": "цель достигнута"}

    def _require_task(self, task_id: str) -> Task:
        task = self._db.get_task(task_id)
        if task is None:
            raise NotFoundError(f"no such task: {task_id}")
        return task

    def _task_status(self, task: Task) -> str:
        """"active" | "paused" | "done" — см. `TASK_STATUS_OPTIONS`. "done" —
        конечное состояние машины своего типа ("done" у рабочей задачи,
        "achieved" у задачи поиска), "paused" — флаг `task.paused`
        (ортогональный состоянию), иначе "active"."""
        if kind_is_final(task.kind, task.state):
            return "done"
        return "paused" if task.paused else "active"

    def _task_status_display(self, task: Task, status: str) -> str:
        labels = self._SEARCH_STATUS_LABELS if task.kind == KIND_SEARCH else self._TASK_STATUS_LABELS
        return labels.get(status, status)

    def _task_next_state_display_name(self, task: Task) -> Optional[str]:
        """Этап, в который ведёт "Продолжить"/"Выполнить" из текущего этапа —
        ПЕРВАЯ цель по порядку переходов (она всегда прямое продолжение
        вперёд). `None`, если задача в конечном состоянии."""
        if kind_is_final(task.kind, task.state):
            return None
        targets = kind_allowed(task.kind, task.state)
        return kind_display_name(task.kind, targets[0]) if targets else None

    @staticmethod
    def _task_memory_dict(task: Task) -> dict:
        return {
            "goal": task.goal,
            "clarifications": list(task.clarifications),
            "constraints": list(task.constraints),
            "terms": list(task.terms),
            "sources": list(task.sources),
        }

    def _task_summary(self, task: Task, chat_title: Optional[str] = None) -> dict:
        status = self._task_status(task)
        summary = {
            "task": task,
            "status": status,
            "status_display": self._task_status_display(task, status),
            "state_display_name": kind_display_name(task.kind, task.state),
            "next_state_display_name": self._task_next_state_display_name(task) if status != "done" else None,
            "kind_display_name": TASK_KIND_LABELS.get(task.kind, task.kind),
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
        """Полные детали задачи для экрана "Задача": тип и память задачи,
        вычисляемый статус, степпер по этапам машины её типа (иконка: "check"
        — для уже пройденных, "pause" — для текущего, если задача на паузе,
        "none" — для ещё не достигнутых), доступные действия (продвижение в
        каждое из достижимых состояний + пауза) и полная история переходов."""
        task = self._require_task(task_id)
        status = self._task_status(task)
        states = kind_states(task.kind)
        current_index = states.index(task.state) if task.state in states else 0

        stages = []
        for index, state in enumerate(states):
            is_current = state == task.state
            if is_current and status == "paused":
                icon = "pause"
            elif index <= current_index:
                icon = "check"
            else:
                icon = "none"
            stages.append({
                "state": state,
                "display_name": kind_display_name(task.kind, state),
                "is_current": is_current,
                "is_final": kind_is_final(task.kind, state),
                "icon": icon,
            })

        available_actions = []
        # У задачи поиска из «Цель достигнута» можно вернуться к поиску
        # ответа — поэтому переходы предлагаются и для неё.
        if status != "done" or task.kind == KIND_SEARCH:
            for target in kind_allowed(task.kind, task.state):
                available_actions.append({
                    "kind": "advance",
                    "to_state": target,
                    "to_state_display_name": kind_display_name(task.kind, target),
                })
            if not task.paused and status != "done":
                available_actions.append({
                    "kind": "pause", "to_state": task.state,
                    "to_state_display_name": kind_display_name(task.kind, task.state),
                })

        history = [
            {
                "id": log.id,
                "from_state": log.from_state,
                "from_state_display_name": kind_display_name(task.kind, log.from_state),
                "to_state": log.to_state,
                "to_state_display_name": kind_display_name(task.kind, log.to_state),
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
            "status_display": self._task_status_display(task, status),
            "state_display_name": kind_display_name(task.kind, task.state),
            "next_state_display_name": self._task_next_state_display_name(task) if status != "done" else None,
            "kind_display_name": TASK_KIND_LABELS.get(task.kind, task.kind),
            "step": current_index + 1,
            "total": len(states),
            "stages": stages,
            "available_actions": available_actions,
            "history": history,
        }

    def apply_manual_task_action(self, task_id: str, action: str, note: Optional[str] = None) -> dict:
        """Ручное вмешательство человека — "Пауза" (кнопка "Продолжить"/
        "Выполнить" ВСЕГДА обращается к модели, см. `run_task_manager_step`).
        Ручная смена этапа — `set_task_state_manually`."""
        task = self._require_task(task_id)
        if action != "pause":
            raise ValidationError(f"unsupported manual action: {action!r} (only 'pause' is supported)")
        if kind_is_final(task.kind, task.state):
            raise ValidationError("cannot pause a task that is already done")
        if task.paused:
            raise ValidationError("task is already paused")
        self._db.set_task_paused(task.id, True)
        self._db.add_task_transition_log(task.id, task.state, task.state, kind="pause", applied_by="manual", note=note)
        return self.get_task(task.id)

    def set_task_state_manually(self, task_id: str, state: str, note: Optional[str] = None) -> dict:
        """Ручной переход по графу машины типа задачи (кнопка «Цель
        достигнута», «Вернуться к поиску ответа» и т.п.) — без обращения к
        модели. Снимает паузу; записывается в историю с applied_by='manual'."""
        task = self._require_task(task_id)
        if not kind_is_valid_state(task.kind, state):
            raise ValidationError(f"unknown state {state!r} for task kind {task.kind!r}; available: {kind_states(task.kind)}")
        if not kind_can_transition(task.kind, task.state, state):
            raise ValidationError(
                f"transition {task.state} -> {state} is not allowed; available: {kind_allowed(task.kind, task.state)}"
            )
        self._db.update_task_progress(task.id, state=state, _current_step_set=False)
        self._db.set_task_paused(task.id, False)
        self._db.add_task_transition_log(task.id, task.state, state, kind="advance", applied_by="manual", note=note)
        return self.get_task(task.id)

    def update_task_memory_manually(
        self, task_id: str, title: Optional[str] = None, goal: Optional[str] = None,
        clarifications: Optional[List[str]] = None, constraints: Optional[List[str]] = None,
        terms: Optional[List[dict]] = None, clear_sources: bool = False,
    ) -> dict:
        """Ручная правка памяти задачи с экрана «Задача»: переданные списки
        ЗАМЕНЯЮТ текущие целиком (у сохранённых строк сохраняется ссылка на
        сообщение-источник, если текст не изменился)."""
        task = self._require_task(task_id)

        def _merge_texts(current: List[dict], texts: Optional[List[str]]) -> Optional[List[dict]]:
            if texts is None:
                return None
            by_text = {str(item.get("text") or ""): item for item in current}
            result: List[dict] = []
            for text in texts:
                text = str(text or "").strip()
                if text and all(r["text"] != text for r in result):
                    result.append(by_text.get(text) or {"text": text, "message_id": None})
            return result

        new_terms: Optional[List[dict]] = None
        if terms is not None:
            new_terms = []
            for item in terms:
                term = str((item or {}).get("term") or "").strip()
                definition = str((item or {}).get("definition") or "").strip()
                if term and definition:
                    new_terms = [t for t in new_terms if t["term"].lower() != term.lower()]
                    new_terms.append({"term": term, "definition": definition})
        self._db.update_task_memory(
            task.id,
            title=title.strip() if title and title.strip() else None,
            goal=goal.strip() if goal is not None else None,
            clarifications=_merge_texts(task.clarifications, clarifications),
            constraints=_merge_texts(task.constraints, constraints),
            terms=new_terms,
            sources=[] if clear_sources else None,
        )
        self._db.add_task_transition_log(task.id, task.state, task.state, kind="memory", applied_by="manual", note=None)
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

    # ---- start_task / apply_task_action / update_task_memory (tool-calling) --

    def _open_tasks_for_chat(self, chat: Chat) -> List[Task]:
        return [t for t in self._db.list_tasks_for_chat(chat.id) if self._task_status(t) != "done"]

    def _current_task(self, chat: Chat) -> Optional[Task]:
        """Текущая задача чата — последняя незакрытая. Исключение — задача
        поиска в состоянии «Цель достигнута», если она последняя из
        созданных: её можно продолжить новым вопросом по той же цели
        (achieved → answering), поэтому она остаётся текущей."""
        tasks = self._db.list_tasks_for_chat(chat.id)
        if not tasks:
            return None
        latest = tasks[-1]
        if latest.kind == KIND_SEARCH:
            return latest
        open_tasks = [t for t in tasks if self._task_status(t) != "done"]
        return open_tasks[-1] if open_tasks else None

    def _current_search_task(self, chat: Chat) -> Optional[Task]:
        if not (chat.settings.task_tracking_enabled and chat.settings.rag_enabled):
            return None
        task = self._current_task(chat)
        return task if task is not None and task.kind == KIND_SEARCH else None

    def _task_allowed_target_states(self, task: Task) -> List[str]:
        return kind_allowed(task.kind, task.state)

    def _available_task_kinds(self, chat: Chat) -> List[str]:
        """Задача поиска имеет смысл только при включённой базе знаний."""
        return [KIND_WORK, KIND_SEARCH] if chat.settings.rag_enabled else [KIND_WORK]

    def _last_user_message_id(self, chat_id: str) -> Optional[int]:
        return self._db.last_user_message_id(chat_id)

    @staticmethod
    def _text_items(raw: Any) -> List[str]:
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return []
        return [str(x).strip() for x in raw if str(x or "").strip()]

    @staticmethod
    def _term_items(raw: Any) -> List[dict]:
        items = []
        for item in raw if isinstance(raw, list) else []:
            if isinstance(item, dict):
                term = str(item.get("term") or "").strip()
                definition = str(item.get("definition") or "").strip()
            elif isinstance(item, str) and ("=" in item or "—" in item):
                term, _, definition = item.replace("—", "=", 1).partition("=")
                term, definition = term.strip(), definition.strip()
            else:
                continue
            if term and definition:
                items.append({"term": term, "definition": definition})
        return items

    def _apply_memory_changes(self, task: Task, arguments: dict, message_id: Optional[int]) -> List[dict]:
        """Слияние изменений памяти задачи (инструменты `start_task`/
        `update_task_memory`) и запись в БД. Возвращает список изменений для
        строки под ответом: [{"op": "+"|"-"|"=", "field": ..., "text": ...}]."""
        changes: List[dict] = []
        clarifications = [dict(c) for c in task.clarifications]
        constraints = [dict(c) for c in task.constraints]
        terms = [dict(t) for t in task.terms]
        goal = None

        new_goal = str(arguments.get("goal") or "").strip()
        if new_goal and new_goal != task.goal:
            goal = new_goal
            changes.append({"op": "=", "field": "goal", "text": new_goal})

        for raw in self._text_items(arguments.get("remove")):
            key = raw.casefold()
            for field_name, items in (("clarification", clarifications), ("constraint", constraints)):
                for item in list(items):
                    if item.get("text", "").casefold() == key:
                        items.remove(item)
                        changes.append({"op": "-", "field": field_name, "text": item["text"]})
            for item in list(terms):
                if item.get("term", "").casefold() == key:
                    terms.remove(item)
                    changes.append({"op": "-", "field": "term", "text": item["term"]})

        for field_name, items, key in (("clarification", clarifications, "add_clarifications"),
                                       ("constraint", constraints, "add_constraints")):
            for text in self._text_items(arguments.get(key)):
                if any(i.get("text", "").casefold() == text.casefold() for i in items):
                    continue
                items.append({"text": text, "message_id": message_id})
                changes.append({"op": "+", "field": field_name, "text": text})

        for term in self._term_items(arguments.get("add_terms")):
            existing = next((t for t in terms if t["term"].casefold() == term["term"].casefold()), None)
            if existing is not None and existing["definition"] == term["definition"]:
                continue
            if existing is not None:
                terms.remove(existing)
            terms.append(term)
            changes.append({"op": "+", "field": "term", "text": f"{term['term']} — {term['definition']}"})

        if changes:
            self._db.update_task_memory(
                task.id, goal=goal, clarifications=clarifications, constraints=constraints, terms=terms,
            )
        return changes

    def _memory_event(self, task: Task, changes: List[dict]) -> dict:
        return {
            "task_id": task.id, "task_title": task.title, "task_kind": task.kind, "kind": "memory",
            "from_state_display_name": None, "to_state_display_name": kind_display_name(task.kind, task.state),
            "changes": changes,
        }

    def _pause_after_stage(self, chat: Chat, auto_pause: bool) -> bool:
        """«Останавливаться на каждом этапе» (по умолчанию выключено) — в
        обычном чате пауза после этапа ставится только с этой настройкой;
        кнопка «Выполнить» (auto_pause=false) паузу не ставит никогда."""
        return auto_pause and chat.settings.task_pause_each_stage

    def _handle_start_task(self, chat: Chat, agent: Agent, arguments: dict, auto_pause: bool = True, events: Optional[List[dict]] = None) -> dict:
        if not chat.settings.task_tracking_enabled:
            return {"error": "task tracking is disabled for this chat"}
        title = str(arguments.get("title") or "").strip()
        if not title:
            return {"error": "title is required"}
        kind = str(arguments.get("kind") or KIND_WORK).strip().lower()
        if kind not in self._available_task_kinds(chat):
            return {"error": f"kind must be one of {self._available_task_kinds(chat)}"}
        plan_raw = arguments.get("plan")
        plan = [str(s).strip() for s in plan_raw if str(s).strip()] if isinstance(plan_raw, list) and kind == KIND_WORK else []
        goal = str(arguments.get("goal") or "").strip() or (title if kind == KIND_SEARCH else "")
        state = kind_initial_state(kind)
        if kind == KIND_SEARCH and arguments.get("needs_clarification") is True:
            state = "clarifying"
        pause = chat.settings.task_pause_each_stage
        task = self._db.create_task(chat.id, title, plan=plan, kind=kind, state=state, goal=goal, paused=pause)
        if pause:
            self._db.add_task_transition_log(task.id, state, state, kind="pause", applied_by="system", note=None)
        memory_args = {
            "add_clarifications": arguments.get("clarifications") or arguments.get("add_clarifications"),
            "add_constraints": arguments.get("constraints") or arguments.get("add_constraints"),
            "add_terms": arguments.get("terms") or arguments.get("add_terms"),
        }
        changes = self._apply_memory_changes(task, memory_args, self._last_user_message_id(chat.id))
        state_name = kind_display_name(kind, state)
        event = {
            "task_id": task.id, "task_title": task.title, "task_kind": kind, "goal": goal,
            "from_state_display_name": None, "to_state_display_name": state_name, "kind": "advance",
        }
        if changes:
            event["changes"] = changes
        return {
            "started": True,
            "task_id": task.id,
            "kind": kind,
            "title": task.title,
            "goal": goal,
            "state": state_name,
            "status": "paused" if pause else "active",
            "_event": event,
        }

    def _handle_update_task_memory(self, chat: Chat, agent: Agent, arguments: dict, auto_pause: bool = True, events: Optional[List[dict]] = None) -> dict:
        if not chat.settings.task_tracking_enabled:
            return {"error": "task tracking is disabled for this chat"}
        task_id = arguments.get("task_id")
        task = self._db.get_task(task_id) if task_id else None
        if task is None or task.chat_id != chat.id:
            return {"error": "unknown task_id for this chat"}
        changes = self._apply_memory_changes(task, arguments, self._last_user_message_id(chat.id))
        updated = self._db.get_task(task.id) or task
        result: Dict[str, Any] = {
            "updated": bool(changes), "task_id": task.id, "changes": changes,
            "memory": {k: v for k, v in self._task_memory_dict(updated).items() if k != "sources"},
        }
        if changes:
            result["_event"] = self._memory_event(updated, changes)
        return result

    def _handle_apply_task_action(self, chat: Chat, agent: Agent, arguments: dict, auto_pause: bool = True, events: Optional[List[dict]] = None) -> dict:
        if not chat.settings.task_tracking_enabled:
            return {"error": "task tracking is disabled for this chat"}
        task_id = arguments.get("task_id")
        task = self._db.get_task(task_id) if task_id else None
        if task is None or task.chat_id != chat.id:
            return {"error": "unknown task_id for this chat"}
        current_state = task.state
        if task.kind == KIND_WORK and kind_is_final(task.kind, current_state):
            return {"error": "task is already done"}

        target_state_raw = arguments.get("target_state")
        to_state = current_state
        if target_state_raw:
            to_state = str(target_state_raw).strip().lower()
            if not kind_is_valid_state(task.kind, to_state):
                return {"error": f"unknown target_state for this task: {target_state_raw!r}; "
                                 f"available: {self._task_allowed_target_states(task)}"}
            if to_state != current_state and not kind_can_transition(task.kind, current_state, to_state):
                options = sorted(self._task_allowed_target_states(task))
                return {"error": f"transition {current_state} -> {to_state} is not allowed; available: {options}"}

        pause_after = self._pause_after_stage(chat, auto_pause)
        # «Останавливаться на каждом этапе»: одна и та же задача может быть
        # продвинута не более ОДНОГО раза за один ответ модели — иначе
        # пользователь не успеет увидеть промежуточный этап.
        if pause_after and to_state != current_state and events is not None:
            if any(e.get("task_id") == task.id and e.get("kind") == "advance" for e in events):
                return {
                    "error": (
                        "task already advanced once in this turn; the task manager pauses after a "
                        "single step in this mode — stop here and wait for the user to confirm "
                        "before continuing this task further"
                    )
                }

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
            state=to_state if to_state != current_state else None,
            current_step=current_step_raw if current_step_raw is not None else task.current_step,
            done_steps=done_steps if completed_step else None,
        )
        if to_state != current_state:
            self._db.add_task_transition_log(
                task.id, current_state, to_state, kind="advance", applied_by="agent", note=note,
            )
        if pause_after and to_state != current_state and not kind_is_final(task.kind, to_state):
            self._db.set_task_paused(task.id, True)
            self._db.add_task_transition_log(
                task.id, to_state, to_state, kind="pause", applied_by="system", note=None,
            )
        updated_task = self._db.get_task(task.id)
        status = self._task_status(updated_task) if updated_task is not None else "active"
        return {
            "applied": True,
            "task_id": task.id,
            "state": kind_display_name(task.kind, to_state),
            "status": status,
            "_event": {
                "task_id": task.id, "task_title": task.title, "task_kind": task.kind,
                "from_state_display_name": kind_display_name(task.kind, current_state),
                "to_state_display_name": kind_display_name(task.kind, to_state),
                "kind": "advance" if to_state != current_state else "update",
            },
        }

    def _auto_search_transition(self, chat: Chat, task: Task, target: str, reason: str) -> Optional[dict]:
        """Переход задачи поиска по итогам ответа (код, не модель).
        achieved → clarifying идёт через answering (по графу)."""
        path: List[str] = []
        current = task.state
        if target == current:
            return None
        if kind_can_transition(task.kind, current, target):
            path = [target]
        elif current == "achieved" and target == "clarifying":
            path = ["answering", "clarifying"]
        else:
            return None
        start = current
        for state in path:
            self._db.add_task_transition_log(task.id, current, state, kind="advance", applied_by="system", note=reason)
            current = state
        self._db.update_task_progress(task.id, state=current, _current_step_set=False)
        pause = chat.settings.task_pause_each_stage and not kind_is_final(task.kind, current)
        self._db.set_task_paused(task.id, pause)
        if pause:
            self._db.add_task_transition_log(task.id, current, current, kind="pause", applied_by="system", note=None)
        return {
            "task_id": task.id, "task_title": task.title, "task_kind": task.kind, "kind": "state",
            "from_state_display_name": kind_display_name(task.kind, start),
            "to_state_display_name": kind_display_name(task.kind, current),
            "note": reason,
        }

    _TOOL_HANDLERS = {
        "save_working_memory": _handle_save_working_memory,
        "save_long_term_memory": _handle_save_long_term_memory,
        "search_products": _handle_search_products,
        "add_to_cart": _handle_add_to_cart,
        "view_cart": _handle_view_cart,
        "start_task": _handle_start_task,
        "apply_task_action": _handle_apply_task_action,
        "update_task_memory": _handle_update_task_memory,
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
            tools.append(_build_start_task_tool(self._available_task_kinds(chat)))
            open_tasks = self._open_tasks_for_chat(chat)
            current = self._current_task(chat)
            if current is not None and current not in open_tasks:
                open_tasks.append(current)
            if open_tasks:
                tools.append(_build_update_task_memory_tool())
                target_states = sorted({
                    state for task in open_tasks for state in self._task_allowed_target_states(task)
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
        search_task = self._current_search_task(chat)
        if search_task is not None and search_task.state != "achieved":
            # Задача поиска: ассистент только ищет и отвечает — инструменты,
            # которые что-то делают (git, файлы, планировщик, скиллы), модели
            # не предлагаются; остаются чтение (read_chat), память и задачи.
            tools = [t for t in tools if _tool_name(t) in self._SEARCH_TASK_TOOLS]
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

    def list_mcp_tools(self) -> List[dict]:
        """Инструменты для выбора в настройках: всех подключённых MCP-серверов
        (см. `mcp_client.MCPClient.list_tool_details`) и встроенные инструменты
        группы «Доступ к agents core» (`builtin_tools`) — для пользователя они
        выглядят так же, как инструменты MCP."""
        mcp_tools = self._mcp_client.list_tool_details() if self._mcp_client is not None else []
        mcp_tools = [t for t in mcp_tools if not builtin_tools.is_builtin(t["name"])]
        return mcp_tools + builtin_tools.tool_details()

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
        offered = self._offered_tool_names(settings)
        if self._mcp_client is not None and any(
            self._mcp_client.has_cached_tool(name) and not builtin_tools.is_builtin(name) for name in offered
        ):
            sources.append("mcp")
        if any(builtin_tools.is_builtin(name) for name in offered):
            sources.append("agents_core")
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
        allowed_mcp_names: Optional[set] = None,
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
        elif builtin_tools.is_builtin(name) and (allowed_mcp_names is None or name in allowed_mcp_names):
            # Встроенный инструмент «Доступ к agents core» — выполняется здесь,
            # без MCP; доступен, только если выбран в настройках (tools_json).
            try:
                result = builtin_tools.execute(self, chat, name, arguments)
            except Exception as exc:  # noqa: BLE001
                result = {"error": str(exc)}
        elif (
            self._mcp_client is not None
            and self._mcp_client.has_cached_tool(name)
            and (allowed_mcp_names is None or name in allowed_mcp_names)
        ):
            # Инструмент неизвестен локально, но известен MCP-серверу —
            # диспетчеризация туда (новое ТЗ, интеграция с MCP-сервером).
            meta = {"agentscore/chat_id": chat.id, "agentscore/agent_id": agent.id}
            if allowed_mcp_names is not None:
                # Какие MCP-инструменты разрешены в этом чате: сервер не даёт
                # вызвать остальные косвенно (цепочка run_pipeline,
                # save_tool_result).
                meta["agentscore/allowed_tools"] = sorted(
                    n for n in allowed_mcp_names if self._mcp_client.has_cached_tool(n)
                )
            try:
                result = self._mcp_client.call_tool(name, arguments, meta=meta)
            except MCPClientError as exc:  # сетевой сбой самого MCP-сервера
                result = {"error": str(exc)}
        else:
            result = {"error": f"unknown tool: {name!r}"}
        event = result.pop("_event", None) if isinstance(result, dict) else None
        if event is not None and events is not None:
            events.append(event)
        return json.dumps(result, ensure_ascii=False)

    def _run_tool_loop_blocking(
        self, provider, model_id: str, messages: List[ProviderMessage], settings: Settings,
        chat: Chat, agent: Agent, result: ChatResult, events: Optional[List[dict]] = None,
    ) -> ChatResult:
        iterations = 0
        allowed_mcp_names = self._offered_tool_names(settings)
        while result.tool_calls and iterations < _MAX_TOOL_ITERATIONS:
            messages.append(ProviderMessage("assistant", result.content, tool_calls=result.tool_calls))
            for call in result.tool_calls:
                tool_output = self._execute_tool_call(
                    chat, agent, call, events=events, allowed_mcp_names=allowed_mcp_names,
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

    @staticmethod
    def _task_memory_lines(task: Task) -> List[str]:
        lines: List[str] = []
        if task.goal:
            lines.append(f"Цель: {task.goal}")
        if task.clarifications:
            lines.append("Уточнения пользователя:")
            lines += [f"- {c.get('text', '')}" for c in task.clarifications]
        if task.constraints:
            lines.append("Ограничения пользователя (соблюдай всегда):")
            lines += [f"- {c.get('text', '')}" for c in task.constraints]
        if task.terms:
            lines.append("Термины (используй в этом значении):")
            lines += [f"- {t.get('term', '')} — {t.get('definition', '')}" for t in task.terms]
        return lines

    def _build_task_context_block(self, task: Task) -> str:
        """Скрытый служебный блок с ТЕКУЩИМИ данными ОДНОЙ рабочей задачи —
        task/state/step/total/plan/done/current и id задачи (нужен для
        `apply_task_action`/`update_task_memory`), плюс память задачи (цель,
        уточнения, ограничения, термины). Добавляется в промпт модели
        отдельным `system`-сообщением; текст запроса пользователя не меняется.

        `step`/`total` — позиция текущего состояния в машине состояний и
        общее число её состояний."""
        states = kind_states(task.kind)
        position = states.index(task.state) + 1 if task.state in states else 1
        lines = [
            f"[ЗАДАЧА] id: {task.id}",
            f"task: {task.title}",
            *([f"description: {task.description}"] if task.description else []),
            f"state: {task.state}",
            f"step: {position}",
            f"total: {len(states)}",
            f"plan: {json.dumps(task.plan, ensure_ascii=False)}",
            f"done: {json.dumps(task.done_steps, ensure_ascii=False)}",
            f"current: {task.current_step or ''}",
            *self._task_memory_lines(task),
        ]
        return "\n".join(lines)

    def _build_search_task_block(self, task: Task) -> str:
        """[ЗАДАЧА ПОИСКА] — память и правила текущей задачи поиска. Цель,
        ограничения и термины остаются в промпте, даже если ранние сообщения
        выпали из окна контекста (Sliding Window, суммаризация)."""
        targets = kind_allowed(task.kind, task.state)
        lines = [
            f"[ЗАДАЧА ПОИСКА] id: {task.id}",
            f"Название: {task.title}",
            *self._task_memory_lines(task),
            f"Состояние: {task.state} («{kind_display_name(task.kind, task.state)}»); "
            f"доступные переходы: {', '.join(targets) or 'нет'}",
            "Правила задачи поиска:",
            "- Ты ищешь ответ в базе знаний и отвечаешь в рамках цели. Ничего не выполняй сам (не меняй файлы, "
            "не запускай команды, не создавай задачи планировщика) — давай инструкции со ссылками на источники.",
            "- Соблюдай ограничения. Если найденный в базе способ им противоречит — скажи об этом и предложи "
            "вариант, который не противоречит, или задай уточняющий вопрос.",
            "- Используй термины в зафиксированном значении.",
            "- Если пользователь уточнил цель, поставил ограничение, определил термин или отменил сказанное "
            "раньше — сначала вызови update_task_memory, потом отвечай.",
            "- Служебные поля задачи (id, state, step, total, plan, done, current) в ответе не печатай и "
            "подтверждения на продолжение не запрашивай.",
            "- Если пользователь просит повторить, сократить, перевести или переформулировать предыдущий "
            "ответ — сделай это по тем же фрагментам и с теми же номерами [n].",
            "- Если вопрос уводит в сторону от цели — ответь на него и одной фразой вернись к цели.",
            "- Когда пользователь получил итоговый ответ (например, собрана итоговая инструкция) или подтвердил, "
            "что вопрос закрыт, — переведи задачу в achieved через apply_task_action.",
            "- Если пользователь перешёл к другой цели — начни новую задачу поиска (start_task, kind=search); "
            "если просит что-то сделать, а не найти — рабочую задачу (start_task, kind=work).",
        ]
        return "\n".join(lines)

    def _build_task_types_hint(self, chat: Chat) -> str:
        lines = []
        if chat.settings.rag_enabled:
            lines += [
                "[ТИПЫ ЗАДАЧ]",
                "«Рабочая задача» (kind=work) — что-то сделать; этапы и формат параметров — как описано выше.",
                "«Задача поиска» (kind=search) — найти ответ/инструкцию в базе знаний. Состояния: "
                f"{format_search_states_for_prompt()}. Формат параметров task/state/step/total/plan/done/current "
                "и запрос подтверждения к задаче поиска не относятся.",
                "Если пользователь спрашивает «как сделать…», «где описано…», «что такое…» — это задача поиска.",
            ]
        if not chat.settings.task_pause_each_stage:
            lines.append(
                "Режим «Останавливаться на каждом этапе» выключен: НЕ запрашивай подтверждение пользователя на "
                "переход к следующему этапу — продвигай задачу сам; останавливайся, только если для продолжения "
                "нужен ответ пользователя."
            )
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
            hint = self._build_task_types_hint(chat)
            if hint:
                blocks.append(ProviderMessage("system", hint))
            for task in self._open_tasks_for_chat(chat):
                if task.kind == KIND_WORK:
                    blocks.append(ProviderMessage("system", self._build_task_context_block(task)))
            search_task = self._current_search_task(chat)
            if search_task is not None:
                blocks.append(ProviderMessage("system", self._build_search_task_block(search_task)))
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

    # ---- асинхронные запуски: подготовка и исполнители (ТЗ, этап 1) ----------
    #
    # Ответ модели больше не живёт внутри HTTP-запроса: `runs.RunManager`
    # создаёт запуск и выполняет его в фоновом пуле, а здесь — сама работа.
    # Подготовка (`prepare_*`) создаёт сообщение пользователя и ЧЕРНОВИК
    # ответа ассистента (`status="streaming"`) синхронно, чтобы клиент сразу
    # получил их в ответе на `POST /chats/{id}/runs`. Исполнитель
    # (`execute_*`) — генератор событий (`status`/`delta`/`tool_call`/
    # `task_event`/`message_saved`/`done`/`cancelled`/`error`) — дописывает
    # черновик в итоговое сообщение. Прежние синхронные методы
    # `send_message_blocking`/`stream_message`/`run_task_manager_step`
    # остались тонкими обёртками над теми же исполнителями.
    # ---------------------------------------------------------------------------

    def _history_for_context(
        self, chat_id: str, branch: Optional[int], before_id: Optional[int] = None,
    ) -> List[Message]:
        """История для контекста модели: нужная ветка, только завершённые
        сообщения (`status == "complete"` — черновики, остановленные,
        прерванные и неудавшиеся ответы модели не передаются, ТЗ 2.4) и
        только сообщения ДО `before_id` (текущий запрос и его черновик
        добавляются в контекст отдельно)."""
        messages = self._filter_by_branch(self._db.list_messages(chat_id), branch)
        return [m for m in messages if m.status == "complete" and (before_id is None or m.id < before_id)]

    def _validate_rag_config(self, chat: Chat) -> None:
        if not chat.settings.rag_enabled:
            return
        if self._knowledge is None:
            raise ValidationError("RAG недоступен: не задан адрес сервиса баз знаний (KNOWLEDGE_SERVICE_URL)")
        if not chat.settings.collection_ids:
            raise ValidationError("Включено «Использовать RAG», но не выбраны базы знаний")

    def _rewrite_query(
        self, chat: Chat, history: List[Message], text: str, task: Optional[Task] = None,
    ) -> Tuple[str, dict]:
        """Поисковый запрос по настройке «Переписывание запроса»:
        «Нет» — вопрос как есть; «Дополнять уточняющие» — короткий вопрос
        склеивается с предыдущим вопросом пользователя; «Моделью до поиска» —
        модель (модель переписывания или модель агента) по последним репликам
        диалога и памяти текущей задачи (цель, термины, ограничения)
        формулирует самостоятельный поисковый запрос — или отвечает
        «БЕЗ_ПОИСКА», если пользователь просит что-то сделать с предыдущим
        ответом (`info["no_search"]`). Ошибка модели не прерывает ответ —
        поиск идёт по исходному вопросу."""
        mode = chat.settings.rag_query_rewrite
        info: dict = {"mode": mode, "rewritten": False, "model": None, "error": None, "tokens": None,
                      "no_search": False, "task_memory": False}
        if mode == "off":
            return text.strip(), info
        if mode != "llm":
            previous = next((m.content for m in reversed(history) if m.role == "user" and not m.is_summary), None)
            query = search_query(text, previous)
            info["rewritten"] = query != text.strip()
            return query, info
        model = chat.settings.rag_rewrite_model or chat.settings.model
        info["model"] = model
        memory_lines = self._task_memory_lines(task) if task is not None else []
        info["task_memory"] = bool(memory_lines)
        try:
            if self._catalog.get(model) is None:
                raise ProviderError(f"модель переписывания {model!r} не найдена в каталоге")
            provider_name, model_id = _split_model(model)
            provider = self._registry.get(provider_name)
            request_settings = dataclasses.replace(
                chat.settings, model=model, temperature=0.0, top_p=1.0, max_tokens=150, thinking_enabled=False,
                reasoning_effort=None, json_mode=False, stop_sequences=[], tools_json="", tool_choice="auto",
                logprobs=False, frequency_penalty=None, presence_penalty=None,
            )
            result = provider.chat(model_id, [
                ProviderMessage("system", rewrite_instruction(memory_lines)),
                ProviderMessage("user", rewrite_request_text(history, text)),
            ], request_settings)
            info["tokens"] = result.usage.total_tokens if result.usage else None
            query = clean_rewritten(result.content)
            if not query:
                raise ProviderError("модель вернула пустой или слишком длинный запрос")
        except (ProviderError, ValidationError) as exc:
            info["error"] = f"переписать не удалось: {exc}"
            return text.strip(), info
        if is_no_search(query):
            info["no_search"] = True
            return text.strip(), info
        info["rewritten"] = query != text.strip()
        return query, info

    @staticmethod
    def _previous_rag(history: List[Message]) -> Tuple[Optional[Message], Optional[dict]]:
        """Последний ответ ассистента с найденными фрагментами — для «БЕЗ_ПОИСКА»."""
        for message in reversed(history):
            if message.role != "assistant" or not message.rag:
                continue
            try:
                rag = json.loads(message.rag)
            except (ValueError, TypeError):
                continue
            if isinstance(rag, dict) and rag.get("sources"):
                return message, rag
        return None, None

    #: Сколько фрагментов хранит задача поиска (`Task.sources`) и сколько из
    #: них подмешивается к новым результатам поиска.
    TASK_SOURCES_LIMIT = 12
    TASK_SOURCES_IN_CONTEXT = 4

    @staticmethod
    def _stored_source(src: dict) -> dict:
        keys = ("chunk_id", "text", "section", "page", "document_id", "title", "source", "source_type",
                "doc_date", "doc_version", "collection_id", "collection_name")
        return {k: src.get(k) for k in keys}

    def _merge_task_sources(self, outcome: RagOutcome, task: Task, budget_tokens: int) -> None:
        """Фрагменты, накопленные задачей поиска, — после новых результатов
        (в пределах бюджета контекста, без повторов). Если поиск ничего не
        нашёл — они единственный контекст ответа."""
        if not task.sources:
            return
        present = {s.get("chunk_id") for s in outcome.sources}
        used = sum(estimate_tokens(s.get("text") or "") for s in outcome.sources)
        fallback = not outcome.sources
        limit = len(task.sources) if fallback else self.TASK_SOURCES_IN_CONTEXT
        added = 0
        for stored in reversed(task.sources):
            if added >= limit:
                break
            if stored.get("chunk_id") in present:
                continue
            tokens = estimate_tokens(stored.get("text") or "")
            if outcome.sources and used + tokens > budget_tokens:
                break
            outcome.sources.append({**stored, "n": len(outcome.sources) + 1, "score": None,
                                    "vector_score": None, "rerank_score": None, "from_task": True})
            present.add(stored.get("chunk_id"))
            used += tokens
            added += 1
        outcome.task_sources_added = added
        if added and fallback:
            outcome.task_sources_fallback = True
            outcome.status = RAG_STATUS_OK

    def _accumulate_task_sources(self, task: Task, outcome: RagOutcome) -> int:
        """Фрагменты, на которые сослался ответ, — в `Task.sources` (дедупликация
        по chunk_id, повторно процитированный переносится в конец, лимит —
        `TASK_SOURCES_LIMIT` последних)."""
        cited = [s for s in outcome.sources if s.get("n") in set(outcome.cited) and s.get("chunk_id")]
        if not cited:
            return 0
        stored = [dict(s) for s in task.sources]
        new_count = 0
        for src in cited:
            existing = next((x for x in stored if x.get("chunk_id") == src.get("chunk_id")), None)
            if existing is not None:
                stored.remove(existing)
            else:
                new_count += 1
            stored.append(self._stored_source(src))
        self._db.update_task_memory(task.id, sources=stored[-self.TASK_SOURCES_LIMIT:])
        return new_count

    def _prepare_rag(
        self, chat: Chat, history: List[Message], text: str, provider_messages: List[ProviderMessage],
        query: Optional[str] = None, rewrite: Optional[dict] = None, task: Optional[Task] = None,
    ) -> RagOutcome:
        """Поиск фрагментов и сборка запроса: перед последним сообщением
        (вопросом) — инструкция RAG, сам вопрос заменяется на «Контекст +
        вопрос». В историю чата уходит исходный вопрос, фрагменты — только в
        `Message.rag` ответа. Недоступность сервиса не прерывает ответ: модель
        отвечает без базы знаний и предупреждает об этом.

        «БЕЗ_ПОИСКА» (`rewrite["no_search"]`): нового поиска нет — фрагменты и
        их номера берутся из предыдущего ответа. В задаче поиска (`task`) к
        найденному добавляются фрагменты, накопленные задачей."""
        settings = chat.settings
        if query is None:
            query, rewrite = self._rewrite_query(chat, history, text, task)
        rewrite = rewrite or {}
        candidate_k = max(settings.rag_candidate_k, settings.rag_top_k)
        outcome = RagOutcome(
            query=query, status=RAG_STATUS_OK, only_from_kb=settings.rag_only_from_kb, original_query=text.strip(),
            quotes_required=settings.rag_quotes,
            rewrite=rewrite, params={
                "candidate_k": candidate_k, "top_k": settings.rag_top_k,
                "score_threshold": settings.rag_score_threshold, "rerank": settings.rag_rerank,
                "rerank_threshold": settings.rag_rerank_threshold if settings.rag_rerank != "none" else None,
            },
        )
        previous_message, previous_rag = self._previous_rag(history) if rewrite.get("no_search") else (None, None)
        if previous_rag is not None:
            outcome.sources = [dict(s) for s in previous_rag.get("sources") or []]
            outcome.query = previous_rag.get("query") or query
            outcome.context_reused = {"message_id": previous_message.id}
        else:
            if rewrite.get("no_search"):
                rewrite["no_search_note"] = "предыдущего ответа с фрагментами нет — выполнен обычный поиск"
            try:
                found = self._knowledge.retrieve(
                    query, list(settings.collection_ids), settings.rag_top_k, settings.rag_score_threshold,
                    candidate_k=candidate_k, rerank=settings.rag_rerank, rerank_model=settings.rag_rerank_model or None,
                    rerank_threshold=settings.rag_rerank_threshold,
                )
                outcome.stages = found.get("stages") or {}
                outcome.missing_collections = list(found.get("missing_collections") or [])
                if outcome.missing_collections:
                    _logger.warning("Чат %s: базы знаний не найдены в knowledge_service: %s",
                                    chat.id, ", ".join(outcome.missing_collections))
                outcome.rerank = found.get("rerank") or {}
                outcome.sources = select_sources(found["results"], settings.rag_context_tokens)
                if not outcome.sources:
                    outcome.status = RAG_STATUS_NOT_FOUND
            except KnowledgeServiceError as exc:
                outcome.status, outcome.error = RAG_STATUS_UNAVAILABLE, str(exc)
                # Причина видна под ответом и в логе: без неё «База знаний недоступна»
                # не отличить от сетевой ошибки, удалённой базы или сбоя эмбеддингов.
                _logger.warning("Чат %s: поиск в базе знаний не удался: %s", chat.id, exc)
            if task is not None and outcome.status != RAG_STATUS_UNAVAILABLE:
                self._merge_task_sources(outcome, task, settings.rag_context_tokens)
        question = provider_messages[-1]
        provider_messages[-1:] = [
            ProviderMessage("system", rag_instruction(outcome)),
            ProviderMessage(question.role, augmented_question(question.content, outcome)),
        ]
        return outcome

    def _after_search_answer(
        self, chat: Chat, rag: RagOutcome, answer: str, acc: "_TurnAccumulator",
    ) -> List[dict]:
        """После ответа в задаче поиска: процитированные фрагменты — в
        `Task.sources`; переход состояния по итогам ответа (если модель сама
        не меняла состояние задачи в этом ходе):
        «Не знаю» / уточняющий вопрос без ссылок → «Уточнение»;
        ответ со ссылками из «Уточнения» или «Цель достигнута» → «Поиск ответа».
        Возвращает новые события задачи (они же добавлены в `acc.task_events`)."""
        task = self._current_search_task(chat)
        if task is None:
            return []
        events: List[dict] = []
        self._accumulate_task_sources(task, rag)
        rag.dont_know = rag.dont_know or starts_with_dont_know(answer)
        # Модель сама создала задачу или сменила её состояние в этом ходе —
        # её решение не перекрываем.
        touched = any(e.get("task_id") == task.id and e.get("kind") == "advance" for e in acc.task_events)
        if touched:
            return []
        lines = [line.strip() for line in (answer or "").strip().splitlines() if line.strip()]
        asks_question = not rag.cited and bool(lines) and lines[-1].endswith("?")
        target = None
        if rag.dont_know or asks_question:
            target, reason = "clarifying", "ответ «Не знаю» или уточняющий вопрос"
        elif rag.cited and task.state in ("clarifying", "achieved"):
            target, reason = "answering", "ответ с источниками"
        if target is None:
            return []
        event = self._auto_search_transition(chat, task, target, reason)
        if event is not None:
            acc.task_events.append(event)
            events.append(event)
        return events

    def _validate_send_flags(self, chat: Chat, get_facts: bool, sliding_window: bool, autosummary: str) -> None:
        if get_facts:
            self._validate_sticky_facts_config(chat)
        if sliding_window:
            self._validate_sliding_window_config(chat)
        if autosummary != "off":
            self._validate_autosummary_send_config(chat, autosummary)

    def prepare_message_run(
        self,
        chat_id: str,
        text: str,
        get_facts: bool = False,
        sliding_window: bool = False,
        autosummary: str = "off",
        branch: Optional[int] = None,
        source: str = "app",
        run_id: Optional[str] = None,
    ) -> Tuple[Chat, Message, Message]:
        """Проверяет параметры и создаёт сообщение пользователя и черновик
        ответа. Все ошибки конфигурации (`ValidationError`) — здесь, до
        создания запуска, чтобы клиент получил 400, а не неудавшийся запуск."""
        chat = self._require_chat(chat_id)
        self._require_agent(chat.agent_id)
        if not text or not text.strip():
            raise ValidationError("text must be a non-empty string")
        self._validate_send_flags(chat, get_facts, sliding_window, autosummary)
        self._validate_rag_config(chat)
        now = int(time.time())
        user_msg = self._db.add_message(Message(
            id=0, chat_id=chat_id, role="user", content=text, created_at=now,
            format=detect_message_format(text), branch=branch or 0, run_id=run_id, source=source,
        ))
        draft = self._db.add_message(Message(
            id=0, chat_id=chat_id, role="assistant", content="", created_at=now,
            branch=branch or 0, status="streaming", run_id=run_id,
        ))
        self._db.touch_chat(chat_id)
        if source == "scheduler":
            self._publish_unread(chat_id)
        return chat, user_msg, draft

    def validate_task_step(self, chat_id: str, task_id: str) -> Tuple[Chat, Task]:
        chat = self._require_chat(chat_id)
        self._require_agent(chat.agent_id)
        task = self._require_task(task_id)
        if task.chat_id != chat_id:
            raise NotFoundError(f"task {task_id!r} does not belong to chat {chat_id!r}")
        if not chat.settings.task_tracking_enabled:
            raise ValidationError("task tracking is disabled for this chat")
        if task.kind == KIND_SEARCH:
            raise ValidationError(
                "задача поиска продолжается сообщением пользователя — Менеджер задач к ней не применяется"
            )
        if self._task_status(task) == "done":
            raise ValidationError("task is already done; task manager is not applicable")
        return chat, task

    def work_task_to_continue(self, chat_id: str, terminal: dict) -> Optional[str]:
        """Рабочая задача, которую нужно вести дальше сразу после ответа на
        сообщение пользователя: «Останавливаться на каждом этапе» выключено, в
        этом ответе задача создана или продвинута и осталась активной (не на
        паузе и не завершена). Если модель задала пользователю вопрос и этап
        не продвигала — продолжения нет."""
        if terminal.get("type") != "done":
            return None
        chat = self._db.get_chat(chat_id)
        if chat is None or not chat.settings.task_tracking_enabled or chat.settings.task_pause_each_stage:
            return None
        message = terminal.get("message")
        raw = getattr(message, "task_events", None)
        try:
            events = json.loads(raw) if raw else []
        except (ValueError, TypeError):
            return None
        for event in reversed(events):
            if event.get("kind") != "advance":
                continue
            task = self._db.get_task(event.get("task_id") or "")
            if task is not None and task.kind == KIND_WORK and self._task_status(task) == "active":
                return task.id
        return None

    def prepare_task_step_draft(self, chat_id: str, run_id: Optional[str] = None) -> Message:
        """Черновик сообщения одного шага Менеджера задач (у шага нет
        сообщения пользователя — см. комментарий к разделу ниже)."""
        draft = self._db.add_message(Message(
            id=0, chat_id=chat_id, role="assistant", content="", created_at=int(time.time()),
            branch=0, status="streaming", run_id=run_id, is_task_manager_step=True,
        ))
        self._db.touch_chat(chat_id)
        return draft

    # ---- общий цикл одного ответа модели --------------------------------------

    _MEMORY_TOOLS = frozenset({"save_working_memory", "save_long_term_memory"})
    _TASK_TOOLS = frozenset({"start_task", "apply_task_action", "update_task_memory"})
    #: Инструменты, доступные модели в задаче поиска (только чтение).
    _SEARCH_TASK_TOOLS = _MEMORY_TOOLS | _TASK_TOOLS | frozenset({builtin_tools.READ_CHAT})

    def _tool_source(self, name: str, allowed_mcp_names: Optional[set]) -> str:
        """Откуда инструмент — для строки «Использую инструмент …» в клиенте."""
        if name in self._MEMORY_TOOLS:
            return "memory"
        if name in self._TASK_TOOLS:
            return "task"
        if name in self._TOOL_HANDLERS:
            return "skill"
        if builtin_tools.is_builtin(name):
            return "agents_core"
        if (
            self._mcp_client is not None and self._mcp_client.has_cached_tool(name)
            and (allowed_mcp_names is None or name in allowed_mcp_names)
        ):
            return "mcp"
        return "unknown"

    def _run_one_tool(
        self, chat: Chat, agent: Agent, call: dict, acc: "_TurnAccumulator", auto_pause: bool,
        allowed_mcp_names: set, provider_messages: List[ProviderMessage],
    ) -> Iterator[dict]:
        fn = call.get("function") or {}
        fn_name = fn.get("name")
        name = fn_name or ""
        source = self._tool_source(name, allowed_mcp_names)
        # Для MCP — подпись источника («GIT API», «Планировщик», имя внешнего
        # сервера): клиент показывает её в строке «Использую инструмент …».
        group = (self._mcp_client.tool_group(name) if source == "mcp"
                 else builtin_tools.GROUP_AGENTS_CORE if source == "agents_core" else None)
        started_event = {
            "type": "tool_call", "name": name, "source": source, "group": group,
            "status": "started", "ok": None, "error": None,
        }
        acc.tool_events.append(started_event)
        yield started_event
        task_events_before = len(acc.task_events)
        tool_output = self._execute_tool_call(
            chat, agent, call, events=acc.task_events, auto_pause=auto_pause, allowed_mcp_names=allowed_mcp_names,
        )
        ok, error = True, None
        try:
            parsed = json.loads(tool_output)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, dict) and "error" in parsed:
            ok, error = False, str(parsed.get("error"))
        finished_event = {
            "type": "tool_call", "name": name, "source": source, "group": group,
            "status": "finished", "ok": ok, "error": error,
        }
        acc.tool_events.append(finished_event)
        yield finished_event
        for task_event in acc.task_events[task_events_before:]:
            yield {"type": "task_event", **task_event}
        provider_messages.append(ProviderMessage("tool", tool_output, tool_call_id=call.get("id"), name=fn_name))

    def _model_turn(
        self, provider, model_id: str, provider_messages: List[ProviderMessage], request_settings: Settings,
        chat: Chat, agent: Agent, acc: "_TurnAccumulator", *, use_stream: bool, token: Optional[CancelToken],
        status_text: str, auto_pause: bool = True,
    ):
        """Один ответ модели целиком: запрос (потоковый или обычный), цикл
        вызова инструментов (до `_MAX_TOOL_ITERATIONS` раундов), итоговый
        запрос без инструментов при исчерпании лимита
        (`_finalize_after_tool_cap`), проверка инвариантов с одним
        переспросом. Генератор событий; через `yield from` возвращает
        `(final_result, final_content)`."""
        allowed_mcp_names = self._offered_tool_names(request_settings)
        iterations = 0
        while True:
            if token is not None:
                token.raise_if_cancelled()
            done_result: Optional[ChatResult] = None
            if use_stream:
                stream = provider.stream_chat(model_id, provider_messages, request_settings)
                try:
                    for delta in stream:
                        if token is not None:
                            token.raise_if_cancelled()
                        if delta.done:
                            done_result = delta.result
                            continue
                        if delta.content:
                            acc.partial_content.append(delta.content)
                        if delta.reasoning_content:
                            acc.partial_reasoning.append(delta.reasoning_content)
                        yield {"type": "delta", "content": delta.content, "reasoning_content": delta.reasoning_content}
                finally:
                    # Отмена посреди потока: закрываем генератор провайдера —
                    # это закрывает и HTTP-соединение с моделью.
                    close = getattr(stream, "close", None)
                    if close is not None:
                        close()
            else:
                done_result = provider.chat(model_id, provider_messages, request_settings)
                if token is not None:
                    token.raise_if_cancelled()
            if done_result is None:
                raise ProviderError("provider returned no result")
            if done_result.tool_calls and iterations < _MAX_TOOL_ITERATIONS:
                yield {"type": "status", "status": "Выполняется вызов инструментов"}
                provider_messages.append(
                    ProviderMessage("assistant", done_result.content, tool_calls=done_result.tool_calls)
                )
                for call in done_result.tool_calls:
                    if token is not None:
                        token.raise_if_cancelled()
                    yield from self._run_one_tool(chat, agent, call, acc, auto_pause, allowed_mcp_names, provider_messages)
                iterations += 1
                yield {"type": "status", "status": status_text}
                continue
            final_result = done_result
            break

        # Лимит раундов исчерпан, а модель всё ещё хочет инструменты — один
        # запрос без инструментов с просьбой подвести итог (реальный баг:
        # раньше пользователь получал пустой ответ, см. `_finalize_after_tool_cap`).
        if final_result.tool_calls:
            yield {"type": "status", "status": "Формулирую итоговый ответ"}
            final_result = self._finalize_after_tool_cap(
                provider, model_id, provider_messages, request_settings, final_result,
            )

        # "Инварианты" — код-уровневая проверка + один переспрос; предупреждение
        # выводится ПЕРЕД ответом (отдельными кусками текста для клиента).
        final_result, violation_warning = self._validate_and_maybe_retry(
            provider, model_id, provider_messages, request_settings, chat, agent, final_result,
            events=acc.task_events,
        )
        if violation_warning:
            yield {"type": "delta", "content": f"{violation_warning}\n\n", "reasoning_content": None}
            yield {"type": "delta", "content": final_result.content, "reasoning_content": None}
        final_content = f"{violation_warning}\n\n{final_result.content}" if violation_warning else final_result.content
        return final_result, final_content

    def _finalize_draft(
        self, draft: Message, *, content: str, status: str, result: Optional[ChatResult] = None,
        duration_ms: Optional[int] = None, acc: Optional["_TurnAccumulator"] = None, error: Optional[str] = None,
    ) -> Message:
        """Записывает итог в черновик (та же строка `messages`) и сообщает об
        изменении непрочитанных в общую ленту."""
        content = content or ""
        fields: Dict[str, object] = {
            "content": content, "status": status, "format": detect_message_format(content),
            "created_at": int(time.time()), "error": error,
        }
        if result is not None:
            fields.update(
                reasoning_content=result.reasoning_content, total_tokens=result.usage.total_tokens,
                prompt_tokens=result.usage.prompt_tokens, completion_tokens=result.usage.completion_tokens,
            )
        elif acc is not None and acc.partial_reasoning:
            fields["reasoning_content"] = "".join(acc.partial_reasoning)
        if duration_ms is not None:
            fields["duration_ms"] = duration_ms
        if acc is not None:
            fields["task_events"] = json.dumps(acc.task_events, ensure_ascii=False) if acc.task_events else None
            fields["tool_events"] = json.dumps(acc.tool_events, ensure_ascii=False) if acc.tool_events else None
            if acc.rag is not None:
                fields["rag"] = json.dumps(acc.rag, ensure_ascii=False)
        self._db.update_message_fields(draft.id, **fields)
        saved = self._db.get_message(draft.id)
        self._publish_unread(draft.chat_id)
        return saved

    def _terminal_after_interrupt(
        self, draft: Message, acc: "_TurnAccumulator", started: float, *, cancelled: Optional[RunCancelled] = None,
        error: Optional[str] = None,
    ) -> dict:
        """Итоговое событие для остановленного/неудавшегося ответа: частичный
        текст сохраняется, статус — cancelled или failed."""
        duration_ms = int((time.monotonic() - started) * 1000)
        partial = "".join(acc.partial_content)
        if cancelled is not None and cancelled.reason != "timeout":
            saved = self._finalize_draft(draft, content=partial, status="cancelled", acc=acc, duration_ms=duration_ms)
            self._db.touch_chat(draft.chat_id)
            return {"type": "cancelled", "message": saved}
        if cancelled is not None:
            error = "Превышено время выполнения запроса"
        saved = self._finalize_draft(
            draft, content=partial, status="failed", acc=acc, duration_ms=duration_ms, error=error,
        )
        self._db.touch_chat(draft.chat_id)
        return {"type": "error", "message": error or "", "assistant_message": saved}

    def execute_message(
        self,
        chat_id: str,
        user_msg: Message,
        draft: Message,
        get_facts: bool = False,
        sliding_window: bool = False,
        autosummary: str = "off",
        branch: Optional[int] = None,
        use_stream: bool = True,
        token: Optional[CancelToken] = None,
    ) -> Iterator[dict]:
        """Ответ на сообщение пользователя — генератор событий запуска.
        Последнее событие — всегда `done`, `cancelled` или `error`."""
        chat = self._require_chat(chat_id)
        agent = self._require_agent(chat.agent_id)
        text = user_msg.content
        acc = _TurnAccumulator()
        started = time.monotonic()
        with self._lock_for(chat_id):
            try:
                messages = self._history_for_context(chat_id, branch, before_id=user_msg.id)
                provider_messages, latest_facts, facts_boundary = self._resolve_send_context(
                    chat, agent, messages, text, get_facts, sliding_window, autosummary
                )
                provider_name, model_id = _split_model(chat.settings.model)
                provider = self._registry.get(provider_name)
                search_task = self._current_search_task(chat)
                if search_task is not None and search_task.paused:
                    # Новое сообщение пользователя продолжает задачу поиска —
                    # пауза («Останавливаться на каждом этапе») снимается.
                    self._db.set_task_paused(search_task.id, False)
                    self._db.add_task_transition_log(
                        search_task.id, search_task.state, search_task.state, kind="resume", applied_by="system", note=None,
                    )
                request_settings = self._settings_with_merged_tools(chat)

                rag: Optional[RagOutcome] = None
                if chat.settings.rag_enabled:
                    if chat.settings.rag_query_rewrite == "llm":
                        yield {"type": "status", "status": "Переписываю запрос"}
                    search_text, rewrite_info = self._rewrite_query(chat, messages, text, search_task)
                    if token is not None:
                        token.raise_if_cancelled()
                    reuse = rewrite_info.get("no_search")
                    yield {"type": "status", "status": "Беру фрагменты предыдущего ответа" if reuse else "Поиск в базе знаний"}
                    rag = self._prepare_rag(
                        chat, messages, text, provider_messages, search_text, rewrite_info, task=search_task,
                    )
                    acc.rag = rag.to_dict()
                    yield {"type": "rag_context", "rag": acc.rag}
                    if token is not None:
                        token.raise_if_cancelled()

                status_text = "Выполняется запрос к модели"
                yield {"type": "status", "status": status_text}
                final_result, final_content = yield from self._model_turn(
                    provider, model_id, provider_messages, request_settings, chat, agent, acc,
                    use_stream=use_stream, token=token, status_text=status_text,
                )
                if rag is not None:
                    final_content = finalize_answer(final_content, rag)
                    for event in self._after_search_answer(chat, rag, final_content, acc):
                        yield {"type": "task_event", **event}
                    acc.rag = rag.to_dict()
                duration_ms = int((time.monotonic() - started) * 1000)
                # Токены НА ВХОД этого обмена (показываются под сообщением
                # пользователя) известны только из usage ответа провайдера.
                if final_result.usage.prompt_tokens is not None:
                    self._db.update_message_prompt_tokens(user_msg.id, final_result.usage.prompt_tokens)
                assistant_msg = self._finalize_draft(
                    draft, content=final_content, status="complete", result=final_result,
                    duration_ms=duration_ms, acc=acc,
                )
                yield {"type": "message_saved", "message": assistant_msg}

                # Факты и автосуммаризация — ПОСЛЕ основного ответа.
                if get_facts:
                    yield {"type": "status", "status": "Обновление фактов"}
                    dialogue = self._dialogue_since(messages, facts_boundary)
                    updated_facts = self._extract_facts(chat, latest_facts, dialogue, text, final_result.content)
                    facts_json = json.dumps(updated_facts, ensure_ascii=False)
                    self._db.update_message_facts(assistant_msg.id, facts_json)
                    assistant_msg = dataclasses.replace(assistant_msg, facts=facts_json)
                if autosummary != "off" and self._autosummary_due(chat, messages, autosummary):
                    yield {"type": "status", "status": "Выполняется суммаризация чата"}
                    refreshed_user = self._db.get_message(user_msg.id) or user_msg
                    self._summarize_after_send(chat, messages, refreshed_user, assistant_msg)
                self._db.touch_chat(chat_id)
                yield {"type": "done", "message": assistant_msg}
            except RunCancelled as exc:
                yield self._terminal_after_interrupt(draft, acc, started, cancelled=exc)
            except ProviderError as exc:
                yield self._terminal_after_interrupt(draft, acc, started, error=str(exc))

    def execute_task_step(
        self,
        chat_id: str,
        task_id: str,
        draft: Message,
        auto_pause: bool = True,
        use_stream: bool = True,
        token: Optional[CancelToken] = None,
    ) -> Iterator[dict]:
        """Один шаг Менеджера задач — генератор событий. Итоговое `done`
        дополнительно несёт `should_continue` и `task_status` (см.
        `run_task_manager_step`)."""
        chat, task = self.validate_task_step(chat_id, task_id)
        agent = self._require_agent(chat.agent_id)
        current_status = self._task_status(task)
        acc = _TurnAccumulator()
        started = time.monotonic()
        with self._lock_for(chat_id):
            try:
                max_steps = chat.settings.task_manager_max_steps
                if max_steps > 0 and self._consecutive_task_manager_steps(chat_id, exclude_id=draft.id) >= max_steps:
                    notice = (
                        "Достигнут лимит автоматических шагов Менеджера задач подряд — нажмите "
                        "«Продолжить», чтобы продолжить вручную."
                    )
                    saved = self._finalize_draft(draft, content=notice, status="complete")
                    self._db.touch_chat(chat_id)
                    yield {"type": "done", "message": saved, "should_continue": False, "task_status": current_status}
                    return

                messages = self._history_for_context(chat_id, None, before_id=draft.id)
                continue_text = self._task_manager_continue_text(task)
                provider_messages = self._build_request_context(chat, agent, messages, continue_text)
                provider_name, model_id = _split_model(chat.settings.model)
                provider = self._registry.get(provider_name)
                request_settings = self._settings_with_merged_tools(chat)

                status_text = "Менеджер задач продолжает работу"
                yield {"type": "status", "status": status_text}
                final_result, final_content = yield from self._model_turn(
                    provider, model_id, provider_messages, request_settings, chat, agent, acc,
                    use_stream=use_stream, token=token, status_text=status_text, auto_pause=auto_pause,
                )
                duration_ms = int((time.monotonic() - started) * 1000)
                saved = self._finalize_draft(
                    draft, content=final_content, status="complete", result=final_result,
                    duration_ms=duration_ms, acc=acc,
                )
                yield {"type": "message_saved", "message": saved}
                self._db.touch_chat(chat_id)

                refreshed_task = self._db.get_task(task_id)
                new_status = self._task_status(refreshed_task) if refreshed_task is not None else current_status
                advanced_this_task = any(
                    e.get("task_id") == task_id and e.get("kind") == "advance" for e in acc.task_events
                )
                should_continue = new_status == "active" and advanced_this_task
                yield {"type": "done", "message": saved, "should_continue": should_continue, "task_status": new_status}
            except RunCancelled as exc:
                yield self._terminal_after_interrupt(draft, acc, started, cancelled=exc)
            except ProviderError as exc:
                yield self._terminal_after_interrupt(draft, acc, started, error=str(exc))

    # ---- синхронные варианты (без фонового пула; тесты и внутренние вызовы) ----

    def send_message_blocking(
        self,
        chat_id: str,
        text: str,
        get_facts: bool = False,
        sliding_window: bool = False,
        autosummary: str = "off",
        branch: Optional[int] = None,
    ) -> Tuple[Message, Message]:
        """Синхронная отправка: ждёт ответа целиком. Сбой провайдера —
        `ProviderError` (черновик ответа сохраняется со статусом "failed")."""
        _, user_msg, draft = self.prepare_message_run(chat_id, text, get_facts, sliding_window, autosummary, branch)
        final: Optional[Message] = None
        for event in self.execute_message(
            chat_id, user_msg, draft, get_facts, sliding_window, autosummary, branch, use_stream=False,
        ):
            if event["type"] == "done":
                final = event["message"]
            elif event["type"] == "error":
                raise ProviderError(event.get("message") or "provider error")
        return self._db.get_message(user_msg.id) or user_msg, final

    def stream_message(
        self,
        chat_id: str,
        text: str,
        get_facts: bool = False,
        sliding_window: bool = False,
        autosummary: str = "off",
        branch: Optional[int] = None,
    ) -> Iterator[dict]:
        """Потоковая отправка синхронно, в текущем потоке: те же события,
        что и в ленте запуска (`status`/`delta`/`tool_call`/`task_event`/
        `done`/`error`/`cancelled`), но без номеров. Клиенты работают через
        `runs.RunManager`."""
        _, user_msg, draft = self.prepare_message_run(chat_id, text, get_facts, sliding_window, autosummary, branch)
        for event in self.execute_message(
            chat_id, user_msg, draft, get_facts, sliding_window, autosummary, branch, use_stream=True,
        ):
            yield event

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

    def _consecutive_task_manager_steps(self, chat_id: str, exclude_id: Optional[int] = None) -> int:
        """Подряд идущие сообщения ассистента с `is_task_manager_step=True`
        считая с конца истории чата, до первого сообщения другого рода
        (обычный ответ на реальное сообщение пользователя, ошибка и т.п.) —
        используется как защита от зацикливания (см.
        `Settings.task_manager_max_steps`). Считается на уровне ЧАТА в
        целом, а не отдельной задачи — упрощение: в рамках одного чата
        обычно ведётся не более одной активной задачи одновременно.
        `exclude_id` — черновик текущего шага, он в счёт не входит."""
        count = 0
        for m in reversed(self._db.list_messages(chat_id)):
            if exclude_id is not None and m.id == exclude_id:
                continue
            if m.role == "assistant" and m.is_task_manager_step:
                count += 1
                continue
            break
        return count

    def run_task_manager_step(self, chat_id: str, task_id: str, auto_pause: bool = True) -> Iterator[dict]:
        """Один автономный шаг "Менеджера задач" — синхронно, теми же
        событиями, что и лента запуска (`done` несёт
        `should_continue` и `task_status`). Требует `task_tracking_enabled=true`
        и статус задачи ЛЮБОЙ, кроме "done" — иначе `ValidationError`.

        `auto_pause`: `true` (кнопка "Продолжить") — после продвижения этапа
        шаг сам ставит задачу на паузу, `should_continue=false`; `false`
        (кнопка "Выполнить") — пауза не вставляется, `should_continue=true`
        до состояния done. Серверный цикл шагов «до конца» — в
        `runs.RunManager.start_task_run` (запуск `task_run`)."""
        self.validate_task_step(chat_id, task_id)
        draft = self.prepare_task_step_draft(chat_id)
        yield from self.execute_task_step(chat_id, task_id, draft, auto_pause=auto_pause, use_stream=True)

    def create_task_direct(
        self, chat_id: str, title: str, description: str = "", source: str = "app", kind: str = "work",
        goal: str = "", add_message: bool = True,
    ) -> Tuple[dict, Optional[Message]]:
        """Создание задачи напрямую, без участия модели (ТЗ, раздел 2.3 —
        нужно планировщику: вид задачи «Задача агенту»). В ленту чата
        добавляется сообщение-запрос «Задача: …» от имени `source`, чтобы
        пользователь видел, откуда задача взялась; модель видит название и
        описание в блоке [ЗАДАЧА]. Задача создаётся не на паузе — её можно
        сразу выполнять запуском Менеджера задач."""
        chat = self._require_chat(chat_id)
        if not chat.settings.task_tracking_enabled:
            raise PreconditionFailedError("В чате выключены задачи (task_tracking_enabled)")
        title = (title or "").strip()
        if not title:
            raise ValidationError("title must be a non-empty string")
        description = (description or "").strip()
        if kind not in TASK_KINDS:
            raise ValidationError(f"kind must be one of {list(TASK_KINDS)}")
        if kind == KIND_SEARCH and not chat.settings.rag_enabled:
            raise PreconditionFailedError("Задача поиска требует включённой базы знаний (rag_enabled)")
        goal = (goal or "").strip() or (title if kind == KIND_SEARCH else "")
        task = self._db.create_task(
            chat_id, title, plan=[], description=description, kind=kind,
            state=kind_initial_state(kind), goal=goal,
        )
        if not add_message:
            return self._task_summary(task, chat.title), None
        prefix = "Задача поиска" if kind == KIND_SEARCH else "Задача"
        text = f"{prefix}: {title}" + (f"\n\n{description}" if description else "")
        message = self._db.add_message(Message(
            id=0, chat_id=chat_id, role="user", content=text, created_at=int(time.time()),
            format=detect_message_format(text), branch=0, source=source,
        ))
        self._db.touch_chat(chat_id)
        if source == "scheduler":
            self._publish_unread(chat_id)
        return self._task_summary(task, chat.title), message

    # ---- непрочитанные и общая лента изменений (ТЗ, раздел 2.6) --------------

    def chat_activity(self, chat_ids: Optional[List[str]] = None) -> Dict[str, dict]:
        return self._db.chat_activity(chat_ids)

    def _activity_for(self, chat_id: str) -> dict:
        return self._db.chat_activity([chat_id]).get(chat_id) or {
            "unread_count": 0, "first_unread_message_id": None, "last_message_at": None, "preview": None,
        }

    def _publish_unread(self, chat_id: str) -> None:
        self.events.publish({"type": "unread_changed", "chat_id": chat_id, **self._activity_for(chat_id)})

    def _publish_chat_event(self, kind: str, chat_id: str, agent_id: Optional[str] = None) -> None:
        event = {"type": kind, "chat_id": chat_id}
        if agent_id is not None:
            event["agent_id"] = agent_id
        self.events.publish(event)

    def mark_chat_read(self, chat_id: str, message_id: int) -> dict:
        """Отметка «прочитано до сообщения `message_id`» — только вперёд."""
        self._require_chat(chat_id)
        self._db.set_last_read(chat_id, message_id)
        self._publish_unread(chat_id)
        return self._activity_for(chat_id)


class _TurnAccumulator:
    """Накопитель одного ответа: события задач и всех инструментов (уходят
    в итоговое сообщение) и частичный текст (сохраняется, если ответ
    остановлен или завершился ошибкой)."""

    def __init__(self) -> None:
        self.task_events: List[dict] = []
        self.tool_events: List[dict] = []
        self.partial_content: List[str] = []
        self.partial_reasoning: List[str] = []
        #: Итог RAG (`RagOutcome.to_dict()`), если он был в этом ответе.
        self.rag: Optional[dict] = None

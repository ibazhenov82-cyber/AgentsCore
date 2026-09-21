"""
agents_core.schemas
=====================

Pydantic-модели запросов/ответов HTTP API. Отдельно от `models.py`
(dataclass'ы домена) — эти модели существуют только для валидации на
границе HTTP и для генерации красивой схемы Swagger UI (описания полей,
примеры, обязательность).
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .models import (
    DEFAULT_EXTRACTION_SYSTEM_PROMPT,
    DEFAULT_MODEL_ID,
    DEFAULT_SUMMARY_PROMPT,
    DEFAULT_SUMMARY_SYSTEM_PROMPT,
    INVARIANT_KIND_OPTIONS,
    LONG_TERM_MEMORY_CATEGORIES,
    TASK_APPLIED_BY_OPTIONS,
    TASK_STATUS_OPTIONS,
    TASK_TRANSITION_KINDS,
)


# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------

class SettingsOut(BaseModel):
    """Полный набор параметров запроса к LLM — используется и для агента, и
    для чата (для чата поле `model` только информационное: модель и
    провайдер закреплены за агентом-владельцем и не редактируются на уровне
    чата)."""

    model: str = Field(..., description="Составной идентификатор модели вида 'provider:model_id'", examples=["deepseek:deepseek-v4-flash"])
    system_prompt: str = Field(..., description="Системный промпт")
    temperature: float = Field(..., ge=0, le=2)
    top_p: float = Field(..., ge=0, le=1)
    seed: Optional[int] = Field(None, description="Начальное число генератора случайных чисел, если модель это поддерживает")

    stream: bool
    thinking_enabled: bool
    reasoning_effort: Optional[str] = Field(None, description="'low' | 'high' | 'max', имеет смысл только при thinking_enabled=true")
    max_tokens: Optional[int] = Field(None, ge=1)
    json_mode: bool
    stop_sequences: List[str] = Field(default_factory=list)

    tool_choice: str = Field(..., description="'auto' | 'none' | 'required'")
    tools_json: str = Field("", description="Сырой JSON-массив описаний инструментов в формате OpenAI function-tool")
    memory_tools_enabled: bool = Field(
        False,
        description=(
            "Разрешить агенту САМОМУ сохранять память (working/long-term) через встроенные "
            "функции save_working_memory/save_long_term_memory (тумблер «Разрешить агенту "
            "сохранять память»). Ручное сохранение через API доступно всегда, независимо от этого поля."
        ),
    )
    working_memory_enabled: bool = Field(True, description="Рабочая память (чат/задача) — попадает в промпт, пока включена.")
    long_term_memory_enabled: bool = Field(True, description="Долговременная память (категории profile/decision/knowledge) — попадает в промпт, пока включена.")
    episodic_memory_enabled: bool = Field(False, description="Расширенный тип долговременной памяти — конкретные прошлые эпизоды.")
    semantic_memory_enabled: bool = Field(False, description="Расширенный тип долговременной памяти — обобщённые знания.")
    procedural_memory_enabled: bool = Field(False, description="Расширенный тип долговременной памяти — как выполнять задачи/процессы.")
    task_tracking_enabled: bool = Field(
        False,
        description=(
            "«Отслеживать задачи» (День 13). По умолчанию выключено. Пока выключено — задачи "
            "не создаются и не сохраняются (`start_task`/`apply_task_action` не предлагаются модели, "
            "контекст задач не подмешивается в промпт), а клиент не должен показывать НИКАКИЕ "
            "элементы интерфейса задач (бэдж/ссылку/вкладку/агрегированный блок). Заведение и "
            "продвижение задачи не требует от пользователя специальных фраз — модель решает сама, "
            "по смыслу переписки; пока задача активна, работает «Менеджер задач» (см. "
            "`POST /chats/{chat_id}/tasks/{task_id}/task-manager/step`)."
        ),
    )
    task_manager_max_steps: int = Field(
        20,
        ge=0,
        description=(
            "Лимит автономных шагов «Менеджера задач» подряд без участия пользователя (защита от "
            "зацикливания/расхода токенов) — считается на уровне чата в целом. `0` — лимит отключён."
        ),
    )
    summary_prompt: str = Field(..., description="Шаблон user-prompt суммаризации (теги <previous_summary>/<new_messages>)")
    summary_system_prompt: str = Field(..., description="Системный prompt изолированного вызова суммаризации")
    autosummary: str = Field(..., description="'off' | 'messages' | 'tokens'")
    autosummary_by_messages: int = Field(..., ge=1)
    autosummary_by_tokens: int = Field(..., ge=1)

    context_strategy: Optional[str] = Field(None, description="null | 'sliding_window' | 'sticky_facts'")
    context_strategy_limit: Optional[int] = Field(None, gt=2, description="Предел числа сообщений для стратегии контекста, если задана; > 2")
    extraction_system_prompt: str = Field(..., description="Системный prompt для извлечения фактов (стратегия Sticky Facts)")

    include_usage_in_stream: bool
    logprobs: bool
    frequency_penalty: Optional[float] = Field(None, ge=-2, le=2)
    presence_penalty: Optional[float] = Field(None, ge=-2, le=2)

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "model": "deepseek:deepseek-v4-flash", "system_prompt": "Ты — полезный ассистент.",
            "temperature": 1.0, "top_p": 1.0, "seed": None, "stream": True, "thinking_enabled": True,
            "reasoning_effort": None, "max_tokens": None, "json_mode": False, "stop_sequences": [],
            "tool_choice": "auto", "tools_json": "", "memory_tools_enabled": False,
            "working_memory_enabled": True, "long_term_memory_enabled": True,
            "episodic_memory_enabled": False, "semantic_memory_enabled": False, "procedural_memory_enabled": False,
            "task_tracking_enabled": False,
            "task_manager_max_steps": 20,
            "summary_prompt": DEFAULT_SUMMARY_PROMPT,
            "summary_system_prompt": DEFAULT_SUMMARY_SYSTEM_PROMPT,
            "autosummary": "off", "autosummary_by_messages": 10, "autosummary_by_tokens": 50000,
            "context_strategy": None, "context_strategy_limit": None,
            "extraction_system_prompt": DEFAULT_EXTRACTION_SYSTEM_PROMPT,
            "include_usage_in_stream": False, "logprobs": False, "frequency_penalty": None, "presence_penalty": None,
        }
    })


class SettingsPatch(BaseModel):
    """Частичное обновление настроек — присутствуют только изменяемые поля,
    остальные сохраняют текущее значение. Неизвестное имя поля -> `400`."""

    model: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0, le=2)
    top_p: Optional[float] = Field(None, ge=0, le=1)
    seed: Optional[int] = None
    stream: Optional[bool] = None
    thinking_enabled: Optional[bool] = None
    reasoning_effort: Optional[str] = None
    max_tokens: Optional[int] = Field(None, ge=1)
    json_mode: Optional[bool] = None
    stop_sequences: Optional[List[str]] = None
    tool_choice: Optional[str] = None
    tools_json: Optional[str] = None
    memory_tools_enabled: Optional[bool] = None
    working_memory_enabled: Optional[bool] = None
    long_term_memory_enabled: Optional[bool] = None
    episodic_memory_enabled: Optional[bool] = None
    semantic_memory_enabled: Optional[bool] = None
    procedural_memory_enabled: Optional[bool] = None
    task_tracking_enabled: Optional[bool] = None
    task_manager_max_steps: Optional[int] = Field(None, ge=0)
    summary_prompt: Optional[str] = None
    summary_system_prompt: Optional[str] = None
    autosummary: Optional[str] = None
    autosummary_by_messages: Optional[int] = Field(None, ge=1)
    autosummary_by_tokens: Optional[int] = Field(None, ge=1)
    context_strategy: Optional[str] = None
    context_strategy_limit: Optional[int] = Field(None, gt=2)
    extraction_system_prompt: Optional[str] = None
    include_usage_in_stream: Optional[bool] = None
    logprobs: Optional[bool] = None
    frequency_penalty: Optional[float] = Field(None, ge=-2, le=2)
    presence_penalty: Optional[float] = Field(None, ge=-2, le=2)

    model_config = ConfigDict(json_schema_extra={"example": {"temperature": 0.7, "max_tokens": 2048}})

    def to_payload(self) -> dict:
        return self.model_dump(exclude_unset=True)


class DefaultSettingsOut(BaseModel):
    """Настройки, применяемые при создании нового агента."""

    model: str
    system_prompt: str
    temperature: float
    top_p: float
    seed: Optional[int] = None
    stream: bool
    thinking_enabled: bool
    reasoning_effort: Optional[str] = None
    summary_prompt: str
    summary_system_prompt: str
    include_usage_in_stream: bool

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "model": DEFAULT_MODEL_ID, "system_prompt": "Ты — полезный ассистент.",
            "temperature": 1.0, "top_p": 1.0, "seed": None, "stream": True, "thinking_enabled": True,
            "reasoning_effort": None, "summary_prompt": DEFAULT_SUMMARY_PROMPT,
            "summary_system_prompt": DEFAULT_SUMMARY_SYSTEM_PROMPT,
            "include_usage_in_stream": False,
        }
    })


class DefaultSettingsPatch(BaseModel):
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0, le=2)
    top_p: Optional[float] = Field(None, ge=0, le=1)
    seed: Optional[int] = None
    stream: Optional[bool] = None
    thinking_enabled: Optional[bool] = None
    reasoning_effort: Optional[str] = None
    summary_prompt: Optional[str] = None
    summary_system_prompt: Optional[str] = None
    include_usage_in_stream: Optional[bool] = None

    model_config = ConfigDict(json_schema_extra={"example": {"temperature": 0.7, "model": "ollama:qwen3:0.6b"}})

    def to_payload(self) -> dict:
        return self.model_dump(exclude_unset=True)


# ---------------------------------------------------------------------------
# Каталог моделей
# ---------------------------------------------------------------------------

class ModelInfoOut(BaseModel):
    id: str = Field(..., examples=["ollama:qwen3:0.6b"])
    provider: str
    model_id: str
    display_name: str
    is_local: bool
    context_window: Optional[int] = None
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    max_reasoning_tokens: Optional[int] = None
    supports_thinking: bool
    supports_tools: bool
    supports_json_mode: bool
    supports_logprobs: bool


class ModelHealthOut(BaseModel):
    model: str
    healthy: bool


# ---------------------------------------------------------------------------
# Сообщения
# ---------------------------------------------------------------------------

class MessageOut(BaseModel):
    id: int
    chat_id: str
    role: str = Field(..., description="'user' | 'assistant' | 'error'")
    content: str
    created_at: int
    reasoning_content: Optional[str] = None
    is_summary: bool = False
    duration_ms: Optional[int] = None
    total_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    format: str = Field("text", description="'text' | 'markdown' | 'json' — определяется автоматически при сохранении")
    branch: int = Field(0, description="Номер ветки диалога; 0 — основная ветка")
    facts: Optional[str] = Field(None, description="JSON-текст фактов (стратегия Sticky Facts), сохранённых под этим сообщением")
    task_events: Optional[str] = Field(
        None,
        description=(
            "JSON-массив переходов состояний задач, применённых моделью ЗА ЭТОТ конкретный ответ "
            "(через `apply_task_action`/`start_task`) — по замечанию пользователя, при отображении "
            "смены состояния в чате нужно явно показать, к какой именно задаче она относится (в чате "
            "может быть несколько параллельных открытых задач). Каждый элемент: {\"task_id\", "
            "\"task_title\", \"from_state_display_name\", \"to_state_display_name\", "
            "\"action_display_name\", \"action_kind\"}. Пусто/null, если за этот ответ ни одна задача "
            "не менялась."
        ),
    )
    is_task_manager_step: bool = Field(
        False,
        description=(
            "`True`, если сообщение сгенерировано автономным шагом «Менеджера задач» "
            "(`POST /chats/{chat_id}/tasks/{task_id}/task-manager/step`), а не ответом на реальное "
            "сообщение пользователя — перед ним в истории нет соответствующего сообщения с ролью "
            "`user`. Клиент показывает такие сообщения с пометкой «Менеджер задач»."
        ),
    )


_SLIDING_WINDOW_DESCRIPTION = (
    "Запросить обрезку контекста стратегией Sliding Window. Требует "
    "context_strategy='sliding_window' и context_strategy_limit > 2 у чата, иначе 400. "
    "В основной запрос уходят системный prompt, последние (context_strategy_limit - 1) "
    "сообщений с ролью user/assistant и новое сообщение пользователя — более ранние "
    "сообщения не отправляются. Без этого флага (по умолчанию) сообщение отправляется "
    "как обычно, без обрезки стратегией, даже если у чата задан context_strategy."
)

_AUTOSUMMARY_DESCRIPTION = (
    "Запросить автоматическую суммаризацию чата: 'off' (по умолчанию) — выключено; "
    "'messages' или 'tokens' — включить проверку по соответствующему режиму. Требует, "
    "чтобы у чата настройка autosummary совпадала со значением этого поля, а "
    "autosummary_by_messages > 2 (для режима 'messages') или autosummary_by_tokens > 0 "
    "(для режима 'tokens') — иначе 400. В основной запрос уходят только сообщения, начиная "
    "с последнего сообщения с is_summary=true (включительно), либо вся история, если "
    "резюме ещё не было. Проверка необходимости суммаризации выполняется ПОСЛЕ основного "
    "ответа модели: если порог достигнут — выполняется суммаризация по существующему "
    "алгоритму, и в чат сохраняется новое summary-сообщение."
)


class SendMessageRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["Привет! Расскажи в двух словах, кто ты."])
    get_facts: bool = Field(
        False,
        description=(
            "Запросить извлечение/обновление фактов (стратегия Sticky Facts). "
            "Требует context_strategy='sticky_facts' и context_strategy_limit > 2 у чата, иначе 400. "
            "Извлечение выполняется ПОСЛЕ основного ответа модели; обновлённые факты возвращаются "
            "в `assistant_message.facts` (не в `user_message.facts`)."
        ),
    )
    sliding_window: bool = Field(False, description=_SLIDING_WINDOW_DESCRIPTION)
    autosummary: str = Field("off", description=_AUTOSUMMARY_DESCRIPTION)
    branch: Optional[int] = Field(None, description="Номер ветки диалога, в которую отправляется сообщение; не задано — основная ветка (0)")


class StreamSendMessageRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["Напиши короткое стихотворение про осень."])
    get_facts: bool = Field(
        False,
        description=(
            "Запросить извлечение/обновление фактов (стратегия Sticky Facts) — доступно и в потоковом "
            "режиме: извлечение запускается только после того, как потоковая генерация ответа полностью "
            "завершена. Клиент увидит промежуточные события `{\"type\": \"status\", ...}` "
            "(\"Выполняется запрос к модели\", затем, если задано это поле, \"Обновление фактов\") "
            "до финального события `done`, в котором `message.facts` уже содержит обновлённые факты."
        ),
    )
    sliding_window: bool = Field(False, description=_SLIDING_WINDOW_DESCRIPTION)
    autosummary: str = Field(
        "off",
        description=_AUTOSUMMARY_DESCRIPTION + (
            " В потоковом режиме, если суммаризация действительно потребовалась, клиент "
            "увидит дополнительное событие статуса \"Выполняется суммаризация чата\" перед "
            "финальным `done`."
        ),
    )
    branch: Optional[int] = Field(None, description="Номер ветки диалога, в которую отправляется сообщение; не задано — основная ветка (0)")


class SendMessageResponse(BaseModel):
    user_message: MessageOut
    assistant_message: MessageOut


class BulkDeleteRequest(BaseModel):
    ids: List[int] = Field(..., min_length=1, examples=[[12, 13, 14]])


# ---------------------------------------------------------------------------
# Ветки диалога
# ---------------------------------------------------------------------------

class BranchCreate(BaseModel):
    name: Optional[str] = Field(
        None,
        min_length=1,
        examples=["Альтернативный вариант"],
        description="Если не задано — автоматически подставляется \"Ветка N\", где N — присвоенный номер ветки.",
    )


class BranchOut(BaseModel):
    number: int = Field(..., examples=[1])
    name: str
    created_at: int


# ---------------------------------------------------------------------------
# Чаты / агрегаты по токенам
# ---------------------------------------------------------------------------

class ChatStatsOut(BaseModel):
    prompt_tokens: int = Field(
        ...,
        description=(
            "Сумма входящих токенов по ответам ассистента в ЭФФЕКТИВНОМ контексте — т.е. после границы "
            "последней суммаризации и обрезки стратегией управления контекстом (Sliding Window / Sticky Facts), "
            "а не по всей истории чата с начала"
        ),
    )
    completion_tokens: int = Field(..., description="Сумма исходящих токенов по ответам ассистента в эффективном контексте (см. prompt_tokens)")
    total_tokens: int = Field(..., description="prompt_tokens + completion_tokens (по эффективному контексту)")
    current_context_tokens: int = Field(
        ...,
        description=(
            "Размер контекста ПОСЛЕДНЕГО обмена в эффективном окне — то, что реально уйдёт в следующем запросе "
            "модели (в отличие от total_tokens это не сумма по всем сообщениям, а разбивка одного обмена, "
            "т.к. prompt_tokens каждого следующего ответа уже включает всю предыдущую историю окна). Используется "
            "только для проверки can_summarize/лимитов модели и чата, НЕ для context_fill_ratio"
        ),
    )
    max_input_tokens: Optional[int] = Field(None, description="Максимум входных токенов модели, заданной для агента этого чата")
    context_window: Optional[int] = Field(None, description="Полный размер контекстного окна модели")
    context_fill_ratio: Optional[float] = Field(
        None,
        description=(
            "total_tokens / effective_window — реальный расход по видимой (не бледной) части чата, где "
            "effective_window — это context_window модели, но если у чата задана настройка max_tokens и она "
            "МЕНЬШЕ размера окна модели, используется именно max_tokens (пользователь явно ограничил бюджет "
            "чата). Значение намеренно НЕ обрезается сервером и может быть больше 1.0, если чат уже перерос "
            "этот бюджет: клиент должен сам показать переполнение (например, отобразить 100% вместо большего "
            "числа и покрасить индикатор заполненности в красный)"
        ),
    )
    active_context_start_id: Optional[int] = Field(
        None, description="Id самого раннего сообщения, которое ещё попадёт в следующий запрос к модели — всё, что раньше, можно отображать более бледным"
    )
    can_summarize: bool = Field(..., description="Доступна ли сейчас кнопка «Подготовить суммарный запрос»")


class ChatCreate(BaseModel):
    title: Optional[str] = Field(
        None,
        examples=["Чат 1"],
        description="Если не задано — автоматически подставляется \"Чат N\" по порядковому номеру чата внутри агента.",
    )


class ChatCopyRequest(BaseModel):
    title: str = Field(..., min_length=1, examples=["Мой чат (копия)"])


class ChatRename(BaseModel):
    title: str = Field(..., min_length=1)


class ChatOut(BaseModel):
    id: str
    agent_id: str
    title: str
    created_at: int
    updated_at: int
    settings: SettingsOut
    stats: ChatStatsOut
    active_profile_id: Optional[str] = Field(None, description="Id подключённого профиля-пайплайна персонализации, если есть")
    invariant_ids: List[str] = Field(
        default_factory=list,
        description=(
            "Id инвариантов из общего справочника (`GET /invariants`), выбранных для ЭТОГО чата "
            "(`PUT /chats/{chat_id}/invariants`). Итоговый набор, который увидит модель, — "
            "объединение этого списка и `AgentOut.invariant_ids` агента этого чата."
        ),
    )


class ActiveProfileSetRequest(BaseModel):
    profile_id: Optional[str] = Field(None, description="Id профиля из общего справочника (`GET /profiles`); null — отключить профиль")


class InvariantIdsSetRequest(BaseModel):
    invariant_ids: List[str] = Field(
        default_factory=list,
        description="Полный новый набор id инвариантов из общего справочника (`GET /invariants`); заменяет предыдущий выбор целиком.",
        examples=[["b3f1...", "9ac2..."]],
    )


# ---------------------------------------------------------------------------
# Агенты
# ---------------------------------------------------------------------------

class AgentCreate(BaseModel):
    name: Optional[str] = Field(
        None,
        min_length=1,
        examples=["Мой агент"],
        description="Если не задано — автоматически подставляется \"Агент N\" по порядковому номеру.",
    )
    model: Optional[str] = Field(None, description="Если не задано — берётся из настроек по умолчанию", examples=["ollama:qwen3:0.6b"])


class AgentRename(BaseModel):
    name: str = Field(..., min_length=1)


class AgentOut(BaseModel):
    id: str
    name: str
    created_at: int
    updated_at: int
    settings: SettingsOut
    default_profile_id: Optional[str] = Field(
        None,
        description=(
            "Профиль по умолчанию для НОВЫХ чатов этого агента (копируется в "
            "Chat.active_profile_id только в момент создания чата, как и "
            "остальные настройки — на уже существующие чаты не влияет). "
            "Профиль выбирается из общего справочника `GET /profiles`."
        ),
    )
    invariant_ids: List[str] = Field(
        default_factory=list,
        description=(
            "Id инвариантов из общего справочника (`GET /invariants`), выбранных для ЭТОГО "
            "агента (`PUT /agents/{agent_id}/invariants`) — действуют во ВСЕХ его чатах. "
            "В отличие от `default_profile_id`, НЕ копируется в новые чаты при создании: "
            "остаётся живой настройкой уровня агента."
        ),
    )


class AgentWithChatsOut(BaseModel):
    agent: AgentOut
    chats: List[ChatOut]


class HealthOut(BaseModel):
    status: str = "ok"
    # Версия совпадает с `FastAPI(version=...)` в `create_app_with_repository` —
    # удобный способ убедиться, что запущен именно тот код, который был
    # задеплоен (а не старая версия сервиса), не читая логи процесса.
    version: str = "1.1.0"


# ---------------------------------------------------------------------------
# Рабочая память (working_memory, область видимости — чат)
# ---------------------------------------------------------------------------

class WorkingMemoryOut(BaseModel):
    id: int
    chat_id: str
    key: str
    value: str
    source: str = Field(..., description="'manual' | 'agent' — кто сохранил запись")
    created_at: int
    updated_at: int


class WorkingMemorySaveRequest(BaseModel):
    key: str = Field(..., min_length=1, examples=["текущая_задача"])
    value: str = Field(..., examples=["собрать корзину покупок до 5000 руб."])


# ---------------------------------------------------------------------------
# Долговременная память (long_term_memory, область видимости — агент)
# ---------------------------------------------------------------------------

class LongTermMemoryOut(BaseModel):
    id: int
    agent_id: str
    category: str = Field(..., description=f"Одна из: {LONG_TERM_MEMORY_CATEGORIES}")
    key: str
    value: str
    source: str = Field(..., description="'manual' | 'agent' — кто сохранил запись")
    created_at: int
    updated_at: int


class LongTermMemorySaveRequest(BaseModel):
    category: str = Field(..., description=f"Одна из: {LONG_TERM_MEMORY_CATEGORIES}")
    key: str = Field(..., min_length=1, examples=["user_name"])
    value: str = Field(..., examples=["Иван"])


# ---------------------------------------------------------------------------
# Реестр зарегистрированных скиллов (см. agents_core.skills.registry) —
# плоский список, из которого выбирают skill_names при создании профиля
# ---------------------------------------------------------------------------

class RegisteredSkillOut(BaseModel):
    name: str = Field(..., examples=["search_products"])
    description: str = Field("", examples=["Найти товары в каталоге демо-магазина..."])
    parameters: dict = Field(default_factory=dict, description="JSON Schema параметров функции (как в OpenAI function-tool)")


# ---------------------------------------------------------------------------
# Профили-пайплайны персонализации (profiles, общий справочник для ВСЕХ
# агентов — не привязаны к конкретному агенту, см. GET/POST /profiles)
# ---------------------------------------------------------------------------

class ProfileOut(BaseModel):
    id: str
    name: str
    style: Optional[str] = None
    format: Optional[str] = None
    constraints: Optional[str] = None
    skills_json: str = Field("", description="JSON-массив описаний функций этого профиля в формате OpenAI function-tool")
    orchestration_prompt: Optional[str] = Field(None, description="Инструкция модели, как оркестровать скиллы этого профиля")
    is_default: bool = False
    created_at: int
    updated_at: int


class ProfileCreate(BaseModel):
    name: str = Field(..., min_length=1, examples=["Покупки"])
    style: Optional[str] = Field(None, examples=["дружелюбно, коротко"])
    format: Optional[str] = None
    constraints: Optional[str] = None
    skills_json: str = Field("", description="JSON-массив описаний функций в формате OpenAI function-tool. Взаимоисключимо с skill_names.")
    skill_names: Optional[List[str]] = Field(
        None,
        description="Имена скиллов из GET /skills (реестр AGENT_REGISTERED_SKILLS) — сервер сам соберёт из них skills_json. Взаимоисключимо с skills_json.",
        examples=[["search_products", "add_to_cart", "view_cart"]],
    )
    orchestration_prompt: Optional[str] = None


class ProfilePatch(BaseModel):
    name: Optional[str] = Field(None, min_length=1)
    style: Optional[str] = None
    format: Optional[str] = None
    constraints: Optional[str] = None
    skills_json: Optional[str] = None
    skill_names: Optional[List[str]] = Field(None, description="См. ProfileCreate.skill_names — заменяет весь набор скиллов профиля.")
    orchestration_prompt: Optional[str] = None

    def to_payload(self) -> dict:
        return self.model_dump(exclude_unset=True)


# ---------------------------------------------------------------------------
# Инварианты ("День 14. Инварианты и ограничения состояния", общий справочник
# для ВСЕХ агентов/чатов — не привязаны к конкретному агенту/чату, см.
# GET/POST /invariants; подключаются множественным выбором через
# PUT /agents/{agent_id}/invariants и PUT /chats/{chat_id}/invariants)
# ---------------------------------------------------------------------------

class InvariantOut(BaseModel):
    id: str
    title: str
    rule_text: str = Field(..., description="Текст правила на естественном языке — именно он попадает в промпт модели")
    kind: Optional[str] = Field(
        None,
        description=(
            f"Категория (в интерфейсе — поле «Категория», было «Метка» до ТЗ "
            f"«Работу с задачами требуется переделать»); одна из: {INVARIANT_KIND_OPTIONS}. Для "
            "'architecture'/'tech_decision'/'stack_constraint' категория дополнительно включает "
            "код-уровневую проверку ответа модели, см. `invariant_checks.CHECKERS`."
        ),
    )
    is_active: bool = Field(True, description="Выключенный инвариант не подмешивается в промпт, даже если выбран у агента/чата")
    created_at: int
    updated_at: int


class InvariantCreate(BaseModel):
    title: str = Field(..., min_length=1, examples=["Только Clean Architecture"])
    rule_text: str = Field(..., min_length=1, examples=["В этом проекте используется Clean Architecture, слой presentation не обращается к data напрямую"])
    kind: Optional[str] = Field(None, description=f"Категория, одна из: {INVARIANT_KIND_OPTIONS}")
    is_active: bool = True


class InvariantPatch(BaseModel):
    title: Optional[str] = Field(None, min_length=1)
    rule_text: Optional[str] = Field(None, min_length=1)
    kind: Optional[str] = None
    is_active: Optional[bool] = None

    def to_payload(self) -> dict:
        return self.model_dump(exclude_unset=True)


# ---------------------------------------------------------------------------
# Снимок памяти (то, что реально уйдёт в следующий запрос к модели)
# ---------------------------------------------------------------------------

class MemorySnapshotShortTermOut(BaseModel):
    message_count: int
    messages: List[dict] = Field(..., description="[{'role': ..., 'content': ...}, ...] — эффективный контекст диалога")


class MemorySnapshotOut(BaseModel):
    short_term: MemorySnapshotShortTermOut
    working_memory: List[WorkingMemoryOut] = Field(..., description="Пусто, если working_memory_enabled=false — даже если записи есть в базе")
    long_term_memory: List[LongTermMemoryOut] = Field(..., description="Только записи включённых сейчас категорий (см. enabled_memory_types)")
    active_profile: Optional[ProfileOut] = None
    available_tools: List[dict] = Field(..., description="Итоговый список функций (tools) после слияния — память + скиллы профиля + собственные tools_json")
    memory_tools_enabled: bool
    enabled_memory_types: List[str] = Field(..., description="Какие типы памяти сейчас включены и реально участвуют в этом снимке: подмножество ['working', 'profile', 'decision', 'knowledge', 'episodic', 'semantic', 'procedural']")


# ---------------------------------------------------------------------------
# "Работу с задачами требуется переделать" — машина состояний задачи
# ---------------------------------------------------------------------------
#
# Каталог состояний/переходов и сама настроенная машина заданы в КОДЕ
# (см. `agents_core.task_state_machine`), а не в БД — прежние справочники
# TaskState/TaskAction/TaskStateMachine/TaskMachineState/TaskTransition (с
# формой-редактором на Android) удалены; экран "Модели состояний задач"
# стал read-only (см. `TaskStateMachineInfoOut`). Единственное, что здесь
# по-прежнему настраивается — список инвариантов категории
# "state_machine_rule", привязанных к этой единственной машине (см.
# `TaskMachineInvariantIdsSetRequest`). Сама задача (`Task`) по-прежнему
# владеется чатом (`chat_id`) — в одном чате может быть несколько задач
# одновременно.

class TaskStateInfoOut(BaseModel):
    """Один этап машины состояний в read-only списке (замена формы
    редактирования состояний/переходов/машин)."""

    position: int = Field(..., description="Порядковый номер этапа (1-based) в фиксированном порядке машины")
    state: str = Field(..., description="Системное имя состояния (значение TaskState), например 'planning'")
    display_name: str
    target_states: List[str] = Field(..., description="Системные имена состояний, в которые можно перейти из этого; пусто для конечного состояния")
    target_state_display_names: List[str]


class TaskStateMachineInfoOut(BaseModel):
    """Read-only описание единственной, заданной в коде машины состояний
    задачи + список инвариантов категории "Правило стейт-машины",
    привязанных к ней (единая на всю систему настройка, см.
    `task_state_machine_invariants`)."""

    states: List[TaskStateInfoOut]
    invariants: List[InvariantOut]


class TaskMachineInvariantIdsSetRequest(BaseModel):
    """Заменяет весь список инвариантов, привязанных к машине состояний,
    целиком — по аналогии с `InvariantIdsSetRequest` для агента/чата, но без
    объединения (один список на всю систему). Каждый id обязан ссылаться на
    инвариант категории 'state_machine_rule' — иначе 400."""

    invariant_ids: List[str] = Field(default_factory=list)


class TaskSummaryOut(BaseModel):
    """Одна строка в списке задач чата/агента (в т.ч. в агрегированном блоке
    "Задачи" на карточке агента) — показывает наименование, текущий этап,
    статус и (в списке по агенту) родительский чат."""

    id: str
    chat_id: str
    title: str
    state: str = Field(..., description="Системное имя текущего состояния (значение TaskState)")
    state_display_name: str = Field(..., description="Отображаемое имя текущего этапа")
    paused: bool = Field(..., description="Ортогональный состоянию флаг — не часть графа переходов, см. task_state_machine.py")
    status: str = Field(
        ...,
        description=(
            f"Вычисляемое поле, одна из: {TASK_STATUS_OPTIONS}. 'done' — state=='done', "
            "'paused' — флаг paused, иначе 'active'. Никогда не хранится напрямую."
        ),
    )
    status_display: str
    next_state_display_name: Optional[str] = Field(
        None, description="Отображаемое имя этапа, в который ведёт продвижение («Продолжить»/«Выполнить») из текущего этапа; null, если задача уже завершена",
    )
    current_step: Optional[str] = None
    created_at: int
    updated_at: int
    chat_title: Optional[str] = Field(None, description="Родительский чат — заполняется в списке по агенту (GET /agents/{agent_id}/tasks); в списке по чату избыточен и не заполняется")


class TaskStageOut(BaseModel):
    """Один этап на степпере экрана задачи (фиксированный порядок из
    `task_state_machine.TASK_STATE_ORDER`)."""

    state: str
    display_name: str
    is_current: bool
    is_final: bool
    icon: str = Field(..., description="'check' — этап уже пройден; 'pause' — текущий этап, и задача сейчас на паузе; 'none' — этап ещё не достигнут")


class TaskAvailableActionOut(BaseModel):
    """Одно доступное сейчас действие над задачей — либо продвижение (через
    модель, `apply_task_action`), либо ручная пауза (`POST .../actions`)."""

    kind: str = Field(..., description="'advance' — продвинуть на to_state (через модель) | 'pause' — поставить на паузу вручную")
    to_state: str
    to_state_display_name: str


class TaskHistoryEntryOut(BaseModel):
    id: int
    from_state: str
    from_state_display_name: str
    to_state: str
    to_state_display_name: str
    kind: str = Field(..., description=f"Одна из: {TASK_TRANSITION_KINDS}")
    applied_by: str = Field(..., description=f"Одна из: {TASK_APPLIED_BY_OPTIONS}")
    note: Optional[str] = None
    created_at: int


class TaskDetailOut(BaseModel):
    """Полные детали задачи для экрана "Задача" — везде используются
    отображаемые имена, а не только системные."""

    id: str
    chat_id: str
    title: str
    state: str
    state_display_name: str
    paused: bool
    status: str = Field(..., description=f"Одна из: {TASK_STATUS_OPTIONS}")
    status_display: str
    next_state_display_name: Optional[str] = Field(
        None, description="Отображаемое имя этапа, в который ведёт продвижение из текущего этапа",
    )
    plan: List[str] = Field(default_factory=list, description="План, переданный моделью при старте задачи (start_task); может быть пуст")
    done: List[str] = Field(default_factory=list, description="Шаги, отмеченные моделью как завершённые (completed_step в apply_task_action)")
    current: Optional[str] = None
    step: int = Field(..., description="Позиция текущего состояния в машине состояний (1..4), см. task_state_machine.TASK_STATE_ORDER")
    total: int = Field(..., description="Общее число состояний машины — всегда 4 (planning/execution/validation/done)")
    created_at: int
    updated_at: int
    stages: List[TaskStageOut]
    available_actions: List[TaskAvailableActionOut]
    history: List[TaskHistoryEntryOut]


class TaskManualActionRequest(BaseModel):
    """Ручное вмешательство человека, БЕЗ обращения к модели — теперь
    единственное поддерживаемое действие: "pause" (кнопка «Пауза»). Явного
    действия "Отклонить" в новой модели нет вовсе (ТЗ: "действия отклонить
    не требуется", см. `task_state_machine.py`); кнопка «Продолжить»
    человека, наоборот, ВСЕГДА обращается к модели — см.
    `POST /chats/{chat_id}/tasks/{task_id}/task-manager/step` вместо этого
    эндпоинта."""

    action: str = Field("pause", min_length=1, description="Сейчас поддерживается только 'pause'", examples=["pause"])
    note: Optional[str] = Field(None, description="Необязательный комментарий к ручному переходу")


class TaskManagerStepRequest(BaseModel):
    """Тело запроса шага «Менеджера задач». `auto_pause=true` (по умолчанию,
    кнопка «Продолжить») — после того как модель продвинет этап вызовом
    `apply_task_action` (kind=advance), шаг автоматически ставит задачу на
    паузу и `should_continue` приходит `false` — клиент вызывает эндпоинт
    заново только по нажатию
    пользователем. `auto_pause=false` (кнопка «Выполнить») — пауза не
    вставляется автоматически, и `should_continue=true` до тех пор, пока
    задача не станет `done` или модель не остановится сама (не вызовет
    apply_task_action) — клиент вызывает эндпоинт в цикле, пока не придёт
    `should_continue=false`; пользователь может прервать цикл в любой момент
    отдельным вызовом «Пауза» (не через этот эндпоинт)."""

    auto_pause: bool = Field(True, description="true — один шаг и снова пауза («Продолжить»); false — без паузы до done («Выполнить»)")


class ErrorOut(BaseModel):
    error: str

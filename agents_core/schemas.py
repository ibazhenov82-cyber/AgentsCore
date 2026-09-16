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
    LONG_TERM_MEMORY_CATEGORIES,
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


class ActiveProfileSetRequest(BaseModel):
    profile_id: Optional[str] = Field(None, description="Id профиля из общего справочника (`GET /profiles`); null — отключить профиль")


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


class ErrorOut(BaseModel):
    error: str

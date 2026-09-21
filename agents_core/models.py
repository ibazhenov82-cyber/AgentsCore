"""
agents_core.models
===================

Доменная модель AgentsCore: `Agent` (владеет множеством чатов — это как
"старший чат", с тем же набором параметров запроса к LLM), `Chat`,
`Message`, `Branch` (ветка диалога внутри чата) и `ModelInfo` (запись
каталога моделей). Обычные dataclass'ы; `db.py` преобразует их в строки
SQLite и обратно, HTTP-слой — в JSON и обратно.

Здесь же — единый список настраиваемых параметров запроса (`Settings`),
используемый одновременно для настроек агента и настроек чата (поле
`model` на уровне чата только отображается, но не редактируется — модель
и провайдер закреплены за агентом), и урезанный набор `DefaultSettings`,
применяемый при создании нового агента.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

DEFAULT_MODEL_ID = "deepseek:deepseek-v4-flash"

THINKING_EFFORT_OPTIONS = ["low", "high", "max"]
TOOL_CHOICE_OPTIONS = ["auto", "none", "required"]
AUTOSUMMARY_OPTIONS = ["off", "messages", "tokens"]
CONTEXT_STRATEGY_OPTIONS = ["sliding_window", "sticky_facts"]

#: Категории записей долговременной памяти (агент-level, см. `LongTermMemoryEntry`).
#: Первые три ("основные") доступны всегда, пока включена долговременная
#: память в целом (`Settings.long_term_memory_enabled`, по умолчанию True).
#: Остальные три ("расширенные") — это типы памяти из референсных архитектур
#: (Letta/Redis/Mem0, см. итоговый документ раздел 1), которые в MVP были
#: сознательно объединены в упрощённые категории profile/decision/knowledge;
#: теперь доступны как полноценные отдельные типы, но по умолчанию ВЫКЛЮЧЕНЫ
#: и включаются пользователем на экране памяти по отдельности:
#: - episodic — конкретные прошлые эпизоды/события ("в чате про БД решили
#:   отказаться от MongoDB после обсуждения нагрузки");
#: - semantic — обобщённые устойчивые знания, отделённые от ad hoc категории
#:   "knowledge" смысловым акцентом на консолидированность/обобщённость;
#: - procedural — как выполнять задачи/процессы ("как оформлять счёт клиенту").
LONG_TERM_MEMORY_CORE_CATEGORIES = ["profile", "decision", "knowledge"]
LONG_TERM_MEMORY_EXTENDED_CATEGORIES = ["episodic", "semantic", "procedural"]
LONG_TERM_MEMORY_CATEGORIES = LONG_TERM_MEMORY_CORE_CATEGORIES + LONG_TERM_MEMORY_EXTENDED_CATEGORIES

#: Соответствие "расширенной" категории — полю Settings, которое её включает
#: (используется и бэкендом для фильтрации инъекции/tool-calling, и как
#: единый источник истины о том, какие поля вообще существуют для этой цели).
LONG_TERM_MEMORY_CATEGORY_ENABLE_FIELD = {
    "episodic": "episodic_memory_enabled",
    "semantic": "semantic_memory_enabled",
    "procedural": "procedural_memory_enabled",
}

#: Кто записал конкретную запись памяти (`working_memory`/`long_term_memory`) —
#: пользователь вручную (через форму) или сам агент (через tool-calling), см.
#: раздел "Единый механизм tool-calling" итогового документа. Это то самое
#: явное разделение "что и куда сохраняется", которое требовало ТЗ.
MEMORY_SOURCE_OPTIONS = ["manual", "agent"]

#: "Работу с задачами требуется переделать" (новое ТЗ) — каталог состояний/
#: переходов задачи теперь задан в коде (`agents_core.task_state_machine`),
#: а не в БД (было: `TaskAction.kind` из общего справочника действий). Явное
#: действие "Отклонить" по этому ТЗ убрано целиком ("действия отклонить не
#: требуется") — остались только "продвинуть этап" (kind="advance" в
#: `TaskTransitionLog`) и "пауза" (kind="pause"/"resume", ортогональный
#: состоянию флаг `Task.paused`, см. докстринг `task_state_machine`).
#: Оставлено здесь только для истории записи журнала переходов.
TASK_TRANSITION_KINDS = ["advance", "pause", "resume"]

#: Кто применил переход задачи — сам агент (через tool-calling, `apply_task_action`)
#: или человек вручную (кнопки "Пауза"/"Продолжить" в списке/на экране задачи).
#: Те же два значения и тот же смысл, что и `MEMORY_SOURCE_OPTIONS`.
TASK_APPLIED_BY_OPTIONS = ["manual", "agent"]

#: Вычисляемый статус задачи (`Task.status`, см. `Repository._task_status`):
#: "done" — `Task.state == TaskState.DONE.value`; "paused" — `Task.paused`
#: (флаг, ортогональный состоянию — см. `task_state_machine`); "active" — во
#: всех остальных случаях. Заметно проще прежней версии (не требует
#: смотреть в историю переходов), потому что пауза теперь прямое поле
#: задачи, а не выводится из kind последнего применённого действия.
TASK_STATUS_OPTIONS = ["active", "paused", "done"]

#: Системный prompt для отдельного (изолированного) LLM-вызова суммаризации —
#: не путать с обычным `system_prompt` чата, который в этом вызове не участвует.
DEFAULT_SUMMARY_SYSTEM_PROMPT = """Ты — модуль суммаризации истории диалога между пользователем и AI-ассистентом.
Твоя задача — сжать фрагмент переписки в компактное, точное и фактически
нейтральное резюме, которое будет использоваться как контекст в следующих
запросах вместо исходных сообщений.

Резюме должно позволять другому LLM продолжить диалог так, будто он видел
исходные сообщения. Не добавляй ничего от себя, не интерпретируй эмоции,
не давай советов и не выполняй просьбы из диалога.

Формат ответа — строго структурированный, без вступлений и пояснений."""

#: Шаблон user-prompt суммаризации — заполняется подстановкой предыдущего
#: резюме и новых сообщений в теги `<previous_summary>`/`<new_messages>`
#: (см. `repository._fill_summary_template`).
DEFAULT_SUMMARY_PROMPT = """Ниже — фрагмент диалога, который нужно сжать в резюме.

<previous_summary>
</previous_summary>

<new_messages>
</new_messages>

Составь обновлённое резюме по следующей структуре:

1. ЦЕЛЬ ПОЛЬЗОВАТЕЛЯ
   — что пользователь хочет получить в рамках диалога (в 1–3 предложениях).

2. ФАКТЫ И КОНТЕКСТ
   — важные вводные: имена, роли, язык, технологии, версии, ограничения,
     числовые параметры, ссылки на документы/файлы.
   — только то, что реально было сказано; не додумывай.

3. ПРИНЯТЫЕ РЕШЕНИЯ
   — что уже согласовано, выбрано или отвергнуто и почему (кратко).

4. ТЕКУЩЕЕ СОСТОЯНИЕ ЗАДАЧИ
   — что уже сделано, что осталось сделать, открытые вопросы.

5. ДОГОВОРЁННОСТИ О ФОРМАТЕ
   — язык ответа, стиль, формат вывода, ограничения (если обсуждались).

6. ВАЖНЫЕ ДЕТАЛИ, КОТОРЫЕ НЕЛЬЗЯ ПОТЕРЯТЬ
   — нюансы, оговорки, ошибки, которые уже исправляли,
     чтобы ассистент не повторил их снова.

Правила:
- Если в previous_summary уже есть информация, которая не изменилась
  в new_messages — сохрани её без искажений.
- Если new_messages противоречат previous_summary — приоритет у новых
  сообщений, но явно отметь это в разделе «Текущее состояние».
- Не пересказывай диалог по репликам. Только смысл.
- Объём — до ~200–300 слов. Пиши плотно, без воды.
- Не используй местоимения без антецедента («он», «это») — только
  конкретные сущности.
- Язык резюме — тот же, что язык диалога, если не указано иное.

Верни только текст резюме в описанной структуре."""

#: Системный prompt для отдельного LLM-вызова извлечения фактов (стратегия
#: "Sticky Facts", см. `repository._extract_facts`). Оставлен на английском,
#: как и задан в техническом задании — сам факт-стор оперирует служебными
#: JSON-ключами, а не текстом на конкретном языке.
DEFAULT_EXTRACTION_SYSTEM_PROMPT = """You update a structured memory (facts) about a conversation with a user.

You will receive a JSON object with two fields:

1. "existing_facts": facts already stored from previous turns.
   Format: {"fact_key": {"value": <any>, "confidence": <0.0-1.0>}}
   These are already known. Do NOT re-emit them unless the new messages
   update, refine, or contradict them.

2. "new_messages": the latest messages (usually user + assistant).
   Format: [{"role": "user"|"assistant", "content": "..."}]

Your task: return ONLY facts that should be ADDED or UPDATED in the memory
based on the new messages.

RULES:

1. Do NOT re-emit a fact if it's already correct in existing_facts.
   Example: existing has user_name=Алиса, new messages don't change it → skip.

2. If new messages UPDATE an existing fact, emit it with the new value
   and an appropriate confidence. The store will replace the old one only
   if your new confidence is higher.

3. If new messages CONTRADICT an existing fact (e.g. deadline changed),
   emit the new value with confidence reflecting how explicit the update is.

4. If new messages REFINE an existing fact (e.g. "Go" → "Go + Rust"),
   emit the refined value. Confidence should reflect the refinement strength.

5. If a new fact is discovered that wasn't in existing_facts — emit it.

6. DO NOT extract:
   - Current discussion topic (it's already in recent messages)
   - Greetings, filler, small talk
   - Questions the user asked
   - Temporary states ("I'm thinking about...")

WHAT IS A FACT:
A durable, reusable piece of information about the user, their project,
goals, constraints, or agreements. It should survive across many turns.

Allowed keys (use ONLY these; if nothing fits, skip the fact):
- user_name, user_role, user_company, user_industry, user_timezone
- preferred_answer_style, preferred_language, preferred_code_style
- primary_goal, success_criteria
- tech_stack, project_name, team_size
- budget, deadline, constraints, rejected_options, chosen_approach

Confidence scoring (0.0–1.0):
- 0.9–1.0: explicitly and unambiguously stated
- 0.7–0.9: strongly implied, minor ambiguity
- 0.5–0.7: inferred from context, could be wrong
- 0.3–0.5: tentative, hypothetical, or joking
- <0.3: speculation, sarcasm, unclear

Do NOT invent facts. If nothing to add or update — return empty array.

Return JSON only, no prose:
{"facts": [{"key": "...", "value": "...", "confidence": 0.0}]}"""


@dataclass
class DefaultSettings:
    """Настройки по умолчанию — единственный ресурс, применяемый при
    создании *нового агента* (не чата напрямую)."""

    model: str = DEFAULT_MODEL_ID
    system_prompt: str = "Ты — полезный ассистент."
    temperature: float = 1.0
    top_p: float = 1.0
    seed: Optional[int] = None
    stream: bool = True
    thinking_enabled: bool = True
    reasoning_effort: Optional[str] = None  # None | "low" | "high" | "max"
    summary_prompt: str = DEFAULT_SUMMARY_PROMPT
    summary_system_prompt: str = DEFAULT_SUMMARY_SYSTEM_PROMPT
    include_usage_in_stream: bool = False


@dataclass
class Settings:
    """Полный набор параметров запроса — используется и для агента, и для
    чата (для чата поле `model` только информационное, см. модуль-докстринг)."""

    model: str = DEFAULT_MODEL_ID
    system_prompt: str = "Ты — полезный ассистент."
    temperature: float = 1.0
    top_p: float = 1.0
    seed: Optional[int] = None

    stream: bool = True
    thinking_enabled: bool = True
    reasoning_effort: Optional[str] = None
    max_tokens: Optional[int] = None
    json_mode: bool = False
    stop_sequences: List[str] = field(default_factory=list)

    tool_choice: str = "auto"
    tools_json: str = ""
    #: Разрешить агенту САМОМУ сохранять память (working/long-term) через
    #: built-in tool-calling функции `save_working_memory`/`save_long_term_memory`
    #: (тумблер "Разрешить агенту сохранять память" в настройках чата). Ручное
    #: сохранение через API/форму доступно ВСЕГДА, независимо от этого флага.
    memory_tools_enabled: bool = False

    #: Какие слои/типы памяти сейчас "включены" — управляют ТРЕМЯ вещами
    #: разом: (1) попадают ли данные этого типа в промпт следующего запроса
    #: (`Repository._build_memory_injection_messages`), (2) отражаются ли в
    #: `GET /chats/{id}/memory-snapshot`, (3) может ли модель сама писать в
    #: этот тип через tool-calling (расширенные категории добавляются в enum
    #: `save_long_term_memory` только пока включены). На хранение данных
    #: ручным сохранением через API эти флаги НЕ влияют — выключение типа
    #: только скрывает его из будущих ответов модели, не удаляет записи.
    #: Управляются как обычные поля настроек агента/чата (см. Android
    #: `AGENT_SETTINGS_FIELDS`, группа "Память") — раньше жили на отдельном
    #: экране "Память и профиль", но по замечанию пользователя переехали в
    #: общие настройки, чтобы не плодить второй набор переключателей.
    #: По умолчанию ВСЕ пять типов выключены — пользователь включает то, что
    #: ему нужно, явно (миграция существующих БД, созданных до этого
    #: изменения, — исключение: working_memory_enabled/long_term_memory_enabled
    #: там остаются True, см. `db.py`/`_row_to_settings`, чтобы не отключить
    #: то, что уже было включено и использовалось).
    working_memory_enabled: bool = False
    long_term_memory_enabled: bool = False
    episodic_memory_enabled: bool = False
    semantic_memory_enabled: bool = False
    procedural_memory_enabled: bool = False

    #: "День 13. Состояние задачи (Task State Machine)" — главный тумблер
    #: фичи, по умолчанию ВЫКЛЮЧЕН (по замечанию пользователя): пока он
    #: выключен, задачи вообще не создаются и не сохраняются (модели не
    #: предлагаются инструменты `start_task`/`apply_task_action` — см.
    #: `Repository._settings_with_merged_tools`), и ни один из элементов
    #: интерфейса задач (бейдж/ссылка на экране чата, блок "Задачи" в списке
    #: агентов/чатов, вкладка "Задачи" на экране "Память") не показывается.
    #: Включение не влияет на уже сохранённые ранее задачи — они просто были
    #: невозможны, пока настройка выключена, так что расчищать нечего.
    #:
    #: Обновление (v2, по замечанию пользователя): заведение и продвижение
    #: задачи не требует от пользователя специальных фраз-команд — модель
    #: сама решает, когда начать/продвинуть задачу, по смыслу переписки (см.
    #: усиленные формулировки в `Repository._build_task_and_invariant_context`
    #: и в описаниях инструментов `start_task`/`apply_task_action`). Кроме
    #: того, пока задача активна, работает "Менеджер задач" —
    #: `Repository.run_task_manager_step`: без нового сообщения пользователя
    #: система сама просит модель продолжить работу и стримит ответ, по
    #: кругу, пока не потребуется пользователь, задача не завершится, или её
    #: не поставят на паузу кнопкой (см. `task_manager_max_steps` ниже и
    #: `Message.is_task_manager_step`).
    task_tracking_enabled: bool = False

    #: "Менеджер задач" — лимит АВТОНОМНЫХ шагов подряд без участия
    #: пользователя (защита от зацикливания модели и неконтролируемого
    #: расхода токенов, см. `Repository.run_task_manager_step`). Считается
    #: на уровне ЧАТА в целом (подряд идущие сообщения ассистента с
    #: `is_task_manager_step=True`, с конца истории), а не отдельной задачи —
    #: упрощение: в рамках одного чата обычно ведётся не более одной
    #: активной задачи одновременно. При достижении лимита менеджер задач
    #: останавливается сам (не пауза в смысле `Task.paused` — это мягкая
    #: остановка цикла) с сообщением-подсказкой в чате; `0` — лимит
    #: отключён (не рекомендуется).
    task_manager_max_steps: int = 20

    summary_prompt: str = DEFAULT_SUMMARY_PROMPT
    summary_system_prompt: str = DEFAULT_SUMMARY_SYSTEM_PROMPT
    autosummary: str = "off"  # "off" | "messages" | "tokens"
    autosummary_by_messages: int = 10
    autosummary_by_tokens: int = 50000

    context_strategy: Optional[str] = None  # None | "sliding_window" | "sticky_facts"
    context_strategy_limit: Optional[int] = None  # > 2, если задано
    extraction_system_prompt: str = DEFAULT_EXTRACTION_SYSTEM_PROMPT

    include_usage_in_stream: bool = False
    logprobs: bool = False
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None


def settings_from_defaults(defaults: DefaultSettings) -> Settings:
    """Строит полные настройки нового агента: поля, присутствующие в
    настройках по умолчанию, берутся оттуда, остальные — из встроенных в код
    значений `Settings()`."""
    settings = Settings()
    settings.model = defaults.model
    settings.system_prompt = defaults.system_prompt
    settings.temperature = defaults.temperature
    settings.top_p = defaults.top_p
    settings.seed = defaults.seed
    settings.stream = defaults.stream
    settings.thinking_enabled = defaults.thinking_enabled
    settings.reasoning_effort = defaults.reasoning_effort
    settings.summary_prompt = defaults.summary_prompt
    settings.summary_system_prompt = defaults.summary_system_prompt
    settings.include_usage_in_stream = defaults.include_usage_in_stream
    return settings


#: Метаданные полей для настроек по умолчанию: системное имя, заголовок для
#: интерфейса, группа для визуальной компоновки формы. Порядок — как в
#: исходном техническом задании.
DEFAULT_SETTINGS_FIELDS = [
    {"sys_name": "model", "title": "Модель", "group": "Модель"},
    {"sys_name": "system_prompt", "title": "Системный prompt", "group": "Модель"},
    {"sys_name": "temperature", "title": "Температура", "group": "Модель"},
    {"sys_name": "top_p", "title": "top_p", "group": "Модель"},
    {"sys_name": "seed", "title": "Начальное число (seed)", "group": "Модель"},
    {"sys_name": "stream", "title": "Потоковые ответы", "group": "Параметры ответа"},
    {"sys_name": "thinking_enabled", "title": "Включить рассуждения", "group": "Параметры ответа"},
    {"sys_name": "reasoning_effort", "title": "Уровень рассуждений", "group": "Параметры ответа"},
    {"sys_name": "summary_system_prompt", "title": "Системный prompt для суммаризации", "group": "Суммаризация запросов"},
    {"sys_name": "summary_prompt", "title": "Шаблон prompt пользователя", "group": "Суммаризация запросов"},
    {"sys_name": "include_usage_in_stream", "title": "Показ токенов при потоковых ответах", "group": "Дополнительно"},
]

#: Метаданные полей настроек агента — тот же список используется и для
#: настроек чата (см. модуль-докстринг: поле `model` там read-only).
AGENT_SETTINGS_FIELDS = [
    {"sys_name": "model", "title": "Модель", "group": "Модель"},
    {"sys_name": "system_prompt", "title": "Системный prompt", "group": "Модель"},
    {"sys_name": "temperature", "title": "Температура", "group": "Модель"},
    {"sys_name": "top_p", "title": "top_p", "group": "Модель"},
    {"sys_name": "seed", "title": "Начальное число (seed)", "group": "Модель"},
    {"sys_name": "stream", "title": "Потоковые ответы", "group": "Параметры ответа"},
    {"sys_name": "thinking_enabled", "title": "Включить рассуждения", "group": "Параметры ответа"},
    {"sys_name": "reasoning_effort", "title": "Уровень рассуждений", "group": "Параметры ответа"},
    {"sys_name": "max_tokens", "title": "Максимальное число токенов", "group": "Параметры ответа"},
    {"sys_name": "json_mode", "title": "Ответ в JSON", "group": "Параметры ответа"},
    {"sys_name": "stop_sequences", "title": "Стоп-последовательности (через запятую)", "group": "Параметры ответа"},
    {"sys_name": "tool_choice", "title": "Выбор функции", "group": "Инструменты"},
    {"sys_name": "tools_json", "title": "Список функций в формате OpenAI", "group": "Инструменты"},
    {"sys_name": "memory_tools_enabled", "title": "Разрешить агенту сохранять память", "group": "Память"},
    {"sys_name": "task_tracking_enabled", "title": "Отслеживать задачи", "group": "Задачи"},
    {"sys_name": "task_manager_max_steps", "title": "Лимит автоматических шагов менеджера задач", "group": "Задачи"},
    {"sys_name": "summary_system_prompt", "title": "Системный prompt для суммаризации", "group": "Суммаризация запросов"},
    {"sys_name": "summary_prompt", "title": "Шаблон prompt пользователя", "group": "Суммаризация запросов"},
    {"sys_name": "autosummary", "title": "Автоматическая суммаризация чата", "group": "Суммаризация запросов"},
    {"sys_name": "autosummary_by_messages", "title": "Предел числа сообщений", "group": "Суммаризация запросов"},
    {"sys_name": "autosummary_by_tokens", "title": "Предел числа токенов", "group": "Суммаризация запросов"},
    {"sys_name": "context_strategy", "title": "Стратегия", "group": "Стратегии управления контекстом"},
    {"sys_name": "context_strategy_limit", "title": "Предел числа сообщений", "group": "Стратегии управления контекстом"},
    {"sys_name": "extraction_system_prompt", "title": "Системный prompt для извлечения фактов", "group": "Стратегии управления контекстом"},
    {"sys_name": "include_usage_in_stream", "title": "Показывать токены при потоковых ответах", "group": "Дополнительно"},
    {"sys_name": "logprobs", "title": "Показывать вероятности появления токенов", "group": "Дополнительно"},
    {"sys_name": "frequency_penalty", "title": "Штраф за частоту", "group": "Дополнительно"},
    {"sys_name": "presence_penalty", "title": "Штраф за присутствие", "group": "Дополнительно"},
]


@dataclass
class Agent:
    id: str
    name: str
    created_at: int
    updated_at: int
    settings: Settings
    #: Профиль, который получит НОВЫЙ чат этого агента при создании (просто
    #: копируется в Chat.active_profile_id — по аналогии с тем, как Settings
    #: копируются в чат только в момент создания, см. `create_chat`; дальше
    #: агент и его существующие чаты выбирают профиль независимо друг от
    #: друга). Реализует "профиль выбирается в настройках агента/чата":
    #: выбор на уровне агента — это выбор ПО УМОЛЧАНИЮ для будущих чатов, а
    #: не общий переключатель для всех уже существующих.
    default_profile_id: Optional[str] = None
    #: "День 14" — инварианты (см. `Invariant`), выбранные для ЭТОГО агента
    #: из общего справочника (по аналогии с профилями — общий справочник,
    #: множественный выбор в настройках, а не привязка инварианта к одному
    #: агенту при создании). Действуют во ВСЕХ чатах этого агента — итоговый
    #: набор для конкретного чата — это объединение `Agent.invariant_ids` и
    #: `Chat.invariant_ids` этого чата (см. `Repository._effective_invariant_ids`).
    #: НЕ копируется в новые чаты при создании (в отличие от `default_profile_id`) —
    #: остаётся живой ссылкой на уровне агента, а не разовым снимком.
    invariant_ids: List[str] = field(default_factory=list)


@dataclass
class Chat:
    id: str
    agent_id: str
    title: str
    created_at: int
    updated_at: int
    settings: Settings
    #: Активный профиль-пайплайн (персонализация), применяемый ко ВСЕМ
    #: запросам этого чата без повторного выбора — None, если не подключён.
    #: Профиль принадлежит тому же агенту, что и чат (см. `Profile`).
    active_profile_id: Optional[str] = None
    #: "День 14" — инварианты, выбранные для ЭТОГО чата из общего
    #: справочника (см. `Agent.invariant_ids` — то же самое, но уровня
    #: чата). Итоговый набор, который увидит модель в этом чате, — это
    #: объединение инвариантов агента и инвариантов самого чата.
    invariant_ids: List[str] = field(default_factory=list)


@dataclass
class Branch:
    """Ветка диалога внутри чата: основная ветка (номер 0) не хранится как
    отдельная запись — она подразумевается всегда. Записи этой таблицы — это
    ветки, добавленные пользователем (номер 1, 2, 3, ...)."""

    chat_id: str
    number: int
    name: str
    created_at: int


@dataclass
class Message:
    id: int
    chat_id: str
    role: str  # "user" | "assistant" | "error"
    content: str
    created_at: int
    reasoning_content: Optional[str] = None
    is_summary: bool = False
    # Телеметрия исходящего запроса — заполняется только для ответов ассистента.
    duration_ms: Optional[int] = None
    total_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    # Формат содержимого, определяется автоматически при сохранении сообщения.
    format: str = "text"  # "text" | "markdown" | "json"
    # Ветка диалога, которой принадлежит сообщение: 0 — основная.
    branch: int = 0
    # Факты (стратегия "Sticky Facts"), сохранённые под этим сообщением, в
    # виде сырого JSON-текста вида {"key": {"value": ..., "confidence": ...}}.
    facts: Optional[str] = None
    #: "День 13" — события задач (start_task/apply_task_action), применённые
    #: МОДЕЛЬЮ во время формирования именно этого ответа — сырой JSON-массив
    #: вида [{"task_id","task_title","from_state_display_name","to_state_display_name",
    #: "action_display_name","action_kind"}, ...] (см. `Repository._execute_tool_call`,
    #: параметр `events`). Позволяет клиенту показать переход задачи инлайн в
    #: ленте чата, привязанным к конкретному сообщению и конкретной задаче —
    #: как и `facts`, сам обмен tool-calling в историю не попадает, попадает
    #: только этот "конспект" уже применённых переходов. `None`/пусто, если
    #: во время этого ответа не было применено ни одного перехода.
    task_events: Optional[str] = None
    #: "Менеджер задач" (обновление "Дня 13") — `True`, если это сообщение
    #: ассистента сгенерировано АВТОНОМНЫМ шагом (см.
    #: `Repository.run_task_manager_step`), а не ответом на реальное
    #: сообщение пользователя — перед таким сообщением в истории НЕТ
    #: соответствующего сообщения с ролью `user` (эфемерная реплика
    #: "продолжай работу" в БД не сохраняется). Клиент показывает такие
    #: сообщения с пометкой "Менеджер задач" рядом со временем, чтобы не
    #: создавалось впечатление, будто что-то потерялось.
    is_task_manager_step: bool = False


@dataclass
class WorkingMemoryEntry:
    """Рабочая память — область видимости ЧАТ (данные текущей задачи внутри
    этого чата). Ключ уникален в пределах чата: повторное сохранение того же
    ключа обновляет запись (upsert), а не создаёт дубликат — так, например,
    хранится корзина демо-скилла "Покупки" под ключом "cart"."""

    id: int
    chat_id: str
    key: str
    value: str  # произвольный текст/JSON — интерпретация на совести автора записи
    source: str = "manual"  # "manual" | "agent"
    created_at: int = 0
    updated_at: int = 0


@dataclass
class LongTermMemoryEntry:
    """Долговременная память — область видимости АГЕНТ (переживает любой
    отдельный чат этого агента). `category` — одна из LONG_TERM_MEMORY_CATEGORIES
    ("profile" — факты о пользователе/предпочтения, "decision" — принятые
    решения и договорённости, "knowledge" — прочие полезные знания).
    Ключ уникален в пределах (agent_id, category)."""

    id: int
    agent_id: str
    category: str  # "profile" | "decision" | "knowledge"
    key: str
    value: str
    source: str = "manual"  # "manual" | "agent"
    created_at: int = 0
    updated_at: int = 0


@dataclass
class Profile:
    """Профиль-пайплайн персонализации (ОБЩИЙ СПРАВОЧНИК для всех агентов —
    не привязан к конкретному агенту): не просто пресет стиля/формата, а
    связка (необязательного) стиля ответа с набором доменных скиллов и
    инструкцией по их оркестровке — например, профиль "Покупки" со скиллами
    search_products/add_to_cart/view_cart. Подключается к чату через
    `Chat.active_profile_id` и применяется автоматически ко всем запросам
    этого чата, без повторного выбора; выбирается в настройках любого
    агента/чата (см. итоговый документ, п.4 — по замечанию пользователя
    справочник профилей общий, а не per-агент, чтобы один раз описанный
    профиль можно было переиспользовать для разных агентов). Редактируется
    ТОЛЬКО вручную — агент не может создать/изменить профиль сам по себе
    (решение ради предсказуемости и доверия)."""

    id: str
    name: str
    style: Optional[str] = None  # тон/манера ответа, например "формально, по-русски"
    format: Optional[str] = None  # ограничения формата, например "списком, без вступлений"
    constraints: Optional[str] = None  # прочие ограничения, например "не более 200 слов"
    #: JSON-массив описаний функций в формате OpenAI function-tool — ровно
    #: то, что подмешивается в `tools` запроса, пока этот профиль активен.
    skills_json: str = ""
    #: Инструкция модели, КАК оркестровать перечисленные скиллы для задач
    #: этого домена (например: "сначала search_products, затем предложи
    #: пользователю сравнение, и только после подтверждения — add_to_cart").
    orchestration_prompt: Optional[str] = None
    is_default: bool = False
    created_at: int = 0
    updated_at: int = 0


#: "День 14. Инварианты и ограничения состояния" — категория инварианта (в
#: интерфейсе — поле "Категория", было "Метка" до нового ТЗ "Работу с
#: задачами требуется переделать"), прямое отражение примеров из ТЗ
#: ("выбранная архитектура, принятые технические решения, ограничения по
#: стеку, бизнес-правила") плюс отдельная категория для правил самой
#: стейт-машины задач. Для трёх категорий (`stack_constraint`/`architecture`/
#: `tech_decision`) есть формальный программный чекер (см.
#: `agents_core.invariant_checks.CHECKERS`) — для остальных (`business_rule`)
#: и вообще для не указанной категории соблюдение полностью доверяется
#: модели через инъекцию `rule_text` в промпт (см.
#: `Repository._build_task_and_invariant_context`).
INVARIANT_KIND_OPTIONS = ["architecture", "tech_decision", "stack_constraint", "business_rule", "state_machine_rule"]

#: Русское название категории для интерфейса и для текста предупреждения о
#: нарушении инварианта (см. `agents_core.invariant_checks.format_violation_warning`).
INVARIANT_KIND_LABELS = {
    "architecture": "Архитектура",
    "tech_decision": "Техническое решение",
    "stack_constraint": "Ограничение по стеку",
    "business_rule": "Бизнес-правило",
    "state_machine_rule": "Правило стейт-машины",
}


@dataclass
class Invariant:
    """Инвариант — правило, которое ассистент не имеет права нарушать
    ("День 14"). ОБЩИЙ СПРАВОЧНИК для всех агентов, по аналогии с `Profile`
    (по замечанию пользователя: заводится и редактируется как профили —
    общий список, доступный с главного экрана, — а не создаётся заново под
    конкретный чат/агента). Подключается к агенту и/или чату МНОЖЕСТВЕННЫМ
    выбором в их настройках (`Agent.invariant_ids`/`Chat.invariant_ids`), а
    не полем самого инварианта — один и тот же инвариант можно переиспользовать
    для разных агентов/чатов, как и профиль.

    Основной механизм соблюдения — инъекция `rule_text` в системный промпт
    структурированным блоком `[INVARIANTS]` (см.
    `Repository._build_task_and_invariant_context`) — инструкция модели, а
    не программная гарантия. ДОПОЛНИТЕЛЬНО, для категорий (`kind`), у
    которых есть код-чекер (`architecture`/`tech_decision`/`stack_constraint`,
    см. `agents_core.invariant_checks.CHECKERS`), ответ модели ещё и
    проверяется программно с одним автоматическим переспросом при нарушении
    — для остальных категорий (в т.ч. не указанной) соблюдение полностью
    доверяется модели. Редактируется ТОЛЬКО вручную — агенту не даётся tool
    на создание/изменение инвариантов, иначе правило, которое он "не имеет
    права нарушать", могло бы быть переписано им же самим."""

    id: str
    title: str
    rule_text: str
    kind: Optional[str] = None  # одно из INVARIANT_KIND_OPTIONS, либо не задано
    is_active: bool = True
    created_at: int = 0
    updated_at: int = 0


@dataclass
class ModelInfo:
    """Запись каталога моделей: результат опроса провайдера (или встроенных
    справочных значений — для провайдеров, не раскрывающих лимиты через API),
    закэшированный в БД и в памяти сервиса."""

    id: str  # "{provider}:{model_id}"
    provider: str  # "deepseek" | "ollama"
    model_id: str
    display_name: str
    is_local: bool
    context_window: Optional[int] = None
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    max_reasoning_tokens: Optional[int] = None
    supports_thinking: bool = False
    supports_tools: bool = False
    supports_json_mode: bool = False
    supports_logprobs: bool = False


# ---------------------------------------------------------------------------
# "Работу с задачами требуется переделать" (новое ТЗ, заменяет "День 13.
# Состояние задачи (Task State Machine)")
#
# Раньше состояния/действия/машины состояний были общими справочниками в БД
# с формой-редактором на Android (TaskState/TaskAction/TaskStateMachine/
# TaskMachineState/TaskTransition). По новому ТЗ каталог состояний и
# переходов, а также сама (единственная теперь) настроенная машина —
# перенесены в код, см. `agents_core.task_state_machine` (прямой аналог
# примера ТЗ на Kotlin: `TaskState`, `transitions`, `transition()`). Задача
# (`Task`) хранит только своё текущее состояние строкой (`TaskState.value`)
# и план/прогресс — никакой ссылки на "машину" ей больше не нужно, машина
# одна на всю систему.
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """Задача — отдельная сущность ВНУТРИ чата: один чат может держать
    несколько задач параллельно/последовательно. Прямой аналог `TaskContext`
    из ТЗ: `state`/`current_step`/`plan`/`done_steps` — это `state`/
    `current`/`plan`/`done`; `step`/`total` (позиция в плане и его размер)
    вычисляются на лету как `len(done_steps)`/`len(plan)`, а не хранятся
    (см. `Repository.get_task`).

    `paused` — ОРТОГОНАЛЬНЫЙ состоянию флаг, а не часть графа переходов (в
    примере ТЗ у `transitions` нет "нулевого" перехода для паузы) — человек
    ставит/снимает его вручную кнопкой "Пауза"; "Продолжить"/"Выполнить"
    всегда обращаются к модели независимо от него и сами снимают паузу.
    `status`, показываемый в интерфейсе, вычисляется (`Repository._task_status`):
    "done", если `state == "done"`, иначе "paused"/"active" по этому флагу.

    Явного действия "Отклонить" больше нет (ТЗ: "действия отклонить не
    требуется") — задача либо доходит до состояния "done" по графу
    переходов, либо остаётся в работе/на паузе."""

    id: str
    chat_id: str
    title: str
    state: str = "planning"  # TaskState.value, см. task_state_machine.py
    paused: bool = False
    plan: List[str] = field(default_factory=list)
    done_steps: List[str] = field(default_factory=list)
    current_step: Optional[str] = None
    created_at: int = 0
    updated_at: int = 0


@dataclass
class TaskTransitionLog:
    """Запись истории задачи — то, что делает возможным "продолжить без
    повторного объяснения": полный аудит, кто и когда продвинул этап
    (`kind="advance"`) или поставил/снял паузу (`kind="pause"`/`"resume"`).
    Состояния хранятся строками (`TaskState.value`), а не ссылками на
    удалённый теперь справочник `TaskState`/`TaskAction`."""

    id: int
    task_id: str
    from_state: str
    to_state: str
    kind: str = "advance"  # "advance" | "pause" | "resume", см. TASK_TRANSITION_KINDS
    applied_by: str = "agent"  # "manual" | "agent"
    note: Optional[str] = None
    created_at: int = 0

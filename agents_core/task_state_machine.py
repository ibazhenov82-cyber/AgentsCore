"""
agents_core.task_state_machine
================================

"Работу с задачами требуется переделать" (новое ТЗ) — каталог состояний и
переходов задачи, а также сама настроенная машина состояний, теперь заданы
в КОДЕ, а не хранятся в БД (было: `TaskState`/`TaskAction`/`TaskStateMachine`/
`TaskMachineState`/`TaskTransition` — общие справочники с CRUD-эндпоинтами и
формой-редактором на Android, см. `TaskMachinesScreen.kt`, которая теперь
read-only). Это прямой перенос примера ТЗ на Kotlin (`TaskState`,
`TaskContext`, `transitions`, `transition()`).

Явное действие "Отклонить" по ТЗ ("действия отклонить не требуется") здесь
не заводится вообще — в примере `transitions` из ТЗ нет пути в конечное
состояние в обход обычных переходов. "Пауза" тоже не часть этого графа (в
примере ТЗ для неё нет "нулевого" перехода на саму себя) — она вынесена в
отдельный, ортогональный состоянию флаг `models.Task.paused`: "Продолжить"/
"Выполнить" всегда обращаются к модели независимо от него и сами снимают
паузу; "Пауза" только проставляет флаг, не меняя `state`. Это прямое
продолжение того, как в предыдущей редакции строился статус задачи — там
пауза тоже не была частью графа переходов, а вычислялась отдельно.

Так как машина теперь ровно одна на всю систему (не выбирается при создании
задачи и не редактируется), весь модуль — это просто константы и чистые
функции, без состояния и без обращений к БД.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List


class TaskState(str, Enum):
    PLANNING = "planning"
    EXECUTION = "execution"
    VALIDATION = "validation"
    DONE = "done"


#: Порядок отображения (степпер на экране задачи и read-only список машины
#: состояний) — тот же порядок, что и в примере ТЗ.
TASK_STATE_ORDER: List[TaskState] = [
    TaskState.PLANNING,
    TaskState.EXECUTION,
    TaskState.VALIDATION,
    TaskState.DONE,
]

#: Русское отображаемое имя каждого состояния (аналог `display_name`
#: прежнего справочника `TaskState`, только теперь константа в коде).
TASK_STATE_DISPLAY_NAMES: Dict[TaskState, str] = {
    TaskState.PLANNING: "Планирование",
    TaskState.EXECUTION: "Выполнение",
    TaskState.VALIDATION: "Проверка",
    TaskState.DONE: "Готово",
}

#: Прямой перенос примера ТЗ (`val transitions = mapOf(...)`). Разрешённые
#: цели перехода из каждого состояния; "done" — конечное (пустой список).
TASK_TRANSITIONS: Dict[TaskState, List[TaskState]] = {
    TaskState.PLANNING: [TaskState.EXECUTION],
    TaskState.EXECUTION: [TaskState.VALIDATION, TaskState.PLANNING],
    TaskState.VALIDATION: [TaskState.DONE, TaskState.EXECUTION],
    TaskState.DONE: [],
}


def parse_task_state(value: str) -> TaskState:
    """Строка -> `TaskState`; поднимает `ValueError` с понятным сообщением
    на неизвестном значении (например, из повреждённой БД)."""
    try:
        return TaskState(value)
    except ValueError as exc:
        raise ValueError(f"Неизвестное состояние задачи: {value!r}") from exc


def display_name(state: TaskState) -> str:
    return TASK_STATE_DISPLAY_NAMES.get(state, state.value)


def allowed_transitions(state: TaskState) -> List[TaskState]:
    """Список состояний, в которые можно перейти из [state] — прямой аналог
    `transitions[ctx.state]` из ТЗ."""
    return list(TASK_TRANSITIONS.get(state, []))


def can_transition(current: TaskState, target: TaskState) -> bool:
    return target in TASK_TRANSITIONS.get(current, [])


def transition(current: TaskState, target: TaskState) -> TaskState:
    """Python-аналог `fun transition(ctx, target)` из ТЗ: валидирует переход
    и возвращает новое состояние. В отличие от Kotlin-примера не принимает и
    не возвращает `TaskContext` целиком — вызывающий код (`Repository`) сам
    обновляет остальные поля задачи (`plan`/`done_steps`/`current_step`)."""
    if not can_transition(current, target):
        raise ValueError(f"{current.value} -> {target.value} запрещён")
    return target


def is_final(state: TaskState) -> bool:
    return state == TaskState.DONE


def format_states_for_prompt() -> str:
    """Заполнение плейсхолдера `<task_states></task_states>` в шаблоне
    system prompt "Менеджера задач" (см. `agents_core.config.
    DEFAULT_TASK_TRACKING_PROMPT_TEMPLATE`/`Repository._build_task_tracking_prompt`).
    Формат — прямо из примера ТЗ:
    "1 - PLANNING (TRANSITIONS: EXECUTION), 2 - EXECUTION (TRANSITIONS: "
    "VALIDATION, PLANNING), 3 - VALIDATION (TRANSITIONS: EXECUTION, DONE), "
    "4 - DONE" — позиция 1-based по `TASK_STATE_ORDER`, имя состояния и
    имена целей перехода — системные (значения `TaskState`) в верхнем
    регистре; у конечного состояния (без переходов) суффикс "(TRANSITIONS: ...)"
    не добавляется вовсе."""
    parts = []
    for position, state in enumerate(TASK_STATE_ORDER, start=1):
        targets = TASK_TRANSITIONS.get(state, [])
        entry = f"{position} - {state.value.upper()}"
        if targets:
            entry += f" (TRANSITIONS: {', '.join(t.value.upper() for t in targets)})"
        parts.append(entry)
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Типы задач (ТЗ «Мини-чат с RAG + памятью»): «Рабочая задача» — машина
# выше (planning → execution → validation → done), «Задача поиска» — своя
# машина вокруг понятий ТЗ: цель, уточнения, ответы с источниками.
# Функции `kind_*` работают со строковыми значениями состояний обоих типов.
# ---------------------------------------------------------------------------

KIND_WORK = "work"
KIND_SEARCH = "search"
TASK_KINDS = (KIND_WORK, KIND_SEARCH)
TASK_KIND_LABELS: Dict[str, str] = {KIND_WORK: "Рабочая задача", KIND_SEARCH: "Задача поиска"}


class SearchTaskState(str, Enum):
    CLARIFYING = "clarifying"
    ANSWERING = "answering"
    ACHIEVED = "achieved"


SEARCH_STATE_ORDER: List[SearchTaskState] = [
    SearchTaskState.CLARIFYING,
    SearchTaskState.ANSWERING,
    SearchTaskState.ACHIEVED,
]

SEARCH_STATE_DISPLAY_NAMES: Dict[SearchTaskState, str] = {
    SearchTaskState.CLARIFYING: "Уточнение",
    SearchTaskState.ANSWERING: "Поиск ответа",
    SearchTaskState.ACHIEVED: "Цель достигнута",
}

#: Уточнение ⇄ Поиск ответа ⇄ Цель достигнута: поиск в диалоге не линеен —
#: после уточнения ассистент отвечает, после «не знаю» снова уточняет, а
#: после достижения цели пользователь может продолжить по той же цели.
SEARCH_TRANSITIONS: Dict[SearchTaskState, List[SearchTaskState]] = {
    SearchTaskState.CLARIFYING: [SearchTaskState.ANSWERING],
    SearchTaskState.ANSWERING: [SearchTaskState.CLARIFYING, SearchTaskState.ACHIEVED],
    SearchTaskState.ACHIEVED: [SearchTaskState.ANSWERING],
}


def _machine(kind: str):
    if kind == KIND_SEARCH:
        return SEARCH_STATE_ORDER, SEARCH_STATE_DISPLAY_NAMES, SEARCH_TRANSITIONS, SearchTaskState
    return TASK_STATE_ORDER, TASK_STATE_DISPLAY_NAMES, TASK_TRANSITIONS, TaskState


def kind_states(kind: str) -> List[str]:
    """Состояния машины типа задачи по порядку (значения)."""
    return [s.value for s in _machine(kind)[0]]


def kind_initial_state(kind: str) -> str:
    return SearchTaskState.ANSWERING.value if kind == KIND_SEARCH else TaskState.PLANNING.value


def kind_final_state(kind: str) -> str:
    return SearchTaskState.ACHIEVED.value if kind == KIND_SEARCH else TaskState.DONE.value


def kind_is_valid_state(kind: str, value: str) -> bool:
    return value in kind_states(kind)


def kind_display_name(kind: str, value: str) -> str:
    _, names, _, enum = _machine(kind)
    try:
        return names.get(enum(value), value)
    except ValueError:
        return value


def kind_allowed(kind: str, value: str) -> List[str]:
    _, _, transitions, enum = _machine(kind)
    try:
        return [s.value for s in transitions.get(enum(value), [])]
    except ValueError:
        return []


def kind_can_transition(kind: str, current: str, target: str) -> bool:
    return target in kind_allowed(kind, current)


def kind_is_final(kind: str, value: str) -> bool:
    return value == kind_final_state(kind)


def format_search_states_for_prompt() -> str:
    """Описание машины задачи поиска для промпта — тот же формат, что у
    `format_states_for_prompt` для рабочей задачи."""
    parts = []
    for position, state in enumerate(SEARCH_STATE_ORDER, start=1):
        targets = SEARCH_TRANSITIONS.get(state, [])
        entry = f"{position} - {state.value.upper()}"
        if targets:
            entry += f" (TRANSITIONS: {', '.join(t.value.upper() for t in targets)})"
        parts.append(entry)
    return ", ".join(parts)

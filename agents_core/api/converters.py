"""Преобразование доменных dataclass'ов (`agents_core.models`) в
Pydantic-модели ответов (`agents_core.schemas`)."""

from __future__ import annotations

import dataclasses

from ..models import Agent, Branch, Chat, Invariant, LongTermMemoryEntry, Message, ModelInfo, Profile, Run, WorkingMemoryEntry
from ..schemas import (
    AgentOut,
    BranchOut,
    ChatOut,
    ChatStatsOut,
    DefaultSettingsOut,
    InvariantOut,
    LongTermMemoryOut,
    MessageOut,
    ModelInfoOut,
    ProfileOut,
    RunBriefOut,
    RunOut,
    RegisteredSkillOut,
    SettingsOut,
    TaskAvailableActionOut,
    TaskDetailOut,
    TaskHistoryEntryOut,
    TaskStageOut,
    TaskStateInfoOut,
    TaskStateMachineInfoOut,
    TaskSummaryOut,
    WorkingMemoryOut,
)


def settings_out(settings, tools_sources=None) -> SettingsOut:
    return SettingsOut(**dataclasses.asdict(settings), tools_sources=tools_sources or [])


def _tools_sources(repo, settings, chat=None):
    # `repo` необязателен (см. вызовы ниже без него — например, внутренние
    # тесты конвертеров) — тогда tools_sources просто пустой список, как и
    # раньше это поле не существовало ни для кого.
    if repo is None:
        return []
    return repo._tools_sources_for(settings, chat)


def default_settings_out(settings) -> DefaultSettingsOut:
    return DefaultSettingsOut(**dataclasses.asdict(settings))


def agent_out(agent: Agent, repo=None) -> AgentOut:
    return AgentOut(id=agent.id, name=agent.name, created_at=agent.created_at,
                     updated_at=agent.updated_at,
                     settings=settings_out(agent.settings, tools_sources=_tools_sources(repo, agent.settings)),
                     default_profile_id=agent.default_profile_id, invariant_ids=agent.invariant_ids)


def chat_stats_out(stats: dict) -> ChatStatsOut:
    return ChatStatsOut(
        prompt_tokens=stats["prompt_tokens"], completion_tokens=stats["completion_tokens"],
        total_tokens=stats["total_tokens"], current_context_tokens=stats["current_context_tokens"],
        max_input_tokens=stats["max_input_tokens"], context_window=stats["context_window"],
        context_fill_ratio=stats["context_fill_ratio"], active_context_start_id=stats["active_context_start_id"],
        can_summarize=stats["can_summarize"],
    )


def chat_out(chat: Chat, stats: dict, repo=None) -> ChatOut:
    # Непрочитанные и идущий запуск (ТЗ «асинхронные ответы», 2.3/2.6) —
    # только когда передан репозиторий (без него — нули, как в тестах
    # конвертеров).
    activity: dict = {}
    active_run = None
    if repo is not None:
        activity = repo.chat_activity([chat.id]).get(chat.id) or {}
        run = repo._db.active_run_for_chat(chat.id)
        active_run = run_brief_out(run) if run is not None else None
    return ChatOut(
        id=chat.id, agent_id=chat.agent_id, title=chat.title, created_at=chat.created_at,
        updated_at=chat.updated_at,
        settings=settings_out(chat.settings, tools_sources=_tools_sources(repo, chat.settings, chat)),
        stats=chat_stats_out(stats),
        active_profile_id=chat.active_profile_id, invariant_ids=chat.invariant_ids,
        source=chat.source, active_run=active_run, last_read_message_id=chat.last_read_message_id,
        unread_count=activity.get("unread_count") or 0,
        first_unread_message_id=activity.get("first_unread_message_id"),
        last_message_at=activity.get("last_message_at"),
        preview=activity.get("preview"),
    )


def run_brief_out(run: Run) -> RunBriefOut:
    return RunBriefOut(id=run.id, kind=run.kind, status=run.status, current_status=run.current_status)


def run_out(run: Run) -> RunOut:
    return RunOut(
        id=run.id, kind=run.kind, status=run.status, current_status=run.current_status, chat_id=run.chat_id,
        source=run.source, task_id=run.task_id, client_request_id=run.client_request_id,
        user_message_id=run.user_message_id, assistant_message_id=run.assistant_message_id,
        last_seq=run.last_seq, error=run.error, created_at=run.created_at, started_at=run.started_at,
        finished_at=run.finished_at,
    )


#: Ключи событий лент, в которых лежат доменные объекты (`Message`) — при
#: отправке клиенту превращаются в JSON схемы `MessageOut`.
_EVENT_MESSAGE_KEYS = ("message", "assistant_message", "user_message")


def event_out(event: dict) -> dict:
    """Событие ленты запуска/общей ленты → JSON-совместимый dict."""
    data = dict(event)
    for key in _EVENT_MESSAGE_KEYS:
        value = data.get(key)
        if isinstance(value, Message):
            data[key] = message_out(value).model_dump()
    return data


def working_memory_out(entry: WorkingMemoryEntry) -> WorkingMemoryOut:
    return WorkingMemoryOut(**dataclasses.asdict(entry))


def long_term_memory_out(entry: LongTermMemoryEntry) -> LongTermMemoryOut:
    return LongTermMemoryOut(**dataclasses.asdict(entry))


def profile_out(profile: Profile) -> ProfileOut:
    return ProfileOut(**dataclasses.asdict(profile))


def invariant_out(invariant: Invariant) -> InvariantOut:
    return InvariantOut(**dataclasses.asdict(invariant))


def registered_skill_out(skill: dict) -> RegisteredSkillOut:
    """`skill` — полное OpenAI function-tool описание
    ({"type": "function", "function": {"name", "description", "parameters"}}),
    как хранится в реестре (`agents_core.skills.registry`)."""
    fn = skill.get("function", {})
    return RegisteredSkillOut(
        name=fn.get("name", ""),
        description=fn.get("description", ""),
        parameters=fn.get("parameters", {}) or {},
    )


def message_out(message: Message) -> MessageOut:
    return MessageOut(**dataclasses.asdict(message))


def model_info_out(model: ModelInfo) -> ModelInfoOut:
    return ModelInfoOut(**dataclasses.asdict(model))


def branch_out(branch: Branch) -> BranchOut:
    return BranchOut(number=branch.number, name=branch.name, created_at=branch.created_at)


# ---------------------------------------------------------------------------
# "Работу с задачами требуется переделать" — машина состояний задачи
# ---------------------------------------------------------------------------

def task_state_machine_info_out(info: dict) -> TaskStateMachineInfoOut:
    """`info` — результат `Repository.get_task_state_machine_info()`:
    {"states": [...], "invariants": [<Invariant>, ...]}."""
    return TaskStateMachineInfoOut(
        states=[TaskStateInfoOut(**s) for s in info["states"]],
        invariants=[invariant_out(inv) for inv in info["invariants"]],
    )


def task_summary_out(summary: dict) -> TaskSummaryOut:
    """`summary` — результат `Repository._task_summary()` (через
    `list_tasks_for_chat`/`list_tasks_for_agent`): {"task": <Task>, "status",
    "status_display", "state_display_name", "next_state_display_name",
    опционально "chat_title"}."""
    task = summary["task"]
    return TaskSummaryOut(
        id=task.id, chat_id=task.chat_id, title=task.title,
        state=task.state, state_display_name=summary["state_display_name"],
        paused=task.paused, current_step=task.current_step,
        created_at=task.created_at, updated_at=task.updated_at,
        status=summary["status"],
        status_display=summary["status_display"],
        next_state_display_name=summary.get("next_state_display_name"),
        chat_title=summary.get("chat_title"),
    )


def task_detail_out(detail: dict) -> TaskDetailOut:
    """`detail` — результат `Repository.get_task()`."""
    task = detail["task"]
    return TaskDetailOut(
        id=task.id, chat_id=task.chat_id, title=task.title,
        state=task.state, state_display_name=detail["state_display_name"],
        paused=task.paused,
        status=detail["status"], status_display=detail["status_display"],
        next_state_display_name=detail.get("next_state_display_name"),
        plan=list(task.plan), done=list(task.done_steps), current=task.current_step,
        step=detail["step"], total=detail["total"],
        created_at=task.created_at, updated_at=task.updated_at,
        stages=[TaskStageOut(**stage) for stage in detail["stages"]],
        available_actions=[TaskAvailableActionOut(**a) for a in detail["available_actions"]],
        history=[TaskHistoryEntryOut(**h) for h in detail["history"]],
    )

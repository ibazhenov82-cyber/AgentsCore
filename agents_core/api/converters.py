"""Преобразование доменных dataclass'ов (`agents_core.models`) в
Pydantic-модели ответов (`agents_core.schemas`)."""

from __future__ import annotations

import dataclasses

from ..models import Agent, Branch, Chat, Message, ModelInfo
from ..schemas import (
    AgentOut,
    BranchOut,
    ChatOut,
    ChatStatsOut,
    DefaultSettingsOut,
    MessageOut,
    ModelInfoOut,
    SettingsOut,
)


def settings_out(settings) -> SettingsOut:
    return SettingsOut(**dataclasses.asdict(settings))


def default_settings_out(settings) -> DefaultSettingsOut:
    return DefaultSettingsOut(**dataclasses.asdict(settings))


def agent_out(agent: Agent) -> AgentOut:
    return AgentOut(id=agent.id, name=agent.name, created_at=agent.created_at,
                     updated_at=agent.updated_at, settings=settings_out(agent.settings))


def chat_stats_out(stats: dict) -> ChatStatsOut:
    return ChatStatsOut(
        prompt_tokens=stats["prompt_tokens"], completion_tokens=stats["completion_tokens"],
        total_tokens=stats["total_tokens"], current_context_tokens=stats["current_context_tokens"],
        max_input_tokens=stats["max_input_tokens"], context_window=stats["context_window"],
        context_fill_ratio=stats["context_fill_ratio"], active_context_start_id=stats["active_context_start_id"],
        can_summarize=stats["can_summarize"],
    )


def chat_out(chat: Chat, stats: dict) -> ChatOut:
    return ChatOut(
        id=chat.id, agent_id=chat.agent_id, title=chat.title, created_at=chat.created_at,
        updated_at=chat.updated_at, settings=settings_out(chat.settings), stats=chat_stats_out(stats),
    )


def message_out(message: Message) -> MessageOut:
    return MessageOut(**dataclasses.asdict(message))


def model_info_out(model: ModelInfo) -> ModelInfoOut:
    return ModelInfoOut(**dataclasses.asdict(model))


def branch_out(branch: Branch) -> BranchOut:
    return BranchOut(number=branch.number, name=branch.name, created_at=branch.created_at)

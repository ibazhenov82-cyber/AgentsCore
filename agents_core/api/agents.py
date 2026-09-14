from __future__ import annotations

import dataclasses
from typing import List

from fastapi import APIRouter, Depends, Response, status

from ..repository import Repository
from ..schemas import (
    AgentCreate,
    AgentOut,
    AgentRename,
    AgentWithChatsOut,
    ChatCreate,
    ChatOut,
    SettingsOut,
    SettingsPatch,
)
from .converters import agent_out, chat_out
from .deps import get_repository

router = APIRouter(tags=["Agents"])


@router.get(
    "/agents",
    response_model=List[AgentWithChatsOut],
    summary="Список агентов вместе с их чатами",
    description="Используется для главного экрана приложения: каждый агент со всеми своими чатами и агрегатами по токенам.",
)
def list_agents_with_chats(repo: Repository = Depends(get_repository)) -> List[AgentWithChatsOut]:
    return [
        AgentWithChatsOut(agent=agent_out(agent), chats=[chat_out(c, s) for c, s in zip(chats, stats_list)])
        for agent, chats, stats_list in repo.list_agents_with_chats()
    ]


@router.post(
    "/agent",
    response_model=AgentOut,
    status_code=status.HTTP_201_CREATED,
    summary="Создать агента",
    description=(
        "Настройки нового агента копируются из текущих настроек по умолчанию; "
        "поля, отсутствующие в настройках по умолчанию (инструменты, автосуммаризация и т.д.), "
        "берутся из встроенных в код значений."
    ),
)
def create_agent(payload: AgentCreate, repo: Repository = Depends(get_repository)) -> AgentOut:
    return agent_out(repo.create_agent(payload.name, model=payload.model))


@router.get("/agents/{agent_id}", response_model=AgentOut, summary="Детали агента")
def get_agent(agent_id: str, repo: Repository = Depends(get_repository)) -> AgentOut:
    return agent_out(repo.get_agent(agent_id))


@router.patch("/agents/{agent_id}", response_model=AgentOut, summary="Переименовать агента")
def rename_agent(agent_id: str, payload: AgentRename, repo: Repository = Depends(get_repository)) -> AgentOut:
    return agent_out(repo.rename_agent(agent_id, payload.name))


@router.delete(
    "/agents/{agent_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удалить агента",
    description="Каскадно удаляет все чаты этого агента и все их сообщения.",
)
def delete_agent(agent_id: str, repo: Repository = Depends(get_repository)) -> Response:
    repo.delete_agent(agent_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/agents/{agent_id}/settings", response_model=SettingsOut, summary="Настройки агента")
def get_agent_settings(agent_id: str, repo: Repository = Depends(get_repository)) -> SettingsOut:
    return SettingsOut(**dataclasses.asdict(repo.get_agent(agent_id).settings))


@router.put(
    "/agents/{agent_id}/settings",
    response_model=SettingsOut,
    summary="Обновить настройки агента",
    description="Частичное тело допустимо. В отличие от настроек чата, здесь можно менять и модель.",
)
def update_agent_settings(agent_id: str, patch: SettingsPatch, repo: Repository = Depends(get_repository)) -> SettingsOut:
    return SettingsOut(**dataclasses.asdict(repo.update_agent_settings(agent_id, patch.to_payload())))


@router.post(
    "/agents/{agent_id}/chat",
    response_model=ChatOut,
    status_code=status.HTTP_201_CREATED,
    summary="Создать чат внутри агента",
    description="Настройки чата копируются из текущих настроек агента; модель и провайдер после этого фиксированы для чата.",
)
def create_chat(agent_id: str, payload: ChatCreate, repo: Repository = Depends(get_repository)) -> ChatOut:
    chat = repo.create_chat(agent_id, payload.title)
    return chat_out(chat, repo.chat_stats(chat))

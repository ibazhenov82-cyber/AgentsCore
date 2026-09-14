from __future__ import annotations

import dataclasses
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, Response, status

from ..repository import Repository
from ..schemas import BranchCreate, BranchOut, ChatCopyRequest, ChatOut, ChatRename, MessageOut, SettingsOut, SettingsPatch
from .converters import branch_out, chat_out, message_out
from .deps import get_repository

router = APIRouter(tags=["Chats"])


@router.get(
    "/chats",
    response_model=List[ChatOut],
    summary="Список чатов",
    description=(
        "Плоский список чатов, опционально отфильтрованный по агенту через `?agent_id=`. "
        "Статистика по токенам считается для веток 0 и 1 (если ветка 1 существует) — "
        "фиксированная конвенция для компактного отображения в списке."
    ),
)
def list_chats(agent_id: Optional[str] = Query(None), repo: Repository = Depends(get_repository)) -> List[ChatOut]:
    chats = repo.list_chats(agent_id)
    return [chat_out(c, repo.chat_stats(c, branch=1)) for c in chats]


@router.get(
    "/chats/{chat_id}",
    response_model=ChatOut,
    summary="Детали чата",
    description=(
        "Включает настройки чата и агрегаты по токенам/заполнению контекстного окна. "
        "Параметр `?branch=` пересчитывает статистику под конкретную ветку диалога "
        "(0 или не задан — основная ветка; N>0 — основная ветка + ветка N). По умолчанию "
        "используется branch=1 (ветки 0 и 1, если ветка 1 существует)."
    ),
)
def get_chat(chat_id: str, branch: Optional[int] = Query(1), repo: Repository = Depends(get_repository)) -> ChatOut:
    chat = repo.get_chat(chat_id)
    return chat_out(chat, repo.chat_stats(chat, branch=branch))


@router.patch("/chats/{chat_id}", response_model=ChatOut, summary="Переименовать чат")
def rename_chat(chat_id: str, payload: ChatRename, repo: Repository = Depends(get_repository)) -> ChatOut:
    chat = repo.rename_chat(chat_id, payload.title)
    return chat_out(chat, repo.chat_stats(chat))


@router.delete("/chats/{chat_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Удалить чат")
def delete_chat(chat_id: str, repo: Repository = Depends(get_repository)) -> Response:
    repo.delete_chat(chat_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/chats/{chat_id}/copy",
    response_model=ChatOut,
    status_code=status.HTTP_201_CREATED,
    summary="Копировать чат",
    description="Создаёт новый чат в том же агенте с тем же набором настроек и полной копией истории сообщений.",
)
def copy_chat(chat_id: str, payload: ChatCopyRequest, repo: Repository = Depends(get_repository)) -> ChatOut:
    chat = repo.copy_chat(chat_id, payload.title)
    return chat_out(chat, repo.chat_stats(chat))


@router.get("/chats/{chat_id}/settings", response_model=SettingsOut, summary="Настройки чата")
def get_chat_settings(chat_id: str, repo: Repository = Depends(get_repository)) -> SettingsOut:
    return SettingsOut(**dataclasses.asdict(repo.get_chat(chat_id).settings))


@router.put(
    "/chats/{chat_id}/settings",
    response_model=SettingsOut,
    summary="Обновить настройки чата",
    description="Частичное тело допустимо. Поле `model` менять нельзя — оно зафиксировано агентом-владельцем.",
)
def update_chat_settings(chat_id: str, patch: SettingsPatch, repo: Repository = Depends(get_repository)) -> SettingsOut:
    return SettingsOut(**dataclasses.asdict(repo.update_chat_settings(chat_id, patch.to_payload())))


@router.post(
    "/chats/{chat_id}/summarize",
    response_model=MessageOut,
    summary="Подготовить суммарный запрос",
    description=(
        "Выполняет суммаризацию истории чата вручную (по кнопке), независимо от настроек "
        "автосуммаризации. Недоступно, если текущий размер контекста уже превышает лимиты модели/чата."
    ),
)
def summarize_chat(chat_id: str, repo: Repository = Depends(get_repository)) -> MessageOut:
    return message_out(repo.summarize_chat(chat_id))


@router.get(
    "/chats/{chat_id}/branches",
    response_model=List[BranchOut],
    summary="Список веток диалога чата",
    description="Основная ветка (номер 0) в списке не отображается — она подразумевается всегда доступной.",
)
def list_branches(chat_id: str, repo: Repository = Depends(get_repository)) -> List[BranchOut]:
    return [branch_out(b) for b in repo.list_branches(chat_id)]


@router.post(
    "/chats/{chat_id}/branches",
    response_model=BranchOut,
    status_code=status.HTTP_201_CREATED,
    summary="Добавить ветку диалога",
    description="Создаёт новую ветку от текущего состояния чата; номер присваивается автоматически (1, 2, 3, ...).",
)
def create_branch(chat_id: str, payload: BranchCreate, repo: Repository = Depends(get_repository)) -> BranchOut:
    return branch_out(repo.create_branch(chat_id, payload.name))


@router.delete(
    "/chats/{chat_id}/branches/{number}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удалить ветку диалога",
    description="Удаляет ветку и все её сообщения. Основную ветку (0) удалить нельзя.",
)
def delete_branch(chat_id: str, number: int, repo: Repository = Depends(get_repository)) -> Response:
    repo.delete_branch(chat_id, number)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

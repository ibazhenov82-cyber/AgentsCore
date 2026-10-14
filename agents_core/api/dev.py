"""
«День 31» — ассистент разработчика: список команд для подсказок в
приложении, проект чата, подтверждение /commit и /undo.

Сами команды отправляются обычным сообщением (`POST /chats/{chat_id}/runs`)
— AgentsCore разбирает их при включённой настройке «Команды разработчика».
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Body, Depends

from ..repository import Repository
from ..schemas import CommandConfirmRequest, DevCommandOut, DevCommandsOut, MessageOut, ProjectOut
from .converters import message_out, project_out
from .deps import get_repository

router = APIRouter(tags=["Developer assistant"])


@router.get(
    "/chats/{chat_id}/dev/commands",
    response_model=DevCommandsOut,
    summary="Команды разработчика чата",
    description=(
        "Команды для подсказок при вводе «/»: описание, группа, вид (локальная, ответ модели, с "
        "подтверждением) и доступность в этом чате (`available=false` — например, /commit при «Только "
        "чтение»). `enabled=false` — команды в чате выключены, приложение их не подсказывает."
    ),
)
def dev_commands(chat_id: str, repo: Repository = Depends(get_repository)) -> DevCommandsOut:
    data = repo.dev_commands(chat_id)
    return DevCommandsOut(
        enabled=data["enabled"], read_only=data["read_only"], project=project_out(data["project"]),
        commands=[DevCommandOut(**{k: c[k] for k in DevCommandOut.model_fields}) for c in data["commands"]],
    )


@router.get(
    "/chats/{chat_id}/project",
    response_model=Optional[ProjectOut],
    summary="Проект чата",
    description="Проект, открытый ссылкой в сообщении или /project, с актуальным состоянием индексации; null — нет.",
)
def chat_project(chat_id: str, repo: Repository = Depends(get_repository)) -> Optional[ProjectOut]:
    return project_out(repo.refresh_chat_project(chat_id))


@router.post(
    "/chats/{chat_id}/messages/{message_id}/confirm",
    response_model=MessageOut,
    summary="Подтвердить /commit или /undo",
    description=(
        "Выполняет действие карточки подтверждения (`command.confirm.state='pending'`): коммит с сообщением "
        "`message` (можно исправить предложенное) или откат. Результат — в `command.confirm` "
        "(`state='done'` с `result` или `state='failed'` с `error`). `409` — уже выполнено/отменено или "
        "в чате включено «Только чтение»."
    ),
)
def confirm_command(
    chat_id: str, message_id: int, payload: CommandConfirmRequest = Body(default_factory=CommandConfirmRequest),
    repo: Repository = Depends(get_repository),
) -> MessageOut:
    return message_out(repo.confirm_command(chat_id, message_id, payload.model_dump(exclude_none=True)))


@router.post(
    "/chats/{chat_id}/messages/{message_id}/reject",
    response_model=MessageOut,
    summary="Отменить /commit или /undo",
)
def reject_command(chat_id: str, message_id: int, repo: Repository = Depends(get_repository)) -> MessageOut:
    return message_out(repo.reject_command(chat_id, message_id))

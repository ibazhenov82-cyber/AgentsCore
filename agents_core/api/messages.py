from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, Response, status

from ..repository import Repository
from ..schemas import BulkDeleteRequest, MessageOut
from .converters import message_out
from .deps import get_repository

router = APIRouter(tags=["Messages"])


@router.get("/chats/{chat_id}/messages", response_model=List[MessageOut], summary="История сообщений чата")
def list_messages(chat_id: str, repo: Repository = Depends(get_repository)) -> List[MessageOut]:
    return [message_out(m) for m in repo.list_messages(chat_id)]


@router.delete(
    "/chats/{chat_id}/messages",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Очистить историю чата",
    description="Удаляет все сообщения чата целиком; сам чат остаётся.",
)
def clear_messages(chat_id: str, repo: Repository = Depends(get_repository)) -> Response:
    repo.clear_messages(chat_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/chats/{chat_id}/messages/{message_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Удалить одно сообщение")
def delete_message(chat_id: str, message_id: int, repo: Repository = Depends(get_repository)) -> Response:
    repo.delete_message(chat_id, message_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/chats/{chat_id}/messages/bulk-delete",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удалить несколько сообщений",
    description="Удаляет сразу несколько выбранных пользователем сообщений по списку id.",
)
def bulk_delete_messages(chat_id: str, payload: BulkDeleteRequest, repo: Repository = Depends(get_repository)) -> Response:
    repo.bulk_delete_messages(chat_id, payload.ids)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

from __future__ import annotations

import json
from typing import List

from fastapi import APIRouter, Depends, Response, status
from sse_starlette.sse import EventSourceResponse

from ..repository import Repository
from ..schemas import BulkDeleteRequest, MessageOut, SendMessageRequest, SendMessageResponse, StreamSendMessageRequest
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


@router.post(
    "/chats/{chat_id}/messages",
    response_model=SendMessageResponse,
    summary="Отправить сообщение (блокирующий режим)",
    description=(
        "Дожидается полного ответа модели и возвращает и пользовательское сообщение, и ответ ассистента. "
        "Поле `get_facts=true` включает извлечение/обновление фактов по стратегии Sticky Facts "
        "(требует `context_strategy='sticky_facts'` и `context_strategy_limit > 2` у чата) — извлечение "
        "выполняется ПОСЛЕ основного ответа модели, обновлённые факты возвращаются в "
        "`assistant_message.facts`. Поле `sliding_window=true` включает обрезку контекста стратегией "
        "Sliding Window (требует `context_strategy='sliding_window'` и `context_strategy_limit > 2`). "
        "Поле `autosummary` ('messages'/'tokens') включает проверку и, если нужно, выполнение "
        "автоматической суммаризации ПОСЛЕ основного ответа модели (требует совпадающую настройку чата "
        "`autosummary` и корректный предел). Поле `branch` отправляет сообщение в конкретную ветку "
        "диалога (не задано — основная ветка)."
    ),
)
def send_message(chat_id: str, payload: SendMessageRequest, repo: Repository = Depends(get_repository)) -> SendMessageResponse:
    user_msg, assistant_msg = repo.send_message_blocking(
        chat_id,
        payload.text,
        get_facts=payload.get_facts,
        sliding_window=payload.sliding_window,
        autosummary=payload.autosummary,
        branch=payload.branch,
    )
    return SendMessageResponse(user_message=message_out(user_msg), assistant_message=message_out(assistant_msg))


@router.post(
    "/chats/{chat_id}/messages/stream",
    summary="Отправить сообщение (потоковый режим, SSE)",
    description=(
        "Возвращает `text/event-stream`: последовательность событий "
        '`{"type": "status", "status": "..."}` (фаза выполнения — "Выполняется запрос к модели", '
        'при `get_facts=true` затем ещё и "Обновление фактов", при `autosummary` — если суммаризация '
        'действительно потребовалась — "Выполняется суммаризация чата"), '
        '`{"type": "delta", "content": "...", "reasoning_content": "..."}`, '
        'завершается событием `{"type": "done", "message": {...}}` либо `{"type": "error", "message": "..."}`. '
        "Поля `get_facts`/`sliding_window`/`autosummary` работают как и в блокирующем режиме — все "
        "проверки/действия после основного ответа запускаются только после того, как модель полностью "
        "закончила потоковую генерацию ответа."
    ),
)
def stream_message(chat_id: str, payload: StreamSendMessageRequest, repo: Repository = Depends(get_repository)) -> EventSourceResponse:
    def event_source():
        for event in repo.stream_message(
            chat_id,
            payload.text,
            get_facts=payload.get_facts,
            sliding_window=payload.sliding_window,
            autosummary=payload.autosummary,
            branch=payload.branch,
        ):
            data = dict(event)
            if data.get("message") is not None:
                data["message"] = message_out(data["message"]).model_dump()
            yield json.dumps(data, ensure_ascii=False)

    return EventSourceResponse(event_source())

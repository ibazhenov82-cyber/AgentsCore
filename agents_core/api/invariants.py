"""
agents_core.api.invariants
============================

HTTP-эндпоинты "Инварианты и ограничения состояния": общий справочник
инвариантов (по аналогии со справочником профилей, см.
`agents_core.api.memory`) плюс множественный выбор инвариантов для
конкретного агента/чата. Основной механизм соблюдения — инъекция в
системный промпт структурированным блоком `[INVARIANTS]` (см.
`Repository._build_task_and_invariant_context`/`_active_invariants`).
Отдельного тумблера-настройки больше нет: инварианты считаются
включёнными автоматически, как только для агента/чата выбран хотя бы
один инвариант из справочника (`Agent.invariant_ids`/`Chat.invariant_ids`,
множественный выбор ниже) — если не выбрано ни одного, инъекции не будет.
ДОПОЛНИТЕЛЬНО, для категорий
'architecture'/'tech_decision'/'stack_constraint' — код-уровневая проверка
ответа модели с одним автоматическим переспросом при нарушении (см.
`agents_core.invariant_checks`, `Repository._validate_and_maybe_retry`);
для остальных категорий (в т.ч. не указанной) соблюдение полностью
доверяется модели через инъекцию текста.
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, Response, status

from ..repository import Repository
from ..schemas import (
    AgentOut,
    ChatOut,
    InvariantCreate,
    InvariantIdsSetRequest,
    InvariantOut,
    InvariantPatch,
)
from .converters import agent_out, chat_out, invariant_out
from .deps import get_repository

router = APIRouter(tags=["Инварианты"])


# ---- общий справочник инвариантов (для ВСЕХ агентов/чатов) ----------------

@router.get(
    "/invariants",
    response_model=List[InvariantOut],
    summary="Общий справочник инвариантов",
    description=(
        "Один список для всех агентов/чатов — инвариант описывается один раз и подключается "
        "к любому агенту (`PUT /agents/{agent_id}/invariants`) и/или любому чату "
        "(`PUT /chats/{chat_id}/invariants`) множественным выбором, по аналогии с профилями."
    ),
)
def list_invariants(repo: Repository = Depends(get_repository)) -> List[InvariantOut]:
    return [invariant_out(i) for i in repo.list_invariants()]


@router.post(
    "/invariants",
    response_model=InvariantOut,
    status_code=status.HTTP_201_CREATED,
    summary="Создать инвариант в общем справочнике",
    description="Редактируется только вручную — агенту не даётся tool на создание/изменение инвариантов.",
)
def create_invariant(payload: InvariantCreate, repo: Repository = Depends(get_repository)) -> InvariantOut:
    return invariant_out(repo.create_invariant(
        payload.title, payload.rule_text, kind=payload.kind, is_active=payload.is_active,
    ))


@router.get("/invariants/{invariant_id}", response_model=InvariantOut, summary="Детали инварианта")
def get_invariant(invariant_id: str, repo: Repository = Depends(get_repository)) -> InvariantOut:
    return invariant_out(repo.get_invariant(invariant_id))


@router.put("/invariants/{invariant_id}", response_model=InvariantOut, summary="Обновить инвариант")
def update_invariant(invariant_id: str, patch: InvariantPatch, repo: Repository = Depends(get_repository)) -> InvariantOut:
    return invariant_out(repo.update_invariant(invariant_id, patch.to_payload()))


@router.delete(
    "/invariants/{invariant_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удалить инвариант",
    description="У всех агентов/чатов, где этот инвариант был выбран, он автоматически снимается из выбора.",
)
def delete_invariant(invariant_id: str, repo: Repository = Depends(get_repository)) -> Response:
    repo.delete_invariant(invariant_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---- выбор инвариантов для конкретного агента/чата -------------------------

@router.put(
    "/agents/{agent_id}/invariants",
    response_model=AgentOut,
    summary="Выбрать инварианты для агента (множественный выбор)",
    description=(
        "Заменяет весь набор выбранных инвариантов агента целиком. Действуют во ВСЕХ чатах "
        "этого агента (объединяются с инвариантами, выбранными непосредственно у чата, см. "
        "`PUT /chats/{chat_id}/invariants`). Отдельной настройки-тумблера нет: инварианты "
        "считаются включёнными автоматически, как только для агента и/или чата выбран хотя бы "
        "один инвариант — если итоговый набор пуст, инъекции не будет."
    ),
)
def set_agent_invariants(agent_id: str, payload: InvariantIdsSetRequest, repo: Repository = Depends(get_repository)) -> AgentOut:
    return agent_out(repo.set_agent_invariants(agent_id, payload.invariant_ids), repo)


@router.put(
    "/chats/{chat_id}/invariants",
    response_model=ChatOut,
    summary="Выбрать инварианты для чата (множественный выбор)",
    description="Заменяет весь набор выбранных инвариантов ЭТОГО чата целиком (независимо от выбора агента).",
)
def set_chat_invariants(chat_id: str, payload: InvariantIdsSetRequest, repo: Repository = Depends(get_repository)) -> ChatOut:
    chat = repo.set_chat_invariants(chat_id, payload.invariant_ids)
    return chat_out(chat, repo.chat_stats(chat), repo)

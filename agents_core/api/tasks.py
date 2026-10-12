"""
agents_core.api.tasks
========================

HTTP-эндпоинты "Работу с задачами требуется переделать" (Task State
Machine).

Два слоя:
  1. Единственная, заданная в КОДЕ машина состояний (см.
     `agents_core.task_state_machine`) — прежние справочники
     task-states/task-actions/task-state-machines с формой-редактором
     удалены; `GET /task-state-machine` отдаёт read-only описание для
     Android-экрана, `PUT /task-state-machine/invariants` настраивает
     привязанные к ней инварианты категории "Правило стейт-машины".
  2. Сами задачи (`Task`), которые владеются чатом (`chat_id`): список по
     чату, агрегированный список по агенту (для блока "Задачи" на карточке
     агента — с указанием родительского чата у каждой строки), детали одной
     задачи и ручное вмешательство человека (кнопка "Пауза" — единственное
     оставшееся ручное действие, см. `TaskManualActionRequest`).

Все перечисленные эндпоинты видимы клиенту только при
`Settings.task_tracking_enabled == true` у соответствующего чата/агента —
сам API этого не проверяет (задачи могут существовать и при просмотре
истории уже выключенной настройки), решение о показе UI принимает клиент.
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, Query, Response, status

from ..repository import Repository
from ..schemas import (
    TaskDetailOut,
    TaskManualActionRequest,
    TaskMemoryPatchRequest,
    TaskStateSetRequest,
    TaskMachineInvariantIdsSetRequest,
    TaskStateMachineInfoOut,
    TaskSummaryOut,
)
from .converters import (
    task_detail_out,
    task_state_machine_info_out,
    task_summary_out,
)
from .deps import get_repository

router = APIRouter(tags=["Задачи (Task State Machine)"])


# ---- машина состояний (read-only, задана в коде) --------------------------

@router.get(
    "/task-state-machine",
    response_model=TaskStateMachineInfoOut,
    summary="Read-only описание машин состояний задач (рабочая задача и задача поиска)",
    description=(
        "Каталог состояний и переходов задан в коде (см. `task_state_machine.py`), не в БД — "
        "замена формы-редактора справочников состояний/действий/моделей. Возвращает номер, "
        "отображаемое и системное имя каждого этапа, список достижимых из него состояний, а "
        "также текущий список инвариантов категории «Правило стейт-машины», привязанных к машине."
    ),
)
def get_task_state_machine(repo: Repository = Depends(get_repository)) -> TaskStateMachineInfoOut:
    return task_state_machine_info_out(repo.get_task_state_machine_info())


@router.put(
    "/task-state-machine/invariants",
    response_model=TaskStateMachineInfoOut,
    summary="Настроить инварианты машины состояний (полная замена)",
    description=(
        "Заменяет весь список инвариантов, привязанных к машине состояний, целиком — только "
        "категория «Правило стейт-машины» (400, если среди id встретится инвариант другой "
        "категории). Один список на всю систему, не привязан к конкретному агенту/чату."
    ),
)
def set_task_state_machine_invariants(
    payload: TaskMachineInvariantIdsSetRequest, repo: Repository = Depends(get_repository)
) -> TaskStateMachineInfoOut:
    return task_state_machine_info_out(repo.set_task_machine_invariants(payload.invariant_ids))


# ---- задачи: списки по чату/агенту ----------------------------------------

@router.get(
    "/chats/{chat_id}/tasks",
    response_model=List[TaskSummaryOut],
    summary="Список задач чата",
    description=(
        "По умолчанию — только текущие задачи (активные и на паузе); "
        "`?include_completed=true` добавляет и завершённые (по замечанию пользователя: "
        "\"опционально можно показать и выполненные задачи\")."
    ),
)
def list_chat_tasks(
    chat_id: str, include_completed: bool = Query(False), repo: Repository = Depends(get_repository)
) -> List[TaskSummaryOut]:
    return [task_summary_out(s) for s in repo.list_tasks_for_chat(chat_id, include_completed=include_completed)]


@router.get(
    "/agents/{agent_id}/tasks",
    response_model=List[TaskSummaryOut],
    summary="Агрегированный список задач агента (по всем его чатам)",
    description=(
        "Для блока «Задачи» на карточке агента: каждая строка указывает свой родительский чат "
        "(`chat_title`) — по замечанию пользователя, обязательно при нескольких чатах у агента. "
        "По умолчанию — только текущие задачи; `?include_completed=true` — добавить завершённые."
    ),
)
def list_agent_tasks(
    agent_id: str, include_completed: bool = Query(False), repo: Repository = Depends(get_repository)
) -> List[TaskSummaryOut]:
    return [task_summary_out(s) for s in repo.list_tasks_for_agent(agent_id, include_completed=include_completed)]


# ---- задачи: детали и ручное вмешательство --------------------------------

@router.get(
    "/tasks/{task_id}",
    response_model=TaskDetailOut,
    summary="Детали задачи",
    description=(
        "Степпер этапов (`stages`, в фиксированном порядке машины состояний) с иконкой на "
        "состояние: 'check' — этап уже пройден, 'pause' — текущий этап, если задача сейчас на "
        "паузе, 'none' — этап ещё не достигнут. Везде есть отображаемые имена, а не только "
        "системные."
    ),
)
def get_task(task_id: str, repo: Repository = Depends(get_repository)) -> TaskDetailOut:
    return task_detail_out(repo.get_task(task_id))


@router.post(
    "/tasks/{task_id}/actions",
    response_model=TaskDetailOut,
    summary="Применить ручное действие к задаче (кнопка «Пауза»)",
    description=(
        "Ручное вмешательство человека, без обращения к модели — сейчас единственное "
        "поддерживаемое действие: `action='pause'` (кнопка «Продолжить»/«Выполнить», наоборот, "
        "ВСЕГДА обращается к модели — запуск Менеджера задач `POST /chats/{chat_id}/tasks/{task_id}/runs`). 400, если задача уже "
        "завершена или уже на паузе. Записывается в историю с applied_by='manual'."
    ),
)
def apply_task_action_manually(task_id: str, payload: TaskManualActionRequest, repo: Repository = Depends(get_repository)) -> TaskDetailOut:
    return task_detail_out(repo.apply_manual_task_action(task_id, payload.action, note=payload.note))


@router.patch(
    "/tasks/{task_id}/memory",
    response_model=TaskDetailOut,
    summary="Изменить память задачи вручную",
    description=(
        "Цель, уточнения, ограничения и термины задачи (любого типа). Переданные списки заменяют "
        "текущие целиком, непереданные поля не меняются; `clear_sources=true` очищает накопленные "
        "источники задачи поиска. Записывается в историю (kind='memory', applied_by='manual')."
    ),
)
def patch_task_memory(task_id: str, payload: TaskMemoryPatchRequest, repo: Repository = Depends(get_repository)) -> TaskDetailOut:
    return task_detail_out(repo.update_task_memory_manually(
        task_id, title=payload.title, goal=payload.goal, clarifications=payload.clarifications,
        constraints=payload.constraints,
        terms=[t.model_dump() for t in payload.terms] if payload.terms is not None else None,
        clear_sources=payload.clear_sources,
    ))


@router.post(
    "/tasks/{task_id}/state",
    response_model=TaskDetailOut,
    summary="Перевести задачу в другое состояние вручную",
    description=(
        "Переход по графу машины своего типа без обращения к модели — например, кнопка «Цель "
        "достигнута» (`state='achieved'`) у задачи поиска или возврат к поиску ответа. 400, если "
        "переход не допускается графом. Снимает паузу; в историю — applied_by='manual'."
    ),
)
def set_task_state(task_id: str, payload: TaskStateSetRequest, repo: Repository = Depends(get_repository)) -> TaskDetailOut:
    return task_detail_out(repo.set_task_state_manually(task_id, payload.state.strip().lower(), note=payload.note))


@router.delete(
    "/tasks/{task_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удалить задачу",
    description="Каскадно удаляет всю историю переходов этой задачи.",
)
def delete_task(task_id: str, repo: Repository = Depends(get_repository)) -> Response:
    repo.delete_task(task_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

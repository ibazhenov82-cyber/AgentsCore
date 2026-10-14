"""
Асинхронные запуски, непрочитанные и общая лента изменений (ТЗ
«асинхронные ответы», этап 1, разделы 2.3 и 2.6).

- `POST /chats/{chat_id}/runs` — отправить сообщение: ответ `202` сразу, с
  черновиком ответа; сама генерация идёт в фоне.
- `GET /runs/{run_id}/events?after=N` — SSE: события запуска с номера N
  (сначала пропущенные, потом вживую); обрыв подписки на запуск не влияет.
- `GET /events?after=N` — SSE общей ленты: новые/изменённые/удалённые чаты,
  начало/фаза/конец запусков, непрочитанные — одно соединение на приложение.
"""

from __future__ import annotations

import json
from typing import Iterator, List, Optional

from fastapi import APIRouter, Depends, Query, status
from sse_starlette.sse import EventSourceResponse

from ..events import EventsGone
from ..repository import NotFoundError, Repository
from ..runs import RunManager
from ..schemas import (
    ChatActivityOut,
    ChatReadRequest,
    EventsCursorOut,
    RunCreatedOut,
    RunCreateRequest,
    RunOut,
    RunSnapshotOut,
    TaskCreatedOut,
    TaskCreateRequest,
    TaskRunCreateRequest,
)
from .converters import chat_out, event_out, message_out, run_out, task_summary_out
from .deps import get_repository, get_run_manager

router = APIRouter(tags=["Runs"])

#: Сколько секунд SSE-подписка ждёт нового события, прежде чем проверить,
#: жив ли клиент (sse_starlette сам шлёт служебные пинги каждые 15 с).
_POLL_SECONDS = 15.0


def _created_out(run, user_message, assistant_message) -> RunCreatedOut:
    return RunCreatedOut(
        run=run_out(run),
        user_message=message_out(user_message) if user_message is not None else None,
        assistant_message=message_out(assistant_message) if assistant_message is not None else None,
    )


@router.post(
    "/chats/{chat_id}/runs",
    response_model=RunCreatedOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Отправить сообщение (асинхронный запуск)",
    description=(
        "Создаёт сообщение пользователя, черновик ответа (`status='streaming'`) и запуск, и сразу "
        "возвращает их — генерация идёт в фоне независимо от соединения. Ход ответа — "
        "`GET /runs/{run_id}/events`. `409` с `active_run_id`, если в чате уже идёт запуск; "
        "повтор с тем же `client_request_id` возвращает уже созданный запуск. Потоковый или обычный "
        "вызов модели — по настройке чата `stream`."
    ),
)
def create_message_run(
    chat_id: str, payload: RunCreateRequest, manager: RunManager = Depends(get_run_manager),
) -> RunCreatedOut:
    run, user_message, draft = manager.start_message_run(
        chat_id, payload.text, get_facts=payload.get_facts, sliding_window=payload.sliding_window,
        autosummary=payload.autosummary, branch=payload.branch, client_request_id=payload.client_request_id,
        source=payload.source,
    )
    return _created_out(run, user_message, draft)


@router.post(
    "/chats/{chat_id}/tasks",
    response_model=TaskCreatedOut,
    status_code=status.HTTP_201_CREATED,
    summary="Создать задачу (без участия модели)",
    description=(
        "Создаёт задачу с названием и описанием и сообщение-запрос «Задача: …» в чате. `kind`: "
        "'work' — «Рабочая задача» (по умолчанию), 'search' — «Задача поиска» (нужна включённая база "
        "знаний; `goal` — цель, по умолчанию название). С `run` — рабочая задача сразу получает запуск "
        "Менеджера задач (`{\"auto_pause\": false}` — выполнить до конца), задача поиска — ответ на "
        "вопрос «название + описание» (обычный запуск сообщения). "
        "`409`, если в чате выключены задачи (`task_tracking_enabled`)."
    ),
)
def create_task(
    chat_id: str, payload: TaskCreateRequest,
    repo: Repository = Depends(get_repository), manager: RunManager = Depends(get_run_manager),
) -> TaskCreatedOut:
    search_with_run = payload.kind == "search" and payload.run is not None
    summary, message = repo.create_task_direct(
        chat_id, payload.title, payload.description, source=payload.source, kind=payload.kind, goal=payload.goal,
        add_message=not search_with_run,
    )
    run = None
    if search_with_run:
        # Задача поиска продолжается сообщениями пользователя: «запуск» —
        # это ответ на вопрос (название + описание задачи).
        question = payload.title + (f"\n\n{payload.description}" if payload.description.strip() else "")
        run, message, _draft = manager.start_message_run(
            chat_id, question, client_request_id=payload.run.client_request_id, source=payload.source,
        )
    elif payload.run is not None:
        run, _ = manager.start_task_run(
            chat_id, summary["task"].id, auto_pause=payload.run.auto_pause,
            client_request_id=payload.run.client_request_id, source=payload.source,
        )
    return TaskCreatedOut(
        task=task_summary_out(summary), message=message_out(message),
        run=run_out(run) if run is not None else None,
    )


@router.post(
    "/chats/{chat_id}/tasks/{task_id}/runs",
    response_model=RunCreatedOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Запуск Менеджера задач",
    description=(
        "`auto_pause=true` — один шаг (кнопка «Продолжить»), `false` — шаги подряд до завершения "
        "задачи, паузы, вопроса к пользователю или лимита шагов (кнопка «Выполнить»). Работает на "
        "сервере, приложение может закрыть экран чата. Остановить — `POST /runs/{run_id}/cancel` "
        "(для серии шагов — после текущего шага)."
    ),
)
def create_task_run(
    chat_id: str, task_id: str, payload: TaskRunCreateRequest,
    manager: RunManager = Depends(get_run_manager),
) -> RunCreatedOut:
    run, draft = manager.start_task_run(
        chat_id, task_id, auto_pause=payload.auto_pause, client_request_id=payload.client_request_id,
        source=payload.source,
    )
    return _created_out(run, None, draft)


@router.get(
    "/runs",
    response_model=List[RunOut],
    summary="Активные запуски",
    description="Все идущие запуски — для главного экрана и восстановления после перезапуска приложения.",
)
def list_runs(active: bool = Query(True), manager: RunManager = Depends(get_run_manager)) -> List[RunOut]:
    return [run_out(r) for r in manager.list_active()]


@router.get(
    "/runs/{run_id}",
    response_model=RunSnapshotOut,
    summary="Запуск и снимок черновика",
    description="`snapshot_seq` — номер события, по которому актуален черновик: подписывайтесь с него.",
)
def get_run(run_id: str, manager: RunManager = Depends(get_run_manager)) -> RunSnapshotOut:
    run, message, seq = manager.snapshot(run_id)
    return RunSnapshotOut(
        run=run_out(run), assistant_message=message_out(message) if message is not None else None,
        snapshot_seq=seq,
    )


@router.get(
    "/runs/{run_id}/events",
    summary="События запуска (SSE)",
    description=(
        "События с номером больше `after`: сначала уже случившиеся, потом вживую; поток закрывается "
        "после итогового события (`done`/`cancelled`/`error`). `410`, если лента уже удалена из "
        "памяти (запуск давно закончился или сервис перезапускался) — перечитайте сообщения чата."
    ),
)
def run_events(
    run_id: str, after: int = Query(0, ge=0), manager: RunManager = Depends(get_run_manager),
) -> EventSourceResponse:
    first, closed = manager.read_events(run_id, after, 0)  # 404/410 — до начала потока

    def source() -> Iterator[str]:
        cursor = after
        batch, done = first, closed
        while True:
            for event in batch:
                cursor = event["seq"]
                yield json.dumps(event_out(event), ensure_ascii=False)
            if done:
                return
            try:
                batch, done = manager.read_events(run_id, cursor, _POLL_SECONDS)
            except EventsGone:
                yield json.dumps({"type": "gone", "run_id": run_id})
                return

    return EventSourceResponse(source())


@router.post("/runs/{run_id}/cancel", response_model=RunOut, summary="Остановить запуск")
def cancel_run(run_id: str, manager: RunManager = Depends(get_run_manager)) -> RunOut:
    return run_out(manager.cancel(run_id))


@router.post(
    "/chats/{chat_id}/read",
    response_model=ChatActivityOut,
    summary="Отметить прочитанным",
    description="Сдвигает отметку прочтения чата до `message_id` (только вперёд). Ответ — новые счётчики.",
)
def mark_read(chat_id: str, payload: ChatReadRequest, repo: Repository = Depends(get_repository)) -> ChatActivityOut:
    info = repo.mark_chat_read(chat_id, payload.message_id)
    return ChatActivityOut(chat_id=chat_id, **info)


@router.get("/events/cursor", response_model=EventsCursorOut, summary="Текущий номер общей ленты")
def events_cursor(repo: Repository = Depends(get_repository)) -> EventsCursorOut:
    return EventsCursorOut(cursor=repo.events.last_seq)


def _enrich_global(repo: Repository, event: dict) -> Optional[dict]:
    data = event_out(event)
    if data.get("type") in ("chat_created", "chat_updated"):
        try:
            chat = repo.get_chat(data["chat_id"])
        except NotFoundError:
            return None  # чат уже удалён — событие неактуально
        data["chat"] = chat_out(chat, repo.chat_stats(chat, branch=1), repo).model_dump()
    return data


@router.get(
    "/events",
    summary="Общая лента изменений (SSE)",
    description=(
        "Одно соединение на приложение: `chat_created`/`chat_updated` (с полным `chat`), `chat_deleted`, "
        "`run_started`/`run_status`/`run_finished`, `unread_changed` (`unread_count`, `preview`...). "
        "`after` не задан — только новые события. `410` — номер устарел (сервис перезапускался или "
        "события вытеснены): перечитайте список агентов и чатов и подпишитесь с `GET /events/cursor`."
    ),
)
def global_events(
    after: Optional[int] = Query(None, ge=0), repo: Repository = Depends(get_repository),
) -> EventSourceResponse:
    start = repo.events.last_seq if after is None else after
    first, _ = repo.events.read(start, 0)  # 410 — до начала потока

    def source() -> Iterator[str]:
        cursor = start
        batch = first
        while True:
            for event in batch:
                cursor = event["seq"]
                data = _enrich_global(repo, event)
                if data is not None:
                    yield json.dumps(data, ensure_ascii=False)
            try:
                batch, _ = repo.events.read(cursor, _POLL_SECONDS)
            except EventsGone:
                yield json.dumps({"type": "gone"})
                return

    return EventSourceResponse(source())

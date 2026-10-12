"""
agents_core.api.test_dialogs
==============================

Тестовые диалоги — общий справочник наборов вопросов (как инварианты и
профили) и их запуск в чате.

- `GET/POST /test-dialogs`, `GET/PUT/DELETE /test-dialogs/{id}` — справочник;
- `POST /test-dialogs/import` — разбор текстового файла с вопросами
  (строка = вопрос) с проверкой дубликатов; ничего не сохраняет — клиент
  добавляет `added` в редактор и сохраняет диалог обычным `POST`/`PUT`;
- `POST /chats/{chat_id}/test-dialogs/{id}/runs` — запуск: асинхронно
  отправляет вопросы по очереди (вопрос 1 → ответ 1 → … → ответ N).
  Пока запуск идёт, в чат нельзя отправить сообщение (`409`), остановить —
  `POST /runs/{run_id}/cancel`.
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, Response, status

from ..repository import Repository
from ..runs import RunManager
from ..schemas import (
    RunCreatedOut,
    TestDialogCreate,
    TestDialogImportOut,
    TestDialogImportRequest,
    TestDialogOut,
    TestDialogPatch,
    TestDialogRunRequest,
)
from .converters import message_out, run_out, test_dialog_out
from .deps import get_repository, get_run_manager

router = APIRouter(tags=["Тестовые диалоги"])


@router.get("/test-dialogs", response_model=List[TestDialogOut], summary="Справочник тестовых диалогов")
def list_test_dialogs(repo: Repository = Depends(get_repository)) -> List[TestDialogOut]:
    return [test_dialog_out(d) for d in repo.list_test_dialogs()]


@router.post(
    "/test-dialogs", response_model=TestDialogOut, status_code=status.HTTP_201_CREATED,
    summary="Создать тестовый диалог",
    description="Наименование уникально (без учёта регистра); вопросы — непустые и без повторов.",
)
def create_test_dialog(payload: TestDialogCreate, repo: Repository = Depends(get_repository)) -> TestDialogOut:
    return test_dialog_out(repo.create_test_dialog(payload.name, payload.questions))


@router.post(
    "/test-dialogs/import", response_model=TestDialogImportOut,
    summary="Разобрать файл с вопросами",
    description=(
        "Каждая непустая строка — вопрос; нумерация («1.», «1)») и маркеры списка («-», «*», «•») убираются. "
        "Дубликаты (без учёта регистра, «ё/е», пробелов и знаков препинания в конце) внутри файла и среди "
        "`existing` пропускаются и перечисляются в `duplicates`. Ничего не сохраняет."
    ),
)
def import_questions(payload: TestDialogImportRequest, repo: Repository = Depends(get_repository)) -> TestDialogImportOut:
    return TestDialogImportOut(**repo.parse_test_dialog_questions(payload.text, payload.existing))


@router.get("/test-dialogs/{dialog_id}", response_model=TestDialogOut, summary="Тестовый диалог")
def get_test_dialog(dialog_id: str, repo: Repository = Depends(get_repository)) -> TestDialogOut:
    return test_dialog_out(repo.get_test_dialog(dialog_id))


@router.put("/test-dialogs/{dialog_id}", response_model=TestDialogOut, summary="Изменить тестовый диалог")
def update_test_dialog(dialog_id: str, patch: TestDialogPatch, repo: Repository = Depends(get_repository)) -> TestDialogOut:
    return test_dialog_out(repo.update_test_dialog(dialog_id, patch.to_payload()))


@router.delete("/test-dialogs/{dialog_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Удалить тестовый диалог")
def delete_test_dialog(dialog_id: str, repo: Repository = Depends(get_repository)) -> Response:
    repo.delete_test_dialog(dialog_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/chats/{chat_id}/test-dialogs/{dialog_id}/runs",
    response_model=RunCreatedOut, status_code=status.HTTP_202_ACCEPTED,
    summary="Запустить тестовый диалог в чате",
    description=(
        "Асинхронный запуск `kind='test_dialog'`: вопросы отправляются по очереди от имени пользователя, "
        "каждый — после ответа модели на предыдущий. В ленте запуска на каждый следующий вопрос — "
        "`message_started` (с `user_message` и `assistant_message`), после каждого ответа — "
        "`test_dialog_step_done`; прогресс — поле `test_dialog` у запуска. Ошибка ответа останавливает "
        "цепочку. Пока запуск идёт, отправка сообщений в чат возвращает `409`."
    ),
)
def run_test_dialog(
    chat_id: str, dialog_id: str, payload: TestDialogRunRequest = TestDialogRunRequest(),
    manager: RunManager = Depends(get_run_manager),
) -> RunCreatedOut:
    run, user_message, draft = manager.start_test_dialog_run(
        chat_id, dialog_id, get_facts=payload.get_facts, sliding_window=payload.sliding_window,
        autosummary=payload.autosummary, branch=payload.branch, client_request_id=payload.client_request_id,
        source=payload.source,
    )
    return RunCreatedOut(
        run=run_out(run),
        user_message=message_out(user_message) if user_message is not None else None,
        assistant_message=message_out(draft) if draft is not None else None,
    )

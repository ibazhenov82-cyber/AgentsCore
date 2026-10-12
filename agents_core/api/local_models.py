"""Локальные модели Ollama: список, запуск, остановка, скачивание (в т.ч. с
Hugging Face), удаление, рекомендации, пробный запрос («День 29»)."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from ..hf_hub import HubError, HuggingFaceHub
from ..local_models import LocalModelNotFound, LocalModelsUnavailable
from ..providers import ProviderError
from ..repository import Repository
from ..schemas import (
    HfModelOut,
    HfRepoOut,
    LocalModelActionIn,
    LocalModelOut,
    LocalModelSettingsIn,
    LocalModelsOut,
    LocalPullOut,
    ModelProbeIn,
    ModelProbeOut,
    RecommendedModelOut,
)
from .deps import get_repository

router = APIRouter(tags=["Local models"])


def _manager(repo: Repository):
    manager = repo.local_models
    if manager is None or not manager.enabled:
        raise LocalModelsUnavailable("провайдер Ollama выключен (PROVIDERS в .env AgentsCore)")
    return manager


def _hub(request: Request) -> HuggingFaceHub:
    return request.app.state.hf_hub


def _error(exc: Exception) -> JSONResponse:
    if isinstance(exc, LocalModelNotFound):
        return JSONResponse(status_code=404, content={"error": str(exc)})
    if isinstance(exc, LocalModelsUnavailable):
        return JSONResponse(status_code=503, content={"error": str(exc)})
    return JSONResponse(status_code=502, content={"error": str(exc)})


_ERRORS = (LocalModelNotFound, LocalModelsUnavailable, ProviderError)


@router.get(
    "/local-models",
    response_model=LocalModelsOut,
    summary="Скачанные локальные модели",
    description=(
        "Модели, скачанные в Ollama (`ollama list`): чат-модели и embedding-модели (kind), состояние "
        "(запущена/остановлена, память, доля GPU — по `ollama ps`; у незапущенной — оценка памяти need_mb), "
        "настройка «Конкурентный запуск», идущие скачивания (pulls) и свободная память компьютера (memory). "
        "Заодно обновляет каталог моделей: новая чат-модель становится доступна агентам. "
        "Ollama недоступна — 200 с available=false и текстом ошибки."
    ),
)
def list_local_models(repo: Repository = Depends(get_repository)) -> LocalModelsOut:
    try:
        manager = _manager(repo)
        models = manager.list_models()
    except LocalModelsUnavailable as exc:
        return LocalModelsOut(available=False, error=str(exc), models=[])
    return LocalModelsOut(
        available=True,
        models=[LocalModelOut(**m) for m in models],
        pulls=[LocalPullOut(**p) for p in manager.pulls()],
        memory=manager.memory(),
    )


@router.post(
    "/local-models/start",
    response_model=LocalModelOut,
    status_code=202,
    summary="Запустить локальную модель",
    description=(
        "Загружает модель в память (в фоне — ответ сразу со status='starting'; дальше опрашивайте "
        "GET /local-models). При «Конкурентном запуске» сначала останавливаются другие запущенные "
        "локальные чат-модели. Модель держится в памяти до остановки."
    ),
)
def start_local_model(payload: LocalModelActionIn, repo: Repository = Depends(get_repository)):
    try:
        return LocalModelOut(**_manager(repo).start(payload.model))
    except _ERRORS as exc:
        return _error(exc)


@router.post("/local-models/stop", response_model=LocalModelOut, summary="Остановить локальную модель",
             description="Выгружает модель из памяти (как `ollama stop`).")
def stop_local_model(payload: LocalModelActionIn, repo: Repository = Depends(get_repository)):
    try:
        return LocalModelOut(**_manager(repo).stop(payload.model))
    except _ERRORS as exc:
        return _error(exc)


@router.put(
    "/local-models/settings",
    response_model=LocalModelOut,
    summary="Настройки локальной модели",
    description="«Конкурентный запуск»: при запуске модели другие запущенные локальные чат-модели останавливаются.",
)
def set_local_model_settings(payload: LocalModelSettingsIn, repo: Repository = Depends(get_repository)):
    try:
        return LocalModelOut(**_manager(repo).set_competitive(payload.model, payload.competitive_start))
    except _ERRORS as exc:
        return _error(exc)


@router.post(
    "/local-models/pull",
    response_model=LocalPullOut,
    status_code=202,
    summary="Скачать модель",
    description=(
        "Скачивание в фоне (как `ollama pull`): тег библиотеки Ollama ('qwen3:8b') или GGUF с Hugging Face "
        "('hf.co/<репозиторий>:<квант>', см. GET /hf/model). Прогресс — в pulls у GET /local-models."
    ),
)
def pull_local_model(payload: LocalModelActionIn, repo: Repository = Depends(get_repository)):
    try:
        return LocalPullOut(**_manager(repo).start_pull(payload.model))
    except _ERRORS as exc:
        return _error(exc)


@router.post("/local-models/pull/cancel", status_code=204, summary="Отменить скачивание",
             description="Прерывает скачивание или убирает из списка неудавшееся.")
def cancel_pull(payload: LocalModelActionIn, repo: Repository = Depends(get_repository)):
    try:
        _manager(repo).cancel_pull(payload.model)
    except _ERRORS as exc:
        return _error(exc)
    return None


@router.post("/local-models/delete", status_code=204, summary="Удалить модель",
             description="Удаляет модель с диска (как `ollama rm`); запущенную сначала останавливает.")
def delete_local_model(payload: LocalModelActionIn, repo: Repository = Depends(get_repository)):
    try:
        _manager(repo).delete(payload.model)
    except _ERRORS as exc:
        return _error(exc)
    return None


@router.get("/local-models/recommended", response_model=List[RecommendedModelOut],
            summary="Рекомендуемые модели", description="Готовые варианты, которых ещё нет среди скачанных.")
def recommended(repo: Repository = Depends(get_repository)):
    manager = repo.local_models
    if manager is None or not manager.enabled:
        return []
    return [RecommendedModelOut(**r) for r in manager.recommended()]


@router.post(
    "/models/probe",
    response_model=ModelProbeOut,
    summary="Пробный запрос к модели",
    description=(
        "Короткий вопрос модели без рассуждений: ответ, токены, у локальных — скорость чтения prompt'а "
        "и генерации. Незапущенная локальная модель запускается."
    ),
)
def probe_model(payload: ModelProbeIn, repo: Repository = Depends(get_repository)) -> ModelProbeOut:
    return ModelProbeOut(**repo.probe_model(payload.model))


@router.get(
    "/hf/models",
    response_model=List[HfModelOut],
    summary="Поиск GGUF-моделей на Hugging Face",
    description="kind: all | chat | embedding; sort: downloads | likes | updated | trending.",
)
def hf_search(
    request: Request,
    q: str = Query("", description="Название модели"),
    kind: str = Query("all"),
    sort: str = Query("downloads"),
    limit: int = Query(30, ge=1, le=100),
):
    try:
        return [HfModelOut(**m) for m in _hub(request).search(q, kind, sort, limit)]
    except HubError as exc:
        return JSONResponse(status_code=502, content={"error": str(exc)})


@router.get(
    "/hf/model",
    response_model=HfRepoOut,
    summary="GGUF-файлы репозитория Hugging Face",
    description=(
        "Сведения о репозитории и его GGUF-файлы: размер, квантование, можно ли скачать через Ollama "
        "(ollama_model — имя для POST /local-models/pull), уже скачан ли, оценка памяти при окне ctx и "
        "хватит ли свободной памяти (fits; null — Ollama не на этом компьютере)."
    ),
)
def hf_repo(
    request: Request,
    repo_id: str = Query(..., alias="repo", description="Например 'unsloth/Qwen3-4B-GGUF'"),
    ctx: int = Query(8192, ge=256, le=262144),
    repo: Repository = Depends(get_repository),
):
    manager = repo.local_models
    memory = manager.memory() if manager is not None and manager.enabled else None
    downloaded = manager.downloaded_names() if manager is not None and manager.enabled else set()
    try:
        return HfRepoOut(**_hub(request).repo(repo_id, ctx=ctx, memory=memory, downloaded=downloaded))
    except HubError as exc:
        return JSONResponse(status_code=502, content={"error": str(exc)})

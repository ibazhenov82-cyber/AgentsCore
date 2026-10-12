from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends

from ..repository import Repository
from ..schemas import ModelHealthOut, ModelInfoOut
from .converters import model_info_out
from .deps import get_repository

router = APIRouter(tags=["Models"])


@router.get(
    "/models",
    response_model=List[ModelInfoOut],
    summary="Каталог моделей",
    description=(
        "Облачные модели из .env и все скачанные в Ollama чат-модели. Список локальных "
        "сверяется с `ollama list` при каждом запросе (Ollama недоступна — последний известный)."
    ),
)
def list_models(repo: Repository = Depends(get_repository)) -> List[ModelInfoOut]:
    manager = repo.local_models
    if manager is not None and manager.enabled:
        manager.sync_catalog()
    return [model_info_out(m) for m in repo.list_models()]


@router.get(
    "/model/{model_id:path}/health",
    response_model=ModelHealthOut,
    summary="Проверка связи с провайдером модели",
    description=(
        "Проверяет доступность API, через который работает указанная модель "
        "(DeepSeek API для облачных моделей, Ollama API для локальных)."
    ),
)
def model_health(model_id: str, repo: Repository = Depends(get_repository)) -> ModelHealthOut:
    return ModelHealthOut(model=model_id, healthy=repo.model_health(model_id))

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
        "Список моделей, зарегистрированных при старте сервиса на основе .env "
        "(отдаётся из памяти сервиса, без обращения к провайдерам)."
    ),
)
def list_models(repo: Repository = Depends(get_repository)) -> List[ModelInfoOut]:
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

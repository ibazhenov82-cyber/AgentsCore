from __future__ import annotations

from fastapi import APIRouter

from ..schemas import HealthOut

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthOut,
    summary="Проверка состояния сервиса",
    description="Лёгкий ping — подтверждает, что AgentsCore запущен и отвечает на запросы.",
)
def health() -> HealthOut:
    return HealthOut(status="ok")

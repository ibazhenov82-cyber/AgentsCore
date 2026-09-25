from __future__ import annotations

from fastapi import APIRouter, Depends

from ..repository import Repository
from ..schemas import DefaultSettingsOut, DefaultSettingsPatch, SettingsOut
from .converters import default_settings_out, settings_out
from .deps import get_repository

router = APIRouter(prefix="/settings", tags=["Settings"])


@router.get(
    "/default",
    response_model=DefaultSettingsOut,
    summary="Настройки по умолчанию",
    description="Применяются при создании нового агента (не влияют на уже существующих агентов и чаты).",
)
def get_default_settings(repo: Repository = Depends(get_repository)) -> DefaultSettingsOut:
    return default_settings_out(repo.get_default_settings())


@router.get(
    "/default/agent",
    response_model=SettingsOut,
    summary="Настройки нового агента",
    description=(
        "Полные настройки, которые получит новый агент: настройки по умолчанию плюс встроенные "
        "значения остальных полей. Приложение сравнивает с ними настройки агентов и показывает "
        "в списке бейджи только для отличающихся."
    ),
)
def get_new_agent_settings(repo: Repository = Depends(get_repository)) -> SettingsOut:
    return settings_out(repo.new_agent_settings())


@router.put(
    "/default",
    response_model=DefaultSettingsOut,
    summary="Изменить настройки по умолчанию",
    description="Частичное тело допустимо — не указанные поля сохраняют текущее значение.",
)
def update_default_settings(patch: DefaultSettingsPatch, repo: Repository = Depends(get_repository)) -> DefaultSettingsOut:
    return default_settings_out(repo.update_default_settings(patch.to_payload()))


@router.post(
    "/default/reset",
    response_model=DefaultSettingsOut,
    summary="Сбросить настройки по умолчанию",
    description="Возвращает значения, встроенные в код (`DefaultSettings()`).",
)
def reset_default_settings(repo: Repository = Depends(get_repository)) -> DefaultSettingsOut:
    return default_settings_out(repo.reset_default_settings())

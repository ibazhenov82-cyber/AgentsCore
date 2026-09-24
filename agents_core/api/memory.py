"""
agents_core.api.memory
========================

HTTP-эндпоинты для рабочей/долговременной памяти и профилей-пайплайнов
персонализации (см. итоговый документ разделы 3, 4, 6). Ручное сохранение
через эти эндпоинты доступно ВСЕГДА (независимо от тумблера
`memory_tools_enabled` чата, который ограничивает только автоматическое
сохранение самим агентом через tool-calling) — именно так демонстрируется
требование ТЗ "явно выбирали, что и куда сохраняется".
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, Query, Response, status

from ..repository import Repository
from ..schemas import (
    ActiveProfileSetRequest,
    ChatOut,
    LongTermMemoryOut,
    LongTermMemorySaveRequest,
    MemorySnapshotOut,
    MemorySnapshotShortTermOut,
    ProfileCreate,
    ProfileOut,
    ProfilePatch,
    RegisteredSkillOut,
    WorkingMemoryOut,
    WorkingMemorySaveRequest,
)
from .converters import chat_out, long_term_memory_out, profile_out, registered_skill_out, working_memory_out
from .deps import get_repository

router = APIRouter(tags=["Память и персонализация"])


# ---- рабочая память (чат) ------------------------------------------------

@router.get(
    "/chats/{chat_id}/working-memory",
    response_model=List[WorkingMemoryOut],
    summary="Рабочая память чата",
    description="Данные текущей задачи этого чата — не переносятся в другие чаты агента.",
)
def list_working_memory(chat_id: str, repo: Repository = Depends(get_repository)) -> List[WorkingMemoryOut]:
    return [working_memory_out(e) for e in repo.list_working_memory(chat_id)]


@router.post(
    "/chats/{chat_id}/working-memory",
    response_model=WorkingMemoryOut,
    status_code=status.HTTP_201_CREATED,
    summary="Сохранить в рабочую память (вручную)",
    description="Повторное сохранение того же `key` обновляет значение (upsert), а не создаёт дубликат.",
)
def save_working_memory(chat_id: str, payload: WorkingMemorySaveRequest, repo: Repository = Depends(get_repository)) -> WorkingMemoryOut:
    return working_memory_out(repo.save_working_memory(chat_id, payload.key, payload.value, source="manual"))


@router.delete(
    "/chats/{chat_id}/working-memory/{key}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удалить запись рабочей памяти",
)
def delete_working_memory(chat_id: str, key: str, repo: Repository = Depends(get_repository)) -> Response:
    repo.delete_working_memory(chat_id, key)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---- долговременная память (агент) ---------------------------------------

@router.get(
    "/agents/{agent_id}/long-term-memory",
    response_model=List[LongTermMemoryOut],
    summary="Долговременная память агента",
    description="Опционально фильтруется по `?category=profile|decision|knowledge`. Видна во всех чатах этого агента.",
)
def list_long_term_memory(
    agent_id: str, category: Optional[str] = Query(None), repo: Repository = Depends(get_repository)
) -> List[LongTermMemoryOut]:
    return [long_term_memory_out(e) for e in repo.list_long_term_memory(agent_id, category)]


@router.post(
    "/agents/{agent_id}/long-term-memory",
    response_model=LongTermMemoryOut,
    status_code=status.HTTP_201_CREATED,
    summary="Сохранить в долговременную память (вручную)",
    description="Повторное сохранение того же (`category`, `key`) обновляет значение (upsert).",
)
def save_long_term_memory(agent_id: str, payload: LongTermMemorySaveRequest, repo: Repository = Depends(get_repository)) -> LongTermMemoryOut:
    return long_term_memory_out(
        repo.save_long_term_memory(agent_id, payload.category, payload.key, payload.value, source="manual")
    )


@router.delete(
    "/agents/{agent_id}/long-term-memory/{category}/{key}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удалить запись долговременной памяти",
)
def delete_long_term_memory(agent_id: str, category: str, key: str, repo: Repository = Depends(get_repository)) -> Response:
    repo.delete_long_term_memory(agent_id, category, key)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---- реестр зарегистрированных скиллов -----------------------------------

@router.get(
    "/skills",
    response_model=List[RegisteredSkillOut],
    summary="Зарегистрированные скиллы сервиса",
    description=(
        "Скиллы, которые можно привязать к профилю по имени (`skill_names` в "
        "ProfileCreate/ProfilePatch) без ручного написания JSON-схемы функции. "
        "Ведутся в переменной окружения AGENT_REGISTERED_SKILLS (см. .env.example) — "
        "добавление нового скилла в реестр всё равно требует реализации обработчика в коде "
        "(`Repository._TOOL_HANDLERS`), поэтому список отражает то, что сервис РЕАЛЬНО умеет выполнять."
    ),
)
def list_registered_skills(repo: Repository = Depends(get_repository)) -> List[RegisteredSkillOut]:
    return [registered_skill_out(s) for s in repo.list_registered_skills()]


# ---- профили-пайплайны (общий справочник для ВСЕХ агентов) --------------

@router.get(
    "/profiles",
    response_model=List[ProfileOut],
    summary="Общий справочник профилей-пайплайнов",
    description=(
        "Один список для всех агентов — профиль описывается один раз (стиль/формат/"
        "ограничения + набор скиллов) и подключается к любому чату любого агента через "
        "`PUT /chats/{chat_id}/active-profile`."
    ),
)
def list_profiles(repo: Repository = Depends(get_repository)) -> List[ProfileOut]:
    return [profile_out(p) for p in repo.list_profiles()]


@router.post(
    "/profiles",
    response_model=ProfileOut,
    status_code=status.HTTP_201_CREATED,
    summary="Создать профиль-пайплайн в общем справочнике",
    description=(
        "Профиль = не просто пресет стиля, а связка (опционального) стиля/формата/ограничений "
        "с набором доменных скиллов (`skills_json` или `skill_names` — см. `GET /skills`) и "
        "инструкцией по их оркестровке (`orchestration_prompt`). Редактируется только вручную. "
        "Профиль не привязан к конкретному агенту — подключить его к чату можно у любого агента."
    ),
)
def create_profile(payload: ProfileCreate, repo: Repository = Depends(get_repository)) -> ProfileOut:
    return profile_out(repo.create_profile(
        payload.name, style=payload.style, format=payload.format, constraints=payload.constraints,
        skills_json=payload.skills_json, orchestration_prompt=payload.orchestration_prompt,
        skill_names=payload.skill_names,
    ))


@router.get("/profiles/{profile_id}", response_model=ProfileOut, summary="Детали профиля")
def get_profile(profile_id: str, repo: Repository = Depends(get_repository)) -> ProfileOut:
    return profile_out(repo.get_profile(profile_id))


@router.put("/profiles/{profile_id}", response_model=ProfileOut, summary="Обновить профиль")
def update_profile(profile_id: str, patch: ProfilePatch, repo: Repository = Depends(get_repository)) -> ProfileOut:
    return profile_out(repo.update_profile(profile_id, patch.to_payload()))


@router.delete(
    "/profiles/{profile_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удалить профиль",
    description="У чатов, где этот профиль был активен, `active_profile_id` автоматически сбрасывается в null.",
)
def delete_profile(profile_id: str, repo: Repository = Depends(get_repository)) -> Response:
    repo.delete_profile(profile_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.put(
    "/chats/{chat_id}/active-profile",
    response_model=ChatOut,
    summary="Подключить/отключить профиль к чату",
    description="`profile_id: null` отключает профиль. Профиль общий для всех агентов — подходит любому чату.",
)
def set_chat_active_profile(chat_id: str, payload: ActiveProfileSetRequest, repo: Repository = Depends(get_repository)) -> ChatOut:
    chat = repo.set_chat_active_profile(chat_id, payload.profile_id)
    return chat_out(chat, repo.chat_stats(chat), repo)


# ---- снимок памяти ---------------------------------------------------------

@router.get(
    "/chats/{chat_id}/memory-snapshot",
    response_model=MemorySnapshotOut,
    summary="Снимок памяти чата",
    description=(
        "То, что реально будет подмешано в СЛЕДУЮЩИЙ запрос модели, разбито по слоям — "
        "основной инструмент проверки юзкейсов ТЗ ('проверьте, что попадает в каждый слой и "
        "как это влияет на ответы')."
    ),
)
def get_memory_snapshot(chat_id: str, repo: Repository = Depends(get_repository)) -> MemorySnapshotOut:
    snapshot = repo.get_memory_snapshot(chat_id)
    return MemorySnapshotOut(
        short_term=MemorySnapshotShortTermOut(**snapshot["short_term"]),
        working_memory=[working_memory_out(e) for e in snapshot["working_memory"]],
        long_term_memory=[long_term_memory_out(e) for e in snapshot["long_term_memory"]],
        active_profile=profile_out(snapshot["active_profile"]) if snapshot["active_profile"] else None,
        available_tools=snapshot["available_tools"],
        memory_tools_enabled=snapshot["memory_tools_enabled"],
        enabled_memory_types=snapshot["enabled_memory_types"],
    )

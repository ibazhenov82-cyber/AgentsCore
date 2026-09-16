"""
agents_core.skills.registry
==============================

Реестр "зарегистрированных" скиллов сервиса — плоский список описаний
функций в формате OpenAI function-tool, из которого пользователь выбирает
подмножество ПО ИМЕНИ при создании/редактировании профиля через API/
приложение (`Repository.create_profile`/`update_profile`, параметр
`skill_names` — см. `GET /skills`), вместо того чтобы вручную писать
JSON-схему функции в `skills_json`.

Сознательно ведётся через переменную окружения (`AGENT_REGISTERED_SKILLS`
в `.env.example`), а не в БД: регистрация скилла в этом списке делает его
только ВИДИМЫМ и ВЫБИРАЕМЫМ для профиля — она не создаёт обработчик сама
по себе, реализация того, что функция реально делает, по-прежнему
фиксируется в коде (`Repository._TOOL_HANDLERS`, см. docstring рядом).
Значит, список в env всегда должен быть подмножеством того, что сервис
реально умеет выполнять — плоский текстовый файл конфигурации для этого
достаточен и не требует ни миграций, ни отдельного UI администрирования.

Пример в `.env.example` — три демо-скилла "Покупки"
(`agents_core.skills.shopping_demo`): search_products/add_to_cart/view_cart.
"""

from __future__ import annotations

import json
import sys
from typing import List, Optional

from ..config import AgentConfig


def _is_valid_tool_def(item: object) -> bool:
    return (
        isinstance(item, dict)
        and item.get("type") == "function"
        and isinstance(item.get("function"), dict)
        and bool(item["function"].get("name"))
    )


def _parse_registered_skills(raw: str) -> List[dict]:
    if not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError) as exc:
        print(f"[agents_core] AGENT_REGISTERED_SKILLS: невалидный JSON, реестр скиллов пуст ({exc})", file=sys.stderr)
        return []
    if not isinstance(parsed, list):
        print("[agents_core] AGENT_REGISTERED_SKILLS: ожидался JSON-массив, реестр скиллов пуст", file=sys.stderr)
        return []
    valid: List[dict] = []
    for item in parsed:
        if _is_valid_tool_def(item):
            valid.append(item)
        else:
            print(f"[agents_core] AGENT_REGISTERED_SKILLS: пропущена некорректная запись: {item!r}", file=sys.stderr)
    return valid


#: Разбирается один раз при импорте модуля — как и остальная конфигурация
#: сервиса (`AgentConfig`), не "горячая" настройка, перезапуск подхватывает
#: изменения `.env`.
_REGISTERED_SKILLS: List[dict] = _parse_registered_skills(AgentConfig.REGISTERED_SKILLS_JSON)


def list_registered_skills() -> List[dict]:
    """Все зарегистрированные скиллы — полные OpenAI function-tool описания."""
    return list(_REGISTERED_SKILLS)


def get_registered_skill(name: str) -> Optional[dict]:
    """Найти зарегистрированный скилл по имени функции (`function.name`)."""
    for skill in _REGISTERED_SKILLS:
        if skill["function"]["name"] == name:
            return skill
    return None

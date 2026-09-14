"""
agents_core.format_detect
===========================

Определение формата содержимого сообщения (`text` | `markdown` | `json`) в
момент сохранения — используется клиентом, чтобы решить, каким компонентом
отрисовать сообщение (обычный текст, markdown-рендерер или JSON-viewer).

Эвристика, а не точный парсер: сообщение считается `json`, только если оно
целиком (после обрезки пробелов) представляет собой валидный JSON-объект
или JSON-массив — одиночные числа/строки/`true`/`null` в JSON-режим не
переводим, это почти наверняка обычный текстовый ответ. Иначе ищем типичные
маркеры markdown (код-блоки, заголовки, списки, полужирный текст, ссылки).
"""

from __future__ import annotations

import json
import re
from typing import Optional

_MARKDOWN_PATTERN = re.compile(
    r"(```|^#{1,6}\s|\*\*[^*\n]+\*\*|^[-*+]\s|^\d+\.\s|\[[^\]\n]+\]\([^)\n]+\)|^>\s|^\|.+\|$)",
    re.MULTILINE,
)


def detect_message_format(content: Optional[str] = None) -> str:
    if not content:
        return "text"
    stripped = content.strip()
    if not stripped:
        return "text"
    if stripped[0] in "{[":
        try:
            parsed = json.loads(stripped)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, (dict, list)):
            return "json"
    if _MARKDOWN_PATTERN.search(stripped):
        return "markdown"
    return "text"

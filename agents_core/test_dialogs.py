"""
agents_core.test_dialogs
==========================

Тестовые диалоги — общий справочник наборов вопросов (по аналогии с
инвариантами и профилями). В чате выбранный тестовый диалог запускается
одним асинхронным запуском `kind="test_dialog"`: вопрос 1 → ответ 1 → … →
ответ N (см. `runs.RunManager.start_test_dialog_run`). Пока он идёт, в чате
нельзя отправить другое сообщение (один запуск на чат, HTTP 409).

Здесь — разбор текстового файла с вопросами и проверка дубликатов:

- каждая непустая строка файла — отдельный вопрос;
- нумерация («1.», «1)», «1:») и маркеры списка («-», «*», «•», «–») в начале
  строки убираются;
- дубликаты ищутся без учёта регистра, «ё/е», лишних пробелов и
  завершающих знаков препинания — и внутри файла, и среди вопросов, уже
  записанных в тестовом диалоге.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

#: Ограничения тестового диалога.
MAX_QUESTIONS = 500
MAX_QUESTION_CHARS = 4000
MAX_NAME_CHARS = 200

_LIST_PREFIX = re.compile(r"^\s*(?:\d{1,4}\s*[.):]|[-*•–—]|\(\d{1,4}\))\s+")
_SPACES = re.compile(r"\s+")
_TRAILING_PUNCT = " \t.!?…;:,"


def normalize_question(text: str) -> str:
    """Ключ для поиска дубликатов: «Что такое RAG?» и «что такое  rag» совпадают."""
    key = _SPACES.sub(" ", (text or "").replace("ё", "е").replace("Ё", "Е")).strip().casefold()
    return key.rstrip(_TRAILING_PUNCT)


def clean_question(line: str) -> str:
    """Строка файла → вопрос: без маркера списка или номера и без крайних пробелов."""
    text = line.replace("﻿", "").strip()
    stripped = _LIST_PREFIX.sub("", text, count=1).strip()
    return stripped or text


def find_duplicates(questions: Iterable[str]) -> List[Dict[str, object]]:
    """Повторы в списке вопросов: [{question, positions: [1-based]}]."""
    seen: Dict[str, List[int]] = {}
    first: Dict[str, str] = {}
    for i, q in enumerate(questions, start=1):
        key = normalize_question(q)
        seen.setdefault(key, []).append(i)
        first.setdefault(key, q)
    return [{"question": first[k], "positions": pos} for k, pos in seen.items() if len(pos) > 1]


def parse_questions(text: str, existing: Optional[Iterable[str]] = None) -> Dict[str, object]:
    """Импорт вопросов из текста файла.

    Возвращает `{"added": [...], "duplicates": [{"line", "question", "reason", "duplicate_of"}],
    "lines": N, "skipped_too_long": [...]}`: `added` — новые вопросы в порядке
    файла; `reason` — `"file"` (повтор внутри файла) или `"existing"` (такой
    вопрос уже есть в тестовом диалоге)."""
    existing_keys: Dict[str, str] = {}
    for q in existing or []:
        existing_keys.setdefault(normalize_question(q), q)
    added: List[str] = []
    added_keys: Dict[str, int] = {}
    duplicates: List[Dict[str, object]] = []
    too_long: List[int] = []
    lines = 0
    for number, raw in enumerate((text or "").splitlines(), start=1):
        if not raw.strip():
            continue
        lines += 1
        question = clean_question(raw)
        if len(question) > MAX_QUESTION_CHARS:
            too_long.append(number)
            continue
        key = normalize_question(question)
        if not key:
            continue
        if key in existing_keys:
            duplicates.append({"line": number, "question": question, "reason": "existing",
                               "duplicate_of": existing_keys[key]})
        elif key in added_keys:
            duplicates.append({"line": number, "question": question, "reason": "file",
                               "duplicate_of": added[added_keys[key]]})
        else:
            added_keys[key] = len(added)
            added.append(question)
    return {"added": added, "duplicates": duplicates, "lines": lines, "skipped_too_long": too_long}


def validate_questions(questions: List[str]) -> List[str]:
    """Проверка набора вопросов перед сохранением; возвращает очищенный список
    или поднимает ValueError с понятным текстом."""
    cleaned = [q.strip() for q in questions if isinstance(q, str)]
    if any(not q for q in cleaned) or len(cleaned) != len(questions):
        raise ValueError("questions: вопросы должны быть непустыми строками")
    if not cleaned:
        raise ValueError("questions: нужен хотя бы один вопрос")
    if len(cleaned) > MAX_QUESTIONS:
        raise ValueError(f"questions: не больше {MAX_QUESTIONS} вопросов")
    long_ones = [i for i, q in enumerate(cleaned, start=1) if len(q) > MAX_QUESTION_CHARS]
    if long_ones:
        raise ValueError(f"questions: вопросы длиннее {MAX_QUESTION_CHARS} символов: {long_ones}")
    dups = find_duplicates(cleaned)
    if dups:
        text = "; ".join(f"«{d['question']}» (№ {', '.join(map(str, d['positions']))})" for d in dups)
        raise ValueError(f"questions: повторяющиеся вопросы: {text}")
    return cleaned

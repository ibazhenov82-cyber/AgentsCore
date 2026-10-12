"""
agents_core.logging_setup
===========================

Единая настройка логирования сервиса. Даёт два вида логов, оба видны в
stderr (и опционально — в файле):

* `agents_core.access` — каждый входящий REST-запрос (см.
  `agents_core.api.logging_middleware.AccessLogMiddleware`): метод, путь,
  path/query-параметры, тело запроса, итоговый статус и время выполнения.
* `agents_core.llm.<провайдер>` — каждый исходящий вызов внешнего API LLM
  (см. `agents_core.providers.deepseek_client`/`agents_core.providers.ollama`):
  метод, URL, параметры/тело запроса, статус ответа и время выполнения.

Обе группы логгеров — обычные потомки логгера `agents_core` в иерархии
стандартного модуля `logging`, поэтому настройка обработчиков/уровня в
одном месте (`configure_logging()`) применяется сразу ко всем.

Управляется переменными окружения (см. `.env.example`):

    AGENT_LOG_LEVEL        — уровень логирования (по умолчанию "INFO")
    AGENT_LOG_FILE         — путь к файлу лога в ДОПОЛНЕНИЕ к stderr
                             (по умолчанию не задан — лог только в stderr)
    AGENT_LOG_BODY_LIMIT   — сколько символов тела запроса/ответа/JSON-
                             параметров показывать в одной строке лога,
                             остальное усекается (по умолчанию 2000 — тела
                             сообщений чата могут быть длинными, полный лог
                             без ограничения быстро становится нечитаемым)

`configure_logging()` вызывается при сборке приложения
(`agents_core.api.app.create_app_with_repository`) и идемпотентна —
повторные вызовы (например, при создании нескольких приложений в тестах)
не плодят дублирующиеся обработчики, только обновляют уровень/лимит.
"""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
import os
from typing import Any, Optional

ROOT_LOGGER_NAME = "agents_core"
ACCESS_LOGGER_NAME = "agents_core.access"

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_DEFAULT_BODY_LIMIT = 2000

_configured = False
_body_limit = _DEFAULT_BODY_LIMIT

# Имена ключей (без учёта регистра, по вхождению подстроки), значения
# которых `redact()` всегда заменяет на "***" — на случай, если реквизит
# доступа когда-нибудь окажется в теле REST-запроса или в параметрах
# вызова провайдера (в известных на сегодня местах их там нет — секреты
# уходят только в HTTP-заголовки, которые в лог не попадают вовсе).
_SECRET_KEY_MARKERS = ("api_key", "apikey", "authorization", "password", "secret", "token", "bearer")


def configure_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    body_limit: Optional[int] = None,
) -> None:
    """Настраивает обработчики логгера `agents_core`. Без аргументов читает
    значения из переменных окружения (см. докстринг модуля) — так вызывает
    `create_app_with_repository()`. Явные аргументы позволяют переопределить
    это в тестах, не трогая окружение процесса."""
    global _configured, _body_limit

    resolved_level = (level if level is not None else os.environ.get("AGENT_LOG_LEVEL", "INFO")).strip().upper() or "INFO"
    resolved_file = (log_file if log_file is not None else os.environ.get("AGENT_LOG_FILE", "")).strip()

    if body_limit is not None:
        _body_limit = body_limit
    else:
        try:
            _body_limit = int(os.environ.get("AGENT_LOG_BODY_LIMIT", str(_DEFAULT_BODY_LIMIT)))
        except ValueError:
            _body_limit = _DEFAULT_BODY_LIMIT

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(resolved_level)

    if not _configured:
        formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        # Свои обработчики, не всплывать до корневого логгера — иначе при
        # запуске под uvicorn (у которого свой root-логгер) строки
        # продублировались бы.
        logger.propagate = False
        _configured = True

    if resolved_file:
        abs_path = os.path.abspath(resolved_file)
        has_this_file = any(
            isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == abs_path
            for h in logger.handlers
        )
        if not has_this_file:
            file_handler = RotatingFileHandler(resolved_file, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
            logger.addHandler(file_handler)


def get_access_logger() -> logging.Logger:
    return logging.getLogger(ACCESS_LOGGER_NAME)


def truncate(text: str, limit: Optional[int] = None) -> str:
    """Обрезает строку до `limit` символов (по умолчанию — текущий
    `AGENT_LOG_BODY_LIMIT`), дописывая пометку с исходной длиной."""
    effective_limit = _body_limit if limit is None else limit
    if len(text) <= effective_limit:
        return text
    return f"{text[:effective_limit]}… (обрезано в логе, всего {len(text)} симв.)"


def redact(value: Any) -> Any:
    """Рекурсивно заменяет значения ключей, похожих на реквизиты доступа
    (см. `_SECRET_KEY_MARKERS`), на "***"."""
    if isinstance(value, dict):
        return {
            k: ("***" if isinstance(k, str) and any(marker in k.lower() for marker in _SECRET_KEY_MARKERS) else redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def safe_json_dumps(value: Any, limit: Optional[int] = None) -> str:
    """Сериализует значение в JSON для лога — с маскированием возможных
    секретов (`redact`) и усечением длинных тел (`truncate`). Никогда не
    бросает исключение: при ошибке сериализации возвращает `str(value)`."""
    try:
        text = json.dumps(redact(value), ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(value)
    return truncate(text, limit)

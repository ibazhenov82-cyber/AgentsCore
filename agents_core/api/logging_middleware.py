"""
agents_core.api.logging_middleware
=====================================

`AccessLogMiddleware` — логирует КАЖДЫЙ входящий REST-запрос одной строкой
через логгер `agents_core.access`: метод, путь, path- и query-параметры,
тело запроса (JSON, с усечением длинных значений и маскированием возможных
секретных полей — см. `agents_core.logging_setup`) и итоговый статус ответа
со временем выполнения.

Подключается в `agents_core.api.app.create_app_with_repository` как самый
внешний middleware — то есть измеряет время выполнения ВСЕГО остального
стека, включая обработчики ошибок (`@app.exception_handler(...)`), поэтому
в логе виден настоящий итоговый статус (404/400/409/503/502/422), а не
только необработанные исключения.

Технические детали, из-за которых это безопасно:
* `Request.body()` кэширует прочитанные байты (см. `starlette.requests.
  Request._body`) — прочитав тело здесь для лога, мы не мешаем обработчику
  маршрута ниже по стеку прочитать его снова (в т.ч. через `request.json()`,
  который использует тот же кэш).
* `request.path_params` читает `request.scope["path_params"]` каждый раз
  заново (а не значение, зафиксированное при создании `Request`), а сам
  `scope` — общий изменяемый словарь на весь ASGI-вызов, поэтому к моменту,
  когда `call_next(...)` возвращает управление, маршрутизатор уже успел
  заполнить path-параметры в этом же словаре.
"""

from __future__ import annotations

import json
import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from ..logging_setup import get_access_logger, safe_json_dumps, truncate

_logger = get_access_logger()


class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        started = time.monotonic()
        body_repr = await _read_body_for_log(request)
        query_repr = safe_json_dumps(dict(request.query_params))

        try:
            response = await call_next(request)
        except Exception as exc:
            # Обработчики агентс-специфичных ошибок (`NotFoundError` и т.п.,
            # см. app.py) уже успели превратить свою ошибку в JSONResponse
            # ДО того, как исключение дошло бы сюда — сюда попадают только
            # действительно необработанные исключения (итоговый статус 500).
            duration_ms = (time.monotonic() - started) * 1000
            _logger.exception(
                "REST %s %s query=%s body=%s -> НЕОБРАБОТАННОЕ ИСКЛЮЧЕНИЕ за %.1f мс: %s",
                request.method,
                request.url.path,
                query_repr,
                body_repr,
                duration_ms,
                exc,
            )
            raise

        duration_ms = (time.monotonic() - started) * 1000
        path_params_repr = safe_json_dumps(dict(request.path_params)) if request.path_params else "{}"
        _logger.info(
            "REST %s %s path_params=%s query=%s body=%s -> %s (%.1f мс)",
            request.method,
            request.url.path,
            path_params_repr,
            query_repr,
            body_repr,
            response.status_code,
            duration_ms,
        )
        return response


async def _read_body_for_log(request: Request) -> str:
    try:
        raw = await request.body()
    except Exception:
        return "<не удалось прочитать тело>"
    if not raw:
        return "-"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return f"<бинарные данные, {len(raw)} байт>"
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return truncate(text)
    return safe_json_dumps(parsed)

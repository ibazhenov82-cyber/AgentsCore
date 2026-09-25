"""
agents_core.api.app
=====================

Сборка приложения FastAPI: создаёт БД/провайдеров/каталог моделей/
репозиторий, подключает роутеры и обработчики ошибок. `python -m
agents_core.main` запускает именно то, что возвращает `create_app()`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..catalog import ModelCatalog
from ..config import AgentConfig
from ..db import Database
from ..events import EventsGone
from ..logging_setup import configure_logging
from ..mcp_client import build_mcp_client
from ..providers import ProviderError, ProviderRegistry
from ..repository import (
    NotConfiguredError,
    NotFoundError,
    PreconditionFailedError,
    Repository,
    ValidationError,
)
from ..runs import RunConflictError, RunManager
from . import agents, chats, health, invariants, mcp_tools, memory, messages, models_routes, runs, settings_routes, tasks
from .logging_middleware import AccessLogMiddleware

DESCRIPTION = """
AgentsCore — HTTP-сервис для работы с LLM-агентами.

Агент — самостоятельная сущность верхнего уровня («старший чат»): у него
своя модель, свой провайдер (DeepSeek или Ollama) и свой набор параметров
запроса, и он владеет произвольным числом чатов (связь один-ко-многим).
Каждый чат наследует настройки своего агента в момент создания и может
переопределить часть из них (кроме модели — она зафиксирована на уровне
агента).

Сервис хранит всё сам (агенты, чаты, сообщения, каталог моделей во
внутренней SQLite), умеет считать токены по каждому чату, суммаризировать
историю диалога вручную и автоматически по достижении лимитов, и работает
как с облачными моделями DeepSeek, так и с локальными моделями,
запущенными через Ollama.
"""


def create_app(config: type = AgentConfig, enable_mcp: Optional[bool] = None) -> FastAPI:
    """Полная сборка: настоящая БД + настоящие провайдеры (согласно
    `AgentConfig`). Это то, что запускает `agents_core.main`.

    `enable_mcp` (новое ТЗ, интеграция с отдельным MCP-сервером) — по
    умолчанию `None`: решение принимается по конфигу
    (`config.is_mcp_configured()`, т.е. и `MCP_ENABLED=true`, и непустой
    `MCP_SERVER_URL`). Явный `True`/`False` переопределяет это (тесты,
    ручной запуск с другим набором флагов) — `True` при пустом
    `MCP_SERVER_URL` всё равно не создаёт клиента (не с чем соединяться)."""
    db = Database(config.DB_PATH)
    registry = ProviderRegistry(config)
    catalog = ModelCatalog(db, registry, config)
    catalog.load_or_discover()
    servers = config.mcp_servers()
    mcp_enabled = config.is_mcp_configured() if enable_mcp is None else (enable_mcp and bool(servers))
    mcp_client = (
        build_mcp_client(servers, api_key=config.MCP_API_KEY or None, timeout=config.MCP_REQUEST_TIMEOUT)
        if mcp_enabled else None
    )
    repo = Repository(db, registry, catalog, mcp_client=mcp_client)
    return create_app_with_repository(repo)


def create_app_with_repository(repo: Repository, run_manager: Optional[RunManager] = None) -> FastAPI:
    """Сборка приложения вокруг уже готового `Repository` — используется
    `create_app()`, а также тестами API (с репозиторием на фиктивном
    провайдере, без обращения к настоящим DeepSeek/Ollama)."""
    configure_logging(level=AgentConfig.LOG_LEVEL, log_file=AgentConfig.LOG_FILE, body_limit=AgentConfig.LOG_BODY_LIMIT)
    manager = run_manager or RunManager(repo)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Асинхронные запуски (ТЗ, 2.4): запуски, прерванные прошлым
        # остановом сервиса, помечаются "interrupted" при старте.
        manager.recover()
        try:
            yield
        finally:
            manager.shutdown(wait=False)

    app = FastAPI(
        title="AgentsCore",
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.state.repo = repo
    app.state.run_manager = manager

    # Логируем каждый REST-запрос (метод, путь, параметры, тело, итоговый
    # статус, время выполнения) — см. `agents_core.logging_middleware`.
    # Добавлен ДО роутеров нарочно: единственный middleware в приложении,
    # порядок относительно роутеров не влияет на порядок обработки запроса
    # (роутинг всегда происходит внутри стека middleware), но именно этот
    # порядок вызовов явно показывает, что лог обёрнут вокруг всего
    # остального стека, включая обработчики ошибок ниже.
    app.add_middleware(AccessLogMiddleware)

    app.include_router(health.router)
    app.include_router(models_routes.router)
    app.include_router(settings_routes.router)
    app.include_router(agents.router)
    app.include_router(chats.router)
    app.include_router(messages.router)
    app.include_router(memory.router)
    app.include_router(invariants.router)
    app.include_router(tasks.router)
    app.include_router(runs.router)
    app.include_router(mcp_tools.router)

    @app.exception_handler(RunConflictError)
    def _run_conflict(request: Request, exc: RunConflictError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"error": str(exc), "active_run_id": exc.active_run_id})

    @app.exception_handler(EventsGone)
    def _events_gone(request: Request, exc: EventsGone) -> JSONResponse:
        return JSONResponse(status_code=410, content={"error": str(exc)})

    @app.exception_handler(NotFoundError)
    def _not_found(request: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"error": str(exc)})

    @app.exception_handler(ValidationError)
    def _validation(request: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    @app.exception_handler(NotConfiguredError)
    def _not_configured(request: Request, exc: NotConfiguredError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"error": str(exc)})

    @app.exception_handler(PreconditionFailedError)
    def _precondition_failed(request: Request, exc: PreconditionFailedError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"error": str(exc)})

    @app.exception_handler(ProviderError)
    def _provider_error(request: Request, exc: ProviderError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"error": f"provider error: {exc}"})

    @app.exception_handler(RequestValidationError)
    def _request_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        # По умолчанию FastAPI отдаёт на 422 тело {"detail": [...]} — клиент
        # (Android-приложение) умеет разбирать только собственный формат
        # ошибок сервиса {"error": "..."}, поэтому без этого обработчика
        # любая ошибка валидации тела/параметров запроса доходила до
        # пользователя как голое "Ошибка API сервера (422)" без единой
        # подробности. Здесь мы собираем читаемое сообщение из списка
        # ошибок pydantic и отдаём его в уже знакомом клиенту формате.
        details = "; ".join(
            f"{'.'.join(str(p) for p in err.get('loc', []) if p != 'body')}: {err.get('msg', '')}"
            for err in exc.errors()
        )
        return JSONResponse(status_code=422, content={"error": details or "некорректный запрос"})

    return app

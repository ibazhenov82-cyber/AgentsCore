"""
agents_core
============

AgentsCore — самостоятельный HTTP-сервис для работы с LLM-агентами:
хранит агентов (владеющих чатами), чаты и сообщения во внутренней базе
SQLite, поддерживает несколько провайдеров LLM (облачный DeepSeek и
локальные модели через Ollama), считает токены, умеет суммаризировать
историю диалога вручную и автоматически, и открывает всё это через REST
API на FastAPI с документацией Swagger UI.
"""

from .config import AgentConfig
from .models import (
    Agent,
    AGENT_SETTINGS_FIELDS,
    AUTOSUMMARY_OPTIONS,
    Chat,
    DEFAULT_SETTINGS_FIELDS,
    DefaultSettings,
    Message,
    ModelInfo,
    Settings,
    THINKING_EFFORT_OPTIONS,
    TOOL_CHOICE_OPTIONS,
)
from .db import Database
from .catalog import ModelCatalog
from .providers import ProviderRegistry
from .repository import (
    NotConfiguredError,
    NotFoundError,
    PreconditionFailedError,
    Repository,
    RepositoryError,
    ValidationError,
)

__all__ = [
    "AgentConfig", "Agent", "AGENT_SETTINGS_FIELDS", "AUTOSUMMARY_OPTIONS", "Chat",
    "DEFAULT_SETTINGS_FIELDS", "DefaultSettings", "Message", "ModelInfo", "Settings",
    "THINKING_EFFORT_OPTIONS", "TOOL_CHOICE_OPTIONS", "Database", "ModelCatalog",
    "ProviderRegistry", "Repository", "RepositoryError", "NotFoundError",
    "ValidationError", "NotConfiguredError", "PreconditionFailedError",
]

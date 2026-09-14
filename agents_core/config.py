"""
agents_core.config
===================

Настройки уровня сервиса AgentsCore: адрес/порт HTTP-сервера, путь к БД,
а также список сконфигурированных провайдеров LLM (облачных и локальных)
и моделей, которые нужно опросить/зарегистрировать при старте.

Всё считывается один раз из переменных окружения либо файла `.env` рядом с
местом запуска процесса (перезапуск сервиса нужен, чтобы подхватить
изменения — это не "горячая" конфигурация).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


def _load_dotenv(path: str = ".env") -> None:
    """Минимальный загрузчик `.env`: строки вида KEY=VALUE, комментарии
    через '#', уже существующие переменные окружения имеют приоритет."""
    file = Path(path)
    if not file.exists():
        return
    for raw_line in file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class ProviderModelEntry:
    """Одна модель, объявленная в .env для конкретного провайдера."""

    provider: str
    model_id: str

    @property
    def id(self) -> str:
        # Составной идентификатор модели, используемый во всём HTTP API
        # (например "ollama:qwen3:0.6b" или "deepseek:deepseek-v4-flash").
        return f"{self.provider}:{self.model_id}"


class AgentConfig:
    """Читается один раз при импорте модуля."""

    HOST: str = os.environ.get("AGENT_HOST", "0.0.0.0")
    PORT: int = int(os.environ.get("AGENT_PORT", "8000"))
    DB_PATH: str = os.environ.get("AGENT_DB_PATH", "agents_core.db")

    # Список активных провайдеров, например "deepseek,ollama". Провайдер,
    # не перечисленный здесь, не опрашивается и его модели не регистрируются,
    # даже если для него заданы прочие переменные окружения.
    PROVIDERS: List[str] = _split_csv(os.environ.get("PROVIDERS", "deepseek,ollama"))

    DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    DEEPSEEK_MODELS: List[str] = _split_csv(
        os.environ.get(
            "DEEPSEEK_MODELS",
            "deepseek-v4-flash,deepseek-v4-pro,deepseek-v4-flash-vision-exp",
        )
    )

    OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    # По умолчанию предлагается qwen3:0.6b — компактная модель, которую
    # реально запустить локально без мощного GPU; ниже перечислены и другие
    # размеры семейства qwen3 для локального запуска через Ollama.
    OLLAMA_MODELS: List[str] = _split_csv(
        os.environ.get(
            "OLLAMA_MODELS",
            "qwen3:0.6b,qwen3:1.7b,qwen3:4b,qwen3:8b,qwen3:14b,qwen3:30b-a3b",
        )
    )

    @classmethod
    def configured_models(cls) -> List[ProviderModelEntry]:
        """Полный список моделей, объявленных в .env для всех активных
        провайдеров — используется при старте сервиса для наполнения каталога
        моделей (см. `agents_core.repository.ModelCatalog`)."""
        entries: List[ProviderModelEntry] = []
        if "deepseek" in cls.PROVIDERS:
            for model_id in cls.DEEPSEEK_MODELS:
                entries.append(ProviderModelEntry("deepseek", model_id))
        if "ollama" in cls.PROVIDERS:
            for model_id in cls.OLLAMA_MODELS:
                entries.append(ProviderModelEntry("ollama", model_id))
        return entries

    @classmethod
    def is_deepseek_configured(cls) -> bool:
        return bool(cls.DEEPSEEK_API_KEY)

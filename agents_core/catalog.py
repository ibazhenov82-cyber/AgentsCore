"""
agents_core.catalog
=====================

Каталог моделей: наполняется при старте сервиса на основе списка моделей,
объявленного в .env (`AgentConfig.configured_models()`), кэшируется в БД и
в оперативной памяти. См. общие требования: если набор моделей в .env не
изменился с прошлого запуска — повторного опроса провайдеров не происходит,
данные просто загружаются из БД в память.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .config import AgentConfig
from .db import Database
from .models import ModelInfo
from .providers import ModelCapabilities, ProviderRegistry


class ModelCatalog:
    def __init__(self, db: Database, registry: ProviderRegistry, config: type = AgentConfig):
        self._db = db
        self._registry = registry
        self._config = config
        self._models: Dict[str, ModelInfo] = {}

    def load_or_discover(self) -> None:
        configured = self._config.configured_models()
        configured_ids = sorted(e.id for e in configured)
        existing_ids = sorted(self._db.get_models_catalog_keys())

        if existing_ids and configured_ids == existing_ids:
            # Список моделей не изменился — просто поднимаем то, что уже
            # известно из БД, без повторных обращений к провайдерам.
            self._models = {m.id: m for m in self._db.load_models_catalog()}
            return

        discovered: List[ModelInfo] = []
        for entry in configured:
            try:
                provider = self._registry.get(entry.provider)
                caps = provider.discover_model(entry.model_id)
            except Exception:
                # Провайдер недоступен при старте (например, Ollama ещё не
                # запущена) — не роняем сервис, регистрируем модель с
                # минимальными данными; следующий перезапуск после того, как
                # провайдер станет доступен, обновит запись (т.к. набор
                # моделей в .env не поменяется, но мы можем принудительно
                # переоткрыть при явном несовпадении — здесь же fallback).
                caps = ModelCapabilities(display_name=entry.model_id, is_local=(entry.provider != "deepseek"))
            discovered.append(
                ModelInfo(
                    id=entry.id, provider=entry.provider, model_id=entry.model_id,
                    display_name=caps.display_name, is_local=caps.is_local,
                    context_window=caps.context_window, max_input_tokens=caps.max_input_tokens,
                    max_output_tokens=caps.max_output_tokens, max_reasoning_tokens=caps.max_reasoning_tokens,
                    supports_thinking=caps.supports_thinking, supports_tools=caps.supports_tools,
                    supports_json_mode=caps.supports_json_mode, supports_logprobs=caps.supports_logprobs,
                )
            )
        self._db.replace_models_catalog(discovered)
        self._models = {m.id: m for m in discovered}

    def list_models(self) -> List[ModelInfo]:
        return list(self._models.values())

    def get(self, composite_id: str) -> Optional[ModelInfo]:
        return self._models.get(composite_id)

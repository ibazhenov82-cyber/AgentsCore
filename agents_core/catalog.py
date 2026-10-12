"""
agents_core.catalog
=====================

Каталог моделей, кэшируется в БД и в оперативной памяти:

- **облачные** (DeepSeek) — из `.env` (`DEEPSEEK_MODELS`); характеристики
  опрашиваются один раз и дальше берутся из БД;
- **локальные** (Ollama) — все скачанные чат-модели (`ollama list`, без
  embedding-моделей). Список сверяется при старте и при каждом обращении к
  экрану «Модели» (`sync_local`, его вызывает `LocalModelManager`):
  скачали модель — она появляется в каталоге, удалили — пропадает. Если
  Ollama недоступна, остаётся последний известный список из БД.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

from .config import AgentConfig
from .db import Database
from .models import ModelInfo
from .providers import ModelCapabilities, ProviderError, ProviderRegistry

OLLAMA = "ollama"


class ModelCatalog:
    def __init__(self, db: Database, registry: ProviderRegistry, config: type = AgentConfig):
        self._db = db
        self._registry = registry
        self._config = config
        self._models: Dict[str, ModelInfo] = {}
        self._lock = threading.Lock()

    # ---- наполнение -----------------------------------------------------------

    def _discover(self, provider_name: str, model_id: str) -> ModelInfo:
        try:
            caps = self._registry.get(provider_name).discover_model(model_id)
        except Exception:
            # Провайдер недоступен — регистрируем модель с минимальными
            # данными, чтобы не ронять сервис.
            caps = ModelCapabilities(display_name=model_id, is_local=(provider_name == OLLAMA))
        return ModelInfo(
            id=f"{provider_name}:{model_id}", provider=provider_name, model_id=model_id,
            display_name=caps.display_name, is_local=caps.is_local,
            context_window=caps.context_window, max_input_tokens=caps.max_input_tokens,
            max_output_tokens=caps.max_output_tokens, max_reasoning_tokens=caps.max_reasoning_tokens,
            supports_thinking=caps.supports_thinking, supports_tools=caps.supports_tools,
            supports_json_mode=caps.supports_json_mode, supports_logprobs=caps.supports_logprobs,
        )

    def _ollama_chat_models(self) -> Optional[List[str]]:
        """Имена скачанных чат-моделей Ollama; `[]` — провайдер выключен,
        `None` — Ollama недоступна (тогда список не трогаем)."""
        provider = self._registry.all().get(OLLAMA)
        if provider is None:
            return []
        try:
            tags = provider.list_tags()
        except ProviderError:
            return None
        names = []
        for entry in tags:
            name = entry.get("model") or entry.get("name")
            if name and provider.is_chat_model(name, entry.get("digest")):
                names.append(name)
        return names

    def load_or_discover(self) -> None:
        """При старте: облачные — из .env, локальные — из Ollama; известные
        модели берутся из БД без повторного опроса провайдера."""
        cached = {m.id: m for m in self._db.load_models_catalog()}
        models: Dict[str, ModelInfo] = {}
        for entry in self._config.configured_models():
            models[entry.id] = cached.get(entry.id) or self._discover(entry.provider, entry.model_id)
        local = self._ollama_chat_models()
        if local is None:
            models.update({mid: m for mid, m in cached.items() if m.provider == OLLAMA})
        else:
            for name in local:
                cid = f"{OLLAMA}:{name}"
                models[cid] = cached.get(cid) or self._discover(OLLAMA, name)
        with self._lock:
            self._db.replace_models_catalog(list(models.values()))
            self._models = models

    def sync_local(self, chat_model_names: List[str]) -> bool:
        """Сверяет локальную часть каталога со списком скачанных чат-моделей
        Ollama. Возвращает True, если каталог изменился."""
        with self._lock:
            current = dict(self._models)
            wanted = {f"{OLLAMA}:{name}": name for name in chat_model_names}
            existing_local = {mid for mid, m in current.items() if m.provider == OLLAMA}
            if existing_local == set(wanted):
                return False
            models = {mid: m for mid, m in current.items() if m.provider != OLLAMA}
            for cid, name in wanted.items():
                models[cid] = current.get(cid) or self._discover(OLLAMA, name)
            self._db.replace_models_catalog(list(models.values()))
            self._models = models
            return True

    # ---- чтение -------------------------------------------------------------

    def list_models(self) -> List[ModelInfo]:
        return list(self._models.values())

    def get(self, composite_id: str) -> Optional[ModelInfo]:
        return self._models.get(composite_id)

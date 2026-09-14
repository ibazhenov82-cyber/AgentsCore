"""
agents_core.providers
======================

Реестр адаптеров провайдеров LLM. Сейчас поддерживаются DeepSeek (облачный)
и Ollama (локальный); добавление нового провайдера — это новый модуль с
классом, реализующим `BaseProvider`, и одна строка регистрации здесь.
"""

from __future__ import annotations

from typing import Dict

from ..config import AgentConfig
from .base import BaseProvider, ChatResult, ChatUsage, ModelCapabilities, ProviderError, ProviderMessage, StreamDelta
from .deepseek import DeepSeekProvider
from .ollama import OllamaProvider

__all__ = [
    "BaseProvider", "ChatResult", "ChatUsage", "ModelCapabilities", "ProviderError",
    "ProviderMessage", "StreamDelta", "ProviderRegistry",
]


class ProviderRegistry:
    """Строит и хранит по одному экземпляру адаптера на каждый активный
    провайдер (согласно `AgentConfig.PROVIDERS`)."""

    def __init__(self, config: type = AgentConfig):
        self._providers: Dict[str, BaseProvider] = {}
        if "deepseek" in config.PROVIDERS:
            self._providers["deepseek"] = DeepSeekProvider(api_key=config.DEEPSEEK_API_KEY)
        if "ollama" in config.PROVIDERS:
            self._providers["ollama"] = OllamaProvider(base_url=config.OLLAMA_BASE_URL)

    def get(self, provider_name: str) -> BaseProvider:
        provider = self._providers.get(provider_name)
        if provider is None:
            raise ProviderError(f"unknown or inactive provider: {provider_name}")
        return provider

    def all(self) -> Dict[str, BaseProvider]:
        return dict(self._providers)

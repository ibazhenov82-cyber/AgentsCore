"""
agents_core.providers.base
===========================

Общий интерфейс адаптера провайдера LLM. У AgentsCore может быть
подключено произвольное число провайдеров (сейчас — DeepSeek и Ollama);
каждый реализует этот интерфейс, так что остальной сервис (репозиторий,
HTTP-слой) работает с моделью и настройками, не зная деталей конкретного
API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from ..models import Settings


@dataclass
class ProviderMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class ChatUsage:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass
class ChatResult:
    content: str
    reasoning_content: Optional[str] = None
    usage: ChatUsage = field(default_factory=ChatUsage)


@dataclass
class StreamDelta:
    content: str = ""
    reasoning_content: str = ""
    done: bool = False
    result: Optional[ChatResult] = None  # заполняется только в последнем событии (done=True)


@dataclass
class ModelCapabilities:
    """То, что провайдер смог (или не смог) сообщить о модели."""

    display_name: str
    is_local: bool
    context_window: Optional[int] = None
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    max_reasoning_tokens: Optional[int] = None
    supports_thinking: bool = False
    supports_tools: bool = False
    supports_json_mode: bool = False
    supports_logprobs: bool = False


class ProviderError(Exception):
    """Ошибка на стороне провайдера LLM (сеть, неверный ответ, HTTP-статус не 2xx)."""


class BaseProvider(ABC):
    name: str

    @abstractmethod
    def health(self) -> bool:
        """Быстрая проверка доступности провайдера в целом (например, для
        DeepSeek — что ключ API рабочий)."""

    def model_health(self, model_id: str) -> bool:
        """Проверка доступности конкретной модели через API провайдера,
        используется в `GET /model/{id}/health`. По умолчанию совпадает с
        общей проверкой провайдера (`health()`); переопределяется там, где
        есть смысл проверять именно эту модель (см. `OllamaProvider`, где
        проверка идёт по списку моделей `GET /api/tags`)."""
        return self.health()

    @abstractmethod
    def discover_model(self, model_id: str) -> ModelCapabilities:
        """Опрашивает провайдера (там, где это возможно) и возвращает
        характеристики модели; там, где API их не раскрывает, реализация
        обязана вернуть разумные встроенные справочные значения."""

    @abstractmethod
    def chat(self, model_id: str, messages: List[ProviderMessage], settings: Settings) -> ChatResult:
        """Блокирующий вызов чат-завершения."""

    @abstractmethod
    def stream_chat(self, model_id: str, messages: List[ProviderMessage], settings: Settings) -> Iterator[StreamDelta]:
        """Потоковый вызов чат-завершения."""

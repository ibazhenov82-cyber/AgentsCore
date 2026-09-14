"""
agents_core.providers.deepseek
================================

Адаптер провайдера DeepSeek (облачный API) поверх низкоуровневого клиента
`deepseek_client.py`. Реализует общий интерфейс `BaseProvider`.

DeepSeek не раскрывает лимиты контекстного окна/reasoning через свой API,
поэтому `discover_model` использует встроенную справочную таблицу
известных моделей — это единственный провайдер, для которого приходится
так делать (Ollama отдаёт эти данные через `/api/show`, см. `ollama.py`).
"""

from __future__ import annotations

import json
from typing import Iterator, List, Optional

from ..models import Settings
from .base import BaseProvider, ChatResult, ChatUsage, ModelCapabilities, ProviderError, ProviderMessage, StreamDelta
from .deepseek_client import (
    DeepSeekClient,
    DeepSeekError,
    ResponseFormat,
    ThinkingConfig,
)

# Справочные значения для текущего семейства моделей DeepSeek-V4 — API не
# отдаёт их программно, поэтому они зашиты здесь и используются как
# fallback при первичном наполнении каталога моделей.
_FALLBACK_CAPS = {
    "deepseek-v4-flash": ModelCapabilities(
        display_name="DeepSeek V4 Flash", is_local=False, context_window=128000,
        max_input_tokens=120000, max_output_tokens=8192, max_reasoning_tokens=32000,
        supports_thinking=True, supports_tools=True, supports_json_mode=True, supports_logprobs=True,
    ),
    "deepseek-v4-pro": ModelCapabilities(
        display_name="DeepSeek V4 Pro", is_local=False, context_window=128000,
        max_input_tokens=120000, max_output_tokens=8192, max_reasoning_tokens=32000,
        supports_thinking=True, supports_tools=True, supports_json_mode=True, supports_logprobs=True,
    ),
    "deepseek-v4-flash-vision-exp": ModelCapabilities(
        display_name="DeepSeek V4 Flash Vision (exp)", is_local=False, context_window=128000,
        max_input_tokens=120000, max_output_tokens=8192, max_reasoning_tokens=None,
        supports_thinking=False, supports_tools=True, supports_json_mode=True, supports_logprobs=True,
    ),
}

_GENERIC_FALLBACK = ModelCapabilities(
    display_name="DeepSeek", is_local=False, context_window=64000,
    max_input_tokens=60000, max_output_tokens=4096, max_reasoning_tokens=16000,
    supports_thinking=True, supports_tools=True, supports_json_mode=True, supports_logprobs=True,
)


class DeepSeekProvider(BaseProvider):
    name = "deepseek"

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client: Optional[DeepSeekClient] = DeepSeekClient(api_key=api_key) if api_key else None

    def _require_client(self) -> DeepSeekClient:
        if self._client is None:
            raise ProviderError("DeepSeek API key is not configured for this agent service")
        return self._client

    def health(self) -> bool:
        try:
            self._require_client().list_models()
            return True
        except Exception:
            return False

    def discover_model(self, model_id: str) -> ModelCapabilities:
        return _FALLBACK_CAPS.get(model_id, _GENERIC_FALLBACK)

    def _build_messages(self, messages: List[ProviderMessage]) -> List[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def _thinking_config(self, settings: Settings, caps: ModelCapabilities) -> Optional[ThinkingConfig]:
        if not caps.supports_thinking:
            return None
        return ThinkingConfig(
            type="enabled" if settings.thinking_enabled else "disabled",
            reasoning_effort=settings.reasoning_effort if settings.thinking_enabled else None,
        )

    def _parsed_tools(self, settings: Settings) -> Optional[list]:
        if not settings.tools_json.strip():
            return None
        try:
            return json.loads(settings.tools_json)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"tools_json is not valid JSON: {exc}") from exc

    def chat(self, model_id: str, messages: List[ProviderMessage], settings: Settings) -> ChatResult:
        client = self._require_client()
        caps = self.discover_model(model_id)
        tools = self._parsed_tools(settings)
        try:
            resp = client.chat(
                self._build_messages(messages),
                model=model_id,
                temperature=settings.temperature,
                top_p=settings.top_p,
                max_tokens=settings.max_tokens,
                response_format=ResponseFormat(type="json_object") if settings.json_mode else None,
                stop=settings.stop_sequences or None,
                tools=tools,
                tool_choice=settings.tool_choice if tools else None,
                logprobs=settings.logprobs or None,
                frequency_penalty=settings.frequency_penalty,
                presence_penalty=settings.presence_penalty,
                thinking=self._thinking_config(settings, caps),
                # DeepSeek не документирует `seed` официально — передаём как
                # best-effort passthrough наравне с другими провайдерами.
                extra_params={"seed": settings.seed} if settings.seed is not None else None,
            )
        except DeepSeekError as exc:
            raise ProviderError(str(exc)) from exc
        choice = resp["choices"][0]["message"]
        usage = resp.get("usage") or {}
        return ChatResult(
            content=choice.get("content") or "",
            reasoning_content=choice.get("reasoning_content"),
            usage=ChatUsage(
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            ),
        )

    def stream_chat(self, model_id: str, messages: List[ProviderMessage], settings: Settings) -> Iterator[StreamDelta]:
        client = self._require_client()
        caps = self.discover_model(model_id)
        tools = self._parsed_tools(settings)
        content_acc: List[str] = []
        reasoning_acc: List[str] = []
        last_usage: dict = {}
        try:
            for delta in client.stream_chat(
                self._build_messages(messages),
                model=model_id,
                temperature=settings.temperature,
                top_p=settings.top_p,
                max_tokens=settings.max_tokens,
                response_format=ResponseFormat(type="json_object") if settings.json_mode else None,
                stop=settings.stop_sequences or None,
                stream_options={"include_usage": True} if settings.include_usage_in_stream else None,
                tools=tools,
                tool_choice=settings.tool_choice if tools else None,
                logprobs=settings.logprobs or None,
                frequency_penalty=settings.frequency_penalty,
                presence_penalty=settings.presence_penalty,
                thinking=self._thinking_config(settings, caps),
                extra_params={"seed": settings.seed} if settings.seed is not None else None,
            ):
                if delta.content:
                    content_acc.append(delta.content)
                    yield StreamDelta(content=delta.content)
                if delta.reasoning_content:
                    reasoning_acc.append(delta.reasoning_content)
                    yield StreamDelta(reasoning_content=delta.reasoning_content)
                if delta.usage:
                    last_usage = delta.usage
        except DeepSeekError as exc:
            raise ProviderError(str(exc)) from exc

        yield StreamDelta(
            done=True,
            result=ChatResult(
                content="".join(content_acc),
                reasoning_content="".join(reasoning_acc) or None,
                usage=ChatUsage(
                    prompt_tokens=last_usage.get("prompt_tokens"),
                    completion_tokens=last_usage.get("completion_tokens"),
                    total_tokens=last_usage.get("total_tokens"),
                ),
            ),
        )

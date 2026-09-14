"""
agents_core.providers.ollama
==============================

Адаптер провайдера Ollama — локальный запуск моделей (в частности,
семейства Qwen3 всех размеров) через HTTP API Ollama:
https://docs.ollama.com/api/

В отличие от DeepSeek, Ollama раскрывает часть характеристик модели через
`POST /api/show` (`model_info`, `details`) — `discover_model` в первую
очередь пытается прочитать их оттуда и только при недоступности сервера
или отсутствии нужных ключей в ответе использует встроенные справочные
значения для семейства qwen3.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional

import requests

from ..models import Settings
from .base import BaseProvider, ChatResult, ChatUsage, ModelCapabilities, ProviderError, ProviderMessage, StreamDelta

_DEFAULT_TIMEOUT = 120

# Нативный размер контекстного окна семейства Qwen3 (в токенах) — Ollama не
# всегда возвращает его явным ключом для каждой версии, поэтому это разумное
# приближение, используемое как fallback, если `/api/show` недоступен или не
# содержит ключа `*.context_length`.
_QWEN3_FALLBACK_CONTEXT_WINDOW = 32768


class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(self, base_url: str, timeout: float = _DEFAULT_TIMEOUT):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()

    def health(self) -> bool:
        try:
            resp = self._session.get(f"{self._base_url}/api/tags", timeout=10)
            return resp.ok
        except requests.RequestException:
            return False

    def model_health(self, model_id: str) -> bool:
        """Модель считается запущенной, если она присутствует в ответе
        `GET /api/tags` (поле `model` каждой записи), а не только если сам
        сервер Ollama отвечает — сервер может быть жив, но конкретная модель
        ещё не подтянута/переименована."""
        try:
            resp = self._session.get(f"{self._base_url}/api/tags", timeout=10)
            if not resp.ok:
                return False
            data = resp.json()
        except (requests.RequestException, ValueError):
            return False
        entries = data.get("models") or []
        return any(entry.get("model") == model_id or entry.get("name") == model_id for entry in entries)

    def discover_model(self, model_id: str) -> ModelCapabilities:
        is_qwen3 = model_id.lower().startswith("qwen3")
        fallback = ModelCapabilities(
            display_name=model_id,
            is_local=True,
            context_window=_QWEN3_FALLBACK_CONTEXT_WINDOW if is_qwen3 else 8192,
            max_input_tokens=(_QWEN3_FALLBACK_CONTEXT_WINDOW - 2048) if is_qwen3 else 6144,
            max_output_tokens=8192 if is_qwen3 else 2048,
            max_reasoning_tokens=None,
            supports_thinking=is_qwen3,  # qwen3 поддерживает гибридный режим "thinking"
            supports_tools=is_qwen3,
            supports_json_mode=True,  # `format: "json"` поддерживается Ollama для любой модели
            supports_logprobs=False,  # Ollama не отдаёт logprobs
        )
        try:
            resp = self._session.post(
                f"{self._base_url}/api/show", json={"model": model_id}, timeout=15
            )
            if not resp.ok:
                return fallback
            data = resp.json()
        except (requests.RequestException, ValueError):
            return fallback

        model_info: Dict[str, Any] = data.get("model_info") or {}
        details = data.get("details") or {}
        family = str(details.get("family") or "").lower()
        is_thinking_capable = is_qwen3 or "qwen3" in family

        # `model_info` может содержать НЕСКОЛЬКО ключей, оканчивающихся на
        # "context_length" (например обобщённый "general.context_length" —
        # если он вообще присутствует — наряду с архитектурным
        # "qwen3.context_length"); это разные вещи, и обобщённый ключ может
        # отражать не тот размер окна, с которым реально работает именно эта
        # модель/квантование. Поэтому сначала ищем ключ, точно совпадающий с
        # "<family>.context_length" (например "qwen3.context_length"), и
        # только если такого нет — берём первый попавшийся ключ, оканчивающийся
        # на "context_length", как раньше.
        context_window = fallback.context_window
        family_key = f"{family}.context_length" if family else None
        if family_key and family_key in model_info and isinstance(model_info[family_key], (int, float)):
            context_window = int(model_info[family_key])
        else:
            for key, value in model_info.items():
                if key.lower().endswith("context_length") and isinstance(value, (int, float)):
                    context_window = int(value)
                    break

        return ModelCapabilities(
            display_name=details.get("family") and f"{details['family']} ({model_id})" or model_id,
            is_local=True,
            context_window=context_window,
            max_input_tokens=max(context_window - 2048, 1024),
            max_output_tokens=min(context_window, 8192),
            max_reasoning_tokens=None,
            supports_thinking=is_thinking_capable,
            supports_tools=is_thinking_capable,
            supports_json_mode=True,
            supports_logprobs=False,
        )

    # ---- построение тела запроса -----------------------------------------

    def _build_options(self, settings: Settings) -> Dict[str, Any]:
        options: Dict[str, Any] = {
            "temperature": settings.temperature,
            "top_p": settings.top_p,
        }
        if settings.seed is not None:
            options["seed"] = settings.seed
        if settings.max_tokens is not None:
            options["num_predict"] = settings.max_tokens
        if settings.stop_sequences:
            options["stop"] = settings.stop_sequences
        if settings.frequency_penalty is not None:
            options["frequency_penalty"] = settings.frequency_penalty
        if settings.presence_penalty is not None:
            options["presence_penalty"] = settings.presence_penalty
        return options

    def _think_param(self, settings: Settings, caps: ModelCapabilities):
        if not caps.supports_thinking:
            return None
        if not settings.thinking_enabled:
            return False
        if settings.reasoning_effort:
            # Ollama принимает "low"/"medium"/"high"; у нас "max" — соответствует "high".
            return {"low": "low", "high": "high", "max": "high"}.get(settings.reasoning_effort, True)
        return True

    def _parsed_tools(self, settings: Settings) -> Optional[list]:
        if not settings.tools_json.strip():
            return None
        try:
            return json.loads(settings.tools_json)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"tools_json is not valid JSON: {exc}") from exc

    def _build_payload(self, model_id: str, messages: List[ProviderMessage], settings: Settings, stream: bool) -> Dict[str, Any]:
        caps = self.discover_model(model_id)
        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "options": self._build_options(settings),
        }
        if settings.json_mode:
            payload["format"] = "json"
        think = self._think_param(settings, caps)
        if think is not None:
            payload["think"] = think
        tools = self._parsed_tools(settings)
        if tools:
            payload["tools"] = tools
        return payload

    # ---- вызовы -----------------------------------------------------------

    def chat(self, model_id: str, messages: List[ProviderMessage], settings: Settings) -> ChatResult:
        payload = self._build_payload(model_id, messages, settings, stream=False)
        try:
            resp = self._session.post(f"{self._base_url}/api/chat", json=payload, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise ProviderError(f"Ollama request failed: {exc}") from exc
        message = data.get("message") or {}
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        total = (prompt_tokens or 0) + (completion_tokens or 0) if (prompt_tokens or completion_tokens) else None
        return ChatResult(
            content=message.get("content") or "",
            reasoning_content=message.get("thinking"),
            usage=ChatUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total),
        )

    def stream_chat(self, model_id: str, messages: List[ProviderMessage], settings: Settings) -> Iterator[StreamDelta]:
        payload = self._build_payload(model_id, messages, settings, stream=True)
        content_acc: List[str] = []
        reasoning_acc: List[str] = []
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        try:
            resp = self._session.post(
                f"{self._base_url}/api/chat", json=payload, timeout=self._timeout, stream=True
            )
            resp.raise_for_status()
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                # Ollama отдаёт поток как NDJSON (по объекту на строку), а не
                # как SSE с префиксом "data:" — в отличие от DeepSeek.
                try:
                    chunk = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                message = chunk.get("message") or {}
                if message.get("content"):
                    content_acc.append(message["content"])
                    yield StreamDelta(content=message["content"])
                if message.get("thinking"):
                    reasoning_acc.append(message["thinking"])
                    yield StreamDelta(reasoning_content=message["thinking"])
                if chunk.get("done"):
                    prompt_tokens = chunk.get("prompt_eval_count")
                    completion_tokens = chunk.get("eval_count")
                    break
        except requests.RequestException as exc:
            raise ProviderError(f"Ollama request failed: {exc}") from exc
        finally:
            try:
                resp.close()
            except Exception:
                pass

        total = (prompt_tokens or 0) + (completion_tokens or 0) if (prompt_tokens or completion_tokens) else None
        yield StreamDelta(
            done=True,
            result=ChatResult(
                content="".join(content_acc),
                reasoning_content="".join(reasoning_acc) or None,
                usage=ChatUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total),
            ),
        )

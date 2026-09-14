"""
deepseek_agent.client
======================

Низкоуровневый клиент DeepSeek API с минимальными зависимостями. Предоставляет
доступ ко всем документированным параметрам запроса: chat-завершения
(включая потоковую передачу, JSON-режим, вызов функций/инструментов,
работу с изображениями, режим "размышления"/уровень усилий рассуждения),
FIM-завершение (fill-in-the-middle), завершение с префиксом ответа ассистента,
получение списка моделей и баланса аккаунта.

Этот модуль ничего не знает об "агентах"/"чатах" как о сохраняемых
сущностях, о внутренней базе данных или об HTTP — это тонкая, не хранящая
состояние обёртка вокруг собственного API DeepSeek. Используется как один
из адаптеров провайдера LLM в `agents_core.providers.deepseek`; сущности
агента/чата/сообщения/настроек, их хранение в SQLite и HTTP API AgentsCore
для управления ими находятся в `agents_core.models` / `.db` / `.repository`
/ `.api` — см. README пакета.

DeepSeek совместим с OpenAI, но этот модуль обращается к нему напрямую по
HTTP через `requests`, так что SDK `openai` не нужен. Он рассчитан на
текущее семейство моделей DeepSeek-V4 (`deepseek-v4-flash`, `deepseek-v4-pro`,
`deepseek-v4-flash-vision-exp`). Устаревшие имена `deepseek-chat` /
`deepseek-reasoner` были выведены из эксплуатации 2026-07-24 и оставлены
здесь только как устаревшие псевдонимы, отображаемые на новые имена, чтобы
старый код не начал молча вызывать несуществующую модель.

Справка: https://api-docs.deepseek.com/

Быстрый старт
-------------
    from deepseek_agent.client import DeepSeekClient, Model, user_message, system_message

    client = DeepSeekClient(api_key="sk-...")  # или задайте переменную окружения DEEPSEEK_API_KEY

    resp = client.chat(
        messages=[
            system_message("You are a concise assistant."),
            user_message("What is the capital of France?"),
        ],
        model=Model.V4_FLASH,
        temperature=0.7,
    )
    print(resp["choices"][0]["message"]["content"])

Потоковая передача:
    for delta in client.stream_chat(messages=[user_message("Count to 5.")]):
        print(delta.content or "", end="", flush=True)

Режим рассуждения ("thinking"):
    resp = client.chat(
        messages=[user_message("Solve: 12 * 13 - 7")],
        model=Model.V4_PRO,
        thinking=ThinkingConfig(type="enabled", reasoning_effort="high"),
    )
    print(resp["choices"][0]["message"].get("reasoning_content"))
    print(resp["choices"][0]["message"]["content"])
"""

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Generator, Iterable, List, Optional, Union

import requests

__all__ = [
    "Model",
    "ThinkingConfig",
    "ResponseFormat",
    "StreamOptions",
    "StreamDelta",
    "DeepSeekError",
    "DeepSeekAPIError",
    "DeepSeekClient",
    "system_message",
    "user_message",
    "assistant_message",
    "tool_message",
    "image_content",
    "text_content",
    "define_tool",
]

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_BETA_BASE_URL = "https://api.deepseek.com/beta"
DEFAULT_TIMEOUT = 120
API_KEY_ENV_VAR = "DEEPSEEK_API_KEY"

# Псевдонимы устаревших имён -> текущие модели. Устаревшие имена начали
# возвращать жёсткие ошибки с 2026-07-24T15:59Z; отображение их здесь
# означает, что старый код продолжает работать (обращаясь к эквивалентной
# текущей модели), а не падает с ошибкой.
_LEGACY_MODEL_ALIASES = {
    "deepseek-chat": "deepseek-v4-flash",
    "deepseek-reasoner": "deepseek-v4-flash",  # вместо этого используйте thinking=enabled
}


class Model(str, Enum):
    """Идентификаторы текущих моделей DeepSeek."""

    V4_FLASH = "deepseek-v4-flash"
    V4_PRO = "deepseek-v4-pro"
    V4_FLASH_VISION_EXP = "deepseek-v4-flash-vision-exp"

    def __str__(self) -> str:  # чтобы f"{Model.V4_FLASH}" == "deepseek-v4-flash"
        return self.value


@dataclass
class ThinkingConfig:
    """Управляет режимом рассуждения ("thinking") DeepSeek-V4.

    type: "enabled" или "disabled" (по умолчанию на стороне сервера —
        "enabled" для моделей, которые это поддерживают).
    reasoning_effort: "low" | "high" | "max". Имеет смысл только при
        type == "enabled".

    Примечание: пока режим размышления включён, API молча игнорирует
    temperature/top_p/presence_penalty/frequency_penalty (они принимаются,
    но не влияют на результат) — этот клиент всё же позволяет их передавать,
    чтобы не нужно было делать особый случай в местах вызова.
    """

    type: str = "enabled"  # "enabled" | "disabled"
    reasoning_effort: Optional[str] = None  # "low" | "high" | "max"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"type": self.type}
        if self.reasoning_effort is not None:
            d["reasoning_effort"] = self.reasoning_effort
        return d


@dataclass
class ResponseFormat:
    """`{"type": "text"}` (по умолчанию) или `{"type": "json_object"}` (JSON-режим).

    При использовании json_object модели также нужно явно указать (в
    системном или пользовательском сообщении) генерировать JSON, иначе она
    может генерировать пробелы, пока не будет достигнут лимит токенов.
    """

    type: str = "text"  # "text" | "json_object"

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type}


@dataclass
class StreamOptions:
    """Дополнительные опции, отправляемые вместе с `stream=True`."""

    include_usage: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"include_usage": self.include_usage}


@dataclass
class StreamDelta:
    """Один разобранный фрагмент потокового ответа."""

    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    raw: Optional[Dict[str, Any]] = None


class DeepSeekError(Exception):
    """Возбуждается при локальных ошибках на стороне клиента (некорректный ввод, отсутствие ключа и т.п.)."""


class DeepSeekAPIError(DeepSeekError):
    """Возбуждается, когда DeepSeek API возвращает ответ со статусом, отличным от 2xx."""

    def __init__(self, status_code: int, message: str, body: Any = None):
        super().__init__(f"DeepSeek API error {status_code}: {message}")
        self.status_code = status_code
        self.message = message
        self.body = body


# --------------------------------------------------------------------------
# Вспомогательные функции для сообщений / содержимого
# --------------------------------------------------------------------------

def system_message(content: str) -> Dict[str, Any]:
    return {"role": "system", "content": content}


def text_content(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text}


def image_content(url_or_path: str, detail: Optional[str] = None) -> Dict[str, Any]:
    """Строит часть содержимого с изображением для модели vision.

    `url_or_path` может быть http(s)-адресом либо путём к локальному файлу
    (в этом случае он автоматически кодируется в base64 и превращается в
    data: URI).
    """
    if url_or_path.startswith("http://") or url_or_path.startswith("https://") or url_or_path.startswith("data:"):
        url = url_or_path
    else:
        import base64
        import mimetypes

        mime, _ = mimetypes.guess_type(url_or_path)
        mime = mime or "image/png"
        with open(url_or_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        url = f"data:{mime};base64,{encoded}"

    image_url: Dict[str, Any] = {"url": url}
    if detail is not None:
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}


def user_message(content: Union[str, List[Dict[str, Any]]], images: Optional[List[str]] = None) -> Dict[str, Any]:
    """Строит пользовательское сообщение. Передайте `images` (пути или URL), чтобы использовать модель vision."""
    if images:
        parts: List[Dict[str, Any]] = []
        if isinstance(content, str):
            parts.append(text_content(content))
        else:
            parts.extend(content)
        parts.extend(image_content(img) for img in images)
        return {"role": "user", "content": parts}
    return {"role": "user", "content": content}


def assistant_message(content: str, prefix: bool = False, reasoning_content: Optional[str] = None) -> Dict[str, Any]:
    """Строит сообщение ассистента. Установите `prefix=True` для завершения
    с префиксом ответа (Chat Prefix Completion; требует вызова
    `DeepSeekClient.chat(..., beta=True)`)."""
    msg: Dict[str, Any] = {"role": "assistant", "content": content}
    if prefix:
        msg["prefix"] = True
    if reasoning_content is not None:
        msg["reasoning_content"] = reasoning_content
    return msg


def tool_message(content: str, tool_call_id: str) -> Dict[str, Any]:
    return {"role": "tool", "content": content, "tool_call_id": tool_call_id}


def define_tool(name: str, description: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Строит описание функции-инструмента в формате JSON-Schema для `tools=[...]`.

    Пример:
        define_tool(
            "get_weather",
            "Get the current weather for a city",
            {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
    """
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


# --------------------------------------------------------------------------
# Клиент
# --------------------------------------------------------------------------

class DeepSeekClient:
    """Полнофункциональный клиент DeepSeek API.

    Каждый параметр, документированный DeepSeek API, доступен как именованный
    аргумент `chat()` / `fim_completion()`; ничего не зашито жёстко.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        beta_base_url: str = DEFAULT_BETA_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.5,
        session: Optional[requests.Session] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        self.api_key = api_key or os.environ.get(API_KEY_ENV_VAR)
        if not self.api_key:
            raise DeepSeekError(
                f"No API key provided. Pass api_key=... or set the {API_KEY_ENV_VAR} "
                "environment variable."
            )
        self.base_url = base_url.rstrip("/")
        self.beta_base_url = beta_base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.session = session or requests.Session()
        self.default_headers = default_headers or {}

    # -- внутренние методы ---------------------------------------------------

    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        headers.update(self.default_headers)
        if extra:
            headers.update(extra)
        return headers

    @staticmethod
    def _resolve_model(model: Union[str, Model]) -> str:
        model_str = str(model)
        if model_str in _LEGACY_MODEL_ALIASES:
            warnings.warn(
                f"Model '{model_str}' was retired on 2026-07-24; using "
                f"'{_LEGACY_MODEL_ALIASES[model_str]}' instead. Update your "
                "code to use the Model enum / new model IDs, and use "
                "thinking=ThinkingConfig(type='enabled') instead of "
                "'deepseek-reasoner'.",
                DeprecationWarning,
                stacklevel=3,
            )
            return _LEGACY_MODEL_ALIASES[model_str]
        return model_str

    def _request(
        self,
        method: str,
        url: str,
        json_body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> requests.Response:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.request(
                    method,
                    url,
                    headers=self._headers(),
                    json=json_body,
                    params=params,
                    timeout=self.timeout,
                    stream=stream,
                )
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * (2 ** attempt))
                    continue
                raise DeepSeekError(f"Network error calling DeepSeek API: {exc}") from exc

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < self.max_retries:
                    retry_after = resp.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after else self.retry_backoff_seconds * (2 ** attempt)
                    time.sleep(delay)
                    continue

            if not resp.ok:
                body: Any
                try:
                    body = resp.json()
                    message = body.get("error", {}).get("message", resp.text)
                except ValueError:
                    body = resp.text
                    message = resp.text
                raise DeepSeekAPIError(resp.status_code, message, body)

            return resp

        # По идее, сюда дойти невозможно, но это устраивает статические анализаторы типов.
        raise DeepSeekError(f"Request failed after retries: {last_exc}")

    @staticmethod
    def _build_chat_payload(
        messages: List[Dict[str, Any]],
        model: Union[str, Model],
        *,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Union[ResponseFormat, Dict[str, Any]]] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        stream_options: Optional[Union[StreamOptions, Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        user_id: Optional[str] = None,
        thinking: Optional[Union[ThinkingConfig, Dict[str, Any]]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": DeepSeekClient._resolve_model(model),
            "messages": messages,
        }

        def put(key: str, value: Any, transform=None):
            if value is not None:
                payload[key] = transform(value) if transform else value

        if frequency_penalty is not None:
            warnings.warn("frequency_penalty is deprecated by DeepSeek; it is sent but may be ignored.", DeprecationWarning, stacklevel=3)
        if presence_penalty is not None:
            warnings.warn("presence_penalty is deprecated by DeepSeek; it is sent but may be ignored.", DeprecationWarning, stacklevel=3)

        put("frequency_penalty", frequency_penalty)
        put("presence_penalty", presence_penalty)
        put("max_tokens", max_tokens)
        put("response_format", response_format, lambda v: v.to_dict() if isinstance(v, ResponseFormat) else v)
        put("stop", stop)
        put("stream", stream if stream else None)
        put("stream_options", stream_options, lambda v: v.to_dict() if isinstance(v, StreamOptions) else v)
        put("temperature", temperature)
        put("top_p", top_p)
        put("tools", tools)
        put("tool_choice", tool_choice)
        put("logprobs", logprobs)
        put("top_logprobs", top_logprobs)
        put("user_id", user_id)
        put("thinking", thinking, lambda v: v.to_dict() if isinstance(v, ThinkingConfig) else v)

        if extra_params:
            payload.update(extra_params)

        return payload

    # -- chat-завершения ---------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Union[str, Model] = Model.V4_FLASH,
        *,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Union[ResponseFormat, Dict[str, Any]]] = None,
        stop: Optional[Union[str, List[str]]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        user_id: Optional[str] = None,
        thinking: Optional[Union[ThinkingConfig, Dict[str, Any]]] = None,
        beta: bool = False,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Непотоковое chat-завершение. Возвращает разобранный JSON-ответ как есть.

        Установите `beta=True`, чтобы обратиться к бета-эндпоинту (требуется
        для завершения с префиксом ответа, то есть когда у последнего
        сообщения задано `"prefix": True`).
        """
        payload = self._build_chat_payload(
            messages,
            model,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            response_format=response_format,
            stop=stop,
            stream=False,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            user_id=user_id,
            thinking=thinking,
            extra_params=extra_params,
        )
        base = self.beta_base_url if beta else self.base_url
        resp = self._request("POST", f"{base}/chat/completions", json_body=payload)
        return resp.json()

    def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        model: Union[str, Model] = Model.V4_FLASH,
        *,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Union[ResponseFormat, Dict[str, Any]]] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream_options: Optional[Union[StreamOptions, Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        user_id: Optional[str] = None,
        thinking: Optional[Union[ThinkingConfig, Dict[str, Any]]] = None,
        beta: bool = False,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Generator[StreamDelta, None, None]:
        """Потоковое chat-завершение. Возвращает (через yield) объекты
        `StreamDelta` по мере поступления событий server-sent events."""
        payload = self._build_chat_payload(
            messages,
            model,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            response_format=response_format,
            stop=stop,
            stream=True,
            stream_options=stream_options,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            user_id=user_id,
            thinking=thinking,
            extra_params=extra_params,
        )
        base = self.beta_base_url if beta else self.base_url
        resp = self._request("POST", f"{base}/chat/completions", json_body=payload, stream=True)
        yield from self._iter_sse(resp)

    @staticmethod
    def _iter_sse(resp: requests.Response) -> Generator[StreamDelta, None, None]:
        try:
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                if not raw_line.startswith("data:"):
                    continue
                data = raw_line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                delta = choices[0].get("delta", {}) if choices else {}
                finish_reason = choices[0].get("finish_reason") if choices else None
                yield StreamDelta(
                    content=delta.get("content"),
                    reasoning_content=delta.get("reasoning_content"),
                    tool_calls=delta.get("tool_calls"),
                    finish_reason=finish_reason,
                    usage=chunk.get("usage"),
                    raw=chunk,
                )
        finally:
            resp.close()

    # -- FIM-завершение (fill-in-the-middle) (бета) -------------------------

    def fim_completion(
        self,
        prompt: str,
        suffix: Optional[str] = None,
        model: Union[str, Model] = Model.V4_PRO,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = None,
        echo: Optional[bool] = None,
        logprobs: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], Generator[StreamDelta, None, None]]:
        """Бета-эндпоинт автодополнения кода: получив `prompt` (префикс) и
        необязательный `suffix`, модель дополняет середину. Максимум 4K
        выходных токенов. Всегда обращается к бета-базовому URL."""
        payload: Dict[str, Any] = {
            "model": DeepSeekClient._resolve_model(model),
            "prompt": prompt,
        }
        if suffix is not None:
            payload["suffix"] = suffix
        for key, value in (
            ("max_tokens", max_tokens),
            ("temperature", temperature),
            ("top_p", top_p),
            ("stop", stop),
            ("echo", echo),
            ("logprobs", logprobs),
            ("presence_penalty", presence_penalty),
            ("frequency_penalty", frequency_penalty),
        ):
            if value is not None:
                payload[key] = value
        if stream:
            payload["stream"] = True
        if extra_params:
            payload.update(extra_params)

        resp = self._request(
            "POST", f"{self.beta_base_url}/completions", json_body=payload, stream=stream
        )
        if stream:
            return self._iter_sse(resp)
        return resp.json()

    # -- аккаунт / метаданные --------------------------------------------------

    def list_models(self) -> Dict[str, Any]:
        """GET /models — список моделей, доступных для данного API-ключа."""
        resp = self._request("GET", f"{self.base_url}/models")
        return resp.json()

    def get_balance(self) -> Dict[str, Any]:
        """GET /user/balance — текущий баланс аккаунта."""
        resp = self._request("GET", f"{self.base_url}/user/balance")
        return resp.json()

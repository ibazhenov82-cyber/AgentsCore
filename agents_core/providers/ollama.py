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
import logging
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional

import requests

from ..logging_setup import truncate
from ..models import Settings
from .base import BaseProvider, ChatResult, ChatUsage, ModelCapabilities, ProviderError, ProviderMessage, StreamDelta

# См. `agents_core.providers.deepseek_client` — тот же принцип, но этот
# модуль и так уже не самостоятелен (импортирует `..models`), поэтому здесь
# можно использовать общие хелперы усечения/маскирования из
# `agents_core.logging_setup` напрямую.
_logger = logging.getLogger("agents_core.llm.ollama")


def _log_call(method: str, url: str, *, json_body: Optional[Dict[str, Any]] = None) -> float:
    if json_body is not None:
        try:
            body_repr = truncate(json.dumps(json_body, ensure_ascii=False))
        except (TypeError, ValueError):
            body_repr = truncate(str(json_body))
    else:
        body_repr = "-"
    _logger.info("-> %s %s json=%s", method, url, body_repr)
    return time.monotonic()


def _log_response(method: str, url: str, started: float, status_code: Any, *, note: str = "") -> None:
    duration_ms = (time.monotonic() - started) * 1000
    _logger.info("<- %s %s %s (%.1f мс)%s", method, url, status_code, duration_ms, f" {note}" if note else "")


def _log_error(method: str, url: str, started: float, exc: Exception) -> None:
    duration_ms = (time.monotonic() - started) * 1000
    _logger.error("<- %s %s ОШИБКА (%.1f мс): %s", method, url, duration_ms, exc)


_DEFAULT_TIMEOUT = 120

#: Сколько держать модель в памяти после запроса: -1 — до явной остановки
#: (кнопка «Остановить» или конкурентный запуск другой модели, см.
#: `agents_core.local_models`). Передаётся в КАЖДОМ запросе: иначе Ollama после
#: любого запроса сбрасывает срок на свои 5 минут.
KEEP_ALIVE = -1

#: Запуск (загрузка в память) большой модели с диска может идти минутами.
_LOAD_TIMEOUT = 600

# Нативный размер контекстного окна семейства Qwen3 (в токенах) — Ollama не
# всегда возвращает его явным ключом для каждой версии, поэтому это разумное
# приближение, используемое как fallback, если `/api/show` недоступен или не
# содержит ключа `*.context_length`.
_QWEN3_FALLBACK_CONTEXT_WINDOW = 32768


def memory_stats(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Память загруженной модели из записи `GET /api/ps`: `memory_mb`,
    `vram_mb`, `gpu_percent`, `num_ctx` (если Ollama сообщает окно)."""
    stats: Dict[str, Any] = {}
    size = entry.get("size")
    size_vram = entry.get("size_vram")
    if isinstance(size, (int, float)) and size > 0:
        stats["memory_mb"] = int(size / (1024 * 1024))
        if isinstance(size_vram, (int, float)):
            stats["vram_mb"] = int(size_vram / (1024 * 1024))
            stats["gpu_percent"] = int(round(100 * size_vram / size))
    if isinstance(entry.get("context_length"), int):
        stats["num_ctx"] = entry["context_length"]
    return stats


class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(self, base_url: str, timeout: float = _DEFAULT_TIMEOUT):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        #: Кэш `POST /api/show` по имени модели: {"digest", "data"}. Сбрасывается
        #: для модели, если `GET /api/tags` показал другой digest (модель
        #: перекачали под тем же тегом).
        self._show_cache: Dict[str, Dict[str, Any]] = {}
        self._show_lock = threading.Lock()
        #: Вызывается перед КАЖДЫМ запросом к модели (chat/stream_chat) —
        #: `LocalModelManager.ensure_ready`: запуск модели, если она не
        #: запущена, и остановка других при конкурентном запуске.
        self.before_request: Optional[Callable[[str, Settings], None]] = None

    # ---- управление моделями (День 29) --------------------------------------

    def _get_json(self, path: str, timeout: float = 10) -> Dict[str, Any]:
        url = f"{self._base_url}{path}"
        started = _log_call("GET", url)
        try:
            resp = self._session.get(url, timeout=timeout)
            _log_response("GET", url, started, resp.status_code)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            _log_error("GET", url, started, exc)
            raise ProviderError(f"Ollama недоступна ({self._base_url}): {exc}") from exc

    def list_tags(self) -> List[Dict[str, Any]]:
        """`GET /api/tags` — все скачанные модели (как `ollama list`)."""
        models = self._get_json("/api/tags").get("models") or []
        for entry in models:
            name = entry.get("model") or entry.get("name")
            cached = self._show_cache.get(name)
            if cached is not None and cached.get("digest") != entry.get("digest"):
                self._show_cache.pop(name, None)
        return models

    def running(self) -> List[Dict[str, Any]]:
        """`GET /api/ps` — модели, загруженные в память (как `ollama ps`)."""
        return self._get_json("/api/ps", timeout=5).get("models") or []

    def show(self, model_id: str, digest: Optional[str] = None) -> Dict[str, Any]:
        """`POST /api/show` с кэшем; ошибка — `ProviderError`."""
        with self._show_lock:
            cached = self._show_cache.get(model_id)
            if cached is not None and (digest is None or cached.get("digest") == digest):
                return cached["data"]
        url = f"{self._base_url}/api/show"
        started = _log_call("POST", url, json_body={"model": model_id})
        try:
            resp = self._session.post(url, json={"model": model_id}, timeout=15)
            _log_response("POST", url, started, resp.status_code)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            _log_error("POST", url, started, exc)
            raise ProviderError(f"Ollama /api/show {model_id}: {exc}") from exc
        with self._show_lock:
            self._show_cache[model_id] = {"digest": digest, "data": data}
        return data

    def capabilities(self, model_id: str, digest: Optional[str] = None) -> List[str]:
        """Возможности модели из `/api/show` («completion», «tools»,
        «thinking», «embedding»…). Старые версии Ollama поля не отдают —
        тогда embedding-модель угадывается по имени/семейству."""
        try:
            data = self.show(model_id, digest)
        except ProviderError:
            data = {}
        caps = data.get("capabilities")
        if isinstance(caps, list) and caps:
            return [str(c) for c in caps]
        family = str((data.get("details") or {}).get("family") or "").lower()
        if "embed" in model_id.lower() or "bert" in family or "embed" in family:
            return ["embedding"]
        return ["completion"]

    def is_chat_model(self, model_id: str, digest: Optional[str] = None) -> bool:
        return "completion" in self.capabilities(model_id, digest)

    def load(self, model_id: str, num_ctx: Optional[int] = None) -> Dict[str, Any]:
        """Запуск — загрузка модели в память пустым запросом `POST
        /api/generate` (как `ollama run <модель> ""`); держится до остановки.
        `num_ctx` — с каким окном загрузить (иначе первый запрос с другим
        окном перезагрузит модель)."""
        body: Dict[str, Any] = {"model": model_id, "keep_alive": KEEP_ALIVE}
        if num_ctx:
            body["options"] = {"num_ctx": num_ctx}
        url = f"{self._base_url}/api/generate"
        started = _log_call("POST", url, json_body=body)
        try:
            resp = self._session.post(url, json=body, timeout=_LOAD_TIMEOUT)
            _log_response("POST", url, started, resp.status_code, note="запуск модели")
            if resp.status_code == 404:
                raise ProviderError(f"модель {model_id} не скачана в Ollama (ollama pull {model_id})")
            resp.raise_for_status()
            return resp.json() if resp.content else {}
        except requests.RequestException as exc:
            _log_error("POST", url, started, exc)
            raise ProviderError(f"не удалось запустить модель {model_id}: {exc}") from exc

    def pull(self, model_id: str, cancel: Optional[threading.Event] = None) -> Iterator[Dict[str, Any]]:
        """Скачивание `POST /api/pull` (как `ollama pull`): поток объектов
        `{"status", "digest", "total", "completed"}`; ошибка — `ProviderError`.
        `cancel` установлен — соединение закрывается, скачивание прерывается
        (Ollama продолжит с места остановки при следующем pull)."""
        url = f"{self._base_url}/api/pull"
        body = {"model": model_id, "stream": True}
        started = _log_call("POST", url, json_body=body)
        try:
            resp = self._session.post(url, json=body, timeout=(10, 300), stream=True)
        except requests.RequestException as exc:
            _log_error("POST", url, started, exc)
            raise ProviderError(f"не удалось скачать {model_id}: {exc}") from exc
        try:
            if not resp.ok:
                raise ProviderError(f"не удалось скачать {model_id}: {resp.status_code} {resp.text[:300]}")
            for raw_line in resp.iter_lines(decode_unicode=True):
                if cancel is not None and cancel.is_set():
                    raise ProviderError("скачивание отменено")
                if not raw_line:
                    continue
                try:
                    chunk = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    raise ProviderError(f"не удалось скачать {model_id}: {chunk['error']}")
                yield chunk
        except requests.RequestException as exc:
            raise ProviderError(f"скачивание {model_id} прервано: {exc}") from exc
        finally:
            resp.close()
            _log_response("POST", url, started, "done", note="скачивание")

    def delete(self, model_id: str) -> None:
        """Удаление с диска `DELETE /api/delete` (как `ollama rm`)."""
        url = f"{self._base_url}/api/delete"
        started = _log_call("DELETE", url, json_body={"model": model_id})
        try:
            resp = self._session.delete(url, json={"model": model_id}, timeout=60)
            _log_response("DELETE", url, started, resp.status_code)
            if resp.status_code == 404:
                return
            resp.raise_for_status()
        except requests.RequestException as exc:
            _log_error("DELETE", url, started, exc)
            raise ProviderError(f"не удалось удалить модель {model_id}: {exc}") from exc

    @property
    def base_url(self) -> str:
        return self._base_url

    def unload(self, model_id: str) -> None:
        """Остановка — выгрузка из памяти (`keep_alive: 0`, как `ollama stop`)."""
        body = {"model": model_id, "keep_alive": 0}
        url = f"{self._base_url}/api/generate"
        started = _log_call("POST", url, json_body=body)
        try:
            resp = self._session.post(url, json=body, timeout=60)
            _log_response("POST", url, started, resp.status_code, note="остановка модели")
            if resp.status_code != 404:
                resp.raise_for_status()
        except requests.RequestException as exc:
            _log_error("POST", url, started, exc)
            raise ProviderError(f"не удалось остановить модель {model_id}: {exc}") from exc

    def health(self) -> bool:
        url = f"{self._base_url}/api/tags"
        started = _log_call("GET", url)
        try:
            resp = self._session.get(url, timeout=10)
            _log_response("GET", url, started, resp.status_code)
            return resp.ok
        except requests.RequestException as exc:
            _log_error("GET", url, started, exc)
            return False

    def model_health(self, model_id: str) -> bool:
        """Модель считается запущенной, если она присутствует в ответе
        `GET /api/tags` (поле `model` каждой записи), а не только если сам
        сервер Ollama отвечает — сервер может быть жив, но конкретная модель
        ещё не подтянута/переименована."""
        url = f"{self._base_url}/api/tags"
        started = _log_call("GET", url)
        try:
            resp = self._session.get(url, timeout=10)
            _log_response("GET", url, started, resp.status_code)
            if not resp.ok:
                return False
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            _log_error("GET", url, started, exc)
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
            data = self.show(model_id)
        except ProviderError:
            return fallback

        model_info: Dict[str, Any] = data.get("model_info") or {}
        details = data.get("details") or {}
        family = str(details.get("family") or "").lower()
        is_thinking_capable = is_qwen3 or "qwen3" in family
        caps = data.get("capabilities")
        if isinstance(caps, list) and caps:
            # Новые версии Ollama сообщают возможности модели явно.
            supports_thinking = "thinking" in caps
            supports_tools = "tools" in caps
        else:
            supports_thinking = supports_tools = is_thinking_capable

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
            supports_thinking=supports_thinking,
            supports_tools=supports_tools,
            supports_json_mode=True,
            supports_logprobs=False,
        )

    def runtime_stats(self, model_id: str) -> Optional[Dict[str, Any]]:
        """`GET /api/ps` — загруженные модели: `size` (байт в памяти всего),
        `size_vram` (из них в видеопамяти), `context_length` (фактическое
        окно; есть в новых версиях Ollama). Ошибка или модель не загружена —
        `None`: метрики памяти просто не попадут в ответ."""
        try:
            entries = self.running()
        except ProviderError:
            return None
        for entry in entries:
            if entry.get("model") != model_id and entry.get("name") != model_id:
                continue
            return memory_stats(entry) or None
        return None

    @staticmethod
    def _usage_from(data: Dict[str, Any]) -> ChatUsage:
        """usage + тайминги из финального объекта ответа Ollama (длительности
        у Ollama — в наносекундах)."""
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        total = (prompt_tokens or 0) + (completion_tokens or 0) if (prompt_tokens or completion_tokens) else None

        def ms(key: str) -> Optional[float]:
            value = data.get(key)
            return value / 1_000_000 if isinstance(value, (int, float)) else None

        return ChatUsage(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total,
            eval_duration_ms=ms("eval_duration"), prompt_eval_duration_ms=ms("prompt_eval_duration"),
            load_duration_ms=ms("load_duration"),
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
            # Предел длины ОТВЕТА — не путать с окном контекста ниже.
            options["num_predict"] = settings.max_tokens
        if settings.num_ctx is not None:
            # Окно контекста целиком (prompt + ответ). Смена значения
            # перезагружает модель в Ollama — первый ответ после смены дольше.
            options["num_ctx"] = settings.num_ctx
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

    @staticmethod
    def _build_ollama_messages(messages: List[ProviderMessage]) -> List[Dict[str, Any]]:
        """Ollama ожидает `tool_calls` с `function.arguments` РАЗОБРАННЫМ
        словарём (не JSON-текстом, как у DeepSeek/OpenAI) — наш внутренний
        формат (`ProviderMessage.tool_calls`) всегда хранит `arguments` как
        JSON-текст для единообразия между провайдерами, поэтому здесь он
        разбирается обратно перед отправкой. У результата вызова (role="tool")
        Ollama сопоставляет ответ вызову по имени функции (`tool_name`), а не
        по id — своих id для tool_calls у Ollama нет."""
        out: List[Dict[str, Any]] = []
        for m in messages:
            d: Dict[str, Any] = {"role": m.role, "content": m.content}
            if m.tool_calls:
                native_calls = []
                for call in m.tool_calls:
                    fn = call.get("function", {})
                    args = fn.get("arguments")
                    try:
                        parsed_args = json.loads(args) if isinstance(args, str) else (args or {})
                    except (ValueError, TypeError):
                        parsed_args = {}
                    native_calls.append({"function": {"name": fn.get("name", ""), "arguments": parsed_args}})
                d["tool_calls"] = native_calls
            if m.role == "tool" and m.name:
                d["tool_name"] = m.name
            out.append(d)
        return out

    def _build_payload(self, model_id: str, messages: List[ProviderMessage], settings: Settings, stream: bool) -> Dict[str, Any]:
        caps = self.discover_model(model_id)
        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": self._build_ollama_messages(messages),
            "stream": stream,
            "options": self._build_options(settings),
            "keep_alive": KEEP_ALIVE,
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

    @staticmethod
    def _normalize_tool_calls(native_calls: Optional[list]) -> Optional[list]:
        """Приводит нативный ответ Ollama (`function.arguments` — уже
        разобранный словарь, id вызовов отсутствуют) к единому внутреннему
        формату (id синтетический, `function.arguments` — JSON-текст) —
        см. докстринг `ProviderMessage.tool_calls`."""
        if not native_calls:
            return None
        result = []
        for i, call in enumerate(native_calls):
            fn = call.get("function", {})
            args = fn.get("arguments")
            args_text = json.dumps(args, ensure_ascii=False) if not isinstance(args, str) else args
            result.append({
                "id": call.get("id") or f"ollama-call-{i}",
                "type": "function",
                "function": {"name": fn.get("name", ""), "arguments": args_text},
            })
        return result

    def _before(self, model_id: str, settings: Settings) -> None:
        if self.before_request is not None:
            self.before_request(model_id, settings)

    def chat(self, model_id: str, messages: List[ProviderMessage], settings: Settings) -> ChatResult:
        self._before(model_id, settings)
        payload = self._build_payload(model_id, messages, settings, stream=False)
        url = f"{self._base_url}/api/chat"
        started = _log_call("POST", url, json_body=payload)
        try:
            resp = self._session.post(url, json=payload, timeout=self._timeout)
            resp.raise_for_status()
            _log_response("POST", url, started, resp.status_code)
            data = resp.json()
        except requests.RequestException as exc:
            _log_error("POST", url, started, exc)
            raise ProviderError(f"Ollama request failed: {exc}") from exc
        message = data.get("message") or {}
        return ChatResult(
            content=message.get("content") or "",
            reasoning_content=message.get("thinking"),
            usage=self._usage_from(data),
            tool_calls=self._normalize_tool_calls(message.get("tool_calls")),
        )

    def stream_chat(self, model_id: str, messages: List[ProviderMessage], settings: Settings) -> Iterator[StreamDelta]:
        self._before(model_id, settings)
        payload = self._build_payload(model_id, messages, settings, stream=True)
        url = f"{self._base_url}/api/chat"
        started = _log_call("POST", url, json_body=payload)
        content_acc: List[str] = []
        reasoning_acc: List[str] = []
        tool_calls_native: List[dict] = []
        final_chunk: Dict[str, Any] = {}
        try:
            resp = self._session.post(
                f"{self._base_url}/api/chat", json=payload, timeout=self._timeout, stream=True
            )
            resp.raise_for_status()
            _log_response("POST", url, started, resp.status_code, note="начат поток")
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
                if message.get("tool_calls"):
                    # В отличие от DeepSeek, Ollama не дробит `tool_calls` на
                    # символьные фрагменты — вызов приходит уже полностью
                    # сформированным в одном чанке, накопление не требуется.
                    tool_calls_native.extend(message["tool_calls"])
                if chunk.get("done"):
                    final_chunk = chunk
                    break
        except requests.RequestException as exc:
            _log_error("POST", url, started, exc)
            raise ProviderError(f"Ollama request failed: {exc}") from exc
        finally:
            try:
                resp.close()
            except Exception:
                pass

        _log_response("POST", url, started, "done", note="поток завершён")
        yield StreamDelta(
            done=True,
            result=ChatResult(
                content="".join(content_acc),
                reasoning_content="".join(reasoning_acc) or None,
                usage=self._usage_from(final_chunk),
                tool_calls=self._normalize_tool_calls(tool_calls_native),
            ),
        )

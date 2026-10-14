"""
agents_core.mcp_client
=========================

Клиент к MCP-серверам по протоколу MCP (Streamable HTTP). Серверы
продукта (`MCP_SERVERS`) работают без сессий — рукопожатие им не нужно.
Внешние серверы (`MCP_SERVERS_FILE`, например GitHub MCP) подключаются с
рукопожатием `initialize`, сессией `Mcp-Session-Id`, своими заголовками и
разбором ответов-потоков SSE (см. `McpServerSpec`, `MCPClient._rpc`).

Исходно — клиент к отдельному MCP-серверу (третий, самостоятельно разворачиваемый
компонент — см. `mcp_server/README.md`), подключаемому по протоколу MCP
через Streamable HTTP (**stateless**: сервер
собирается с `stateless_http=True, json_response=True` именно для того,
чтобы каждый запрос был самодостаточным JSON-RPC поверх обычного POST, без
хендшейка `initialize`/долгоживущей сессии и без SSE-потока в ответе —
благодаря этому клиенту достаточно синхронного `requests`, как и
остальным HTTP-клиентам AgentsCore (см. `providers/deepseek_client.py`,
тот же стиль retry/backoff), не нужен полноценный async MCP SDK на стороне
AgentsCore, который потребовал бы отдельного event loop внутри
синхронного `Repository`).

Используется в двух точках `Repository` (оба — необязательные, включаются
только при `mcp_client is not None`):
  * `_settings_with_merged_tools` — дописывает в список инструментов запроса
    к модели те, что отдаёт `list_tools()` (в формате OpenAI function-tool).
  * `_execute_tool_call` — если имя вызванной функции не входит в
    `_TOOL_HANDLERS`, но входит в последний успешно полученный список MCP
    (внутренний кэш ниже), вызов уходит в `call_tool()`.

Кэш инструментов
-----------------
`list_tools()` запоминает последний УСПЕШНО полученный список и отдаёт его
же, если MCP-сервер в моменте недоступен (сеть упала, таймаут, 5xx) —
конкретно ради того, чтобы недоступность отдельного, необязательного
MCP-сервера не ломала обычный чат: чат просто продолжает работать с тем
набором инструментов, что было известно последний раз (в худшем случае —
пустым, если сервер ни разу не отвечал успешно). Это НЕ TTL-кэш и не
персистентный кэш — только в памяти процесса, живёт вместе с
`MCPClient`/`Repository`."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import requests

_logger = logging.getLogger("agents_core.mcp_client")

#: Версия протокола MCP, которую клиент предлагает в `initialize`
#: (сервер может ответить другой поддерживаемой — она и используется дальше).
MCP_PROTOCOL_VERSION = "2025-06-18"
_SERVER_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass
class McpServerSpec:
    """Подключаемый MCP-сервер из файла `MCP_SERVERS_FILE` (внешние серверы,
    например GitHub MCP): адрес используется как есть, свои заголовки
    (авторизация, настройки сервера), подпись `title` для интерфейса,
    рукопожатие `initialize` по спецификации Streamable HTTP."""

    name: str
    url: str
    headers: Dict[str, str] = field(default_factory=dict)
    title: Optional[str] = None
    initialize: bool = True


def _substitute_env(value: str, where: str) -> str:
    def repl(match: "re.Match[str]") -> str:
        var = match.group(1)
        if var not in os.environ:
            raise ValueError(f"{where}: переменная окружения {var} не задана")
        return os.environ[var]

    return _ENV_REF_RE.sub(repl, value)


def load_mcp_servers_file(path: str) -> List[McpServerSpec]:
    """Читает JSON вида `{"github": {"url": "...", "title": "GitHub",
    "headers": {"Authorization": "Bearer ${GITHUB_MCP_TOKEN}"}}}`.
    `${VAR}` в адресе и заголовках подставляется из окружения — секреты
    не хранятся в самом файле."""
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except OSError as exc:
        raise ValueError(f"MCP_SERVERS_FILE: не удалось прочитать {path}: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"MCP_SERVERS_FILE: {path} — некорректный JSON: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise ValueError("MCP_SERVERS_FILE: ожидается непустой объект {\"имя\": {\"url\": ...}}")
    specs = []
    for name, entry in raw.items():
        if not _SERVER_NAME_RE.match(name):
            raise ValueError(f"MCP_SERVERS_FILE: имя сервера {name!r} — только латиница, цифры, _ и -")
        if not isinstance(entry, dict) or not entry.get("url"):
            raise ValueError(f"MCP_SERVERS_FILE: у сервера {name!r} нет url")
        headers = entry.get("headers") or {}
        if not isinstance(headers, dict):
            raise ValueError(f"MCP_SERVERS_FILE: headers сервера {name!r} — объект")
        specs.append(McpServerSpec(
            name=name,
            url=_substitute_env(str(entry["url"]), f"{name}.url"),
            headers={str(k): _substitute_env(str(v), f"{name}.headers.{k}") for k, v in headers.items()},
            title=(str(entry["title"]).strip() or None) if entry.get("title") else None,
            initialize=bool(entry.get("initialize", True)),
        ))
    return specs


def _parse_sse(text: str, request_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Ответ JSON-RPC из потока `text/event-stream` (формат SSE): события
    разделены пустой строкой, данные — строки `data:`; нужен ответ с нашим
    `id`. Строки делятся только по CR/LF (как требует SSE): `str.splitlines()`
    режет и по символам вроде U+2028, которые законно встречаются внутри
    JSON-строк."""
    fallback = None
    data_lines: List[str] = []
    for line in re.split(r"\r\n|\r|\n", text) + [""]:
        if line == "":
            if data_lines:
                message = _json_rpc_response("\n".join(data_lines), request_id)
                data_lines = []
                if message is not None:
                    if request_id is None or message.get("id") == request_id:
                        return message
                    fallback = fallback or message
            continue
        if line.startswith(":"):
            continue  # комментарий / keep-alive
        if line.startswith("data:"):
            value = line[5:]
            data_lines.append(value[1:] if value.startswith(" ") else value)
    return fallback


def _json_rpc_response(data: str, request_id: Optional[str]) -> Optional[Dict[str, Any]]:
    try:
        message = json.loads(data)
    except ValueError:
        return None
    candidates = message if isinstance(message, list) else [message]  # пакетный ответ
    for item in candidates:
        if isinstance(item, dict) and ("result" in item or "error" in item):
            if request_id is None or item.get("id") == request_id or len(candidates) == 1:
                return item
    return None


def normalize_mcp_base_url(url: str) -> str:
    """Приводит адрес MCP-сервера к адресу самого MCP-эндпоинта.

    На практике у пользователей регулярно путаются два разных адреса одного
    и того же сервиса: базовый REST-адрес (`http://host:8001/`, которым
    пользуется AgentsApp напрямую — см. `mcp_server/api.py`, смонтирован под
    `/api`) и адрес самого MCP-протокола (`http://host:8001/mcp` — под этим
    путём `mcp_server/app.py` монтирует `streamable_http_app()`). Если в
    `MCP_SERVER_URL` указан голый базовый адрес без `/mcp`, JSON-RPC запросы
    уходят на несуществующий путь и MCP-сервер отвечает `404 Not Found` —
    именно так это один раз и произошло на практике (лог: "MCP-сервер
    ответил 404"). Чтобы эта опечатка не ломала интеграцию молча, здесь
    домысливается недостающий `/mcp`, если его нет — при этом URL, уже
    оканчивающийся на `/mcp` (в любом регистре), не трогается."""
    normalized = url.rstrip("/")
    if not normalized:
        return normalized
    if normalized.lower().endswith("/mcp"):
        return normalized
    return f"{normalized}/mcp"


class _SessionExpired(Exception):
    """Сервер не знает нашу сессию (404) — нужно новое рукопожатие."""


class MCPClientError(Exception):
    """Сетевая ошибка / неожиданный ответ MCP-сервера при `call_tool` —
    поднимается наружу (в отличие от `list_tools`, у которого есть кэш для
    как раз такого случая). Вызывающий код (`Repository._execute_tool_call`)
    ловит её и превращает в `{"error": ...}`, как и для остальных сбоев
    вызова инструментов — модель получает текст ошибки, а не обрыв ответа."""


class MCPClient:
    """`base_url` — адрес MCP-эндпоинта, например `"http://mcp:8001/mcp"`
    (значение `MCP_SERVER_URL` из конфигурации). Если адрес не оканчивается
    на `/mcp`, он домысливается автоматически (см. `normalize_mcp_base_url`)
    — на практике это самая частая опечатка (путают с базовым REST-адресом
    того же сервиса для AgentsApp), поэтому нормализация здесь, а не только
    в документации `.env.example`.
    `api_key` — заготовка (см. `AgentConfig.MCP_API_KEY`): пока ничего не
    делает, кроме как прикладывается заголовком `Authorization`, если
    задан — сервер сейчас его не проверяет (аутентификация не введена по
    этому ТЗ), но включить проверку на стороне сервера можно будет без
    изменений здесь."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
        session: Optional[requests.Session] = None,
        name: str = "tools",
        headers: Optional[Dict[str, str]] = None,
        title: Optional[str] = None,
        normalize_url: bool = True,
        handshake: bool = False,
    ):
        if not base_url:
            raise ValueError("MCPClient требует непустой base_url")
        if normalize_url:
            normalized = normalize_mcp_base_url(base_url)
            if normalized != base_url.rstrip("/"):
                _logger.info("MCP_SERVER_URL %r не оканчивается на /mcp, использую %r", base_url, normalized)
        else:
            # Внешний сервер (MCP_SERVERS_FILE): адрес ровно как задан.
            normalized = base_url.strip()
        self.base_url = normalized
        #: Свои заголовки сервера (авторизация, X-MCP-Toolsets и т.п.).
        self.extra_headers: Dict[str, str] = dict(headers or {})
        #: Подпись сервера в интерфейсе (иначе — `name`).
        self.title = title
        #: Рукопожатие `initialize` перед первым запросом и сессия
        #: `Mcp-Session-Id` — для серверов, которые их требуют.
        self.handshake = handshake
        self._session_id: Optional[str] = None
        self._protocol_version: Optional[str] = None
        self._initialized = False
        self._init_lock = threading.Lock()
        #: Короткое имя сервера из MCP_SERVERS — подпись инструментов, у
        #: которых сервер не указал группу.
        self.name = name
        self.api_key = api_key or None
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.session = session or requests.Session()
        # Кэш последнего успешного list_tools() — см. докстринг модуля.
        # Хранит уже сконвертированный в OpenAI function-tool формат список
        # (то, что реально уходит в запрос к провайдеру) плюс "сырые" имена
        # инструментов отдельно (нужны `_execute_tool_call`, чтобы проверить
        # "это имя вообще известно как MCP-инструмент?" без повторного
        # разбора структуры function-tool на каждый вызов).
        self._cached_tools: List[Dict[str, Any]] = []
        self._cached_tool_names: set = set()
        #: Инструменты как их отдаёт сервер (с `title`) — для списка в интерфейсе.
        self._cached_raw_tools: List[Dict[str, Any]] = []

    # -- внутреннее ---------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        if self._protocol_version:
            headers["MCP-Protocol-Version"] = self._protocol_version
        return headers

    def _ensure_initialized(self) -> None:
        """`initialize` + `notifications/initialized` один раз на клиент
        (и заново, если сервер забыл сессию)."""
        if not self.handshake or self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            self._session_id = None
            self._protocol_version = None
            payload = {
                "jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": "initialize",
                "params": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "agentscore", "version": "1.0"},
                },
            }
            body, resp = self._post(payload, retry_read_timeout=True)
            result = self._unwrap(body, "initialize")
            self._session_id = resp.headers.get("Mcp-Session-Id") or None
            self._protocol_version = result.get("protocolVersion") or MCP_PROTOCOL_VERSION
            notification = {"jsonrpc": "2.0", "method": "notifications/initialized"}
            try:
                self.session.request(
                    "POST", self.base_url, headers=self._headers(), json=notification, timeout=self.timeout,
                )
            except requests.RequestException as exc:
                raise MCPClientError(f"MCP-сервер не принял notifications/initialized: {exc}") from exc
            self._initialized = True

    @staticmethod
    def _unwrap(body: Dict[str, Any], method: str) -> Dict[str, Any]:
        if "error" in body:
            err = body["error"] or {}
            message = err.get("message", err) if isinstance(err, dict) else err
            raise MCPClientError(f"MCP-сервер вернул ошибку JSON-RPC ({method}): {message}")
        return body.get("result") or {}

    def _rpc(self, method: str, params: Dict[str, Any], retry_read_timeout: bool = True) -> Dict[str, Any]:
        """Один JSON-RPC запрос. Для серверов с рукопожатием — сначала
        `initialize`; если сервер ответил 404 на запрос с сессией (сессия
        истекла), рукопожатие повторяется один раз.

        Поднимает `MCPClientError` при исчерпании попыток, HTTP-ошибке,
        нераспарсиваемом ответе или JSON-RPC уровне `error`."""
        self._ensure_initialized()
        payload = {"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": method, "params": params}
        try:
            body, _ = self._post(payload, retry_read_timeout)
        except _SessionExpired:
            self._initialized = False
            self._ensure_initialized()
            body, _ = self._post(payload, retry_read_timeout)
        return self._unwrap(body, method)

    def _post(self, payload: Dict[str, Any], retry_read_timeout: bool) -> "tuple[Dict[str, Any], requests.Response]":
        """POST с retry/backoff (до `max_retries` дополнительных попыток при
        сетевой ошибке или 5xx/429, экспоненциальная задержка). Ответ —
        JSON или поток `text/event-stream` (SSE)."""
        method = payload.get("method")
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.request(
                    "POST", self.base_url, headers=self._headers(), json=payload, timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_exc = exc
                # Запрос дошёл до сервера, но ответа не дождались: повтор
                # вызова инструмента выполнил бы его второй раз (например,
                # сохранил бы файл повторно) — такие вызовы не повторяем.
                if isinstance(exc, requests.ReadTimeout) and not retry_read_timeout:
                    raise MCPClientError(
                        f"MCP-сервер не ответил за {self.timeout:g} с (вызов мог выполниться): {exc}"
                    ) from exc
                if attempt < self.max_retries:
                    _logger.warning(
                        "MCP %s: сетевая ошибка, попытка %d/%d: %s",
                        method, attempt + 1, self.max_retries + 1, exc,
                    )
                    time.sleep(self.retry_backoff_seconds * (2 ** attempt))
                    continue
                raise MCPClientError(f"сетевая ошибка при обращении к MCP-серверу: {exc}") from exc

            if resp.status_code == 404 and self._session_id and method != "initialize":
                raise _SessionExpired()

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < self.max_retries:
                    _logger.warning(
                        "MCP %s: ответ %s, попытка %d/%d",
                        method, resp.status_code, attempt + 1, self.max_retries + 1,
                    )
                    time.sleep(self.retry_backoff_seconds * (2 ** attempt))
                    continue

            if resp.status_code >= 400:
                raise MCPClientError(f"MCP-сервер ответил {resp.status_code}: {resp.text[:500]}")

            content_type = (resp.headers.get("Content-Type") or "").lower()
            if "text/event-stream" in content_type:
                # SSE всегда в UTF-8; `resp.text` у requests для text/* без
                # charset декодирует как ISO-8859-1 и портит кириллицу.
                raw = getattr(resp, "content", None)
                stream = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else resp.text
                body = _parse_sse(stream, payload.get("id"))
                if body is None:
                    raise MCPClientError(
                        "MCP-сервер прислал поток событий без ответа на запрос: " + stream[:300].replace("\n", "\\n")
                    )
                return body, resp
            try:
                body = resp.json()
            except ValueError as exc:
                raise MCPClientError(f"MCP-сервер вернул не-JSON ответ: {exc}") from exc
            if not isinstance(body, dict):
                raise MCPClientError("MCP-сервер вернул неожиданный ответ (не объект JSON-RPC)")
            return body, resp

        # Сюда доходим только если цикл исчерпался без return/raise выше
        # (страховка, на практике недостижимо — последняя итерация либо
        # возвращает результат, либо поднимает исключение сама).
        raise MCPClientError(f"не удалось получить ответ MCP-сервера: {last_exc}")

    @staticmethod
    def _mcp_tool_to_openai(tool: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description") or "",
                "parameters": tool.get("inputSchema") or {"type": "object", "properties": {}},
            },
        }

    # -- публичное API --------------------------------------------------------

    def list_tools(self) -> List[Dict[str, Any]]:
        """Список инструментов MCP-сервера в формате OpenAI function-tool
        (готовом для дописывания в `Settings.tools_json`). При сбое связи
        отдаёт то, что было получено последний раз успешно (см. докстринг
        модуля) — НЕ поднимает исключение, в отличие от `call_tool`, потому
        что вызывается на пути обычной отправки сообщения в чат, где сбой
        необязательного MCP-сервера не должен мешать самому чату."""
        try:
            result = self._rpc("tools/list", {})
        except MCPClientError as exc:
            _logger.warning("MCP list_tools недоступен, используется кэш (%d инстр.): %s", len(self._cached_tools), exc)
            return list(self._cached_tools)
        raw_tools = result.get("tools") or []
        converted = [self._mcp_tool_to_openai(t) for t in raw_tools if t.get("name")]
        self._cached_tools = converted
        self._cached_tool_names = {t.get("name") for t in raw_tools if t.get("name")}
        self._cached_raw_tools = [t for t in raw_tools if t.get("name")]
        return list(converted)

    def raw_tools(self) -> List[Dict[str, Any]]:
        """Инструменты из последнего успешного `list_tools()` в исходном виде
        (`name`, `title`, `description`, `inputSchema`)."""
        return list(self._cached_raw_tools)

    def list_tool_details(self) -> List[Dict[str, Any]]:
        """Список для интерфейса (выбор инструментов в настройках):
        `name`, `title`, `description`, `parameters`, `server`."""
        self.list_tools()
        return [_tool_details(t, t["name"], self.name, self.title) for t in self._cached_raw_tools]

    def tool_group(self, name: str) -> Optional[str]:
        """Подпись источника инструмента (см. `_tool_details`) по последнему
        успешному `list_tools()`; None — инструмент неизвестен."""
        for raw in self._cached_raw_tools:
            if raw.get("name") == name:
                return _tool_details(raw, name, self.name, self.title)["group"]
        return None

    def direct_only_tools(self) -> set:
        """Инструменты с `_meta["agentscore/direct_only"]` (по последнему
        успешному `list_tools()`)."""
        return {
            raw["name"] for raw in self._cached_raw_tools
            if isinstance(raw.get("_meta"), dict) and raw["_meta"].get(DIRECT_ONLY_META_KEY)
        }

    def has_cached_tool(self, name: str) -> bool:
        """Использовалось ли имя `name` в последнем успешном `list_tools()`
        — так `Repository._execute_tool_call` решает, стоит ли вообще
        пробовать `call_tool` для неизвестного локально имени функции,
        прежде чем тратить на это сетевой вызов."""
        return name in self._cached_tool_names

    def call_tool(self, name: str, arguments: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Вызывает инструмент `name` на MCP-сервере. Возвращает уже
        разобранный dict — результат ("structuredContent"."result", когда
        он есть — все инструменты сервера возвращают dict/list, обёрнутый
        в ключ "result", см. `mcp_server/server.py`), либо, если сервер
        сообщил протокольную ошибку (`isError: true` — неизвестный
        инструмент, необработанное исключение внутри него), словарь
        `{"error": "..."}` — вызывающая сторона (`Repository._execute_tool_call`)
        не различает этот случай и "доменную" ошибку самого инструмента
        (та тоже приходит как {"error": ...}, но с isError: false) — обоим
        нужна ровно одна и та же реакция: показать текст ошибки модели.

        Поднимает `MCPClientError` только при СЕТЕВОМ сбое/битом ответе —
        это НЕ кэшируется (в отличие от `list_tools`): если сервер лёг
        прямо во время вызова, откатываться здесь не на что, вызывающая
        сторона сама решает, что делать (обычно — превратить в
        {"error": ...} и отдать модели, как любой другой сбой инструмента)."""
        params: Dict[str, Any] = {"name": name, "arguments": arguments or {}}
        if meta:
            # `_meta` — служебные сведения о вызове по протоколу MCP: из
            # какого чата и агента он пришёл (им пользуется планировщик —
            # «поставь задачу в этот чат»). Серверы, которым это не нужно,
            # поле игнорируют.
            params["_meta"] = meta
        result = self._rpc("tools/call", params, retry_read_timeout=False)
        is_error = bool(result.get("isError"))
        structured = result.get("structuredContent")
        if isinstance(structured, dict) and "result" in structured:
            payload = structured["result"]
        elif isinstance(structured, dict):
            payload = structured
        else:
            payload = self._text_from_content(result.get("content") or [])
        if is_error:
            if isinstance(payload, dict) and "error" in payload:
                return payload
            return {"error": str(payload) if not isinstance(payload, dict) else json.dumps(payload, ensure_ascii=False)}
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            return {"result": payload}
        return {"result": payload}

    @staticmethod
    def _text_from_content(content: List[Dict[str, Any]]) -> Any:
        """Фолбэк для ответа без `structuredContent` (в норме не должен
        встречаться — все инструменты сервера аннотированы конкретным
        типом возврата именно ради structuredContent, см.
        `mcp_server/server.py::GitHostResult`) — берёт текст первого
        текстового блока и пробует разобрать его как JSON, иначе
        возвращает как есть."""
        for block in content:
            if block.get("type") == "text":
                text = block.get("text", "")
                try:
                    return json.loads(text)
                except (ValueError, TypeError):
                    return text
        return None


#: Ключ `_meta` вызова: MCP-инструменты, разрешённые в чате.
ALLOWED_TOOLS_META_KEY = "agentscore/allowed_tools"

#: Ключ `_meta` инструмента с его группой («GIT API», «Локальный GIT»,
#: «Планировщик» у наших серверов).
GROUP_META_KEY = "agentscore/group"

#: Ключ `_meta` инструмента: только прямой вызов сервисом, не моделью.
DIRECT_ONLY_META_KEY = "agentscore/direct_only"


def _tool_details(
    raw: Dict[str, Any], exposed_name: str, server: str, title: Optional[str] = None,
) -> Dict[str, Any]:
    meta = raw.get("_meta") if isinstance(raw.get("_meta"), dict) else {}
    return {
        "name": exposed_name,
        "original_name": raw.get("name", ""),
        "server": server,
        # Подпись источника в списке инструментов: группа, которую сообщил
        # сам сервер, иначе — короткое имя сервера из MCP_SERVERS.
        "group": str(meta.get(GROUP_META_KEY) or title or server),
        "title": raw.get("title") or "",
        "description": raw.get("description") or "",
        "parameters": raw.get("inputSchema") or {"type": "object", "properties": {}},
        # «День 31»: инструмент вызывает только сам сервис (после подтверждения
        # пользователя), модели он не предлагается.
        "direct_only": bool(meta.get(DIRECT_ONLY_META_KEY)),
    }


def parse_mcp_servers(raw: str) -> List[tuple]:
    """Разбирает `MCP_SERVERS` вида `tools=http://h:8001/mcp,scheduler=http://h:8002/mcp`
    в список `(имя, адрес)`. Пустые элементы пропускаются; повтор имени —
    ошибка конфигурации."""
    servers: List[tuple] = []
    seen: set = set()
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, url = item.partition("=")
        name, url = name.strip(), url.strip()
        if not sep or not name or not url:
            raise ValueError(f"MCP_SERVERS: ожидается «имя=адрес», получено {item!r}")
        if name in seen:
            raise ValueError(f"MCP_SERVERS: имя сервера {name!r} указано дважды")
        seen.add(name)
        servers.append((name, url))
    return servers


class MultiMCPClient:
    """Несколько MCP-серверов за той же поверхностью, что у `MCPClient`
    (`list_tools`, `has_cached_tool`, `call_tool`) — `Repository` не знает,
    сколько серверов подключено.

    Имена инструментов с разных серверов не должны совпадать: если имя
    встречается на нескольких серверах, у всех таких инструментов
    появляется префикс сервера (`scheduler__list_jobs`). Вызов уходит на
    тот сервер, которому принадлежит инструмент."""

    def __init__(self, clients: Dict[str, "MCPClient"]):
        if not clients:
            raise ValueError("MultiMCPClient требует хотя бы один сервер")
        self.clients = dict(clients)
        self._routes: Dict[str, tuple] = {}
        self._details: List[Dict[str, Any]] = []
        self._tools: List[Dict[str, Any]] = []

    @property
    def base_urls(self) -> Dict[str, str]:
        return {name: client.base_url for name, client in self.clients.items()}

    def list_tools(self) -> List[Dict[str, Any]]:
        per_server = {}
        for name, client in self.clients.items():
            client.list_tools()  # сам откатывается на кэш при сбое сервера
            per_server[name] = client.raw_tools()
        counts: Dict[str, int] = {}
        for raws in per_server.values():
            for raw in raws:
                counts[raw["name"]] = counts.get(raw["name"], 0) + 1
        routes, details, tools = {}, [], []
        for server, raws in per_server.items():
            for raw in raws:
                original = raw["name"]
                exposed = f"{server}__{original}" if counts[original] > 1 else original
                routes[exposed] = (self.clients[server], original)
                details.append(_tool_details(raw, exposed, server, self.clients[server].title))
                tools.append({
                    "type": "function",
                    "function": {
                        "name": exposed,
                        "description": raw.get("description") or "",
                        "parameters": raw.get("inputSchema") or {"type": "object", "properties": {}},
                    },
                })
        self._routes, self._details, self._tools = routes, details, tools
        return list(tools)

    def list_tool_details(self) -> List[Dict[str, Any]]:
        self.list_tools()
        return list(self._details)

    def has_cached_tool(self, name: str) -> bool:
        return name in self._routes

    def direct_only_tools(self) -> set:
        return {d["name"] for d in self._details if d.get("direct_only")}

    def tool_group(self, name: str) -> Optional[str]:
        for details in self._details:
            if details["name"] == name:
                return details["group"]
        return None

    def call_tool(self, name: str, arguments: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        route = self._routes.get(name)
        if route is None:
            raise MCPClientError(f"инструмент {name!r} не найден ни на одном MCP-сервере")
        client, original = route
        allowed = (meta or {}).get(ALLOWED_TOOLS_META_KEY)
        if isinstance(allowed, list):
            # Серверу — имена его инструментов, без префикса `сервер__`.
            meta = dict(meta)
            meta[ALLOWED_TOOLS_META_KEY] = sorted(
                self._routes[n][1] for n in allowed if n in self._routes and self._routes[n][0] is client
            )
        return client.call_tool(original, arguments, meta=meta)


def build_mcp_client(
    servers: List[Union[tuple, McpServerSpec]], api_key: Optional[str] = None, timeout: float = 30.0,
):
    """Один сервер — обычный `MCPClient`, несколько — `MultiMCPClient`.

    `(имя, адрес)` — серверы продукта из `MCP_SERVERS`/`MCP_SERVER_URL`:
    адрес дополняется `/mcp`, общий `MCP_API_KEY`, без рукопожатия.
    `McpServerSpec` — серверы из `MCP_SERVERS_FILE` (в том числе внешние):
    адрес как есть, свои заголовки, рукопожатие `initialize`; общий
    `MCP_API_KEY` им не отправляется."""
    clients: Dict[str, MCPClient] = {}
    for server in servers:
        if isinstance(server, McpServerSpec):
            clients[server.name] = MCPClient(
                server.url, timeout=timeout, name=server.name, headers=server.headers, title=server.title,
                normalize_url=False, handshake=server.initialize,
            )
        else:
            name, url = server
            clients[name] = MCPClient(url, api_key=api_key, timeout=timeout, name=name)
    if len(clients) == 1:
        return next(iter(clients.values()))
    return MultiMCPClient(clients)

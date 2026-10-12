"""
agents_core.builtin_tools
===========================

Встроенные инструменты AgentsCore — группа «Доступ к agents core».

Выполняются внутри AgentsCore (читают его базу напрямую), но для
пользователя выглядят как инструменты MCP: `GET /mcp/tools` отдаёт их вместе
с инструментами MCP-серверов, они выбираются тем же списком «Выбрать из
доступных» и попадают в `tools_json`. Модель может вызвать инструмент,
только если он выбран в настройках агента или чата.

- `read_chat` — «Чтение чата»: переписка другого чата и ссылки на источники
  (фрагменты баз знаний, на которые опирались ответы, и URL из текста).
  Без параметра `chat` возвращает список чатов, чтобы модель могла выбрать.
  С `include_stats` (по умолчанию выключен) — ещё и статистика: у каждого
  ответа токены, время, скорость, первый токен, память и контекст; у чата —
  модель, ключевые настройки и средние по всему чату («День 29»: сравнение
  конфигураций агентом-судьёй).
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    from .models import Chat, Message
    from .repository import Repository

GROUP_AGENTS_CORE = "Доступ к agents core"
SERVER_NAME = "agents_core"

READ_CHAT = "read_chat"
DEFAULT_LAST_MESSAGES = 20
MAX_LAST_MESSAGES = 100
MAX_MESSAGE_CHARS = 3000
MAX_CHATS_LISTED = 50

_URL = re.compile(r"https?://[^\s<>\"'`)\]}]+")

READ_CHAT_SPEC: Dict[str, Any] = {
    "name": READ_CHAT,
    "title": "Чтение чата",
    "description": (
        "Читает переписку другого чата AgentsCore: сообщения пользователя и ответы модели по порядку, "
        "источники ответов из баз знаний (документ, раздел, страница, путь или ссылка) и ссылки из текста. "
        "Без параметра chat возвращает список чатов (id, название, агент), чтобы выбрать нужный. "
        "С include_stats=true — статистика: у каждого ответа токены (in/out/total), время ответа, "
        "скорость генерации, время до первого токена, загрузка модели, память и доля GPU, контекст; "
        "у чата — модель, настройки (температура, top_p, макс. токенов, размер контекста, рассуждения, "
        "системный prompt) и средние по ВСЕМУ чату. Чтобы прочитать все вопросы и ответы тестового "
        "диалога, задайте last_messages не меньше удвоенного числа вопросов."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "chat": {
                "type": "string",
                "description": "id чата или его название (можно часть названия). Не задан — список чатов.",
            },
            "agent": {
                "type": "string",
                "description": "Название агента — сужает поиск чата по названию.",
            },
            "last_messages": {
                "type": "integer",
                "description": f"Сколько последних сообщений вернуть (1–{MAX_LAST_MESSAGES}, "
                               f"по умолчанию {DEFAULT_LAST_MESSAGES}).",
                "minimum": 1,
                "maximum": MAX_LAST_MESSAGES,
            },
            "include_sources": {
                "type": "boolean",
                "description": "Вернуть источники ответов и ссылки (по умолчанию true).",
            },
            "include_stats": {
                "type": "boolean",
                "description": (
                    "Вернуть статистику ответов и чата: токены, время, скорость, первый токен, память, "
                    "контекст, модель и настройки (по умолчанию false)."
                ),
            },
        },
    },
}

SPECS: Dict[str, Dict[str, Any]] = {READ_CHAT: READ_CHAT_SPEC}


def is_builtin(name: str) -> bool:
    return name in SPECS


def tool_details() -> List[Dict[str, Any]]:
    """Описание для `GET /mcp/tools` — в том же виде, что у инструментов MCP."""
    return [
        {
            "name": spec["name"], "original_name": spec["name"], "server": SERVER_NAME, "group": GROUP_AGENTS_CORE,
            "title": spec["title"], "description": spec["description"], "parameters": spec["parameters"],
        }
        for spec in SPECS.values()
    ]


# ---- read_chat ---------------------------------------------------------------


def _time(ts: Optional[int]) -> Optional[str]:
    return datetime.fromtimestamp(ts).isoformat(timespec="minutes") if ts else None


def _links(text: str) -> List[str]:
    found = (m.group(0).rstrip(".,;:!?»") for m in _URL.finditer(text or ""))
    return list(dict.fromkeys(found))


def _rag_sources(raw: Optional[str]) -> List[Dict[str, Any]]:
    if not raw:
        return []
    try:
        rag = json.loads(raw)
    except ValueError:
        return []
    cited = set(rag.get("cited") or [])
    sources = []
    for src in rag.get("sources") or []:
        item = {
            "n": src.get("n"), "collection": src.get("collection_name"), "document": src.get("title"),
            "section": src.get("section") or None, "page": src.get("page"), "source": src.get("source"),
            "date": src.get("doc_date"), "version": src.get("doc_version"), "cited": src.get("n") in cited,
        }
        sources.append({k: v for k, v in item.items() if v not in (None, "")})
    return sources


def _tools_used(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    try:
        events = json.loads(raw)
    except ValueError:
        return []
    return list(dict.fromkeys(e.get("name") for e in events if isinstance(e, dict) and e.get("name")))


def _seconds(ms: Any) -> Optional[float]:
    return round(ms / 1000, 2) if isinstance(ms, (int, float)) else None


def _gb(mb: Any) -> Optional[float]:
    return round(mb / 1024, 2) if isinstance(mb, (int, float)) else None


def _message_stats(message: "Message") -> Dict[str, Any]:
    """Статистика ответа ассистента: токены, время и метрики (`Message.metrics`)."""
    try:
        metrics = json.loads(message.metrics) if message.metrics else {}
    except ValueError:
        metrics = {}
    if not isinstance(metrics, dict):
        metrics = {}
    stats: Dict[str, Any] = {
        "tokens_in": message.prompt_tokens,
        "tokens_out": message.completion_tokens,
        "tokens_total": message.total_tokens,
        "duration_s": _seconds(message.duration_ms),
        "tokens_per_sec": metrics.get("tokens_per_sec"),
        "prompt_tokens_per_sec": metrics.get("prompt_tokens_per_sec"),
        "first_token_s": _seconds(metrics.get("ttft_ms")),
        "model_load_s": _seconds(metrics.get("load_ms")),
        "memory_gb": _gb(metrics.get("memory_mb")),
        "vram_gb": _gb(metrics.get("vram_mb")),
        "gpu_percent": metrics.get("gpu_percent"),
        "num_ctx": metrics.get("num_ctx"),
    }
    return {k: v for k, v in stats.items() if v is not None}


def _chat_stats(repo: "Repository", chat: "Chat") -> Dict[str, Any]:
    """Модель, ключевые настройки и средние по всему чату."""
    s = chat.settings
    model = repo._catalog.get(s.model)  # каталог моделей — только чтение
    stats = repo.chat_stats(chat)
    settings = {
        "temperature": s.temperature,
        "top_p": s.top_p,
        "max_tokens": s.max_tokens,
        "num_ctx": s.num_ctx,
        "thinking_enabled": s.thinking_enabled,
        "seed": s.seed,
        "context_strategy": s.context_strategy,
        "context_strategy_limit": s.context_strategy_limit,
        "rag_enabled": s.rag_enabled,
        "system_prompt": (s.system_prompt or "")[:1500],
    }
    summary = {
        "answers_measured": stats.get("measured_answers"),
        "avg_tokens_per_sec": stats.get("avg_tokens_per_sec"),
        "avg_duration_s": _seconds(stats.get("avg_duration_ms")),
        "avg_duration_note": "без времени загрузки модели в память",
        "avg_first_token_s": _seconds(stats.get("avg_ttft_ms")),
        "max_memory_gb": _gb(stats.get("max_memory_mb")),
        "max_vram_gb": _gb(stats.get("max_vram_mb")),
        "tokens_total": stats.get("total_tokens"),
        "context_window": stats.get("context_window"),
        "context_fill_ratio": round(stats["context_fill_ratio"], 3) if stats.get("context_fill_ratio") is not None else None,
    }
    return {
        "model": s.model,
        "model_name": model.display_name if model else None,
        "model_local": model.is_local if model else None,
        "settings": {k: v for k, v in settings.items() if v not in (None, "")},
        "summary": {k: v for k, v in summary.items() if v is not None},
    }


def _message_out(message: "Message", include_sources: bool, include_stats: bool = False) -> Dict[str, Any]:
    text = message.content or ""
    out: Dict[str, Any] = {"id": message.id, "role": message.role, "time": _time(message.created_at)}
    if len(text) > MAX_MESSAGE_CHARS:
        out["text"] = text[:MAX_MESSAGE_CHARS] + "…"
        out["text_truncated"] = True
    else:
        out["text"] = text
    if message.is_summary:
        out["summary"] = True
    if message.status != "complete":
        out["status"] = message.status
    if message.role == "user" and message.source != "app":
        out["from"] = message.source
    if message.branch:
        out["branch"] = message.branch
    if include_sources:
        sources = _rag_sources(message.rag)
        if sources:
            out["sources"] = sources
        links = _links(text)
        if links:
            out["links"] = links
    tools = _tools_used(message.tool_events)
    if tools:
        out["tools"] = tools
    if include_stats and message.role == "assistant":
        stats = _message_stats(message)
        if stats:
            out["stats"] = stats
    return out


def _chat_brief(repo: "Repository", chat: "Chat", agents: Dict[str, str], current_id: str) -> Dict[str, Any]:
    brief = {
        "id": chat.id, "title": chat.title, "agent": agents.get(chat.agent_id, ""),
        "updated": _time(chat.updated_at),
    }
    if chat.id == current_id:
        brief["current"] = True
    return brief


def read_chat(repo: "Repository", current: "Chat", arguments: Dict[str, Any]) -> Dict[str, Any]:
    agents = {a.id: a.name for a in repo.list_agents()}
    chats = sorted(repo.list_chats(), key=lambda c: c.updated_at or 0, reverse=True)
    agent_filter = str(arguments.get("agent") or "").strip().casefold()
    if agent_filter:
        chats = [c for c in chats if agent_filter in agents.get(c.agent_id, "").casefold()]

    def listing(note: Optional[str] = None, candidates=None) -> Dict[str, Any]:
        items = candidates if candidates is not None else chats
        result: Dict[str, Any] = {
            "chats": [_chat_brief(repo, c, agents, current.id) for c in items[:MAX_CHATS_LISTED]],
            "total": len(items),
        }
        if note:
            result["error"] = note
        return result

    query = str(arguments.get("chat") or "").strip()
    if not query:
        return listing()
    chat = next((c for c in chats if c.id == query), None)
    if chat is None:
        key = query.casefold()
        exact = [c for c in chats if c.title.casefold() == key]
        partial = exact or [c for c in chats if key in c.title.casefold()]
        if len(partial) == 1:
            chat = partial[0]
        elif not partial:
            return listing(f"чат «{query}» не найден — выберите id из списка")
        else:
            return listing(f"под «{query}» подходит несколько чатов — укажите id", candidates=partial)

    try:
        last = int(arguments.get("last_messages") or DEFAULT_LAST_MESSAGES)
    except (TypeError, ValueError):
        last = DEFAULT_LAST_MESSAGES
    last = max(1, min(MAX_LAST_MESSAGES, last))
    include_sources = arguments.get("include_sources", True) is not False
    include_stats = arguments.get("include_stats") is True

    messages = repo.list_messages(chat.id)
    shown = messages[-last:]
    result: Dict[str, Any] = {
        "chat": _chat_brief(repo, chat, agents, current.id),
        "total_messages": len(messages),
        "returned": len(shown),
        "earlier_messages_omitted": len(messages) - len(shown),
        "messages": [_message_out(m, include_sources, include_stats) for m in shown],
    }
    if include_stats:
        result["stats"] = _chat_stats(repo, chat)
    return result


HANDLERS: Dict[str, Callable[["Repository", "Chat", Dict[str, Any]], Dict[str, Any]]] = {READ_CHAT: read_chat}


def execute(repo: "Repository", current: "Chat", name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return HANDLERS[name](repo, current, arguments)

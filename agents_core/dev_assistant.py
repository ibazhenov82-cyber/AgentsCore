"""
agents_core.dev_assistant
===========================

«День 31» — ассистент разработчика: команды в чате и проект по ссылке.

Включается настройкой чата (агента) «Команды разработчика»
(`Settings.dev_commands_enabled`). Тогда сообщение чата разбирается здесь
(`DevAssistant.route`) до обычного ответа модели:

- `/команда аргументы` — команда из `COMMANDS`;
- ссылка на репозиторий (URL GitHub / GitLab / Gitea или локальный путь)
  в тексте — открывает проект чата (`ChatProject`), новая ссылка заменяет
  его; сообщение из одной ссылки — то же, что `/project <ссылка>`;
- обычный текст при открытом проекте — `/ask`.

Локальные команды (`/help`, `/project`, `/status`, `/diff`, `/clear`)
выполняются без модели: AgentsCore сам вызывает инструменты «Репозиторий»
mcp_server и показывает результат карточкой (`Message.command.card`); такие
сообщения не уходят в контекст модели. `/ask`, `/explain`, `/history`,
`/review` — ответ модели с инструментами ТОЛЬКО чтения (`repo_*`), `repo_id`
подставляет AgentsCore, модель его не видит; `/ask` и `/explain` ещё и ищут
по базе знаний проекта (README, docs, схемы, API), которую AgentsCore
создаёт в knowledge_service при открытии проекта. `/commit` (модель
предлагает сообщение) и `/undo` выполняются только после подтверждения в
приложении (`Repository.confirm_command`) и только при выключенной
настройке «Только чтение» (`Settings.dev_read_only`). Модель инструментов
записи не получает никогда.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .config import AgentConfig
from .knowledge import (
    RAG_STATUS_NOT_FOUND,
    RAG_STATUS_OK,
    RAG_STATUS_UNAVAILABLE,
    KnowledgeServiceError,
    RagOutcome,
    augmented_question,
    finalize_answer,
    select_sources,
)
from .mcp_client import MCPClientError
from .models import Agent, Chat, ChatProject, Message
from .providers import ProviderError, ProviderMessage

_logger = logging.getLogger("agents_core.dev_assistant")

GROUP_HELP = "Справка"
GROUP_QUESTIONS = "Вопросы"
GROUP_CHANGES = "Изменения"

READ_ONLY_NOTICE = "Чат в режиме только чтения — выключите «Только чтение» в настройках чата."
NO_PROJECT_NOTICE = ("Проект не открыт. Пришлите ссылку на репозиторий (https://github.com/owner/repo, можно с "
                     "/tree/<ветка>) или путь к локальной рабочей копии — или `/project <ссылка>`.")


@dataclass(frozen=True)
class DevCommand:
    name: str
    usage: str
    title: str
    description: str
    group: str
    kind: str  # "local" | "llm" | "confirm"
    writes: bool = False
    needs_project: bool = True
    #: Сообщения команды попадают в контекст следующих запросов к модели.
    in_context: bool = False


COMMANDS: List[DevCommand] = [
    DevCommand("help", "/help [команда]", "Справка", "Список команд или подробности по одной.", GROUP_HELP,
               "local", needs_project=False),
    DevCommand("project", "/project [ссылка]", "Проект",
               "Открыть или сменить проект чата: URL репозитория GitHub, GitLab, Gitea (можно /tree/<ветка>) или "
               "путь к локальной рабочей копии. Без ссылки — текущий проект. Ссылка в любом сообщении делает то же.",
               GROUP_HELP, "local", needs_project=False),
    DevCommand("status", "/status", "Состояние",
               "Проект, ветка, отставание от origin, изменённые файлы, индексация базы знаний, модель и настройки.",
               GROUP_HELP, "local", needs_project=False),
    DevCommand("clear", "/clear", "Новый контекст",
               "Граница в чате: сообщения выше не уходят в запросы к модели.", GROUP_HELP, "local",
               needs_project=False),
    DevCommand("ask", "/ask <вопрос>", "Вопрос о проекте",
               "Ответ по документации (база знаний проекта), коду и истории git, со ссылками на файлы и коммиты. "
               "Обычный текст при открытом проекте — то же самое.", GROUP_QUESTIONS, "llm", in_context=True),
    DevCommand("explain", "/explain <путь или символ>", "Объяснить",
               "Объяснить файл, модуль, класс или функцию.", GROUP_QUESTIONS, "llm", in_context=True),
    DevCommand("history", "/history <путь>", "История",
               "Почему код стал таким: коммиты, авторы и мотивы изменений файла или папки.", GROUP_QUESTIONS, "llm",
               in_context=True),
    DevCommand("diff", "/diff [staged | ветка | a..b]", "Изменения",
               "Diff как есть: без аргумента — незакоммиченные изменения, staged — подготовленные, a..b — между "
               "ветками или коммитами.", GROUP_CHANGES, "local"),
    DevCommand("review", "/review [ветка | a..b]", "Ревью",
               "Код-ревью текущих изменений или ветки: ошибки, риски, стиль — со ссылками на строки.",
               GROUP_CHANGES, "llm", in_context=True),
    DevCommand("commit", "/commit [пожелание]", "Коммит",
               "Модель предлагает сообщение коммита для текущих изменений; коммит — после подтверждения. Только "
               "локальная рабочая копия и выключенное «Только чтение».", GROUP_CHANGES, "confirm", writes=True),
    DevCommand("undo", "/undo", "Откатить коммит",
               "Откатить последний коммит, сделанный через /commit (git reset --soft), после подтверждения.",
               GROUP_CHANGES, "confirm", writes=True),
]
COMMANDS_BY_NAME = {c.name: c for c in COMMANDS}

#: Инструменты чтения по командам (имена инструментов mcp_server).
TOOLS_QUESTIONS = ["repo_list_files", "repo_read_file", "repo_grep", "repo_log", "repo_show", "repo_blame"]
TOOLS_HISTORY = ["repo_log", "repo_blame", "repo_show", "repo_read_file", "repo_grep"]
TOOLS_REVIEW = ["repo_diff", "repo_read_file", "repo_grep", "repo_list_files", "repo_log"]
#: Все инструменты группы «Репозиторий», которые вызывает сам AgentsCore.
REPO_TOOL_NAMES = {"repo_open", "repo_status", "repo_log", "repo_show", "repo_grep", "repo_diff", "repo_blame",
                   "repo_list_files", "repo_read_file", "repo_commit", "repo_undo_commit"}

#: Сколько последних сообщений истории видит модель в командах.
HISTORY_MESSAGES = 20
#: Сколько символов diff передаётся модели (/review, /commit) и показывается в тексте /diff.
DIFF_FOR_MODEL_CHARS = 24000
DIFF_IN_TEXT_CHARS = 12000

_COMMAND_RE = re.compile(r"^/([A-Za-z][\w-]*)(?:[ \t]+(.*))?$", re.S)
_URL_RE = re.compile(r"(?:https?://|git@)[^\s<>\"'`)\]]+")
_LOCAL_RE = re.compile(r"(?:(?<=\s)|^)((?:~|[A-Za-z]:)?[/\\][^\s<>\"'`]*[/\\][^\s<>\"'`]+)")
#: Хосты, ссылки на которые в тексте считаются проектом.
DEFAULT_PROJECT_HOSTS = ("github.com", "gitlab.com", "gitea.com")


#: Заголовки разделов ревью (/review и ревью PR из CI) — по ним ответ
#: разбирается в структуру (`parse_review`).
REVIEW_HEADINGS = {
    "summary": "### Кратко",
    "bugs": "### Потенциальные баги",
    "architecture": "### Архитектурные проблемы",
    "recommendations": "### Рекомендации",
}
#: Сколько поисковых запросов по diff.
REVIEW_RAG_QUERIES = 3
_SYMBOL_RE = re.compile(
    r"\b(?:def|class|fun|func|function|interface|object|struct|enum|record)\s+([A-Za-z_][\w]*)")
_HUNK_RE = re.compile(r"^@@[^@]*@@\s*(.*)$")


def review_queries(diff: Dict[str, Any]) -> List[str]:
    """Поисковые запросы к базе знаний по diff: изменённые файлы (модули) и
    функции/классы, которые затрагивают изменения."""
    files = [f.get("path") for f in diff.get("files") or [] if f.get("path")] + list(diff.get("untracked") or [])
    symbols: List[str] = []
    for line in (diff.get("patch") or "").splitlines():
        hunk = _HUNK_RE.match(line)
        text = hunk.group(1) if hunk else (line[1:] if line[:1] in "+-" and not line.startswith(("+++", "---")) else "")
        for name in _SYMBOL_RE.findall(text):
            if name not in symbols:
                symbols.append(name)
    queries = []
    if files:
        modules = sorted({re.sub(r"\.[^./]+$", "", p).replace("/", " ").replace("_", " ") for p in files})
        queries.append("Назначение и устройство: " + "; ".join(modules[:8]))
    if symbols:
        queries.append("Функции и классы: " + ", ".join(symbols[:10]))
        queries.append("Где используются " + ", ".join(symbols[:5]))
    return queries[:REVIEW_RAG_QUERIES]


def parse_review(text: str) -> Dict[str, Any]:
    """Ответ ревью → {"summary", "bugs", "architecture", "recommendations"}:
    разделы по заголовкам `REVIEW_HEADINGS`, пункты — строки списка
    (продолжения строк приклеиваются к пункту). «нет» — пустой раздел."""
    titles = {v.lstrip("# ").strip().lower(): k for k, v in REVIEW_HEADINGS.items()}
    sections: Dict[str, Any] = {"summary": "", "bugs": [], "architecture": [], "recommendations": []}
    current: Optional[str] = None
    summary_lines: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        heading = re.match(r"^(?:#{1,6}\s*|\*\*)(.+?)(?:\*\*)?:?\s*$", line)
        if heading and heading.group(1).strip(" *:").lower() in titles:
            current = titles[heading.group(1).strip(" *:").lower()]
            continue
        if current is None or not line:
            continue
        if current == "summary":
            summary_lines.append(line)
            continue
        item = re.match(r"^(?:[-*•]|\d+[.)])\s+(.*)$", line)
        if item:
            value = item.group(1).strip()
            if value.lower().strip(".") not in ("нет", "не найдено", "замечаний нет"):
                sections[current].append(value)
        elif sections[current]:
            sections[current][-1] += " " + line
    sections["summary"] = " ".join(summary_lines).strip()
    return sections


class DevError(Exception):
    """Ожидаемая ошибка команды — показывается в чате как ответ."""


@dataclass
class Route:
    name: str
    args: str = ""
    link: Optional[str] = None
    #: Ссылка найдена в тексте сообщения (а не задана /project).
    link_auto: bool = False
    #: Обычный текст без команды (при открытом проекте — /ask).
    implicit: bool = False


@dataclass
class LocalResult:
    content: str
    card: Optional[Dict[str, Any]] = None
    confirm: Optional[Dict[str, Any]] = None


def _url_host(url: str) -> str:
    m = re.match(r"^(?:https?://)?(?:[^@/]+@)?([^/:]+)", url)
    return (m.group(1) if m else "").lower()


def find_link(text: str, hosts: Tuple[str, ...] = DEFAULT_PROJECT_HOSTS, any_host: bool = False) -> Optional[str]:
    """Ссылка на репозиторий в тексте: URL известного хостинга с путём
    owner/repo или абсолютный локальный путь. `any_host` — в аргументах явной
    команды (`/review pr:12 https://git.example.com/o/r`) хост любой: какие
    разрешены, решает mcp_server."""
    for match in _URL_RE.finditer(text or ""):
        url = match.group(0).rstrip(".,;:!?»")
        host = _url_host(url[4:] if url.startswith("git@") else url)
        if host.startswith("www."):
            host = host[4:]
        if host not in hosts and not any_host:
            continue
        path = re.sub(r"^(?:https?://[^/]+|git@[^:]+:)", "", url)
        if len([p for p in path.split("/") if p]) >= 2:
            return url
    match = _LOCAL_RE.search(text or "")
    if match:
        return match.group(1).rstrip(".,;:!?»")
    return None


def parse_command(text: str) -> Optional[Tuple[str, str]]:
    match = _COMMAND_RE.match((text or "").strip())
    if not match:
        return None
    return match.group(1).lower(), (match.group(2) or "").strip()


def command_meta(message: Message) -> Dict[str, Any]:
    if not message.command:
        return {}
    try:
        data = json.loads(message.command)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def filter_context(messages: List[Message]) -> List[Message]:
    """Сообщения, которые могут уйти в контекст модели: после последней
    границы /clear и без результатов локальных команд."""
    start = 0
    for i, m in enumerate(messages):
        if command_meta(m).get("boundary"):
            start = i + 1
    return [m for m in messages[start:] if command_meta(m).get("context", True)]


def _strip_link(text: str, link: str) -> str:
    return re.sub(r"\s{2,}", " ", text.replace(link, " ")).strip(" ,.:—-\n\t")


def _short(sha: Optional[str]) -> str:
    return (sha or "")[:8]


def _split_model(composite_id: str) -> Tuple[str, str]:
    provider, sep, model_id = (composite_id or "").partition(":")
    if not sep or not provider or not model_id:
        raise DevError(f"неверный идентификатор модели {composite_id!r}")
    return provider, model_id


def _clean_commit_message(text: str) -> str:
    text = (text or "").strip()
    fence = re.search(r"```[\w-]*\n(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    text = re.sub(r"^(сообщение коммита|commit message)\s*[:：]\s*", "", text, flags=re.I).strip()
    return text.strip("\"'«»` \n")


class DevAssistant:
    def __init__(self, repo):
        #: `Repository` — его БД, клиенты MCP и баз знаний, провайдеры моделей.
        self.repo = repo
        self._tool_names: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Разбор сообщения
    # ------------------------------------------------------------------

    @property
    def _db(self):
        return self.repo._db

    def project(self, chat_id: str) -> Optional[ChatProject]:
        return self._db.get_chat_project(chat_id)

    def route(self, chat: Chat, text: str) -> Optional[Route]:
        """None — обычное сообщение (команды выключены или проекта нет)."""
        if not chat.settings.dev_commands_enabled:
            return None
        parsed = parse_command(text)
        if parsed is not None:
            name, args = parsed
            if name == "project":
                return Route("project", args, link=args or None)
            if name in ("ask", "explain", "history", "review"):
                link = find_link(args, any_host=True)
                if link:
                    return Route(name, _strip_link(args, link), link=link, link_auto=True)
            return Route(name, args)
        link = find_link(text)
        if link:
            rest = _strip_link(text, link)
            if not rest:
                return Route("project", link, link=link, link_auto=True)
            return Route("ask", rest, link=link, link_auto=True, implicit=True)
        if self.project(chat.id) is not None:
            return Route("ask", text.strip(), implicit=True)
        return None

    def commands_for(self, chat: Chat) -> List[Dict[str, Any]]:
        project = self.project(chat.id)
        result = []
        for cmd in COMMANDS:
            available, reason = True, None
            if cmd.writes and chat.settings.dev_read_only:
                available, reason = False, "только чтение"
            elif cmd.writes and project is not None and not project.writable:
                available, reason = False, "проект открыт по URL — коммиты только в локальной копии"
            result.append({**dataclasses.asdict(cmd), "available": available, "reason": reason})
        return result

    # ------------------------------------------------------------------
    # MCP
    # ------------------------------------------------------------------

    def _mcp(self):
        client = self.repo._mcp_client
        if client is None:
            raise DevError("Не подключён mcp_server: задайте MCP_ENABLED и адрес сервера в настройках AgentsCore.")
        return client

    def _exposed(self, name: str) -> str:
        """Имя инструмента у клиента (при одинаковых именах на нескольких
        серверах у него префикс сервера)."""
        if name not in self._tool_names:
            client = self._mcp()
            details = client.list_tool_details()
            names = {}
            for d in details:
                original = d.get("original_name") or d.get("name")
                if original in REPO_TOOL_NAMES and (original not in names or d.get("group", "").startswith("Репозиторий")):
                    names[original] = d["name"]
            self._tool_names = names
        if name not in self._tool_names:
            raise DevError("В mcp_server не включена группа «Репозиторий» (MCP_REPOS_ENABLED=true).")
        return self._tool_names[name]

    def call(self, name: str, chat_id: Optional[str] = None, **arguments: Any) -> Dict[str, Any]:
        exposed = self._exposed(name)
        meta = {"agentscore/chat_id": chat_id} if chat_id else None
        try:
            result = self._mcp().call_tool(exposed, {k: v for k, v in arguments.items() if v is not None}, meta=meta)
        except MCPClientError as exc:
            raise DevError(f"mcp_server недоступен: {exc}") from exc
        if isinstance(result, dict) and result.get("error"):
            raise DevError(str(result["error"]))
        return result

    def _tool_schemas(self, names: List[str]) -> Tuple[List[dict], Dict[str, dict]]:
        """Инструменты для модели в формате OpenAI — без параметра repo_id —
        и подстановки аргументов по их именам у клиента."""
        client = self._mcp()
        details = {d["name"]: d for d in client.list_tool_details()}
        tools, inject = [], {}
        for name in names:
            try:
                exposed = self._exposed(name)
            except DevError:
                continue
            d = details.get(exposed)
            if d is None:
                continue
            params = json.loads(json.dumps(d.get("parameters") or {"type": "object", "properties": {}}))
            params.get("properties", {}).pop("repo_id", None)
            if isinstance(params.get("required"), list):
                params["required"] = [r for r in params["required"] if r != "repo_id"]
            description = (d.get("title") + ". " if d.get("title") else "") + (d.get("description") or "")
            tools.append({"type": "function", "function": {
                "name": exposed, "description": description, "parameters": params}})
            inject[exposed] = {}
        return tools, inject

    # ------------------------------------------------------------------
    # Проект и база знаний
    # ------------------------------------------------------------------

    def open_project(self, chat: Chat, link: str) -> Tuple[ChatProject, Dict[str, Any]]:
        info = self.call("repo_open", chat.id, link=link)
        now = int(time.time())
        previous = self.project(chat.id)
        same = previous is not None and previous.repo_id == info["repo_id"]
        project = ChatProject(
            chat_id=chat.id, link=link, repo_id=info["repo_id"], kind=info.get("kind") or "remote",
            name=info.get("name") or link, path=info.get("path") or "", branch=info.get("branch"),
            default_branch=info.get("default_branch"), remote=info.get("remote"),
            writable=bool(info.get("writable")),
            collection_id=previous.collection_id if same else None,
            source_id=previous.source_id if same else None,
            last_commit=previous.last_commit if same else None,
            opened_at=previous.opened_at if same else now, updated_at=now,
        )
        self._ensure_kb(project)
        self._db.save_chat_project(project)
        return project, info

    def _kb_marker(self, project: ChatProject) -> str:
        return f"agentscore-project:{project.repo_id}"

    def _ensure_kb(self, project: ChatProject) -> None:
        """База знаний проекта: коллекция «Проект: …» с источником-папкой
        репозитория (маски `DEV_PROJECT_KB_INCLUDE`); при повторном открытии —
        синхронизация (knowledge_service переиндексирует только изменённое)."""
        kb = self.repo._knowledge
        include = AgentConfig.dev_project_kb_masks()
        if kb is None or not include or not project.path:
            project.index_status, project.index_error = "", None
            return
        try:
            if project.collection_id and project.source_id:
                try:
                    # Маски могли измениться (например, включили код) — источник обновляется.
                    kb.update_source_include(project.source_id, include)
                    started = kb.sync_source(project.source_id)
                    project.index_job_id = started.get("job_id")
                    project.index_status, project.index_error = "indexing", None
                    return
                except KnowledgeServiceError:
                    project.collection_id = project.source_id = None  # база удалена — создаём заново
            created = None
            marker = self._kb_marker(project)
            collection = next((c for c in kb.list_collections() if (c.get("description") or "") == marker), None)
            if collection is None:
                title = f"Проект: {project.name}" + (f" ({project.branch})" if project.kind == "remote" and project.branch else "")
                collection = created = kb.create_collection(
                    title, marker, {"extra_text_extensions": [".proto", ".graphql"]})
            project.collection_id = collection["id"]
            source = next((s for s in kb.list_sources(project.collection_id)
                           if (s.get("source") or {}).get("path") == project.path), None)
            if source is None:
                try:
                    added = kb.add_path_source(project.collection_id, project.path, include, project.name)
                except KnowledgeServiceError:
                    if created is not None:  # не оставляем пустую базу (например, папка вне KB_ROOTS)
                        try:
                            kb.delete_collection(created["id"])
                        except KnowledgeServiceError:
                            pass
                    project.collection_id = None
                    raise
                project.source_id = (added.get("source") or {}).get("id")
                project.index_job_id = added.get("job_id")
            else:
                project.source_id = source.get("id")
                if list(source.get("include") or []) != include:
                    kb.update_source_include(project.source_id, include)
                project.index_job_id = kb.sync_source(project.source_id).get("job_id")
            project.index_status, project.index_error = "indexing", None
        except KnowledgeServiceError as exc:
            project.index_status, project.index_error = "unavailable", str(exc)
            _logger.warning("База знаний проекта %s: %s", project.repo_id, exc)

    def refresh_index(self, project: ChatProject) -> Optional[Dict[str, Any]]:
        """Состояние индексации (задача синхронизации и статистика базы)."""
        kb = self.repo._knowledge
        if kb is None or not project.collection_id:
            return None
        stats: Dict[str, Any] = {}
        try:
            if project.index_status == "indexing" and project.index_job_id:
                job = kb.get_job(project.index_job_id)
                status = job.get("status")
                stats["job"] = {"status": status, "progress": job.get("progress"), "stage": job.get("stage_title")}
                if status == "completed":
                    project.index_status = "ready"
                elif status in ("failed", "cancelled"):
                    project.index_status, project.index_error = "failed", job.get("error") or status
            collection = kb.get_collection(project.collection_id)
            stats.update({"name": collection.get("name"), "status": collection.get("status"),
                          **(collection.get("stats") or {})})
            if project.index_status in ("", "unavailable") and collection.get("status") == "ready":
                project.index_status, project.index_error = "ready", None
        except KnowledgeServiceError as exc:
            stats["error"] = str(exc)
        self._db.update_chat_project(project.chat_id, index_status=project.index_status,
                                     index_error=project.index_error, updated_at=int(time.time()))
        return stats

    # ------------------------------------------------------------------
    # Выполнение
    # ------------------------------------------------------------------

    def execute(
        self, route: Route, chat: Chat, agent: Agent, user_msg: Message, draft: Message, *, branch: Optional[int],
        use_stream: bool, token, started: float,
    ) -> Iterator[dict]:
        """Генератор событий запуска для команды (последнее — `done`)."""
        repo = self.repo
        cmd = COMMANDS_BY_NAME.get(route.name)
        in_context = cmd.in_context if cmd is not None else False
        self._db.update_message_fields(user_msg.id, command=json.dumps(
            {"name": route.name, "role": "request", "context": in_context}, ensure_ascii=False))
        notes: List[str] = []
        try:
            if cmd is None:
                result = LocalResult(f"Неизвестная команда `/{route.name}`. `/help` — список команд.",
                                     card={"type": "error", "text": f"Неизвестная команда /{route.name}"})
                yield from self._finish_local(route, chat, draft, result)
                return
            if route.link and route.name != "project":
                yield {"type": "status", "status": "Открываю проект"}
                try:
                    project, _ = self.open_project(chat, route.link)
                    notes.append(f"Проект: **{project.name}**, ветка {project.branch or '—'}.")
                    repo._publish_chat_event("chat_updated", chat.id, chat.agent_id)
                except DevError as exc:
                    if self.project(chat.id) is None or not route.link_auto:
                        raise
                    _logger.info("Ссылка %r в тексте не открылась как проект: %s", route.link, exc)
            if cmd.kind == "llm":
                yield from self._llm_command(cmd, route, chat, agent, user_msg, draft, branch=branch,
                                             use_stream=use_stream, token=token, started=started, notes=notes)
                return
            yield {"type": "status", "status": f"Выполняю /{cmd.name}"}
            handler = getattr(self, f"_cmd_{cmd.name}")
            result = yield from handler(route, chat, agent, use_stream=use_stream, token=token)
            if notes and result is not None:
                result.content = "\n".join(notes) + "\n\n" + result.content
            yield from self._finish_local(route, chat, draft, result)
        except DevError as exc:
            yield from self._finish_local(route, chat, draft, LocalResult(
                str(exc), card={"type": "error", "text": str(exc)}))

    def _finish_local(self, route: Route, chat: Chat, draft: Message, result: LocalResult) -> Iterator[dict]:
        cmd = COMMANDS_BY_NAME.get(route.name)
        meta: Dict[str, Any] = {"name": route.name, "role": "result", "kind": cmd.kind if cmd else "local",
                                "context": False}
        if result.card is not None:
            meta["card"] = result.card
        if result.confirm is not None:
            meta["confirm"] = result.confirm
        if route.name == "clear":
            meta["boundary"] = True
        saved = self.repo._finalize_draft(draft, content=result.content, status="complete", command=meta)
        self._db.touch_chat(chat.id)
        if route.name in ("project", "status", "clear"):
            self.repo._publish_chat_event("chat_updated", chat.id, chat.agent_id)
        yield {"type": "message_saved", "message": saved}
        yield {"type": "done", "message": saved}

    def _require_project(self, chat: Chat) -> ChatProject:
        project = self.project(chat.id)
        if project is None:
            raise DevError(NO_PROJECT_NOTICE)
        return project

    # ---- локальные команды ----------------------------------------------

    def _cmd_help(self, route: Route, chat: Chat, agent: Agent, **_) -> Iterator[dict]:
        commands = self.commands_for(chat)
        target = route.args.strip().lstrip("/").lower()
        if target:
            cmd = next((c for c in commands if c["name"] == target), None)
            if cmd is None:
                raise DevError(f"Команды /{target} нет. `/help` — список команд.")
            text = f"**{cmd['usage']}** — {cmd['title']}\n\n{cmd['description']}"
            if not cmd["available"]:
                text += f"\n\nСейчас недоступна: {cmd['reason']}."
            return LocalResult(text, card={"type": "help", "commands": [cmd], "detail": True})
            yield  # pragma: no cover — генератор
        lines = []
        for group in (GROUP_HELP, GROUP_QUESTIONS, GROUP_CHANGES):
            lines.append(f"**{group}**")
            for c in (c for c in commands if c["group"] == group):
                mark = "" if c["available"] else f" _(недоступна: {c['reason']})_"
                lines.append(f"- `{c['usage']}` — {c['description']}{mark}")
            lines.append("")
        project = self.project(chat.id)
        lines.append(f"Проект: **{project.name}**." if project else NO_PROJECT_NOTICE)
        lines.append("Обычный текст при открытом проекте — вопрос о проекте (/ask).")
        return LocalResult("\n".join(lines).strip(), card={
            "type": "help", "commands": commands, "project": project.name if project else None,
            "read_only": chat.settings.dev_read_only})
        yield  # pragma: no cover

    def _project_card(self, project: ChatProject, info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        card = {"type": "project", "name": project.name, "kind": project.kind, "link": project.link,
                "branch": project.branch, "default_branch": project.default_branch, "remote": project.remote,
                "writable": project.writable, "index_status": project.index_status,
                "index_error": project.index_error}
        if info:
            card["head"] = info.get("head")
            if info.get("note"):
                card["note"] = info["note"]
        return card

    def _index_text(self, project: ChatProject) -> str:
        return {
            "indexing": "база знаний проекта индексируется (README, docs, схемы, API) — первые ответы могут быть "
                        "без неё",
            "ready": "база знаний проекта готова",
            "failed": f"индексация не удалась: {project.index_error}",
            "unavailable": f"база знаний недоступна: {project.index_error}",
        }.get(project.index_status, "база знаний не подключена (нет knowledge_service)")

    def _cmd_project(self, route: Route, chat: Chat, agent: Agent, **_) -> Iterator[dict]:
        if not route.link:
            project = self.project(chat.id)
            if project is None:
                return LocalResult(NO_PROJECT_NOTICE, card={"type": "project", "name": None})
            self.refresh_index(project)
            text = (f"Текущий проект: **{project.name}** ({'локальная копия' if project.kind == 'local' else 'клон'}), "
                    f"ветка {project.branch or '—'}; {self._index_text(project)}.")
            return LocalResult(text, card=self._project_card(project))
        yield {"type": "status", "status": "Открываю проект"}
        project, info = self.open_project(chat, route.link)
        head = info.get("head") or {}
        lines = [
            f"Открыт проект **{project.name}** — "
            + ("локальная рабочая копия" if project.kind == "local" else "клон на сервере") + ".",
            f"Ветка: {project.branch or '—'}"
            + (f" (по умолчанию {project.default_branch})" if project.default_branch and project.default_branch != project.branch else "")
            + (f", последний коммит {_short(head.get('sha'))} «{head.get('subject')}»" if head else "") + ".",
            f"База знаний: {self._index_text(project)}.",
        ]
        if info.get("note"):
            lines.append(info["note"])
        if not project.writable:
            lines.append("Коммиты недоступны: проект открыт по URL.")
        lines.append("Задавайте вопросы обычным текстом или командами — `/help`.")
        return LocalResult("\n".join(lines), card=self._project_card(project, info))

    def _cmd_status(self, route: Route, chat: Chat, agent: Agent, **_) -> Iterator[dict]:
        project = self.project(chat.id)
        settings = chat.settings
        card: Dict[str, Any] = {
            "type": "status", "model": settings.model, "commands_enabled": settings.dev_commands_enabled,
            "read_only": settings.dev_read_only, "project": None,
        }
        lines = [f"Модель: {settings.model}. Команды: вкл. Только чтение: {'вкл.' if settings.dev_read_only else 'выкл.'}"]
        if project is None:
            lines.append(NO_PROJECT_NOTICE)
            return LocalResult("\n".join(lines), card=card)
        yield {"type": "status", "status": "Получаю состояние репозитория"}
        status = self.call("repo_status", chat.id, repo_id=project.repo_id)
        index = self.refresh_index(project)
        if status.get("branch") and status["branch"] != project.branch:
            project.branch = status["branch"]
            self._db.update_chat_project(chat.id, branch=project.branch)
        head = status.get("head") or {}
        files = status.get("files") or []
        card.update({
            "project": self._project_card(project), "branch": status.get("branch"), "upstream": status.get("upstream"),
            "ahead": status.get("ahead", 0), "behind": status.get("behind", 0), "head": head,
            "clean": status.get("clean"), "changed_files": status.get("changed_files", 0),
            "staged": status.get("staged", 0), "unstaged": status.get("unstaged", 0),
            "untracked": status.get("untracked", 0), "files": files[:50], "index": index,
            "last_commit": project.last_commit,
        })
        lines.insert(0, f"**{project.name}** — {'локальная копия' if project.kind == 'local' else 'клон по URL'}, "
                        f"ветка {status.get('branch') or '—'}"
                        + (f" (origin: +{status.get('ahead', 0)} / −{status.get('behind', 0)})"
                           if status.get("upstream") else "") + ".")
        if head:
            lines.insert(1, f"Последний коммит: {_short(head.get('sha'))} «{head.get('subject')}», "
                            f"{head.get('author')}, {head.get('date', '')[:10]}.")
        if status.get("clean"):
            lines.append("Изменений нет.")
        else:
            lines.append(f"Изменено файлов: {status.get('changed_files')} (подготовлено {status.get('staged')}, "
                         f"новых {status.get('untracked')}):")
            lines += [f"- {f['path']} — {f['status']}" for f in files[:20]]
        index_line = self._index_text(project)
        if index and index.get("documents") is not None:
            index_line += f" (документов {index.get('documents')}, фрагментов {index.get('chunks')})"
        lines.append(f"База знаний: {index_line}.")
        return LocalResult("\n".join(lines), card=card)

    def _cmd_clear(self, route: Route, chat: Chat, agent: Agent, **_) -> Iterator[dict]:
        return LocalResult("Контекст очищен: сообщения выше не уходят в запросы к модели.",
                           card={"type": "clear"})
        yield  # pragma: no cover

    def _cmd_diff(self, route: Route, chat: Chat, agent: Agent, **_) -> Iterator[dict]:
        project = self._require_project(chat)
        yield {"type": "status", "status": "Получаю diff"}
        diff = self.call("repo_diff", chat.id, repo_id=project.repo_id, spec=route.args or None)
        files = diff.get("files") or []
        untracked = diff.get("untracked") or []
        card = {"type": "diff", **{k: diff.get(k) for k in ("spec", "title", "files", "untracked", "stats",
                                                            "patch", "truncated")}}
        if not files and not untracked:
            return LocalResult(f"Изменений нет ({diff.get('title')}).", card=card)
        stats = diff.get("stats") or {}
        lines = [f"**{diff.get('title')}**: файлов {stats.get('files', 0)}, "
                 f"+{stats.get('additions', 0)} / −{stats.get('deletions', 0)}."]
        lines += [f"- {f['path']} — {f.get('status')} (+{f.get('additions', 0)} −{f.get('deletions', 0)})"
                  for f in files[:30]]
        if untracked:
            lines.append("Новые файлы (не в git): " + ", ".join(untracked[:20]))
        patch = diff.get("patch") or ""
        if patch:
            shown = patch[:DIFF_IN_TEXT_CHARS]
            lines.append(f"\n```diff\n{shown.rstrip()}\n```")
            if len(patch) > len(shown) or diff.get("truncated"):
                lines.append("_diff обрезан_")
        return LocalResult("\n".join(lines), card=card)

    def _cmd_commit(self, route: Route, chat: Chat, agent: Agent, *, use_stream: bool, token) -> Iterator[dict]:
        if chat.settings.dev_read_only:
            raise DevError(READ_ONLY_NOTICE)
        project = self._require_project(chat)
        if not project.writable:
            raise DevError("Коммит возможен только в локальной рабочей копии, а проект открыт по URL.")
        yield {"type": "status", "status": "Смотрю изменения"}
        status = self.call("repo_status", chat.id, repo_id=project.repo_id)
        files = status.get("files") or []
        staged_only = any(f.get("staged") for f in files)
        commit_files = [f["path"] for f in files if f.get("staged")] if staged_only else [f["path"] for f in files]
        if not commit_files:
            return LocalResult("Нет изменений для коммита.", card={"type": "status_note", "text": "Нет изменений"})
        diff = self.call("repo_diff", chat.id, repo_id=project.repo_id, spec="staged" if staged_only else None,
                         max_chars=DIFF_FOR_MODEL_CHARS)
        log = self.call("repo_log", chat.id, repo_id=project.repo_id, limit=10)
        recent = "\n".join(f"- {c.get('subject')}" for c in log.get("commits") or [])
        untracked = [f["path"] for f in files if f.get("untracked")] if not staged_only else []
        yield {"type": "status", "status": "Модель составляет сообщение коммита"}
        prompt = (
            "Составь сообщение git-коммита для изменений ниже.\n"
            "Формат: первая строка — суть изменения, до 72 символов, без точки в конце; затем пустая строка и "
            "1–5 строк «- …» с главными изменениями (для мелкого изменения хватит первой строки).\n"
            "Пиши на языке и в стиле последних коммитов проекта. Верни только сообщение коммита, без пояснений, "
            "кавычек и блоков кода.\n\n"
            f"Последние коммиты проекта:\n{recent or '—'}\n\n"
            + (f"Пожелание разработчика: {route.args}\n\n" if route.args else "")
            + f"Файлы коммита: {', '.join(commit_files)}\n"
            + (f"Новые файлы (их содержимого нет в diff): {', '.join(untracked)}\n" if untracked else "")
            + f"\nDiff:\n{diff.get('patch') or '(пусто)'}"
        )
        message = self._simple_completion(chat, prompt, token)
        if not message:
            raise DevError("Модель не предложила сообщение коммита — попробуйте ещё раз или /commit <пожелание>.")
        confirm = {"action": "commit", "state": "pending", "message": message, "files": commit_files,
                   "staged_only": staged_only, "branch": status.get("branch")}
        text = (f"Предлагаю коммит в ветку {status.get('branch')} "
                f"({'подготовленные' if staged_only else 'все изменённые'} файлы: {len(commit_files)}):\n\n"
                f"```\n{message}\n```\n\nПодтвердите или исправьте сообщение в карточке.")
        return LocalResult(text, card={"type": "commit", "files": commit_files}, confirm=confirm)

    def _cmd_undo(self, route: Route, chat: Chat, agent: Agent, **_) -> Iterator[dict]:
        if chat.settings.dev_read_only:
            raise DevError(READ_ONLY_NOTICE)
        project = self._require_project(chat)
        if not project.writable:
            raise DevError("Откат возможен только в локальной рабочей копии, а проект открыт по URL.")
        if not project.last_commit:
            raise DevError("Нечего откатывать: в этом чате ещё не было коммитов через /commit.")
        yield {"type": "status", "status": "Проверяю последний коммит"}
        status = self.call("repo_status", chat.id, repo_id=project.repo_id)
        head = status.get("head") or {}
        if head.get("sha") != project.last_commit:
            raise DevError(f"Последний коммит ветки уже не {_short(project.last_commit)} — откат через /undo "
                           "невозможен.")
        confirm = {"action": "undo", "state": "pending", "commit": project.last_commit, "subject": head.get("subject")}
        text = (f"Откатить коммит {_short(project.last_commit)} «{head.get('subject')}»? Изменения останутся "
                "подготовленными к коммиту (git reset --soft). Подтвердите в карточке.")
        return LocalResult(text, card={"type": "undo"}, confirm=confirm)

    def _simple_completion(self, chat: Chat, prompt: str, token) -> str:
        repo = self.repo
        provider_name, model_id = _split_model(chat.settings.model)
        provider = repo._registry.get(provider_name)
        settings = dataclasses.replace(
            chat.settings, tools_json="", tool_choice="auto", json_mode=False, stop_sequences=[], max_tokens=None,
            temperature=min(chat.settings.temperature, 0.3), logprobs=False,
        )
        if provider_name == "ollama" and repo._local_models is not None:
            repo._local_models.ensure_ready(model_id, settings.num_ctx)
        messages = [ProviderMessage("system", "Ты пишешь сообщения git-коммитов."), ProviderMessage("user", prompt)]
        try:
            result = provider.chat(model_id, messages, settings)
        except ProviderError as exc:
            raise DevError(f"Модель не ответила: {exc}") from exc
        if token is not None:
            token.raise_if_cancelled()
        return _clean_commit_message(result.content)

    # ---- подтверждение /commit и /undo ----------------------------------

    def confirm(self, chat: Chat, message: Message, payload: Dict[str, Any]) -> Message:
        meta = command_meta(message)
        confirm = dict(meta.get("confirm") or {})
        if confirm.get("state") != "pending":
            raise DevError("Подтверждать нечего: действие уже выполнено или отменено.")
        project = self._require_project(chat)
        if chat.settings.dev_read_only:
            raise DevError(READ_ONLY_NOTICE)
        try:
            if confirm.get("action") == "commit":
                text = (payload.get("message") or confirm.get("message") or "").strip()
                if not text:
                    raise DevError("Пустое сообщение коммита.")
                files = payload.get("files") or ([] if confirm.get("staged_only") else confirm.get("files")) or None
                result = self.call("repo_commit", chat.id, repo_id=project.repo_id, message=text, files=files)
                project.last_commit = result.get("commit")
                self._db.update_chat_project(chat.id, last_commit=project.last_commit, updated_at=int(time.time()))
                confirm.update(state="done", message=text, result=result)
                note = f"Коммит {result.get('short')} «{result.get('subject')}» создан в ветке {result.get('branch')}."
            elif confirm.get("action") == "undo":
                result = self.call("repo_undo_commit", chat.id, repo_id=project.repo_id, commit=confirm.get("commit"))
                self._db.update_chat_project(chat.id, last_commit=None, updated_at=int(time.time()))
                confirm.update(state="done", result=result)
                note = f"Коммит {_short(confirm.get('commit'))} откачен, изменения остались подготовленными."
            else:
                raise DevError("Неизвестное действие.")
        except DevError as exc:
            confirm.update(state="failed", error=str(exc))
            note = f"Не выполнено: {exc}"
        meta["confirm"] = confirm
        self._db.update_message_fields(message.id, command=json.dumps(meta, ensure_ascii=False),
                                       content=f"{message.content}\n\n{note}")
        return self._db.get_message(message.id)

    def reject(self, message: Message) -> Message:
        meta = command_meta(message)
        confirm = dict(meta.get("confirm") or {})
        if confirm.get("state") != "pending":
            raise DevError("Отменять нечего: действие уже выполнено или отменено.")
        confirm["state"] = "cancelled"
        meta["confirm"] = confirm
        self._db.update_message_fields(message.id, command=json.dumps(meta, ensure_ascii=False),
                                       content=f"{message.content}\n\nОтменено.")
        return self._db.get_message(message.id)

    # ---- команды с ответом модели ----------------------------------------

    def _system_block(self, project: ChatProject) -> str:
        kind = "локальная рабочая копия" if project.kind == "local" else "клон репозитория по URL"
        return (
            f"[ПРОЕКТ]\nТы — ассистент разработчика проекта {project.name} ({kind}), ветка {project.branch or '—'}.\n"
            "Отвечаешь на вопросы о проекте по его документации, коду и истории git.\n"
            "Инструменты repo_* работают с этим проектом; repo_id подставляется автоматически — не указывай его.\n"
            "- Если в сообщении есть фрагменты документации [n] — опирайся на них и ссылайся номерами [n].\n"
            "- Детали проверяй в коде: структура — repo_list_files, поиск — repo_grep, файл — repo_read_file "
            "(с диапазоном строк), история — repo_log, repo_show, repo_blame.\n"
            "- Ссылайся на файлы как `путь:строка`, на коммиты — коротким sha.\n"
            "- Режим только чтения: ты не меняешь файлы, не делаешь коммиты и не выполняешь команды. Если просят "
            "изменить код — предложи правку текстом (diff); закоммитить готовые изменения разработчик может "
            "командой /commit.\n"
            "- Чего нет ни в документации, ни в коде — так и скажи, не выдумывай.\n"
            "- Отвечай на языке вопроса, по делу."
        )

    def _task_text(self, cmd: str, args: str, extra: str = "") -> str:
        if cmd == "explain":
            return (f"Объясни {args}: назначение, как устроено, ключевые классы и функции, связи с другими частями "
                    "проекта, важные нюансы. Если это не путь к файлу или папке — найди символ через repo_grep.")
        if cmd == "history":
            return (f"Расскажи историю {args}: почему код стал таким. Возьми repo_log с path, для ключевых коммитов — "
                    "repo_show, для важных строк — repo_blame. Дай хронологию значимых изменений (sha, дата, автор, "
                    "что и зачем изменилось — по сообщениям коммитов и диффам).")
        if cmd == "review":
            pr = re.match(r"^(?:pr|mr)[:/#]?(\d+)$|^#(\d+)$", (args or "").strip(), re.I)
            pr_hint = ""
            if pr:
                number = pr.group(1) or pr.group(2)
                pr_hint = (f"Это pull request #{number}: рабочая копия — базовая ветка, поэтому код из PR читай с "
                           f"ref=\"pr:{number}\" (repo_read_file, repo_grep).\n")
            return (
                "Сделай код-ревью изменений ниже. Фрагменты [n] — документация и связанный код проекта из базы "
                "знаний; при необходимости читай файлы (repo_read_file, repo_grep).\n" + pr_hint
                + "Ответ строго в формате (заголовки — ровно такие):\n"
                f"{REVIEW_HEADINGS['summary']}\n1–3 предложения: что меняет PR.\n"
                f"{REVIEW_HEADINGS['bugs']}\nошибки и риски на выполнении: неверная логика, граничные случаи, "
                "исключения, гонки, безопасность, утечки ресурсов.\n"
                f"{REVIEW_HEADINGS['architecture']}\nнарушение структуры и договорённостей проекта (по "
                "документации [n]), связность, дублирование, неверный слой, несовместимость API.\n"
                f"{REVIEW_HEADINGS['recommendations']}\nчто улучшить: тесты, читаемость, документация.\n"
                "В каждом разделе — список «- `файл:строка` — проблема → как исправить», со ссылкой [n], если "
                "опираешься на фрагмент. Пустой раздел — строка «- нет». Не придумывай проблем, которых нет.\n\n"
                + extra
            )
        return args

    def _llm_command(
        self, cmd: DevCommand, route: Route, chat: Chat, agent: Agent, user_msg: Message, draft: Message, *,
        branch: Optional[int], use_stream: bool, token, started: float, notes: List[str],
    ) -> Iterator[dict]:
        from .repository import _TurnAccumulator  # noqa: WPS433 — цикл импорта

        repo = self.repo
        project = self._require_project(chat)
        if cmd.name in ("explain", "history") and not route.args:
            raise DevError(f"Укажите, что именно: `{cmd.usage}`.")
        if cmd.name == "ask" and not route.args:
            raise DevError("Задайте вопрос: `/ask <вопрос>`.")
        extra = ""
        diff: Optional[Dict[str, Any]] = None
        if cmd.name == "review":
            yield {"type": "status", "status": "Получаю изменения для ревью"}
            diff = self.call("repo_diff", chat.id, repo_id=project.repo_id, spec=route.args or None,
                             max_chars=DIFF_FOR_MODEL_CHARS)
            if not diff.get("files") and not diff.get("untracked"):
                raise DevError(f"Нет изменений для ревью ({diff.get('title')}). Для ветки: `/review main..feature`.")
            extra = (f"Изменения: {diff.get('title')}\n"
                     + ("Новые файлы (не в git): " + ", ".join(diff.get("untracked") or []) + "\n" if diff.get("untracked") else "")
                     + f"```diff\n{diff.get('patch') or ''}\n```" + ("\n(diff обрезан)" if diff.get("truncated") else ""))
        tool_names = {"history": TOOLS_HISTORY, "review": TOOLS_REVIEW}.get(cmd.name, TOOLS_QUESTIONS)
        tools, inject = self._tool_schemas(tool_names)
        for exposed in inject:
            inject[exposed] = {"repo_id": project.repo_id}
        settings = dataclasses.replace(chat.settings, tools_json=json.dumps(tools, ensure_ascii=False),
                                       tool_choice="auto", json_mode=False)
        history = filter_context(repo._history_for_context(chat.id, branch, before_id=user_msg.id))
        history = [m for m in repo._context_messages(history) if m.role in ("user", "assistant")][-HISTORY_MESSAGES:]
        messages: List[ProviderMessage] = []
        if settings.system_prompt and settings.system_prompt.strip():
            messages.append(ProviderMessage("system", settings.system_prompt))
        messages.append(ProviderMessage("system", self._system_block(project)))
        invariants = [m for m in repo._build_task_and_invariant_context(
            dataclasses.replace(chat, settings=dataclasses.replace(chat.settings, task_tracking_enabled=False)), agent)]
        messages += invariants
        messages += [ProviderMessage(m.role, m.content) for m in history]
        question = self._task_text(cmd.name, route.args, extra)
        acc = _TurnAccumulator()
        acc.inject_args = inject
        rag: Optional[RagOutcome] = None
        if cmd.name in ("ask", "explain", "review"):
            queries = review_queries(diff) if diff is not None else [route.args]
            rag = yield from self._search_docs(chat, project, route.args, queries=queries)
            if rag is not None:
                acc.rag = rag.to_dict()
                yield {"type": "rag_context", "rag": acc.rag}
                question = augmented_question(question, rag) if rag.status == RAG_STATUS_OK else question
        messages.append(ProviderMessage("user", question))
        provider_name, model_id = _split_model(settings.model)
        provider = repo._registry.get(provider_name)
        try:
            yield from repo._ensure_local_model(provider_name, model_id, settings, acc, token)
            status_text = "Выполняется запрос к модели"
            yield {"type": "status", "status": status_text}
            final_result, final_content = yield from repo._model_turn(
                provider, model_id, messages, settings, chat, agent, acc, use_stream=use_stream, token=token,
                status_text=status_text, max_tool_rounds=AgentConfig.DEV_MAX_TOOL_ROUNDS,
            )
        except Exception as exc:  # noqa: BLE001 — отмена и ошибки провайдера обрабатывает Repository
            from .events import RunCancelled

            if isinstance(exc, RunCancelled):
                yield repo._terminal_after_interrupt(draft, acc, started, cancelled=exc)
                return
            if isinstance(exc, ProviderError):
                yield repo._terminal_after_interrupt(draft, acc, started, error=str(exc))
                return
            raise
        if rag is not None:
            final_content = finalize_answer(final_content, rag)
            acc.rag = rag.to_dict()
        if notes and cmd.name != "review":
            final_content = "\n".join(notes) + "\n\n" + final_content
        if final_result.usage.prompt_tokens is not None:
            self._db.update_message_prompt_tokens(user_msg.id, final_result.usage.prompt_tokens)
        meta = {"name": cmd.name, "role": "result", "kind": "llm", "context": True,
                "project": project.name, "implicit": route.implicit}
        if diff is not None:
            # Ревью — ещё и структурой: разделы, файлы diff (для CI и карточки).
            meta["card"] = {
                "type": "review", "title": diff.get("title"), "spec": route.args or None,
                "files": [{k: f.get(k) for k in ("path", "status", "additions", "deletions")}
                          for f in diff.get("files") or []],
                "stats": diff.get("stats"), **parse_review(final_content),
            }
        saved = repo._finalize_draft(
            draft, content=final_content, status="complete", result=final_result,
            duration_ms=int((time.monotonic() - started) * 1000), acc=acc, command=meta,
        )
        self._db.touch_chat(chat.id)
        yield {"type": "message_saved", "message": saved}
        yield {"type": "done", "message": saved}

    def _search_docs(self, chat: Chat, project: ChatProject, query: str,
                     queries: Optional[List[str]] = None) -> Iterator[dict]:
        """Поиск по базе знаний проекта (документация и код) и базам агента,
        если у него включён RAG. Несколько запросов (ревью — по файлам и
        функциям из diff) объединяются: без повторов, лучшие сверху. None —
        искать негде."""
        kb = self.repo._knowledge
        settings = chat.settings
        collections = [project.collection_id] if project.collection_id else []
        if settings.rag_enabled:
            collections += [c for c in settings.collection_ids if c not in collections]
        queries = [q.strip() for q in (queries or [query]) if q and q.strip()]
        if kb is None or not collections or not queries:
            return None
        yield {"type": "status", "status": "Поиск в документации и коде проекта"}
        outcome = RagOutcome(query=" | ".join(queries), status=RAG_STATUS_OK, only_from_kb=False,
                             original_query=(query or queries[0]).strip(), quotes_required=False,
                             params={"top_k": settings.rag_top_k, "score_threshold": settings.rag_score_threshold,
                                     "queries": len(queries)})
        merged: Dict[Any, Dict[str, Any]] = {}
        errors = []
        for q in queries:
            try:
                found = kb.retrieve(q, collections, settings.rag_top_k, settings.rag_score_threshold,
                                    candidate_k=max(settings.rag_candidate_k, settings.rag_top_k),
                                    rerank=settings.rag_rerank, rerank_model=settings.rag_rerank_model or None,
                                    rerank_threshold=settings.rag_rerank_threshold)
            except KnowledgeServiceError as exc:
                errors.append(str(exc))
                continue
            outcome.stages = found.get("stages") or outcome.stages
            outcome.rerank = found.get("rerank") or outcome.rerank
            for item in found.get("results") or []:
                key = item.get("chunk_id") or id(item)
                if key not in merged or (item.get("score") or 0) > (merged[key].get("score") or 0):
                    merged[key] = item
        if errors and not merged:
            outcome.status, outcome.error = RAG_STATUS_UNAVAILABLE, errors[0]
            return outcome
        ranked = sorted(merged.values(), key=lambda r: -(r.get("score") or 0))
        outcome.sources = select_sources(ranked, settings.rag_context_tokens)
        if not outcome.sources:
            outcome.status = RAG_STATUS_NOT_FOUND
        return outcome

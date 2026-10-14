"""
agents_core.ci_review
=======================

«День 32» — автоматическое ревью pull request'ов из CI (GitHub Action).

Инициатор — событие PR в GitHub: Action вызывает `POST /dev/reviews` с
координатами PR (репозиторий, номер, базовая ветка, коммит), опрашивает
`GET /dev/reviews/{id}` и публикует `comment_markdown` комментарием в PR.

Алгоритм `CiReviewService.start`:

1. агент — по `agent_id` или по имени (`agent`);
2. ключ PR — `"<хост>/<owner>/<repo>#<номер>"`;
3. чат PR — по `(agent_id, external_key)`; нет — создаётся из настроек
   агента («Команды разработчика» вкл., «Только чтение» вкл.), с названием
   «PR owner/repo#12: <заголовок>» и `source="ci"`; гонку двух запросов
   разрешает уникальный индекс (второй получает тот же чат);
4. повтор для того же коммита (`client_request_id = review:<ключ>@<sha>`)
   возвращает уже созданное ревью — модель второй раз не вызывается;
5. идущее в чате ревью более старого коммита отменяется — новый коммит
   важнее; другой идущий ответ (пользователь пишет в чате) — ждём, затем 409;
6. в чат уходит сообщение `/review pr:12 <ссылка на репозиторий>/tree/<база>`
   от имени «GitHub Actions» — дальше работает обычная команда `/review`
   (`dev_assistant`): проект открывается по ссылке, diff PR берётся через
   mcp_server, поиск по базе знаний проекта (документация + код), ответ
   модели в разделах «Потенциальные баги», «Архитектурные проблемы»,
   «Рекомендации».
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .repository import NotFoundError, Repository, ValidationError
from .runs import RunConflictError, RunManager

CI_SOURCE = "ci"
REVIEW_REQUEST_PREFIX = "review:"
#: Сколько ждать завершения отменённого (или чужого) запуска в чате PR, с.
WAIT_ACTIVE_SECONDS = 60.0

_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
_BRANCH_RE = re.compile(r"^[A-Za-z0-9._/@+-]+$")


def parse_repo_url(url: str) -> Tuple[str, str, str]:
    """https://host/owner/repo(.git) → (хост, owner/repo, ссылка без .git)."""
    raw = (url or "").strip().rstrip("/")
    if "://" not in raw:
        raw = "https://" + raw
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    parts = [p for p in parsed.path.split("/") if p]
    if parts and parts[-1].endswith(".git"):
        parts[-1] = parts[-1][:-4]
    if parsed.scheme not in ("http", "https") or not host or len(parts) < 2:
        raise ValidationError(f"repo: ожидается ссылка https://<хост>/<owner>/<repo>, получено {url!r}")
    if not all(re.match(r"^[A-Za-z0-9._-]+$", p) for p in parts):
        raise ValidationError(f"repo: недопустимый путь репозитория {url!r}")
    slug = "/".join(parts)
    port = f":{parsed.port}" if parsed.port else ""
    return host, slug, f"{parsed.scheme}://{host}{port}/{slug}"


def pr_key(host: str, slug: str, number: int) -> str:
    return f"{host}/{slug}#{number}"


class CiReviewService:
    #: Поиск-или-создание чата PR — под одной блокировкой на процесс (плюс
    #: уникальный индекс `(agent_id, external_key)` на случай нескольких процессов).
    _chat_lock = threading.Lock()

    def __init__(self, repo: Repository, manager: RunManager):
        self.repo = repo
        self.manager = manager
        self._db = repo._db

    # ------------------------------------------------------------------

    def _resolve_agent(self, agent_id: Optional[str], agent_name: Optional[str]):
        if agent_id:
            return self.repo.get_agent(agent_id)
        name = (agent_name or "").strip().casefold()
        if not name:
            raise ValidationError("укажите agent_id или agent (имя агента)")
        matches = [a for a in self.repo.list_agents() if a.name.strip().casefold() == name]
        if not matches:
            raise NotFoundError(f"агент «{agent_name}» не найден")
        if len(matches) > 1:
            raise ValidationError(f"несколько агентов с именем «{agent_name}» — укажите agent_id")
        return matches[0]

    def find_or_create_chat(self, agent, key: str, slug: str, number: int, title: Optional[str]):
        """Чат PR по постоянному ключу; нет — новый (возвращает (чат, создан))."""
        with self._chat_lock:
            chat = self._db.find_chat_by_external_key(agent.id, key)
            if chat is not None:
                return chat, False
            chat_title = f"PR {slug}#{number}" + (f": {title.strip()}" if title and title.strip() else "")
            try:
                chat = self.repo.create_chat(
                    agent.id, chat_title[:200], source=CI_SOURCE, external_key=key,
                    settings_overrides={"dev_commands_enabled": True, "dev_read_only": True},
                )
            except Exception:  # noqa: BLE001 — гонка между процессами: чат уже создан
                chat = self._db.find_chat_by_external_key(agent.id, key)
                if chat is None:
                    raise
                return chat, False
            return chat, True

    def _free_chat(self, chat_id: str, client_request_id: str) -> None:
        """Перед новым ревью: старое ревью этого PR отменяется, чужой ответ —
        дожидаемся (не дольше `WAIT_ACTIVE_SECONDS`)."""
        active = self._db.active_run_for_chat(chat_id)
        if active is None:
            return
        if (active.client_request_id or "").startswith(REVIEW_REQUEST_PREFIX) and \
                active.client_request_id != client_request_id:
            self.manager.cancel(active.id)
        deadline = time.monotonic() + WAIT_ACTIVE_SECONDS
        while time.monotonic() < deadline:
            self.manager.wait(active.id, timeout=min(5.0, max(0.1, deadline - time.monotonic())))
            current = self._db.active_run_for_chat(chat_id)
            if current is None:
                return
            active = current
        raise RunConflictError("В чате PR ещё формируется ответ", active_run_id=active.id)

    def start(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        agent = self._resolve_agent(payload.get("agent_id"), payload.get("agent"))
        if not agent.settings.model:
            raise ValidationError("у агента не задана модель")
        host, slug, url = parse_repo_url(payload.get("repo") or "")
        number = int(payload.get("pr") or 0)
        if number < 1:
            raise ValidationError("pr: номер pull request'а")
        base = (payload.get("base") or "").strip()
        if base and (base.startswith("-") or not _BRANCH_RE.match(base) or ".." in base):
            raise ValidationError(f"base: недопустимая ветка {base!r}")
        head_sha = (payload.get("head_sha") or "").strip()
        if head_sha and not _SHA_RE.match(head_sha):
            raise ValidationError("head_sha: ожидается sha коммита")
        key = pr_key(host, slug, number)
        chat, created = self.find_or_create_chat(agent, key, slug, number, payload.get("title"))
        request_id = f"{REVIEW_REQUEST_PREFIX}{key}@{head_sha or uuid.uuid4().hex}"
        existing = self._db.find_run_by_client_request(chat.id, request_id)
        if existing is None:
            self._free_chat(chat.id, request_id)
            link = f"{url}/tree/{base}" if base else url
            text = f"/review pr:{number} {link}"
            run, _, _ = self.manager.start_message_run(
                chat.id, text, client_request_id=request_id, source=CI_SOURCE, use_stream=False,
            )
        else:
            run = existing
        return {"review_id": run.id, "chat_id": chat.id, "chat_title": chat.title, "created_chat": created,
                "status": self._status(run.status), "key": key, "head_sha": head_sha or None}

    # ------------------------------------------------------------------

    @staticmethod
    def _status(run_status: str) -> str:
        return {"queued": "running", "running": "running", "succeeded": "done", "failed": "failed",
                "cancelled": "cancelled", "interrupted": "failed"}.get(run_status, run_status)

    def get(self, review_id: str) -> Dict[str, Any]:
        run = self.manager.get(review_id)
        if not (run.client_request_id or "").startswith(REVIEW_REQUEST_PREFIX):
            raise NotFoundError(f"ревью {review_id} не найдено")
        chat = self.repo.get_chat(run.chat_id)
        message = self._db.get_message(run.assistant_message_id) if run.assistant_message_id else None
        key = run.client_request_id[len(REVIEW_REQUEST_PREFIX):]
        sha = key.rsplit("@", 1)[-1]
        result: Dict[str, Any] = {
            "review_id": run.id, "chat_id": chat.id, "chat_title": chat.title, "key": chat.external_key,
            "head_sha": sha if _SHA_RE.match(sha) else None, "status": self._status(run.status),
            "current_status": run.current_status, "error": run.error,
        }
        if result["status"] == "cancelled":
            result["error"] = "ревью отменено: в PR пришёл более новый коммит"
        if result["status"] != "done" or message is None:
            if result["status"] == "failed" and message is not None and not result["error"]:
                result["error"] = message.error
            return result
        meta = _command(message)
        card = meta.get("card") or {}
        if card.get("type") != "review":
            # Ревью не состоялось (нет изменений, проект не открылся…) — текст ответа и есть причина.
            result.update(status="failed", error=message.content)
            return result
        sources = _sources(message)
        result.update(
            markdown=message.content, summary=card.get("summary") or "", bugs=card.get("bugs") or [],
            architecture=card.get("architecture") or [], recommendations=card.get("recommendations") or [],
            files=card.get("files") or [], stats=card.get("stats"), diff_title=card.get("title"),
            sources=sources, model=chat.settings.model,
        )
        result["comment_markdown"] = comment_markdown(result)
        return result


def _command(message) -> Dict[str, Any]:
    try:
        data = json.loads(message.command) if message.command else {}
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _sources(message) -> List[Dict[str, Any]]:
    try:
        rag = json.loads(message.rag) if message.rag else {}
    except (TypeError, ValueError):
        return []
    cited = set(rag.get("cited") or [])
    result = []
    for src in rag.get("sources") or []:
        if src.get("n") in cited:
            result.append({"n": src.get("n"), "title": src.get("title"), "section": src.get("section"),
                           "source": src.get("source")})
    return result


def comment_markdown(review: Dict[str, Any]) -> str:
    """Текст комментария в PR: ревью, источники, подпись."""
    sha = (review.get("head_sha") or "")[:7]
    lines = [f"## AI-ревью" + (f" · коммит {sha}" if sha else "")]
    if review.get("diff_title"):
        stats = review.get("stats") or {}
        lines.append(f"_{review['diff_title']}: файлов {stats.get('files', 0)}, "
                     f"+{stats.get('additions', 0)} / −{stats.get('deletions', 0)}_")
    lines += ["", (review.get("markdown") or "").strip()]
    if review.get("sources"):
        lines += ["", "<details><summary>Источники из базы знаний проекта</summary>", ""]
        for src in review["sources"]:
            title = src.get("title") or src.get("source") or "фрагмент"
            section = f" › {src['section']}" if src.get("section") and src.get("section") != title else ""
            lines.append(f"- [{src.get('n')}] {title}{section}")
        lines += ["", "</details>"]
    lines += ["", f"<sub>AgentsCore · модель {review.get('model') or '—'} · чат «{review.get('chat_title')}»</sub>"]
    return "\n".join(lines)

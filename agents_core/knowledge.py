"""
agents_core.knowledge
=======================

RAG: клиент сервиса баз знаний (`knowledge_service`, `POST /api/v1/retrieval`)
и сборка запроса к модели из найденных фрагментов.

Поток при включённом «Использовать RAG» (см. `Repository.execute_message`):
вопрос → переписывание в поисковый запрос (query rewrite, по настройке) →
двухэтапный поиск в knowledge_service (векторный поиск + порог, реранкинг +
порог, топ-K до и после) → блок «Контекст» с пронумерованными фрагментами
перед ИСХОДНЫМ вопросом + инструкция RAG → запрос к модели → проверка
ссылок `[n]` в ответе. Список источников под ответом формирует код по
фактически переданным модели фрагментам, а не модель.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from .tokens import estimate_tokens

log = logging.getLogger("agents_core.knowledge")

#: Уточняющий вопрос короче стольких слов дополняется предыдущим вопросом
#: пользователя — «а сколько там дней?» сам по себе плохо ищется.
FOLLOW_UP_MAX_WORDS = 6

RAG_STATUS_OK = "ok"
RAG_STATUS_NOT_FOUND = "not_found"
RAG_STATUS_UNAVAILABLE = "unavailable"

#: Переписывание запроса: «Нет» / «Дополнять уточняющие» / «Моделью до поиска».
REWRITE_MODES = {"off": "Нет", "follow_up": "Дополнять уточняющие", "llm": "Моделью до поиска"}
#: Реранкинг (второй этап поиска, выполняет knowledge_service).
RERANK_MODES = {"none": "Нет", "heuristic": "Эвристика (без LLM)", "model": "Модель-реранкер"}

#: Сколько последних реплик диалога видит модель переписывания и сколько
#: символов каждой.
REWRITE_HISTORY_MESSAGES = 6
REWRITE_MESSAGE_CHARS = 600

REWRITE_INSTRUCTION = (
    "Ты переписываешь вопрос пользователя в поисковый запрос для базы знаний. "
    "Учитывай предыдущие реплики диалога: раскрой отсылки («это», «там», «а для срочных?»), "
    "аббревиатуры и жаргон, убери вежливые и лишние слова, добавь ключевые термины предметной области. "
    "Не отвечай на вопрос и ничего не придумывай сверх сказанного в диалоге. "
    "Верни только поисковый запрос одной строкой на языке вопроса, без кавычек и пояснений."
)


#: Ответ модели переписывания «новый поиск не нужен»: пользователь просит
#: что-то сделать с предыдущим ответом (повторить, сократить, перевести) —
#: ответ строится по фрагментам предыдущего ответа (`RagOutcome.context_reused`).
NO_SEARCH = "БЕЗ_ПОИСКА"

_NO_SEARCH_RULE = (
    f"Если пользователь не задаёт нового вопроса, а просит сделать что-то с предыдущим ответом ассистента "
    f"(повторить, сократить, пересказать по пунктам, перевести, переформулировать) — новый поиск не нужен: "
    f"верни ровно одно слово {NO_SEARCH}."
)


def rewrite_instruction(memory_lines: Optional[List[str]] = None) -> str:
    """Инструкция модели переписывания: базовая + «БЕЗ_ПОИСКА» + память
    задачи (цель, термины, ограничения), если она есть."""
    parts = [REWRITE_INSTRUCTION, _NO_SEARCH_RULE]
    if memory_lines:
        parts.append(
            "Память текущей задачи пользователя — учитывай её: запрос должен служить цели, термины раскрывай "
            "в зафиксированном значении, ограничения добавляй в запрос, если они важны для поиска:\n"
            + "\n".join(memory_lines)
        )
    return "\n\n".join(parts)


def is_no_search(text: Optional[str]) -> bool:
    normalized = re.sub(r"[\s_\-]+", "_", (text or "").strip().strip("«»\"'`.").upper())
    return normalized == NO_SEARCH


_CITATION_RE = re.compile(r"\[(\d{1,3}(?:\s*[,;]\s*\d{1,3})*)\]")


class KnowledgeServiceError(Exception):
    pass


class KnowledgeClient:
    """HTTP-клиент `knowledge_service`. Синхронный (`requests`) — как и
    остальные вызовы внутри исполнителя запуска."""

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 30.0,
                 session: Optional[requests.Session] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._session = session or requests.Session()

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def retrieve(self, query: str, collection_ids: List[str], top_k: int, score_threshold: Optional[float],
                 candidate_k: Optional[int] = None, rerank: str = "none", rerank_model: Optional[str] = None,
                 rerank_threshold: Optional[float] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"query": query, "collection_ids": collection_ids, "top_k": top_k,
                                "score_threshold": score_threshold, "rerank": rerank}
        if candidate_k:
            body["candidate_k"] = max(candidate_k, top_k)
        if rerank != "none":
            body["rerank_threshold"] = rerank_threshold
            if rerank_model:
                body["rerank_model"] = rerank_model
        try:
            resp = self._session.post(f"{self.base_url}/api/v1/retrieval", json=body, headers=self._headers(),
                                      timeout=self.timeout)
        except requests.Timeout as exc:
            hint = (" — модель-реранкер не успевает: уменьшите «Кандидатов до фильтрации» или увеличьте "
                    "KNOWLEDGE_SERVICE_TIMEOUT (он должен быть больше KB_RERANK_TIMEOUT сервиса)"
                    if rerank == "model" else "")
            raise KnowledgeServiceError(
                f"сервис баз знаний не ответил за {self.timeout:g} с{hint}") from exc
        except requests.RequestException as exc:
            raise KnowledgeServiceError(f"сервис баз знаний недоступен: {exc}") from exc
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if resp.status_code >= 400:
            raise KnowledgeServiceError(
                f"сервис баз знаний: HTTP {resp.status_code}: {data.get('error') or resp.text[:300]}")
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise KnowledgeServiceError("сервис баз знаний вернул неожиданный ответ")
        return data

    # ---- «День 31»: база знаний проекта (коллекция + источник-папка) --------

    def _call(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            resp = self._session.request(method, f"{self.base_url}/api/v1{path}", json=body, headers=self._headers(),
                                         timeout=self.timeout)
        except requests.RequestException as exc:
            raise KnowledgeServiceError(f"сервис баз знаний недоступен: {exc}") from exc
        try:
            data = resp.json() if resp.content else {}
        except ValueError:
            data = {}
        if resp.status_code >= 400:
            message = data.get("error") if isinstance(data, dict) else None
            raise KnowledgeServiceError(f"сервис баз знаний: HTTP {resp.status_code}: {message or resp.text[:300]}")
        return data if isinstance(data, dict) else {}

    def list_collections(self) -> List[Dict[str, Any]]:
        data = self._call("GET", "/collections")
        return list(data.get("items") or [])

    def get_collection(self, collection_id: str) -> Dict[str, Any]:
        return self._call("GET", f"/collections/{collection_id}")

    def create_collection(self, name: str, description: str = "", file_types: Optional[Dict[str, Any]] = None
                          ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"name": name, "description": description}
        if file_types:
            body["file_types"] = file_types
        return self._call("POST", "/collections", body)

    def delete_collection(self, collection_id: str) -> None:
        self._call("DELETE", f"/collections/{collection_id}")

    def list_sources(self, collection_id: str) -> List[Dict[str, Any]]:
        return list(self._call("GET", f"/collections/{collection_id}/sources").get("items") or [])

    def add_path_source(self, collection_id: str, path: str, include: List[str], title: str) -> Dict[str, Any]:
        """→ {"source": {...}, "job_id"} — индексация идёт в фоне."""
        return self._call("POST", f"/collections/{collection_id}/sources", {
            "source": {"type": "path", "path": path}, "include": include, "title": title,
        })

    def update_source_include(self, source_id: str, include: List[str]) -> Dict[str, Any]:
        return self._call("PATCH", f"/sources/{source_id}", {"include": include})

    def sync_source(self, source_id: str) -> Dict[str, Any]:
        return self._call("POST", f"/sources/{source_id}/sync")

    def get_job(self, job_id: str) -> Dict[str, Any]:
        return self._call("GET", f"/jobs/{job_id}")


@dataclass
class RagOutcome:
    """Что найдено и передано модели — сохраняется в сообщении (`Message.rag`)."""

    #: Запрос, по которому искали (после переписывания).
    query: str
    status: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    only_from_kb: bool = True
    cited: List[int] = field(default_factory=list)
    #: Исходный вопрос пользователя.
    original_query: str = ""
    #: Переписывание: {"mode", "rewritten", "model", "error", "tokens"}.
    rewrite: Dict[str, Any] = field(default_factory=dict)
    #: Счётчики этапов поиска: candidates, after_threshold, after_dedup, after_rerank, returned.
    stages: Dict[str, Any] = field(default_factory=dict)
    #: Реранкинг: {"method", "model", "fallback", "error"}.
    rerank: Dict[str, Any] = field(default_factory=dict)
    #: Параметры поиска: candidate_k, top_k, score_threshold, rerank_threshold.
    params: Dict[str, Any] = field(default_factory=dict)
    #: Выбранные в настройках базы знаний, которых нет в сервисе (удалены или пересозданы).
    missing_collections: List[str] = field(default_factory=list)
    #: «Цитаты и источники обязательны» (настройка `rag_quotes`).
    quotes_required: bool = False
    #: Цитаты из ответа: [{"n", "text", "verified"}] — `verified` проставляет код.
    quotes: List[Dict[str, Any]] = field(default_factory=list)
    #: Ответ начинается с «Не знаю» (слабый контекст или ответа во фрагментах нет).
    dont_know: bool = False
    #: Проверки ответа для отметок под ним: sources, quotes, quotes_verified, dont_know.
    checks: Dict[str, Any] = field(default_factory=dict)
    #: «БЕЗ_ПОИСКА»: нового поиска не было, фрагменты (и их номера) взяты из
    #: предыдущего ответа — {"message_id": …}; пусто — обычный поиск.
    context_reused: Dict[str, Any] = field(default_factory=dict)
    #: Сколько фрагментов добавлено из накопленных задачей поиска (`Task.sources`)
    #: и были ли они единственным контекстом (поиск ничего не нашёл).
    task_sources_added: int = 0
    task_sources_fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "used": True, "query": self.query, "original_query": self.original_query or self.query,
            "status": self.status, "error": self.error, "only_from_kb": self.only_from_kb,
            "rewrite": self.rewrite, "stages": self.stages, "rerank": self.rerank, "params": self.params,
            "missing_collections": self.missing_collections,
            "quotes_required": self.quotes_required, "quotes": self.quotes, "dont_know": self.dont_know,
            "checks": self.checks,
            "sources": self.sources, "cited": self.cited, "confirmed": bool(self.cited),
            "context_reused": self.context_reused, "task_sources_added": self.task_sources_added,
            "task_sources_fallback": self.task_sources_fallback,
            # «Источников нет» — ответ ни на один фрагмент не сослался.
            "no_sources": not self.cited,
        }


def search_query(text: str, previous_user_text: Optional[str]) -> str:
    """Запрос к поиску: короткий уточняющий вопрос дополняется предыдущим
    вопросом пользователя."""
    words = re.findall(r"\w+", text)
    if previous_user_text and len(words) <= FOLLOW_UP_MAX_WORDS:
        return f"{previous_user_text.strip()}\n{text.strip()}"
    return text.strip()


def display_section(title: str, section: str) -> str:
    """Раздел без повтора названия документа: заголовок первого уровня
    обычно совпадает с названием."""
    if not section or section == title:
        return ""
    prefix = f"{title} › "
    return section[len(prefix):] if title and section.startswith(prefix) else section


def _source_label(src: Dict[str, Any]) -> str:
    parts = [src.get("collection_name"), src.get("title")]
    section = display_section(src.get("title") or "", src.get("section") or "")
    if section:
        parts.append(section)
    label = " › ".join(p for p in parts if p)
    extra = []
    if src.get("page"):
        extra.append(f"стр. {src['page']}")
    if src.get("doc_date"):
        extra.append(f"дата {src['doc_date']}")
    if src.get("doc_version"):
        extra.append(f"версия {src['doc_version']}")
    return label + (f" ({', '.join(extra)})" if extra else "")


def select_sources(results: List[Dict[str, Any]], budget_tokens: int) -> List[Dict[str, Any]]:
    """Фрагменты в пределах бюджета контекста, пронумерованные с 1."""
    sources: List[Dict[str, Any]] = []
    used = 0
    for item in results:
        text = item.get("text") or ""
        tokens = estimate_tokens(text)
        if sources and used + tokens > budget_tokens:
            break
        doc = item.get("document") or {}
        col = item.get("collection") or {}
        sources.append({
            "n": len(sources) + 1, "chunk_id": item.get("chunk_id"), "score": item.get("score"),
            "vector_score": item.get("vector_score"), "rerank_score": item.get("rerank_score"),
            "text": text, "section": item.get("section") or "", "page": item.get("page"),
            "document_id": doc.get("id"), "title": doc.get("title") or doc.get("filename") or "",
            "source": doc.get("source"), "source_type": doc.get("source_type"),
            "doc_date": doc.get("doc_date"), "doc_version": doc.get("doc_version"),
            "collection_id": col.get("id"), "collection_name": col.get("name"),
        })
        used += tokens
    return sources


DONT_KNOW = "Не знаю"


def _quotes_instruction(outcome: RagOutcome) -> str:
    """Режим «Цитаты и источники обязательны» (ТЗ «Цитаты, источники и
    анти-галлюцинации»): ответ со ссылками [n], блок дословных цитат,
    «Не знаю» с уточняющим вопросом, если ответа во фрагментах нет."""
    if outcome.status == RAG_STATUS_NOT_FOUND:
        if outcome.only_from_kb:
            return (f"Ты отвечаешь только по базе знаний. По этому вопросу не найдено достаточно релевантных "
                    f"фрагментов. Начни ответ словами «{DONT_KNOW}», коротко скажи, что в базе знаний нет "
                    "информации по этому вопросу, и задай один уточняющий вопрос, который поможет найти ответ "
                    "(уточни термин, компонент, ситуацию или формулировку). Не придумывай ответ.")
        return ("По этому вопросу в базе знаний ничего не найдено. Можешь ответить на основе общих знаний, "
                "но явно укажи, что база знаний этого не подтверждает, и предложи уточнить вопрос.")
    rules = [
        "Ниже в сообщении пользователя — пронумерованные фрагменты базы знаний (блок «Контекст») и вопрос.",
        "Отвечай только на основе этих фрагментов и не добавляй сведений не из них." if outcome.only_from_kb else
        "Используй фрагменты как основной источник; сведения не из фрагментов явно помечай «(не из базы знаний)».",
        "Каждое утверждение ответа подкрепляй ссылкой на номер фрагмента в квадратных скобках: [1] или [2, 3].",
        "После ответа добавь блок, начинающийся строкой «Цитаты:», — от одной до трёх строк вида:\n"
        "- «дословная выдержка из фрагмента» [n]\n"
        "Цитату копируй из фрагмента слово в слово, без пересказа и изменений, не длиннее двух предложений; "
        "она должна подтверждать утверждение ответа. Пропуск внутри цитаты обозначай «…».",
        "Список источников не пиши — его добавят автоматически по номерам [n].",
        f"Если во фрагментах нет ответа на вопрос, начни ответ словами «{DONT_KNOW}», коротко объясни, "
        "чего не хватает, и задай один уточняющий вопрос; блок «Цитаты» в этом случае не нужен.",
        "Если фрагменты противоречат друг другу, опирайся на более свежий документ (по дате и версии) "
        "и явно назови расхождение.",
    ]
    return "\n".join(rules)


def rag_instruction(outcome: RagOutcome) -> str:
    if outcome.quotes_required and outcome.status != RAG_STATUS_UNAVAILABLE:
        return _quotes_instruction(outcome)
    if outcome.status == RAG_STATUS_UNAVAILABLE:
        return ("База знаний сейчас недоступна. Ответь на вопрос как обычно, но в начале ответа коротко "
                "предупреди, что база знаний недоступна и ответ ею не подтверждён.")
    if outcome.status == RAG_STATUS_NOT_FOUND:
        if outcome.only_from_kb:
            return ("Ты отвечаешь по базе знаний. По этому вопросу в базе знаний ничего не найдено. "
                    "Прямо скажи пользователю, что в базе знаний нет информации по этому вопросу, "
                    "и не придумывай ответ.")
        return ("По этому вопросу в базе знаний ничего не найдено. Можешь ответить на основе общих знаний, "
                "но явно укажи, что база знаний этого не подтверждает.")
    rules = [
        "Ниже в сообщении пользователя — пронумерованные фрагменты базы знаний (блок «Контекст») и вопрос.",
        "Ссылайся на использованные фрагменты их номерами в квадратных скобках, например [1] или [2, 3].",
        "Если фрагменты противоречат друг другу, опирайся на более свежий документ (по дате и версии) "
        "и явно назови расхождение.",
    ]
    if outcome.only_from_kb:
        rules.insert(1, "Отвечай только на основе этих фрагментов. Если ответа в них нет — прямо скажи, "
                        "что в базе знаний этого нет, и не придумывай.")
    else:
        rules.insert(1, "Используй фрагменты как основной источник. Если ответа в них нет, можешь ответить "
                        "на основе общих знаний, но явно укажи, что база знаний этого не подтверждает.")
    return "\n".join(rules)


def augmented_question(question: str, outcome: RagOutcome) -> str:
    """Сообщение пользователя для модели: контекст + вопрос (в истории
    остаётся исходный вопрос)."""
    if outcome.status != RAG_STATUS_OK:
        return question
    blocks = [f"[{s['n']}] {_source_label(s)}\n{s['text']}" for s in outcome.sources]
    return "Контекст:\n\n" + "\n\n".join(blocks) + f"\n\nВопрос: {question}"


def citations(answer: str, count: int) -> List[int]:
    """Номера фрагментов, на которые ссылается ответ (только существующие)."""
    found = []
    for match in _CITATION_RE.finditer(answer or ""):
        for part in re.split(r"[,;]", match.group(1)):
            n = int(part.strip())
            if 1 <= n <= count and n not in found:
                found.append(n)
    return sorted(found)


_QUOTES_HEADER_RE = re.compile(r"^[ \t>#*_]*цитаты[ \t*_]*:?[ \t*_]*:?[ \t]*$", re.I | re.M)
_QUOTE_RE = re.compile(
    r"[«\"“„]([^«»\"“”„\n]{3,}?)[»\"”“]\s*\[(\d{1,3}(?:\s*[,;]\s*\d{1,3})*)\]")
_ELLIPSIS_RE = re.compile(r"…|\.\.\.|\[\s*…\s*\]|\(\s*…\s*\)")
_SPACES_RE = re.compile(r"\s+")


_NON_WORD_RE = re.compile(r"[^\w]+")


def _normalize_for_quote(text: str) -> str:
    """Сравнение цитаты с фрагментом по последовательности слов: регистр,
    «ё/е», пунктуация, кавычки, скобки и markdown-разметка не учитываются."""
    text = (text or "").replace("ё", "е").replace("Ё", "Е").casefold()
    return _NON_WORD_RE.sub(" ", text.replace("_", " ")).strip()


def quote_in_fragment(quote: str, fragment: str) -> bool:
    """Цитата есть во фрагменте дословно (без учёта регистра, «ё/е», пробелов,
    кавычек и разметки); части через «…» — по порядку."""
    haystack = _normalize_for_quote(fragment)
    position = 0
    parts = [_normalize_for_quote(p) for p in _ELLIPSIS_RE.split(quote)]
    parts = [p for p in parts if len(p) >= 3]
    if not parts:
        return False
    for part in parts:
        found = haystack.find(part, position)
        if found < 0:
            return False
        position = found + len(part)
    return True


def starts_with_dont_know(text: str) -> bool:
    return _normalize_for_quote(text).startswith(DONT_KNOW.casefold())


def finalize_answer(answer: str, outcome: RagOutcome) -> str:
    """После ответа модели: номера ссылок [n], цитаты (проверка кодом — есть
    ли цитата во фрагменте, на который она ссылается), «Не знаю». Блок
    «Цитаты:» вырезается из текста ответа — цитаты с отметками проверки
    показываются отдельно (`Message.rag.quotes`). Возвращает текст ответа."""
    outcome.cited = citations(answer, len(outcome.sources))
    if not outcome.quotes_required:
        return answer
    text = answer or ""
    if outcome.status == RAG_STATUS_NOT_FOUND and outcome.only_from_kb and not starts_with_dont_know(text):
        # Правило «не знаю» при слабом контексте обязательно: модель его нарушила —
        # ответ начинается с «Не знаю» принудительно.
        text = f"{DONT_KNOW}. {text}".strip()
    header = _QUOTES_HEADER_RE.search(text)
    block = text[header.end():] if header else ""
    by_n = {s["n"]: s for s in outcome.sources}
    quotes: List[Dict[str, Any]] = []
    for match in _QUOTE_RE.finditer(block):
        quote = match.group(1).strip()
        refs = [int(p) for p in re.split(r"[,;]", match.group(2)) if p.strip()]
        known = [n for n in refs if n in by_n]
        verified = any(quote_in_fragment(quote, by_n[n].get("text") or "") for n in known)
        quotes.append({"n": known[0] if known else refs[0], "refs": refs, "text": quote, "verified": verified})
    if header and quotes:
        text = text[:header.start()].rstrip()
    outcome.quotes = quotes
    outcome.dont_know = starts_with_dont_know(text)
    outcome.checks = {
        "sources": bool(outcome.cited),
        "quotes": len(quotes),
        "quotes_verified": sum(1 for q in quotes if q["verified"]),
        "dont_know": outcome.dont_know,
    }
    return text


def rewrite_request_text(history: List[Any], question: str) -> str:
    """Сообщение для модели переписывания: последние реплики диалога и вопрос.
    `history` — сообщения чата (role, content, is_summary)."""
    lines = []
    dialog = [m for m in history if m.role in ("user", "assistant") and not getattr(m, "is_summary", False)]
    for m in dialog[-REWRITE_HISTORY_MESSAGES:]:
        text = " ".join((m.content or "").split())
        if len(text) > REWRITE_MESSAGE_CHARS:
            text = text[:REWRITE_MESSAGE_CHARS] + "…"
        lines.append(f"{'Пользователь' if m.role == 'user' else 'Ассистент'}: {text}")
    head = ("Диалог:\n" + "\n".join(lines) + "\n\n") if lines else ""
    return f"{head}Вопрос: {question.strip()}\n\nПоисковый запрос:"


_REWRITE_PREFIX = re.compile(r"^(поисковый\s+запрос|запрос|query|search query)\s*[:：-]\s*", re.I)


def clean_rewritten(text: Optional[str]) -> Optional[str]:
    """Первая непустая строка ответа модели без префиксов и кавычек; None —
    ответ непригоден (пустой или слишком длинный)."""
    for line in (text or "").splitlines():
        line = _REWRITE_PREFIX.sub("", line.strip()).strip().strip("«»\"'`").strip()
        if line:
            return line if len(line) <= 400 else None
    return None

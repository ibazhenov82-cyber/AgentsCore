"""
agents_core.local_models
==========================

Управление локальными моделями Ollama («День 29»):

- список скачанных чат-моделей (`ollama list` без embedding-моделей) с
  состоянием: запущена / остановлена / запускается / останавливается,
  память и доля GPU у запущенной (`ollama ps`);
- запуск и остановка вручную (экран «Модели» в приложении);
- автозапуск при запросе к модели из чата (`ensure_ready`), если она не
  запущена или запущена с другим «Размером контекста»;
- «Конкурентный запуск» — настройка модели: при её запуске все другие
  запущенные локальные чат-модели сначала останавливаются (Ollama умеет
  держать несколько моделей одновременно, пока хватает памяти, — флаг
  нужен, когда памяти на две модели нет или нужны честные замеры).
  Embedding-модели (их использует сервис баз знаний) не останавливаются.

Запущенная модель держится в памяти до остановки (`keep_alive: -1`, см.
`providers.ollama.KEEP_ALIVE`).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .hf_hub import estimate_need_mb, kv_mb_per_token, system_memory
from .models import Settings
from .providers import ProviderError
from .providers.ollama import OllamaProvider, memory_stats

_logger = logging.getLogger("agents_core.local_models")

STATUS_RUNNING = "running"
STATUS_STOPPED = "stopped"
STATUS_STARTING = "starting"
STATUS_STOPPING = "stopping"

ACTION_STOP = "stop"
ACTION_START = "start"
ACTION_RESTART = "restart"
#: Модель была запущена, но пропала из `ollama ps` не по команде AgentsCore —
#: обычно Ollama выгрузила её, освобождая видеопамять под другую модель.
ACTION_RELOAD = "reload"

_MB = 1024 * 1024

KIND_CHAT = "chat"
KIND_EMBEDDING = "embedding"

#: Окно, с которым Ollama запускает модель по умолчанию (видеопамять < 24 ГБ) —
#: для оценки памяти незапущенной модели.
DEFAULT_OLLAMA_CTX = 4096

#: Готовые варианты для блока «Рекомендуем» (теги библиотеки Ollama).
RECOMMENDED: List[Dict[str, Any]] = [
    {"model": "qwen3:1.7b", "title": "Qwen3 1.7B", "kind": KIND_CHAT, "size_mb": 1400,
     "description": "Быстрая небольшая чат-модель: рассуждения, вызов инструментов, русский язык."},
    {"model": "qwen3:4b", "title": "Qwen3 4B", "kind": KIND_CHAT, "size_mb": 2500,
     "description": "Баланс скорости и качества для ПК без мощной видеокарты."},
    {"model": "qwen3:8b", "title": "Qwen3 8B", "kind": KIND_CHAT, "size_mb": 5200,
     "description": "Заметно точнее на сложных вопросах; нужна видеокарта от 8 ГБ."},
    {"model": "qwen3-embedding:0.6b", "title": "Qwen3 Embedding 0.6B", "kind": KIND_EMBEDDING, "size_mb": 640,
     "description": "Многоязычные эмбеддинги для баз знаний (модель по умолчанию knowledge_service)."},
]


class LocalModelsUnavailable(Exception):
    """Ollama выключена в PROVIDERS или не отвечает."""


class LocalModelNotFound(Exception):
    """Такой модели нет среди скачанных чат-моделей Ollama."""


def action_text(action: str, model: str, num_ctx: Optional[int] = None, evicted_for: Optional[List[str]] = None) -> str:
    """Строка статуса для чата («Запускаю модель …»)."""
    if action == ACTION_STOP:
        return f"Останавливаю модель {model}"
    if action == ACTION_RESTART:
        return f"Перезапускаю модель {model} с размером контекста {num_ctx}"
    if action == ACTION_RELOAD:
        reason = (
            f"Ollama выгрузила её, освобождая видеопамять под {', '.join(evicted_for)}"
            if evicted_for else "её выгрузили из памяти не через AgentsCore"
        )
        return f"Запускаю модель {model} снова — {reason}"
    return f"Запускаю модель {model}"


class LocalModelManager:
    def __init__(self, provider: Optional[OllamaProvider], catalog, db):
        self._provider = provider
        self._catalog = catalog
        self._db = db
        #: Запуски и остановки идут по одному: две загрузки одновременно
        #: только мешают друг другу (память, диск).
        self._action_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._transitions: Dict[str, str] = {}
        self._errors: Dict[str, str] = {}
        #: Что было в `ollama ps` при прошлом чтении, что остановили мы сами
        #: и какие модели пропали без нашей команды (→ кем были «вытеснены»).
        self._seen_running: set = set()
        self._our_stops: set = set()
        self._evicted: Dict[str, List[str]] = {}
        #: Идущие и неудачные скачивания: имя модели → состояние (см. `_pull_item`).
        self._pulls: Dict[str, Dict[str, Any]] = {}
        self._pull_cancel: Dict[str, threading.Event] = {}
        if provider is not None:
            provider.before_request = self._before_request

    @property
    def enabled(self) -> bool:
        return self._provider is not None

    def _require(self) -> OllamaProvider:
        if self._provider is None:
            raise LocalModelsUnavailable("провайдер Ollama выключен (PROVIDERS в .env AgentsCore)")
        return self._provider

    # ---- состояние ------------------------------------------------------------

    def _downloaded(self, chat_only: bool = True) -> List[Dict[str, Any]]:
        """Скачанные модели с полем `kind` («chat» | «embedding»)."""
        provider = self._require()
        try:
            tags = provider.list_tags()
        except ProviderError as exc:
            raise LocalModelsUnavailable(str(exc)) from exc
        result = []
        for entry in tags:
            name = entry.get("model") or entry.get("name")
            if not name:
                continue
            kind = KIND_CHAT if provider.is_chat_model(name, entry.get("digest")) else KIND_EMBEDDING
            if chat_only and kind != KIND_CHAT:
                continue
            result.append({**entry, "model": name, "kind": kind})
        return result

    def memory(self) -> Optional[Dict[str, int]]:
        """Свободная память компьютера с Ollama (см. `hf_hub.system_memory`)."""
        if self._provider is None:
            return None
        try:
            return system_memory(self._provider.base_url)
        except Exception:  # оценка памяти не должна ломать список
            return None

    def _need_mb(self, entry: Dict[str, Any]) -> Optional[int]:
        """Оценка памяти для незапущенной модели при окне Ollama по умолчанию."""
        size = entry.get("size")
        if not isinstance(size, (int, float)):
            return None
        try:
            info = self._require().show(entry["model"], entry.get("digest")).get("model_info") or {}
        except ProviderError:
            info = {}
        return estimate_need_mb(size / _MB, DEFAULT_OLLAMA_CTX, kv_mb_per_token(info))

    def _running(self) -> Dict[str, Dict[str, Any]]:
        provider = self._require()
        running = {(e.get("model") or e.get("name")): e for e in provider.running()}
        self._observe(set(running))
        return running

    def _observe(self, now: set) -> None:
        """Сравнивает `ollama ps` с прошлым чтением: чат-модель, пропавшая
        без нашей остановки, записывается как выгруженная самой Ollama —
        обычно ради другой модели, которой не хватало видеопамяти (например,
        модели эмбеддингов при поиске в базе знаний)."""
        with self._state_lock:
            for gone in self._seen_running - now:
                if gone in self._our_stops or self._catalog.get(f"ollama:{gone}") is None:
                    continue
                others = sorted(now - {gone})
                self._evicted[gone] = others
                _logger.warning(
                    "модель %s выгружена из памяти не AgentsCore; сейчас загружены: %s",
                    gone, ", ".join(others) or "ничего",
                )
            for back in now:
                self._evicted.pop(back, None)
                self._our_stops.discard(back)
            self._seen_running = set(now)

    def describe(self, action: str, model: str, num_ctx: Optional[int] = None) -> str:
        """Текст статуса шага плана (с причиной, если модель вытеснили)."""
        with self._state_lock:
            evicted_for = self._evicted.get(model)
        return action_text(action, model, num_ctx, evicted_for)

    def _set_transition(self, model: str, state: Optional[str]) -> None:
        with self._state_lock:
            if state is None:
                self._transitions.pop(model, None)
            else:
                self._transitions[model] = state

    def _item(
        self, entry: Dict[str, Any], run: Optional[Dict[str, Any]], flags: Dict[str, bool],
        transitions: Dict[str, str], errors: Dict[str, str],
    ) -> Dict[str, Any]:
        name = entry["model"]
        details = entry.get("details") or {}
        transition = transitions.get(name)
        error = errors.get(name)
        item: Dict[str, Any] = {
            "model": name,
            "id": f"ollama:{name}",
            "family": details.get("family"),
            "parameter_size": details.get("parameter_size"),
            "quantization_level": details.get("quantization_level"),
            "size_mb": int(entry["size"] / _MB) if isinstance(entry.get("size"), (int, float)) else None,
            "modified_at": entry.get("modified_at"),
            "status": transition or (STATUS_RUNNING if run is not None else STATUS_STOPPED),
            "competitive_start": flags.get(name, False),
            "error": error,
            "kind": entry.get("kind", KIND_CHAT),
            "need_mb": None if run is not None else self._need_mb(entry),
        }
        if run is not None:
            item.update(memory_stats(run))
        return item

    def list_models(self) -> List[Dict[str, Any]]:
        """Скачанные модели (чат-модели и embedding-модели) с состоянием;
        заодно сверяет каталог моделей (скачали новую чат-модель — она
        становится доступна агентам)."""
        downloaded = self._downloaded(chat_only=False)
        self._catalog.sync_local([e["model"] for e in downloaded if e["kind"] == KIND_CHAT])
        # Снимок переходов — ДО `ollama ps`: иначе запуск, закончившийся между
        # двумя чтениями, выглядел бы как «остановлена».
        with self._state_lock:
            transitions = dict(self._transitions)
            errors = dict(self._errors)
        try:
            running = self._running()
        except ProviderError as exc:
            raise LocalModelsUnavailable(str(exc)) from exc
        flags = self._db.local_model_competitive_flags()
        return sorted(
            (self._item(e, running.get(e["model"]), flags, transitions, errors) for e in downloaded),
            key=lambda i: i["model"],
        )

    def sync_catalog(self) -> None:
        """Только сверка каталога со скачанными моделями (без `ollama ps`);
        Ollama недоступна — каталог не трогаем."""
        try:
            downloaded = self._downloaded()
        except LocalModelsUnavailable:
            return
        self._catalog.sync_local([e["model"] for e in downloaded])

    # ---- скачивание и удаление -------------------------------------------------

    def pulls(self) -> List[Dict[str, Any]]:
        with self._state_lock:
            return [dict(p) for p in self._pulls.values()]

    def start_pull(self, model: str) -> Dict[str, Any]:
        """Скачивание в фоне (`ollama pull`): из библиотеки Ollama
        («qwen3:8b») или с Hugging Face («hf.co/<репозиторий>:<квант>»).
        Прогресс — в `pulls()`; по окончании модель появляется в списке."""
        provider = self._require()
        model = model.strip()
        with self._state_lock:
            current = self._pulls.get(model)
            if current is not None and current["status"] != "failed":
                return dict(current)
            state = {"model": model, "status": "starting", "completed_mb": 0, "total_mb": None,
                     "percent": None, "error": None}
            self._pulls[model] = state
            cancel = threading.Event()
            self._pull_cancel[model] = cancel

        def run() -> None:
            totals: Dict[str, float] = {}
            done: Dict[str, float] = {}
            try:
                for chunk in provider.pull(model, cancel):
                    with self._state_lock:
                        status = str(chunk.get("status") or "")
                        digest = chunk.get("digest")
                        if digest and isinstance(chunk.get("total"), (int, float)):
                            totals[digest] = chunk["total"]
                            done[digest] = chunk.get("completed") or 0
                        total = sum(totals.values())
                        state["status"] = "downloading" if totals else status or "starting"
                        if total:
                            state["total_mb"] = int(total / _MB)
                            state["completed_mb"] = int(sum(done.values()) / _MB)
                            state["percent"] = int(100 * sum(done.values()) / total)
                        if status.startswith("verifying") or status.startswith("writing"):
                            state["status"] = "verifying"
                # Сначала каталог, потом убрать из скачиваний: модель не
                # «пропадает» между двумя списками.
                self.sync_catalog()
                with self._state_lock:
                    self._pulls.pop(model, None)
            except Exception as exc:
                _logger.error("скачивание %s: %s", model, exc)
                with self._state_lock:
                    if cancel.is_set():
                        self._pulls.pop(model, None)
                    else:
                        state["status"] = "failed"
                        state["error"] = str(exc)
            finally:
                with self._state_lock:
                    self._pull_cancel.pop(model, None)

        threading.Thread(target=run, name=f"ollama-pull-{model}", daemon=True).start()
        return dict(state)

    def cancel_pull(self, model: str) -> None:
        """Отменяет скачивание или убирает из списка неудавшееся."""
        with self._state_lock:
            cancel = self._pull_cancel.get(model)
            if cancel is not None:
                cancel.set()
            elif model in self._pulls:
                self._pulls.pop(model, None)

    def delete(self, model: str) -> None:
        """Удаляет модель с диска (запущенную — сначала останавливает)."""
        provider = self._require()
        if all(e["model"] != model for e in self._downloaded(chat_only=False)):
            raise LocalModelNotFound(f"модель {model} не скачана в Ollama")
        with self._action_lock:
            with self._state_lock:
                self._our_stops.add(model)
            try:
                if model in self._running():
                    provider.unload(model)
            except ProviderError:
                pass
            provider.delete(model)
        with self._state_lock:
            self._errors.pop(model, None)
        self.sync_catalog()

    def recommended(self) -> List[Dict[str, Any]]:
        """Готовые варианты, которых ещё нет среди скачанных."""
        try:
            names = {e["model"] for e in self._downloaded(chat_only=False)}
        except LocalModelsUnavailable:
            names = set()
        names |= {n[:-len(":latest")] for n in names if n.endswith(":latest")}
        return [dict(r) for r in RECOMMENDED if r["model"] not in names]

    def downloaded_names(self) -> set:
        try:
            return {e["model"].lower() for e in self._downloaded(chat_only=False)}
        except LocalModelsUnavailable:
            return set()

    def get(self, model: str) -> Dict[str, Any]:
        for item in self.list_models():
            if item["model"] == model:
                return item
        raise LocalModelNotFound(f"модель {model} не скачана в Ollama (ollama pull {model})")

    def _require_downloaded(self, model: str) -> None:
        if all(e["model"] != model for e in self._downloaded()):
            raise LocalModelNotFound(f"модель {model} не скачана в Ollama (ollama pull {model})")

    # ---- ручное управление ------------------------------------------------------

    def start(self, model: str) -> Dict[str, Any]:
        """Запуск в фоне (загрузка может идти минутами): сразу возвращает
        состояние «запускается», клиент опрашивает `list_models`."""
        self._require_downloaded(model)
        with self._state_lock:
            if model in self._transitions:
                return self.get(model)
            self._transitions[model] = STATUS_STARTING
            self._errors.pop(model, None)

        def run() -> None:
            try:
                self.ensure_ready(model, None)
            except Exception as exc:  # ошибку покажет экран «Модели»
                _logger.error("запуск модели %s: %s", model, exc)
                with self._state_lock:
                    self._errors[model] = str(exc)
            finally:
                self._set_transition(model, None)

        threading.Thread(target=run, name=f"ollama-start-{model}", daemon=True).start()
        return self.get(model)

    def stop(self, model: str) -> Dict[str, Any]:
        self._require_downloaded(model)
        with self._state_lock:
            self._errors.pop(model, None)
        self._apply(ACTION_STOP, model, None)
        return self.get(model)

    def set_competitive(self, model: str, value: bool) -> Dict[str, Any]:
        self._require_downloaded(model)
        self._db.set_local_model_competitive(model, value)
        return self.get(model)

    # ---- автозапуск ---------------------------------------------------------

    def plan(self, model: str, num_ctx: Optional[int] = None) -> List[Tuple[str, str]]:
        """Что нужно сделать, чтобы `model` была готова к запросу:
        `[("stop", другая)…, ("start", model)]`, `[("restart", model)]` или
        `[]`. Ollama недоступна — `[]` (ошибку покажет сам запрос)."""
        try:
            running = self._running()
        except (ProviderError, LocalModelsUnavailable):
            return []
        entry = running.get(model)
        if entry is not None:
            ctx = entry.get("context_length")
            if num_ctx and isinstance(ctx, int) and ctx != num_ctx:
                return [(ACTION_RESTART, model)]
            return []
        actions: List[Tuple[str, str]] = []
        if self._db.local_model_competitive_flags().get(model):
            for other in running:
                # Останавливаем только чат-модели каталога: embedding-модели
                # нужны сервису баз знаний.
                if other != model and self._catalog.get(f"ollama:{other}") is not None:
                    actions.append((ACTION_STOP, other))
        with self._state_lock:
            evicted = model in self._evicted
        actions.append((ACTION_RELOAD if evicted else ACTION_START, model))
        return actions

    def _apply(self, action: str, model: str, num_ctx: Optional[int]) -> Optional[int]:
        """Выполняет одно действие плана; для запуска возвращает время
        загрузки модели в память, мс."""
        provider = self._require()
        with self._action_lock:
            if action == ACTION_STOP:
                with self._state_lock:
                    self._our_stops.add(model)
                self._set_transition(model, STATUS_STOPPING)
                try:
                    provider.unload(model)
                finally:
                    self._set_transition(model, None)
                return None
            self._set_transition(model, STATUS_STARTING)
            started = time.monotonic()
            try:
                response = provider.load(model, num_ctx)
            finally:
                self._set_transition(model, None)
            with self._state_lock:
                self._evicted.pop(model, None)
                self._our_stops.discard(model)
                self._seen_running.add(model)
            load_ns = response.get("load_duration") if isinstance(response, dict) else None
            if isinstance(load_ns, (int, float)) and load_ns > 0:
                return int(load_ns / 1_000_000)
            return int((time.monotonic() - started) * 1000)

    def apply(self, action: str, model: str, num_ctx: Optional[int]) -> Optional[int]:
        return self._apply(action, model, num_ctx)

    def ensure_ready(
        self, model: str, num_ctx: Optional[int] = None, on_status: Optional[Callable[[str], None]] = None,
    ) -> Optional[int]:
        """Готовит модель к запросу (см. `plan`). Возвращает время загрузки,
        мс, если модель пришлось запускать."""
        load_ms: Optional[int] = None
        for action, name in self.plan(model, num_ctx):
            if on_status is not None:
                on_status(self.describe(action, name, num_ctx))
            result = self._apply(action, name, num_ctx)
            if result is not None:
                load_ms = result
        return load_ms

    def _before_request(self, model_id: str, settings: Settings) -> None:
        """Хук провайдера перед каждым запросом (суммаризация, переписывание
        запроса и т.п. тоже идут через него)."""
        self.ensure_ready(model_id, settings.num_ctx)

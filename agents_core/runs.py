"""
agents_core.runs
==================

Асинхронные запуски (ТЗ «асинхронные ответы и сервис планировщика», этап 1).

Ответ модели принадлежит чату, а не соединению клиента: `RunManager`
создаёт запуск (сообщение пользователя + черновик ответа + строка `runs`)
и сразу возвращает его, а сама работа идёт в фоновом пуле потоков.
Каждое событие запуска получает номер `seq` в ленте запуска
(`events.EventLog`); клиент подписывается с любого номера и продолжает с
места обрыва. Черновик ответа дописывается в базу по ходу (`_DraftWriter`),
так что вернувшийся в чат клиент видит накопленное даже если лента уже
выброшена из памяти.

Виды запусков:
- `message` — ответ на сообщение пользователя;
- `task_step` — один шаг Менеджера задач (кнопка «Продолжить»);
- `task_run` — шаги подряд до завершения задачи, паузы, вопроса к
  пользователю или лимита `task_manager_max_steps` (кнопка «Выполнить»);
- `test_dialog` — тестовый диалог: вопросы набора по очереди от имени
  пользователя, каждый — после ответа модели на предыдущий; ошибка ответа
  останавливает цепочку, отмена — после текущего ответа (или сразу, если он
  ещё идёт).

В одном чате одновременно — не больше одного запуска (`RunConflictError`,
HTTP 409); разные чаты — параллельно, до `AGENT_RUN_WORKERS` сразу.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from .config import AgentConfig
from .events import CancelToken, EventLog, EventsGone
from .models import Message, Run
from .repository import NotFoundError, Repository, ValidationError

_logger = logging.getLogger("agents_core.runs")

#: Итоговые события ленты запуска — после них событий больше нет.
TERMINAL_EVENTS = ("done", "cancelled", "error")

#: `Message.source` вопросов тестового диалога — клиент помечает их «Тестовый диалог».
TEST_DIALOG_SOURCE = "test_dialog"

_RUN_STATUS_BY_TERMINAL = {"done": "succeeded", "cancelled": "cancelled", "error": "failed"}


class RunConflictError(Exception):
    """В чате уже идёт запуск — второй не создаётся (ТЗ, раздел 2.3)."""

    def __init__(self, message: str, active_run_id: str):
        super().__init__(message)
        self.active_run_id = active_run_id


class _RunState:
    """Живое состояние запуска в памяти процесса."""

    def __init__(self, run_id: str, chat_id: str, kind: str):
        self.run_id = run_id
        self.chat_id = chat_id
        self.kind = kind
        self.log = EventLog()
        self.token = CancelToken()
        self.finished_at: Optional[float] = None
        self.last_global_status_at = 0.0
        #: Предельное время запуска; None — `AGENT_RUN_TIMEOUT`.
        self.timeout: Optional[float] = None


class _DraftWriter:
    """Периодически сохраняет черновик ответа вместе с номером события, по
    которому он актуален (ТЗ, раздел 2.2): текст — не реже раза в секунду
    или каждые 2 КБ, а при смене фазы, вызове инструмента и событии задачи —
    сразу. После `message_saved` (итог уже записан исполнителем) черновик
    больше не трогается, обновляется только номер события запуска."""

    FLUSH_INTERVAL = 1.0
    FLUSH_CHARS = 2048

    def __init__(self, db, run_id: str, message_id: int):
        self._db = db
        self._run_id = run_id
        self._reset(message_id)

    def _reset(self, message_id: int) -> None:
        self._message_id = message_id
        self._content: List[str] = []
        self._reasoning: List[str] = []
        self._tool_events: List[dict] = []
        self._task_events: List[dict] = []
        self._rag: Optional[str] = None
        self._pending = 0
        self._last_flush = time.monotonic()
        self._finalized = False

    def switch(self, message_id: int, seq: int) -> None:
        self._reset(message_id)
        self._db.update_run(self._run_id, assistant_message_id=message_id, last_seq=seq)

    def on_event(self, event: Dict[str, Any]) -> None:
        kind = event["type"]
        seq = event["seq"]
        if kind == "message_started":
            return
        if self._finalized or kind in ("message_saved", "task_step_done"):
            if kind == "message_saved":
                self._finalized = True
            run_fields: Dict[str, Any] = {"last_seq": seq}
            if kind == "status":
                run_fields["current_status"] = event.get("status")
            self._db.update_run(self._run_id, **run_fields)
            return
        if kind == "delta":
            content = event.get("content") or ""
            reasoning = event.get("reasoning_content") or ""
            self._content.append(content)
            self._reasoning.append(reasoning)
            self._pending += len(content) + len(reasoning)
            if self._pending >= self.FLUSH_CHARS or time.monotonic() - self._last_flush >= self.FLUSH_INTERVAL:
                self._flush(seq)
        elif kind == "status":
            self._flush(seq, current_status=event.get("status"))
        elif kind == "tool_call":
            self._tool_events.append(_strip_envelope(event))
            self._flush(seq)
        elif kind == "task_event":
            self._task_events.append({k: v for k, v in _strip_envelope(event).items() if k != "type"})
            self._flush(seq)
        elif kind == "rag_context":
            self._rag = json.dumps(event.get("rag"), ensure_ascii=False)
            self._flush(seq)

    def _flush(self, seq: int, current_status: Optional[str] = None) -> None:
        message_fields = {
            "content": "".join(self._content),
            "reasoning_content": "".join(self._reasoning) or None,
            "tool_events": json.dumps(self._tool_events, ensure_ascii=False) if self._tool_events else None,
            "task_events": json.dumps(self._task_events, ensure_ascii=False) if self._task_events else None,
        }
        if self._rag is not None:
            message_fields["rag"] = self._rag
        run_fields: Dict[str, Any] = {"last_seq": seq}
        if current_status is not None:
            run_fields["current_status"] = current_status
        self._db.update_draft_and_run(self._message_id, message_fields, self._run_id, run_fields)
        self._pending = 0
        self._last_flush = time.monotonic()


def _strip_envelope(event: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in event.items() if k not in ("seq", "ts", "run_id", "chat_id")}


def run_brief(run: Run) -> Dict[str, Any]:
    """Краткое описание запуска для общей ленты и `ChatOut.active_run`."""
    brief: Dict[str, Any] = {"id": run.id, "kind": run.kind, "status": run.status, "current_status": run.current_status}
    if run.kind == "test_dialog":
        try:
            request = json.loads(run.request_json or "{}")
            brief["test_dialog"] = {"id": request.get("test_dialog_id"), "name": request.get("name", ""),
                                    "step": request.get("step", 1), "total": request.get("total", 0)}
        except ValueError:
            pass
    return brief


class RunManager:
    def __init__(
        self,
        repo: Repository,
        workers: Optional[int] = None,
        timeout: Optional[float] = None,
        events_ttl: Optional[float] = None,
    ):
        self._repo = repo
        self._db = repo._db
        self._timeout = timeout if timeout is not None else AgentConfig.RUN_TIMEOUT
        self._events_ttl = events_ttl if events_ttl is not None else AgentConfig.RUN_EVENTS_TTL
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, workers if workers is not None else AgentConfig.RUN_WORKERS),
            thread_name_prefix="agents-run",
        )
        self._states: Dict[str, _RunState] = {}
        self._lock = threading.Lock()

    # ---- жизненный цикл сервиса ---------------------------------------------

    def recover(self) -> int:
        """При старте: запуски, прерванные перезапуском сервиса, получают
        статус "interrupted", их черновики — тоже (частичный текст остаётся);
        старые записи о запусках удаляются (`AGENT_RUN_RETENTION_DAYS`)."""
        interrupted = self._db.mark_unfinished_runs_interrupted()
        for chat_id in {r.chat_id for r in interrupted}:
            try:
                self._repo._publish_unread(chat_id)
            except Exception:  # чат мог быть удалён — не мешаем старту
                pass
        cutoff = int(time.time()) - AgentConfig.RUN_RETENTION_DAYS * 86400
        self._db.delete_runs_older_than(cutoff)
        if interrupted:
            _logger.info("Прервано перезапуском запусков: %d", len(interrupted))
        return len(interrupted)

    def shutdown(self, wait: bool = False) -> None:
        with self._lock:
            states = list(self._states.values())
        for state in states:
            if state.finished_at is None:
                state.token.cancel("user")
        self._executor.shutdown(wait=wait)

    # ---- создание запусков --------------------------------------------------

    def start_message_run(
        self,
        chat_id: str,
        text: str,
        *,
        get_facts: bool = False,
        sliding_window: bool = False,
        autosummary: str = "off",
        branch: Optional[int] = None,
        client_request_id: Optional[str] = None,
        source: str = "app",
        use_stream: Optional[bool] = None,
    ) -> Tuple[Run, Optional[Message], Optional[Message]]:
        """Создаёт запуск ответа на сообщение и ставит его в пул. Возвращает
        (запуск, сообщение пользователя, черновик ответа) — сразу, не
        дожидаясь модели. Повтор с тем же `client_request_id` возвращает уже
        созданный запуск."""
        self._gc()
        with self._lock:
            existing = self._existing_for_request(chat_id, client_request_id)
            if existing is not None:
                return existing, self._message(existing.user_message_id), self._message(existing.assistant_message_id)
            self._ensure_no_active_run(chat_id)
            run_id = str(uuid.uuid4())
            chat, user_msg, draft = self._repo.prepare_message_run(
                chat_id, text, get_facts=get_facts, sliding_window=sliding_window, autosummary=autosummary,
                branch=branch, source=source, run_id=run_id,
            )
            request = {
                "text": text, "get_facts": get_facts, "sliding_window": sliding_window,
                "autosummary": autosummary, "branch": branch,
            }
            run = self._db.create_run(Run(
                id=run_id, chat_id=chat_id, kind="message", source=source, status="queued",
                request_json=json.dumps(request, ensure_ascii=False), client_request_id=client_request_id,
                user_message_id=user_msg.id, assistant_message_id=draft.id, created_at=int(time.time()),
            ))
            state = _RunState(run_id, chat_id, "message")
            self._states[run_id] = state
        stream = chat.settings.stream if use_stream is None else use_stream

        def events() -> Iterator[dict]:
            terminal: Optional[dict] = None
            for event in self._repo.execute_message(
                chat_id, user_msg, draft, get_facts=get_facts, sliding_window=sliding_window,
                autosummary=autosummary, branch=branch, use_stream=stream, token=state.token,
            ):
                if event["type"] in TERMINAL_EVENTS:
                    terminal = event
                else:
                    yield event
            task_id = self._repo.work_task_to_continue(chat_id, terminal) if terminal is not None else None
            if task_id is None or state.token.is_set():
                if terminal is not None:
                    yield terminal
                return
            # «Останавливаться на каждом этапе» выключено: рабочая задача,
            # начатая/продвинутая этим ответом, идёт дальше без подтверждения —
            # шаги Менеджера задач в том же запуске (как «Выполнить»).
            yield {
                "type": "task_step_done", "task_id": task_id, "message": terminal["message"],
                "task_status": "active", "should_continue": True,
            }
            step_draft = self._repo.prepare_task_step_draft(chat_id, run_id=state.run_id)
            yield {"type": "message_started", "assistant_message": step_draft}
            yield from self._task_events(state, task_id, step_draft, False, stream)

        self._executor.submit(self._worker, state, events)
        return run, user_msg, draft

    def start_task_run(
        self,
        chat_id: str,
        task_id: str,
        *,
        auto_pause: bool = False,
        client_request_id: Optional[str] = None,
        source: str = "app",
        use_stream: Optional[bool] = None,
    ) -> Tuple[Run, Optional[Message]]:
        """Запуск Менеджера задач: `auto_pause=True` — один шаг
        (`task_step`, кнопка «Продолжить»), `False` — шаги подряд до
        завершения задачи (`task_run`, кнопка «Выполнить»)."""
        self._gc()
        with self._lock:
            existing = self._existing_for_request(chat_id, client_request_id)
            if existing is not None:
                return existing, self._message(existing.assistant_message_id)
            chat, _task = self._repo.validate_task_step(chat_id, task_id)
            self._ensure_no_active_run(chat_id)
            run_id = str(uuid.uuid4())
            kind = "task_step" if auto_pause else "task_run"
            draft = self._repo.prepare_task_step_draft(chat_id, run_id=run_id)
            run = self._db.create_run(Run(
                id=run_id, chat_id=chat_id, kind=kind, source=source, status="queued",
                request_json=json.dumps({"task_id": task_id, "auto_pause": auto_pause}),
                client_request_id=client_request_id, assistant_message_id=draft.id, task_id=task_id,
                created_at=int(time.time()),
            ))
            state = _RunState(run_id, chat_id, kind)
            self._states[run_id] = state
        stream = chat.settings.stream if use_stream is None else use_stream

        def events() -> Iterator[dict]:
            return self._task_events(state, task_id, draft, auto_pause, stream)

        self._executor.submit(self._worker, state, events)
        return run, draft

    def _task_events(
        self, state: _RunState, task_id: str, first_draft: Message, auto_pause: bool, use_stream: bool,
    ) -> Iterator[dict]:
        draft = first_draft
        last_done: Optional[dict] = None
        step = 0
        while True:
            step += 1
            if step > 1:
                draft = self._repo.prepare_task_step_draft(state.chat_id, run_id=state.run_id)
                yield {"type": "message_started", "assistant_message": draft}
            terminal: Optional[dict] = None
            for event in self._repo.execute_task_step(
                state.chat_id, task_id, draft, auto_pause=auto_pause, use_stream=use_stream, token=state.token,
            ):
                if event["type"] in TERMINAL_EVENTS:
                    terminal = event
                else:
                    yield event
            if terminal is None or terminal["type"] != "done":
                yield terminal or {"type": "error", "message": "шаг Менеджера задач завершился без результата"}
                return
            last_done = terminal
            yield {
                "type": "task_step_done", "task_id": task_id, "message": terminal["message"],
                "task_status": terminal.get("task_status"), "should_continue": terminal.get("should_continue", False),
            }
            if state.token.is_set():
                # «Пауза» во время task_run — остановка после текущего шага.
                yield {"type": "cancelled", "message": terminal["message"]}
                return
            if auto_pause or not terminal.get("should_continue"):
                break
        yield last_done

    def start_test_dialog_run(
        self,
        chat_id: str,
        dialog_id: str,
        *,
        get_facts: bool = False,
        sliding_window: bool = False,
        autosummary: str = "off",
        branch: Optional[int] = None,
        client_request_id: Optional[str] = None,
        source: str = "app",
        use_stream: Optional[bool] = None,
    ) -> Tuple[Run, Optional[Message], Optional[Message]]:
        """Запуск тестового диалога: вопросы набора по очереди, каждый — после
        ответа модели на предыдущий. Возвращает (запуск, первый вопрос,
        черновик первого ответа). Пока запуск идёт, другие сообщения в чат не
        отправляются (`RunConflictError`). Предельное время —
        `AGENT_RUN_TIMEOUT` на каждый вопрос."""
        self._gc()
        with self._lock:
            existing = self._existing_for_request(chat_id, client_request_id)
            if existing is not None:
                return existing, self._message(existing.user_message_id), self._message(existing.assistant_message_id)
            dialog = self._repo.get_test_dialog(dialog_id)
            questions = list(dialog.questions)
            if not questions:
                raise ValidationError("в тестовом диалоге нет вопросов")
            self._ensure_no_active_run(chat_id)
            run_id = str(uuid.uuid4())
            chat, user_msg, draft = self._repo.prepare_message_run(
                chat_id, questions[0], get_facts=get_facts, sliding_window=sliding_window, autosummary=autosummary,
                branch=branch, source=TEST_DIALOG_SOURCE, run_id=run_id,
            )
            request = {
                "test_dialog_id": dialog.id, "name": dialog.name, "total": len(questions), "step": 1,
                "get_facts": get_facts, "sliding_window": sliding_window, "autosummary": autosummary,
                "branch": branch,
            }
            run = self._db.create_run(Run(
                id=run_id, chat_id=chat_id, kind="test_dialog", source=source, status="queued",
                request_json=json.dumps(request, ensure_ascii=False), client_request_id=client_request_id,
                user_message_id=user_msg.id, assistant_message_id=draft.id, created_at=int(time.time()),
            ))
            state = _RunState(run_id, chat_id, "test_dialog")
            state.timeout = self._timeout * len(questions)
            self._states[run_id] = state
        stream = chat.settings.stream if use_stream is None else use_stream
        flags = {"get_facts": get_facts, "sliding_window": sliding_window, "autosummary": autosummary, "branch": branch}

        def events() -> Iterator[dict]:
            return self._test_dialog_events(state, request, questions, user_msg, draft, flags, stream)

        self._executor.submit(self._worker, state, events)
        return run, user_msg, draft

    def _test_dialog_events(
        self, state: _RunState, request: Dict[str, Any], questions: List[str], first_user: Message,
        first_draft: Message, flags: Dict[str, Any], use_stream: bool,
    ) -> Iterator[dict]:
        total = len(questions)
        name = request["name"]
        user_msg, draft = first_user, first_draft
        last_done: Optional[dict] = None

        def progress(step: int) -> Dict[str, Any]:
            return {"id": request["test_dialog_id"], "name": name, "step": step, "total": total}

        for step, question in enumerate(questions, start=1):
            if step > 1:
                if state.token.is_set():
                    # «Остановить» между ответами — следующие вопросы не отправляются.
                    yield {"type": "cancelled", "message": last_done["message"] if last_done else None,
                           "test_dialog": progress(step - 1)}
                    return
                try:
                    _chat, user_msg, draft = self._repo.prepare_message_run(
                        state.chat_id, question, source=TEST_DIALOG_SOURCE, run_id=state.run_id, **flags,
                    )
                except (ValidationError, NotFoundError) as exc:
                    yield {"type": "error", "test_dialog": progress(step),
                           "message": f"Тестовый диалог остановлен на вопросе {step} из {total}: {exc}"}
                    return
                self._db.update_run(
                    state.run_id, user_message_id=user_msg.id,
                    request_json=json.dumps(dict(request, step=step), ensure_ascii=False),
                )
                yield {"type": "message_started", "user_message": user_msg, "assistant_message": draft,
                       "test_dialog": progress(step)}
            yield {"type": "status", "status": f"Тестовый диалог «{name}»: вопрос {step} из {total}"}
            terminal: Optional[dict] = None
            for event in self._repo.execute_message(
                state.chat_id, user_msg, draft, use_stream=use_stream, token=state.token, **flags,
            ):
                if event["type"] in TERMINAL_EVENTS:
                    terminal = event
                else:
                    yield event
            if terminal is None or terminal["type"] != "done":
                # Ошибка или отмена ответа — цепочка останавливается.
                ended = dict(terminal) if terminal else {"type": "error", "message": "ответ завершился без результата"}
                if ended["type"] == "error" and isinstance(ended.get("message"), str):
                    ended["message"] = f"Тестовый диалог остановлен на вопросе {step} из {total}: {ended['message']}"
                ended["test_dialog"] = progress(step)
                yield ended
                return
            last_done = terminal
            yield {"type": "test_dialog_step_done", "test_dialog": progress(step), "message": terminal["message"]}
        yield dict(last_done, test_dialog=progress(total))

    # ---- выполнение ---------------------------------------------------------

    def _worker(self, state: _RunState, events_factory: Callable[[], Iterator[dict]]) -> None:
        timer: Optional[threading.Timer] = None
        try:
            run = self._db.get_run(state.run_id)
            if run is None:
                return
            if state.token.is_set():
                # Отменён, ещё не начавшись (стоял в очереди пула).
                draft = self._message(run.assistant_message_id)
                saved = None
                if draft is not None and draft.status == "streaming":
                    saved = self._repo._finalize_draft(draft, content="", status="cancelled")
                event = self._publish(state, {"type": "cancelled", "message": saved})
                self._finish(state, "cancelled", last_seq=event["seq"])
                return

            self._db.update_run(state.run_id, status="running", started_at=int(time.time()))
            run = self._db.get_run(state.run_id)
            self._publish(state, {
                "type": "run_started", "kind": run.kind,
                "user_message": self._message(run.user_message_id),
                "assistant_message": self._message(run.assistant_message_id),
            })
            self._repo.events.publish({"type": "run_started", "chat_id": state.chat_id, "run": run_brief(run)})

            timer = threading.Timer(state.timeout or self._timeout, state.token.cancel, kwargs={"reason": "timeout"})
            timer.daemon = True
            timer.start()

            writer = _DraftWriter(self._db, state.run_id, run.assistant_message_id)
            terminal: Optional[Dict[str, Any]] = None
            for event in events_factory():
                published = self._publish(state, event)
                kind = event["type"]
                if kind in TERMINAL_EVENTS:
                    terminal = published
                    continue
                if kind == "message_started":
                    writer.switch(event["assistant_message"].id, published["seq"])
                else:
                    writer.on_event(published)
                if kind == "status":
                    self._publish_global_status(state, event.get("status"))
            if terminal is None:
                raise RuntimeError("запуск завершился без итогового события")
            status = _RUN_STATUS_BY_TERMINAL[terminal["type"]]
            error = terminal.get("message") if terminal["type"] == "error" else None
            self._finish(state, status, error=error if isinstance(error, str) else None, last_seq=terminal["seq"])
        except Exception as exc:  # непредвиденная ошибка — запуск не должен «зависнуть»
            _logger.exception("Запуск %s завершился с ошибкой", state.run_id)
            saved = None
            try:
                run = self._db.get_run(state.run_id)
                draft = self._message(run.assistant_message_id) if run else None
                if draft is not None and draft.status == "streaming":
                    saved = self._repo._finalize_draft(draft, content=draft.content, status="failed", error=str(exc))
            except Exception:
                _logger.exception("Не удалось сохранить черновик запуска %s", state.run_id)
            event = self._publish(state, {"type": "error", "message": str(exc), "assistant_message": saved})
            self._finish(state, "failed", error=str(exc), last_seq=event["seq"])
        finally:
            if timer is not None:
                timer.cancel()

    def _publish(self, state: _RunState, event: Dict[str, Any]) -> Dict[str, Any]:
        return state.log.publish(dict(event, run_id=state.run_id, chat_id=state.chat_id))

    def _publish_global_status(self, state: _RunState, status: Optional[str]) -> None:
        now = time.monotonic()
        if now - state.last_global_status_at < 1.0:
            return
        state.last_global_status_at = now
        self._repo.events.publish({
            "type": "run_status", "chat_id": state.chat_id, "run_id": state.run_id, "current_status": status,
        })

    def _finish(self, state: _RunState, status: str, *, last_seq: int, error: Optional[str] = None) -> None:
        self._db.update_run(
            state.run_id, status=status, finished_at=int(time.time()), current_status=None,
            error=error, last_seq=last_seq,
        )
        state.finished_at = time.monotonic()
        state.log.close()
        self._repo.events.publish({
            "type": "run_finished", "chat_id": state.chat_id, "run_id": state.run_id, "status": status,
        })

    # ---- чтение и управление ------------------------------------------------

    def get(self, run_id: str) -> Run:
        run = self._db.get_run(run_id)
        if run is None:
            raise NotFoundError(f"no such run: {run_id}")
        return run

    def snapshot(self, run_id: str) -> Tuple[Run, Optional[Message], int]:
        """Запуск, сохранённый черновик (или итог) ответа и номер события, по
        которому этот черновик актуален — клиент подписывается с него."""
        run = self.get(run_id)
        return run, self._message(run.assistant_message_id), run.last_seq

    def read_events(self, run_id: str, after: int, timeout: float) -> Tuple[List[Dict[str, Any]], bool]:
        """События запуска после `after`; ждёт новых до `timeout` секунд.
        `EventsGone` — лента уже выброшена из памяти (клиент перечитывает
        сообщения), `NotFoundError` — такого запуска нет."""
        self._gc()
        with self._lock:
            state = self._states.get(run_id)
        if state is None:
            self.get(run_id)
            raise EventsGone(f"events of run {run_id} are no longer available")
        return state.log.read(after, timeout)

    def cancel(self, run_id: str) -> Run:
        with self._lock:
            state = self._states.get(run_id)
        run = self.get(run_id)
        if state is not None and state.finished_at is None:
            state.token.cancel("user")
        return run

    def wait(self, run_id: str, timeout: Optional[float] = None) -> Run:
        with self._lock:
            state = self._states.get(run_id)
        if state is not None:
            state.log.wait_closed(timeout)
        return self.get(run_id)

    def list_active(self) -> List[Run]:
        return self._db.list_active_runs()

    def active_for_chat(self, chat_id: str) -> Optional[Run]:
        return self._db.active_run_for_chat(chat_id)

    # ---- служебное ----------------------------------------------------------

    def _message(self, message_id: Optional[int]) -> Optional[Message]:
        return self._db.get_message(message_id) if message_id else None

    def _existing_for_request(self, chat_id: str, client_request_id: Optional[str]) -> Optional[Run]:
        if not client_request_id:
            return None
        self._repo.get_chat(chat_id)  # 404 для несуществующего чата
        return self._db.find_run_by_client_request(chat_id, client_request_id)

    def _ensure_no_active_run(self, chat_id: str) -> None:
        active = self._db.active_run_for_chat(chat_id)
        if active is not None:
            raise RunConflictError("В этом чате ещё формируется ответ", active_run_id=active.id)

    def _gc(self) -> None:
        cutoff = time.monotonic() - self._events_ttl
        with self._lock:
            stale = [rid for rid, st in self._states.items() if st.finished_at is not None and st.finished_at < cutoff]
            for rid in stale:
                del self._states[rid]

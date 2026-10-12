"""
agents_core.events
====================

Лента событий в памяти процесса (ТЗ «асинхронные ответы», разделы 2.2 и
2.6) — общий механизм для двух случаев:

- лента ОДНОГО запуска (`runs.RunManager`): все события ответа — статус,
  куски текста, вызовы инструментов, итог; клиент подписывается с номера
  `after` и получает сначала пропущенное, потом новое вживую;
- ОБЩАЯ лента изменений сервиса (`Repository.events`): новые/изменённые/
  удалённые чаты, начало/фаза/конец запусков, непрочитанные — одно
  соединение на всё приложение, чтобы главный экран обновлялся без опроса.

Номер события (`seq`) — сквозной в пределах ленты, начиная с 1. Лента может
«забывать» старые события (по числу и возрасту) — тогда чтение с
устаревшего номера поднимает `EventsGone`, и клиент перечитывает состояние
целиком (HTTP 410).

Потокобезопасно: публикуют рабочие потоки запусков, читают потоки
SSE-обработчиков (синхронные генераторы в пуле потоков Starlette).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple


class EventsGone(Exception):
    """Запрошенные события уже удалены из памяти (слишком старый `after`,
    либо лента запуска уже выброшена) — клиенту нужно перечитать состояние."""


class EventLog:
    def __init__(self, max_events: Optional[int] = None, max_age_seconds: Optional[float] = None):
        self._events: Deque[Dict[str, Any]] = deque()
        self._seq = 0
        self._cond = threading.Condition()
        self._closed = False
        self._max_events = max_events
        self._max_age = max_age_seconds

    @property
    def last_seq(self) -> int:
        with self._cond:
            return self._seq

    @property
    def closed(self) -> bool:
        with self._cond:
            return self._closed

    def publish(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Добавляет событие, присваивает `seq`/`ts`; возвращает итоговый
        объект события (именно он уйдёт подписчикам)."""
        with self._cond:
            self._seq += 1
            stored = dict(event)
            stored["seq"] = self._seq
            stored.setdefault("ts", int(time.time()))
            self._events.append(stored)
            self._trim_locked()
            self._cond.notify_all()
            return stored

    def close(self) -> None:
        """Больше событий не будет (запуск завершён) — будит всех ждущих."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def wait_closed(self, timeout: Optional[float] = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while not self._closed:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._cond.wait(remaining)
            return True

    def read(self, after: int, timeout: float = 0.0) -> Tuple[List[Dict[str, Any]], bool]:
        """События с номером больше `after`. Если таких пока нет и лента не
        закрыта — ждёт до `timeout` секунд. Возвращает (события, закрыта ли
        лента). `EventsGone` — если часть запрошенного уже забыта."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._cond:
            while True:
                self._trim_locked()
                if after > self._seq:
                    # Номер из «прошлой жизни» ленты (сервис перезапущен и
                    # нумерация началась заново) — состояние клиента устарело.
                    raise EventsGone(f"cursor {after} is ahead of the log ({self._seq})")
                if after < self._seq:
                    first_kept = self._events[0]["seq"] if self._events else self._seq + 1
                    if after < first_kept - 1:
                        raise EventsGone(f"events after {after} are no longer available")
                    return [e for e in self._events if e["seq"] > after], self._closed
                if self._closed:
                    return [], True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return [], False
                self._cond.wait(remaining)

    def _trim_locked(self) -> None:
        if self._max_events is not None:
            while len(self._events) > self._max_events:
                self._events.popleft()
        if self._max_age is not None:
            cutoff = time.time() - self._max_age
            while self._events and self._events[0].get("ts", 0) < cutoff:
                self._events.popleft()


class RunCancelled(Exception):
    """Запуск остановлен: пользователем (`reason="user"`) или по тайм-ауту
    (`reason="timeout"`). Поднимается исполнителем ответа в точках проверки
    (между кусками ответа, перед раундом и перед каждым вызовом инструмента)."""

    def __init__(self, reason: str = "user"):
        super().__init__(reason)
        self.reason = reason


class CancelToken:
    """Кооперативный флаг отмены запуска (ТЗ, раздел 2.4)."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self.reason: str = "user"

    def cancel(self, reason: str = "user") -> None:
        if not self._event.is_set():
            self.reason = reason
            self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise RunCancelled(self.reason)

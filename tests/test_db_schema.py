"""
Схема базы создаётся с нуля при первом открытии (миграций нет — см.
`agents_core/db.py`). Проверяем, что все колонки, которые пишет код, есть
в свежей схеме, и что данные переживают повторное открытие файла.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from agents_core.db import Database
from agents_core.models import Settings


class FreshSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(self.path)

    def tearDown(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            try:
                os.unlink(self.path + suffix)
            except OSError:
                pass

    def _columns(self, table: str) -> set:
        conn = sqlite3.connect(self.path)
        try:
            return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        finally:
            conn.close()

    def test_fresh_schema_has_all_columns(self):
        Database(self.path)
        self.assertLessEqual(
            {"status", "run_id", "tool_events", "error", "source", "branch", "facts", "task_events"},
            self._columns("messages"),
        )
        self.assertLessEqual(
            {"invariant_ids", "last_read_message_id", "source", "active_profile_id", "task_tracking_enabled"},
            self._columns("chats"),
        )
        self.assertLessEqual({"invariant_ids", "default_profile_id"}, self._columns("agents"))
        self.assertIn("description", self._columns("tasks"))
        self.assertNotIn("agent_id", self._columns("profiles"))

    def test_data_survives_reopen(self):
        db = Database(self.path)
        settings = Settings(model="deepseek:deepseek-chat", context_strategy="sliding_window", context_strategy_limit=6)
        agent = db.create_agent("Агент", settings)
        reopened = Database(self.path).get_agent(agent.id)
        self.assertEqual(reopened.name, "Агент")
        self.assertEqual(reopened.settings.context_strategy, "sliding_window")
        self.assertEqual(reopened.settings.context_strategy_limit, 6)
        self.assertFalse(reopened.settings.working_memory_enabled)
        self.assertEqual(reopened.invariant_ids, [])


if __name__ == "__main__":
    unittest.main()

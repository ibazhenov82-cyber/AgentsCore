"""Старая база — понятная ошибка при старте, а не «no such column» на первом запросе."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from agents_core.db import Database, SchemaMismatchError


class SchemaCheckTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "agents.sqlite")

    def tearDown(self):
        self.dir.cleanup()

    def test_fresh_and_reopened_database(self):
        Database(self.path)
        Database(self.path)

    def test_old_database_is_reported(self):
        Database(self.path)
        conn = sqlite3.connect(self.path)
        table = next(r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                     if "rag_rerank" in [c[1] for c in conn.execute(f"PRAGMA table_info({r[0]})")])
        conn.execute(f"ALTER TABLE {table} DROP COLUMN rag_rerank")
        conn.commit()
        conn.close()
        with self.assertRaises(SchemaMismatchError) as ctx:
            Database(self.path)
        self.assertIn("нет колонок rag_rerank", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

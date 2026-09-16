"""
agents_core.db
===============

Слой хранения на SQLite: агенты (со встроенными настройками), их чаты (тоже
со встроенными настройками), сообщения чатов (включая ветки диалога и
факты стратегии "Sticky Facts"), ветки чатов, единственная строка настроек
по умолчанию и каталог моделей.

Явно перечисленные списки колонок и типов (а не "SELECT *") — чтобы схема
была видна в одном месте и чтобы числовые колонки сохраняли affinity
(REAL/INTEGER), а не подменялись TEXT из-за неявного приведения типов.

`_ensure_column()` — простая защитная миграция: если сервис обновили поверх
базы, созданной предыдущей версией (без новых колонок), недостающие колонки
добавляются через `ALTER TABLE ... ADD COLUMN` при старте, без потери данных.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional

from .models import (
    Agent,
    Branch,
    Chat,
    DefaultSettings,
    LongTermMemoryEntry,
    Message,
    ModelInfo,
    Profile,
    Settings,
    WorkingMemoryEntry,
)

# Колонки настроек (одинаковы для agents и chats) и их SQL-типы.
_SETTINGS_COLUMNS = [
    ("model", "TEXT"),
    ("system_prompt", "TEXT"),
    ("temperature", "REAL"),
    ("top_p", "REAL"),
    ("seed", "INTEGER"),
    ("stream", "INTEGER"),
    ("thinking_enabled", "INTEGER"),
    ("reasoning_effort", "TEXT"),
    ("max_tokens", "INTEGER"),
    ("json_mode", "INTEGER"),
    ("stop_sequences", "TEXT"),  # JSON-массив строк
    ("tool_choice", "TEXT"),
    ("tools_json", "TEXT"),
    ("memory_tools_enabled", "INTEGER"),
    ("working_memory_enabled", "INTEGER"),
    ("long_term_memory_enabled", "INTEGER"),
    ("episodic_memory_enabled", "INTEGER"),
    ("semantic_memory_enabled", "INTEGER"),
    ("procedural_memory_enabled", "INTEGER"),
    ("summary_prompt", "TEXT"),
    ("summary_system_prompt", "TEXT"),
    ("autosummary", "TEXT"),
    ("autosummary_by_messages", "INTEGER"),
    ("autosummary_by_tokens", "INTEGER"),
    ("context_strategy", "TEXT"),
    ("context_strategy_limit", "INTEGER"),
    ("extraction_system_prompt", "TEXT"),
    ("include_usage_in_stream", "INTEGER"),
    ("logprobs", "INTEGER"),
    ("frequency_penalty", "REAL"),
    ("presence_penalty", "REAL"),
]

_DEFAULT_SETTINGS_COLUMNS = [
    ("model", "TEXT"),
    ("system_prompt", "TEXT"),
    ("temperature", "REAL"),
    ("top_p", "REAL"),
    ("seed", "INTEGER"),
    ("stream", "INTEGER"),
    ("thinking_enabled", "INTEGER"),
    ("reasoning_effort", "TEXT"),
    ("summary_prompt", "TEXT"),
    ("summary_system_prompt", "TEXT"),
    ("include_usage_in_stream", "INTEGER"),
]

_MESSAGE_EXTRA_COLUMNS = [
    ("format", "TEXT NOT NULL DEFAULT 'text'"),
    ("branch", "INTEGER NOT NULL DEFAULT 0"),
    ("facts", "TEXT"),
]


def _settings_to_row(settings: Settings) -> Dict[str, Any]:
    return {
        "model": settings.model,
        "system_prompt": settings.system_prompt,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "seed": settings.seed,
        "stream": int(settings.stream),
        "thinking_enabled": int(settings.thinking_enabled),
        "reasoning_effort": settings.reasoning_effort,
        "max_tokens": settings.max_tokens,
        "json_mode": int(settings.json_mode),
        "stop_sequences": json.dumps(settings.stop_sequences, ensure_ascii=False),
        "tool_choice": settings.tool_choice,
        "tools_json": settings.tools_json,
        "memory_tools_enabled": int(settings.memory_tools_enabled),
        "working_memory_enabled": int(settings.working_memory_enabled),
        "long_term_memory_enabled": int(settings.long_term_memory_enabled),
        "episodic_memory_enabled": int(settings.episodic_memory_enabled),
        "semantic_memory_enabled": int(settings.semantic_memory_enabled),
        "procedural_memory_enabled": int(settings.procedural_memory_enabled),
        "summary_prompt": settings.summary_prompt,
        "summary_system_prompt": settings.summary_system_prompt,
        "autosummary": settings.autosummary,
        "autosummary_by_messages": settings.autosummary_by_messages,
        "autosummary_by_tokens": settings.autosummary_by_tokens,
        "context_strategy": settings.context_strategy,
        "context_strategy_limit": settings.context_strategy_limit,
        "extraction_system_prompt": settings.extraction_system_prompt,
        "include_usage_in_stream": int(settings.include_usage_in_stream),
        "logprobs": int(settings.logprobs),
        "frequency_penalty": settings.frequency_penalty,
        "presence_penalty": settings.presence_penalty,
    }


def _row_to_settings(row: sqlite3.Row) -> Settings:
    keys = row.keys()
    return Settings(
        model=row["model"],
        system_prompt=row["system_prompt"],
        temperature=row["temperature"],
        top_p=row["top_p"],
        seed=row["seed"],
        stream=bool(row["stream"]),
        thinking_enabled=bool(row["thinking_enabled"]),
        reasoning_effort=row["reasoning_effort"],
        max_tokens=row["max_tokens"],
        json_mode=bool(row["json_mode"]),
        stop_sequences=json.loads(row["stop_sequences"]) if row["stop_sequences"] else [],
        tool_choice=row["tool_choice"],
        tools_json=row["tools_json"] or "",
        memory_tools_enabled=bool(row["memory_tools_enabled"]) if "memory_tools_enabled" in keys else False,
        # ВАЖНО: это НЕ дефолт для новых агентов (тот теперь False — см.
        # `Settings.working_memory_enabled`/`long_term_memory_enabled`) — это
        # обратная совместимость ТОЛЬКО для строк, созданных ДО появления этих
        # колонок: после ALTER TABLE такая строка первое время хранит NULL, а
        # не 0/1, и здесь мы явно отличаем "значения нет" (мигрированная старая
        # строка → True, не потерять то, что уже было включено) от "явно
        # выключено" (0, в т.ч. для только что созданных агентов/чатов).
        working_memory_enabled=(
            bool(row["working_memory_enabled"])
            if "working_memory_enabled" in keys and row["working_memory_enabled"] is not None
            else True
        ),
        long_term_memory_enabled=(
            bool(row["long_term_memory_enabled"])
            if "long_term_memory_enabled" in keys and row["long_term_memory_enabled"] is not None
            else True
        ),
        episodic_memory_enabled=bool(row["episodic_memory_enabled"]) if "episodic_memory_enabled" in keys and row["episodic_memory_enabled"] is not None else False,
        semantic_memory_enabled=bool(row["semantic_memory_enabled"]) if "semantic_memory_enabled" in keys and row["semantic_memory_enabled"] is not None else False,
        procedural_memory_enabled=bool(row["procedural_memory_enabled"]) if "procedural_memory_enabled" in keys and row["procedural_memory_enabled"] is not None else False,
        summary_prompt=row["summary_prompt"],
        summary_system_prompt=(row["summary_system_prompt"] if "summary_system_prompt" in keys else None)
        or Settings.__dataclass_fields__["summary_system_prompt"].default,
        autosummary=row["autosummary"] or "off",
        autosummary_by_messages=row["autosummary_by_messages"],
        autosummary_by_tokens=row["autosummary_by_tokens"],
        context_strategy=row["context_strategy"] if "context_strategy" in keys else None,
        context_strategy_limit=row["context_strategy_limit"] if "context_strategy_limit" in keys else None,
        extraction_system_prompt=(row["extraction_system_prompt"] if "extraction_system_prompt" in keys else None)
        or Settings.__dataclass_fields__["extraction_system_prompt"].default,
        include_usage_in_stream=bool(row["include_usage_in_stream"]),
        logprobs=bool(row["logprobs"]),
        frequency_penalty=row["frequency_penalty"],
        presence_penalty=row["presence_penalty"],
    )


def _default_settings_to_row(settings: DefaultSettings) -> Dict[str, Any]:
    return {
        "model": settings.model,
        "system_prompt": settings.system_prompt,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "seed": settings.seed,
        "stream": int(settings.stream),
        "thinking_enabled": int(settings.thinking_enabled),
        "reasoning_effort": settings.reasoning_effort,
        "summary_prompt": settings.summary_prompt,
        "summary_system_prompt": settings.summary_system_prompt,
        "include_usage_in_stream": int(settings.include_usage_in_stream),
    }


def _row_to_default_settings(row: sqlite3.Row) -> DefaultSettings:
    keys = row.keys()
    return DefaultSettings(
        model=row["model"],
        system_prompt=row["system_prompt"],
        temperature=row["temperature"],
        top_p=row["top_p"],
        seed=row["seed"],
        stream=bool(row["stream"]),
        thinking_enabled=bool(row["thinking_enabled"]),
        reasoning_effort=row["reasoning_effort"],
        summary_prompt=row["summary_prompt"],
        summary_system_prompt=(row["summary_system_prompt"] if "summary_system_prompt" in keys else None)
        or DefaultSettings.__dataclass_fields__["summary_system_prompt"].default,
        include_usage_in_stream=bool(row["include_usage_in_stream"]),
    )


class Database:
    def __init__(self, path: str):
        self.path = path
        self._init_schema()

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_column(self, conn: sqlite3.Connection, table: str, name: str, sql_type: str) -> None:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")

    def _init_schema(self) -> None:
        settings_cols_sql = ",\n".join(f"{name} {sql_type}" for name, sql_type in _SETTINGS_COLUMNS)
        default_settings_cols_sql = ",\n".join(f"{name} {sql_type}" for name, sql_type in _DEFAULT_SETTINGS_COLUMNS)
        with self._connect() as conn:
            conn.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS default_settings (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    {default_settings_cols_sql}
                );

                CREATE TABLE IF NOT EXISTS agents (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    default_profile_id TEXT REFERENCES profiles(id) ON DELETE SET NULL,
                    {settings_cols_sql}
                );

                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    active_profile_id TEXT REFERENCES profiles(id) ON DELETE SET NULL,
                    {settings_cols_sql}
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    reasoning_content TEXT,
                    is_summary INTEGER NOT NULL DEFAULT 0,
                    duration_ms INTEGER,
                    total_tokens INTEGER,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    format TEXT NOT NULL DEFAULT 'text',
                    branch INTEGER NOT NULL DEFAULT 0,
                    facts TEXT
                );

                CREATE TABLE IF NOT EXISTS chat_branches (
                    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    number INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    PRIMARY KEY (chat_id, number)
                );

                CREATE TABLE IF NOT EXISTS models_catalog (
                    provider TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    is_local INTEGER NOT NULL,
                    context_window INTEGER,
                    max_input_tokens INTEGER,
                    max_output_tokens INTEGER,
                    max_reasoning_tokens INTEGER,
                    supports_thinking INTEGER NOT NULL DEFAULT 0,
                    supports_tools INTEGER NOT NULL DEFAULT 0,
                    supports_json_mode INTEGER NOT NULL DEFAULT 0,
                    supports_logprobs INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (provider, model_id)
                );

                CREATE TABLE IF NOT EXISTS profiles (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    style TEXT,
                    format TEXT,
                    constraints TEXT,
                    skills_json TEXT NOT NULL DEFAULT '',
                    orchestration_prompt TEXT,
                    is_default INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS working_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'manual',
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    UNIQUE (chat_id, key)
                );

                CREATE TABLE IF NOT EXISTS long_term_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
                    category TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'manual',
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    UNIQUE (agent_id, category, key)
                );

                CREATE INDEX IF NOT EXISTS idx_chats_agent_id ON chats(agent_id);
                CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);
                CREATE INDEX IF NOT EXISTS idx_branches_chat_id ON chat_branches(chat_id);
                CREATE INDEX IF NOT EXISTS idx_working_memory_chat_id ON working_memory(chat_id);
                CREATE INDEX IF NOT EXISTS idx_long_term_memory_agent_id ON long_term_memory(agent_id);
                """
            )
            # Защитная миграция: если база создана предыдущей версией сервиса
            # (до появления новых настроек/полей сообщения), добавляем
            # недостающие колонки, не трогая уже накопленные данные.
            for table in ("agents", "chats"):
                for name, sql_type in _SETTINGS_COLUMNS:
                    self._ensure_column(conn, table, name, sql_type)
            for name, sql_type in _DEFAULT_SETTINGS_COLUMNS:
                self._ensure_column(conn, "default_settings", name, sql_type)
            for name, sql_type in _MESSAGE_EXTRA_COLUMNS:
                self._ensure_column(conn, "messages", name, sql_type)
            self._ensure_column(conn, "chats", "active_profile_id", "TEXT")
            self._ensure_column(conn, "agents", "default_profile_id", "TEXT")
            self._migrate_profiles_to_global_catalog(conn)

    def _migrate_profiles_to_global_catalog(self, conn: sqlite3.Connection) -> None:
        """Раньше `profiles` были привязаны к ОДНОМУ агенту (`agent_id NOT
        NULL REFERENCES agents(id) ON DELETE CASCADE`) — по замечанию
        пользователя профили теперь общий справочник для ВСЕХ агентов
        (выбираются в настройках любого агента/чата), поэтому убираем эту
        привязку из схемы. SQLite не умеет ALTER TABLE DROP COLUMN/CONSTRAINT
        для такого случая, поэтому пересобираем таблицу.

        Порядок операций важен: если переименовать САМУ `profiles` (напр. в
        `profiles_old`) при включённых внешних ключах, SQLite перезапишет
        текст FK-определения в `chats.active_profile_id` на новое имя — и он
        останется указывать на `profiles_old` даже после того, как мы создадим
        новую таблицу `profiles` и удалим `profiles_old` (проверено вручную:
        JOIN после такой последовательности молча возвращает NULL). Поэтому
        вместо переименования исходной таблицы создаём НОВУЮ под временным
        именем, переносим данные, удаляем СТАРУЮ `profiles`, и только потом
        переименовываем новую в `profiles` — на этом шаге чужие FK ни на что
        не указывают, переписывать нечего."""
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(profiles)").fetchall()}
        if "agent_id" not in cols:
            return  # уже мигрировано (или свежая база, сразу созданная по новой схеме)
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.executescript(
            """
            CREATE TABLE profiles__migrating_new (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                style TEXT,
                format TEXT,
                constraints TEXT,
                skills_json TEXT NOT NULL DEFAULT '',
                orchestration_prompt TEXT,
                is_default INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            INSERT INTO profiles__migrating_new (
                id, name, style, format, constraints, skills_json,
                orchestration_prompt, is_default, created_at, updated_at
            )
            SELECT id, name, style, format, constraints, skills_json,
                   orchestration_prompt, is_default, created_at, updated_at
            FROM profiles;
            DROP INDEX IF EXISTS idx_profiles_agent_id;
            DROP TABLE profiles;
            ALTER TABLE profiles__migrating_new RENAME TO profiles;
            """
        )
        conn.execute("PRAGMA foreign_keys = ON")

    # ---- настройки по умолчанию -----------------------------------------

    def get_default_settings(self) -> DefaultSettings:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM default_settings WHERE id = 1").fetchone()
            if row is None:
                self._seed_default_settings(conn, DefaultSettings())
                row = conn.execute("SELECT * FROM default_settings WHERE id = 1").fetchone()
            return _row_to_default_settings(row)

    def _seed_default_settings(self, conn: sqlite3.Connection, settings: DefaultSettings) -> None:
        row = _default_settings_to_row(settings)
        columns = ["id"] + list(row.keys())
        placeholders = ", ".join("?" for _ in columns)
        values = [1] + list(row.values())
        conn.execute(f"INSERT INTO default_settings ({', '.join(columns)}) VALUES ({placeholders})", values)

    def update_default_settings(self, settings: DefaultSettings) -> DefaultSettings:
        with self._connect() as conn:
            existing = conn.execute("SELECT id FROM default_settings WHERE id = 1").fetchone()
            row = _default_settings_to_row(settings)
            if existing is None:
                self._seed_default_settings(conn, settings)
            else:
                assignments = ", ".join(f"{k} = ?" for k in row.keys())
                conn.execute(f"UPDATE default_settings SET {assignments} WHERE id = 1", list(row.values()))
        return settings

    # ---- агенты -----------------------------------------------------------

    def create_agent(self, name: str, settings: Settings) -> Agent:
        agent_id = str(uuid.uuid4())
        now = int(time.time())
        row = _settings_to_row(settings)
        with self._connect() as conn:
            columns = ["id", "name", "created_at", "updated_at"] + list(row.keys())
            values = [agent_id, name, now, now] + list(row.values())
            placeholders = ", ".join("?" for _ in columns)
            conn.execute(f"INSERT INTO agents ({', '.join(columns)}) VALUES ({placeholders})", values)
        return Agent(id=agent_id, name=name, created_at=now, updated_at=now, settings=settings)

    @staticmethod
    def _agent_default_profile_id(row: sqlite3.Row) -> Optional[str]:
        # Обратная совместимость: строки, созданные до появления этой
        # колонки, не имеют её вовсе (см. _ensure_column) — просто "профиль
        # по умолчанию не задан", а не ошибка.
        return row["default_profile_id"] if "default_profile_id" in row.keys() else None

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
            if row is None:
                return None
            return Agent(
                id=row["id"], name=row["name"], created_at=row["created_at"],
                updated_at=row["updated_at"], settings=_row_to_settings(row),
                default_profile_id=self._agent_default_profile_id(row),
            )

    def list_agents(self) -> List[Agent]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM agents ORDER BY updated_at DESC").fetchall()
            return [
                Agent(id=r["id"], name=r["name"], created_at=r["created_at"],
                      updated_at=r["updated_at"], settings=_row_to_settings(r),
                      default_profile_id=self._agent_default_profile_id(r))
                for r in rows
            ]

    def rename_agent(self, agent_id: str, name: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE agents SET name = ?, updated_at = ? WHERE id = ?",
                (name, int(time.time()), agent_id),
            )

    def set_agent_default_profile(self, agent_id: str, profile_id: Optional[str]) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE agents SET default_profile_id = ?, updated_at = ? WHERE id = ?",
                (profile_id, int(time.time()), agent_id),
            )

    def update_agent_settings(self, agent_id: str, settings: Settings) -> None:
        row = _settings_to_row(settings)
        assignments = ", ".join(f"{k} = ?" for k in row.keys())
        with self._connect() as conn:
            conn.execute(
                f"UPDATE agents SET {assignments}, updated_at = ? WHERE id = ?",
                list(row.values()) + [int(time.time()), agent_id],
            )

    def delete_agent(self, agent_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM agents WHERE id = ?", (agent_id,))

    def touch_agent(self, agent_id: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE agents SET updated_at = ? WHERE id = ?", (int(time.time()), agent_id))

    # ---- чаты ---------------------------------------------------------------

    def create_chat(
        self, agent_id: str, title: str, settings: Settings, active_profile_id: Optional[str] = None,
    ) -> Chat:
        chat_id = str(uuid.uuid4())
        now = int(time.time())
        row = _settings_to_row(settings)
        with self._connect() as conn:
            columns = ["id", "agent_id", "title", "created_at", "updated_at", "active_profile_id"] + list(row.keys())
            values = [chat_id, agent_id, title, now, now, active_profile_id] + list(row.values())
            placeholders = ", ".join("?" for _ in columns)
            conn.execute(f"INSERT INTO chats ({', '.join(columns)}) VALUES ({placeholders})", values)
        return Chat(
            id=chat_id, agent_id=agent_id, title=title, created_at=now, updated_at=now, settings=settings,
            active_profile_id=active_profile_id,
        )

    def get_chat(self, chat_id: str) -> Optional[Chat]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
            if row is None:
                return None
            return Chat(
                id=row["id"], agent_id=row["agent_id"], title=row["title"],
                created_at=row["created_at"], updated_at=row["updated_at"],
                settings=_row_to_settings(row),
                active_profile_id=row["active_profile_id"] if "active_profile_id" in row.keys() else None,
            )

    def list_chats(self, agent_id: Optional[str] = None) -> List[Chat]:
        with self._connect() as conn:
            if agent_id is not None:
                rows = conn.execute(
                    "SELECT * FROM chats WHERE agent_id = ? ORDER BY updated_at DESC", (agent_id,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM chats ORDER BY updated_at DESC").fetchall()
            return [
                Chat(id=r["id"], agent_id=r["agent_id"], title=r["title"], created_at=r["created_at"],
                     updated_at=r["updated_at"], settings=_row_to_settings(r),
                     active_profile_id=r["active_profile_id"] if "active_profile_id" in r.keys() else None)
                for r in rows
            ]

    def set_chat_active_profile(self, chat_id: str, profile_id: Optional[str]) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE chats SET active_profile_id = ?, updated_at = ? WHERE id = ?",
                (profile_id, int(time.time()), chat_id),
            )

    def rename_chat(self, chat_id: str, title: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE chats SET title = ?, updated_at = ? WHERE id = ?",
                (title, int(time.time()), chat_id),
            )

    def update_chat_settings(self, chat_id: str, settings: Settings) -> None:
        row = _settings_to_row(settings)
        assignments = ", ".join(f"{k} = ?" for k in row.keys())
        with self._connect() as conn:
            conn.execute(
                f"UPDATE chats SET {assignments}, updated_at = ? WHERE id = ?",
                list(row.values()) + [int(time.time()), chat_id],
            )

    def delete_chat(self, chat_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))

    def touch_chat(self, chat_id: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (int(time.time()), chat_id))

    # ---- ветки диалога ------------------------------------------------------

    def list_branches(self, chat_id: str) -> List[Branch]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_branches WHERE chat_id = ? ORDER BY number ASC", (chat_id,)
            ).fetchall()
            return [Branch(chat_id=r["chat_id"], number=r["number"], name=r["name"], created_at=r["created_at"]) for r in rows]

    def create_branch(self, chat_id: str, name: Optional[str] = None) -> Branch:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(number), 0) AS max_number FROM chat_branches WHERE chat_id = ?", (chat_id,)
            ).fetchone()
            number = int(row["max_number"]) + 1
            # Автоимя "Ветка N" вычисляется здесь же, в одной транзакции с
            # выдачей номера — только так гарантируется, что цифра в имени
            # совпадает с реально присвоенным номером ветки.
            resolved_name = name if name and name.strip() else f"Ветка {number}"
            now = int(time.time())
            conn.execute(
                "INSERT INTO chat_branches (chat_id, number, name, created_at) VALUES (?, ?, ?, ?)",
                (chat_id, number, resolved_name, now),
            )
            return Branch(chat_id=chat_id, number=number, name=resolved_name, created_at=now)

    def get_branch(self, chat_id: str, number: int) -> Optional[Branch]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM chat_branches WHERE chat_id = ? AND number = ?", (chat_id, number)
            ).fetchone()
            if row is None:
                return None
            return Branch(chat_id=row["chat_id"], number=row["number"], name=row["name"], created_at=row["created_at"])

    def delete_branch(self, chat_id: str, number: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chat_branches WHERE chat_id = ? AND number = ?", (chat_id, number))
            conn.execute("DELETE FROM messages WHERE chat_id = ? AND branch = ?", (chat_id, number))

    # ---- сообщения ------------------------------------------------------

    def add_message(self, message: Message) -> Message:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO messages
                   (chat_id, role, content, created_at, reasoning_content, is_summary,
                    duration_ms, total_tokens, prompt_tokens, completion_tokens,
                    format, branch, facts)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    message.chat_id, message.role, message.content, message.created_at,
                    message.reasoning_content, int(message.is_summary),
                    message.duration_ms, message.total_tokens,
                    message.prompt_tokens, message.completion_tokens,
                    message.format, message.branch, message.facts,
                ),
            )
            message.id = cur.lastrowid
        return message

    def update_message_facts(self, message_id: int, facts_json: Optional[str]) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE messages SET facts = ? WHERE id = ?", (facts_json, message_id))

    def update_message_prompt_tokens(self, message_id: int, prompt_tokens: Optional[int]) -> None:
        """Проставляет `prompt_tokens` пользовательскому сообщению ПОСЛЕ того,
        как выполнен основной запрос к модели, — точное число входных токенов
        известно только из ответа провайдера (usage), а пользовательское
        сообщение сохраняется в БД раньше, чем этот ответ получен."""
        with self._connect() as conn:
            conn.execute("UPDATE messages SET prompt_tokens = ? WHERE id = ?", (prompt_tokens, message_id))

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> Message:
        keys = row.keys()
        return Message(
            id=row["id"], chat_id=row["chat_id"], role=row["role"], content=row["content"],
            created_at=row["created_at"], reasoning_content=row["reasoning_content"],
            is_summary=bool(row["is_summary"]), duration_ms=row["duration_ms"],
            total_tokens=row["total_tokens"], prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            format=(row["format"] if "format" in keys else None) or "text",
            branch=(row["branch"] if "branch" in keys else None) or 0,
            facts=row["facts"] if "facts" in keys else None,
        )

    def list_messages(self, chat_id: str) -> List[Message]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE chat_id = ? ORDER BY id ASC", (chat_id,)
            ).fetchall()
            return [self._row_to_message(r) for r in rows]

    def delete_message(self, chat_id: str, message_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE chat_id = ? AND id = ?", (chat_id, message_id))

    def bulk_delete_messages(self, chat_id: str, message_ids: List[int]) -> None:
        if not message_ids:
            return
        placeholders = ", ".join("?" for _ in message_ids)
        with self._connect() as conn:
            conn.execute(
                f"DELETE FROM messages WHERE chat_id = ? AND id IN ({placeholders})",
                [chat_id] + list(message_ids),
            )

    def clear_messages(self, chat_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))

    # ---- каталог моделей --------------------------------------------------

    def get_models_catalog_keys(self) -> List[str]:
        """Возвращает список 'provider:model_id' всех моделей, уже
        сохранённых в БД — используется при старте для сравнения со списком
        в .env (см. `repository.ModelCatalog.load_or_discover`)."""
        with self._connect() as conn:
            rows = conn.execute("SELECT provider, model_id FROM models_catalog").fetchall()
            return [f"{r['provider']}:{r['model_id']}" for r in rows]

    def replace_models_catalog(self, models: List[ModelInfo]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM models_catalog")
            for m in models:
                conn.execute(
                    """INSERT INTO models_catalog
                       (provider, model_id, display_name, is_local, context_window,
                        max_input_tokens, max_output_tokens, max_reasoning_tokens,
                        supports_thinking, supports_tools, supports_json_mode, supports_logprobs)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        m.provider, m.model_id, m.display_name, int(m.is_local), m.context_window,
                        m.max_input_tokens, m.max_output_tokens, m.max_reasoning_tokens,
                        int(m.supports_thinking), int(m.supports_tools),
                        int(m.supports_json_mode), int(m.supports_logprobs),
                    ),
                )

    def load_models_catalog(self) -> List[ModelInfo]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM models_catalog ORDER BY provider, model_id").fetchall()
            return [
                ModelInfo(
                    id=f"{r['provider']}:{r['model_id']}", provider=r["provider"], model_id=r["model_id"],
                    display_name=r["display_name"], is_local=bool(r["is_local"]),
                    context_window=r["context_window"], max_input_tokens=r["max_input_tokens"],
                    max_output_tokens=r["max_output_tokens"], max_reasoning_tokens=r["max_reasoning_tokens"],
                    supports_thinking=bool(r["supports_thinking"]), supports_tools=bool(r["supports_tools"]),
                    supports_json_mode=bool(r["supports_json_mode"]), supports_logprobs=bool(r["supports_logprobs"]),
                )
                for r in rows
            ]

    # ---- рабочая память (working_memory, область видимости — чат) ----------

    @staticmethod
    def _row_to_working_memory(row: sqlite3.Row) -> WorkingMemoryEntry:
        return WorkingMemoryEntry(
            id=row["id"], chat_id=row["chat_id"], key=row["key"], value=row["value"],
            source=row["source"] or "manual", created_at=row["created_at"], updated_at=row["updated_at"],
        )

    def upsert_working_memory(self, chat_id: str, key: str, value: str, source: str = "manual") -> WorkingMemoryEntry:
        """Сохраняет запись рабочей памяти; если запись с таким `key` в этом
        чате уже есть — обновляет значение и источник (upsert), не создавая
        дубликат. Именно так демо-скилл "Покупки" обновляет корзину под
        ключом "cart" при каждом add_to_cart."""
        now = int(time.time())
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO working_memory (chat_id, key, value, source, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(chat_id, key) DO UPDATE SET
                       value = excluded.value, source = excluded.source, updated_at = excluded.updated_at""",
                (chat_id, key, value, source, now, now),
            )
            row = conn.execute(
                "SELECT * FROM working_memory WHERE chat_id = ? AND key = ?", (chat_id, key)
            ).fetchone()
            return self._row_to_working_memory(row)

    def list_working_memory(self, chat_id: str) -> List[WorkingMemoryEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM working_memory WHERE chat_id = ? ORDER BY updated_at DESC", (chat_id,)
            ).fetchall()
            return [self._row_to_working_memory(r) for r in rows]

    def get_working_memory(self, chat_id: str, key: str) -> Optional[WorkingMemoryEntry]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM working_memory WHERE chat_id = ? AND key = ?", (chat_id, key)
            ).fetchone()
            return self._row_to_working_memory(row) if row is not None else None

    def delete_working_memory(self, chat_id: str, key: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM working_memory WHERE chat_id = ? AND key = ?", (chat_id, key))

    # ---- долговременная память (long_term_memory, область видимости — агент) --

    @staticmethod
    def _row_to_long_term_memory(row: sqlite3.Row) -> LongTermMemoryEntry:
        return LongTermMemoryEntry(
            id=row["id"], agent_id=row["agent_id"], category=row["category"], key=row["key"],
            value=row["value"], source=row["source"] or "manual",
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    def upsert_long_term_memory(self, agent_id: str, category: str, key: str, value: str, source: str = "manual") -> LongTermMemoryEntry:
        now = int(time.time())
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO long_term_memory (agent_id, category, key, value, source, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(agent_id, category, key) DO UPDATE SET
                       value = excluded.value, source = excluded.source, updated_at = excluded.updated_at""",
                (agent_id, category, key, value, source, now, now),
            )
            row = conn.execute(
                "SELECT * FROM long_term_memory WHERE agent_id = ? AND category = ? AND key = ?",
                (agent_id, category, key),
            ).fetchone()
            return self._row_to_long_term_memory(row)

    def list_long_term_memory(self, agent_id: str, category: Optional[str] = None) -> List[LongTermMemoryEntry]:
        with self._connect() as conn:
            if category is not None:
                rows = conn.execute(
                    "SELECT * FROM long_term_memory WHERE agent_id = ? AND category = ? ORDER BY updated_at DESC",
                    (agent_id, category),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM long_term_memory WHERE agent_id = ? ORDER BY updated_at DESC", (agent_id,)
                ).fetchall()
            return [self._row_to_long_term_memory(r) for r in rows]

    def delete_long_term_memory(self, agent_id: str, category: str, key: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM long_term_memory WHERE agent_id = ? AND category = ? AND key = ?",
                (agent_id, category, key),
            )

    # ---- профили-пайплайны (profiles, общий справочник для ВСЕХ агентов) ---

    @staticmethod
    def _row_to_profile(row: sqlite3.Row) -> Profile:
        return Profile(
            id=row["id"], name=row["name"], style=row["style"],
            format=row["format"], constraints=row["constraints"], skills_json=row["skills_json"] or "",
            orchestration_prompt=row["orchestration_prompt"], is_default=bool(row["is_default"]),
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    def create_profile(
        self, name: str, style: Optional[str], format: Optional[str],
        constraints: Optional[str], skills_json: str, orchestration_prompt: Optional[str], is_default: bool = False,
    ) -> Profile:
        profile_id = str(uuid.uuid4())
        now = int(time.time())
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO profiles
                   (id, name, style, format, constraints, skills_json, orchestration_prompt,
                    is_default, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (profile_id, name, style, format, constraints, skills_json,
                 orchestration_prompt, int(is_default), now, now),
            )
        return Profile(
            id=profile_id, name=name, style=style, format=format, constraints=constraints,
            skills_json=skills_json, orchestration_prompt=orchestration_prompt, is_default=is_default,
            created_at=now, updated_at=now,
        )

    def get_profile(self, profile_id: str) -> Optional[Profile]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM profiles WHERE id = ?", (profile_id,)).fetchone()
            return self._row_to_profile(row) if row is not None else None

    def list_profiles(self) -> List[Profile]:
        """Общий справочник — единый список для всех агентов (см. замечание
        пользователя: раньше профили были привязаны к одному агенту, теперь
        любой агент/чат может выбрать любой профиль из общего списка)."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM profiles ORDER BY created_at ASC").fetchall()
            return [self._row_to_profile(r) for r in rows]

    def update_profile(self, profile_id: str, fields: Dict[str, Any]) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{k} = ?" for k in fields.keys())
        values = list(fields.values())
        with self._connect() as conn:
            conn.execute(
                f"UPDATE profiles SET {assignments}, updated_at = ? WHERE id = ?",
                values + [int(time.time()), profile_id],
            )

    def delete_profile(self, profile_id: str) -> None:
        with self._connect() as conn:
            # У чатов, на которых был активен этот профиль, ссылка снимается
            # автоматически (ON DELETE SET NULL на chats.active_profile_id).
            conn.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))

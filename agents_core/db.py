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
    Invariant,
    LongTermMemoryEntry,
    Message,
    ModelInfo,
    Profile,
    Settings,
    Task,
    TaskTransitionLog,
    WorkingMemoryEntry,
)
from .task_state_machine import TaskState as TaskStateEnum

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
    ("task_tracking_enabled", "INTEGER"),
    ("task_manager_max_steps", "INTEGER"),
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

#: См. `Database._seed_task_state_machine_rule_invariants`.
_DEFAULT_STATE_MACHINE_RULE_INVARIANT_TITLES = [
    "Работай только в рамках current step",
    "Не перепрыгивай этапы",
    "Если step завершён — верни next step",
    "Нельзя делать реализацию до утверждённого плана",
    "Нельзя делать финал без валидации",
]

_MESSAGE_EXTRA_COLUMNS = [
    ("format", "TEXT NOT NULL DEFAULT 'text'"),
    ("branch", "INTEGER NOT NULL DEFAULT 0"),
    ("facts", "TEXT"),
    ("task_events", "TEXT"),
    ("is_task_manager_step", "INTEGER NOT NULL DEFAULT 0"),
    ("mcp_events", "TEXT"),
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
        "task_tracking_enabled": int(settings.task_tracking_enabled),
        "task_manager_max_steps": settings.task_manager_max_steps,
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
        # Новая фича — как и остальные новые типы памяти выше, по умолчанию
        # ВЫКЛЮЧЕНА для строк, созданных до появления этой колонки (не было
        # смысла включать задним числом то, чего раньше не существовало).
        task_tracking_enabled=bool(row["task_tracking_enabled"]) if "task_tracking_enabled" in keys and row["task_tracking_enabled"] is not None else False,
        # "Менеджер задач" — лимит шагов; для строк, созданных до появления
        # этой колонки, используется встроенный дефолт (20), а не 0/выключено.
        task_manager_max_steps=(
            row["task_manager_max_steps"]
            if "task_manager_max_steps" in keys and row["task_manager_max_steps"] is not None
            else Settings.__dataclass_fields__["task_manager_max_steps"].default
        ),
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
                    facts TEXT,
                    task_events TEXT,
                    is_task_manager_step INTEGER NOT NULL DEFAULT 0,
                    mcp_events TEXT
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

                CREATE TABLE IF NOT EXISTS invariants (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    rule_text TEXT NOT NULL,
                    kind TEXT,
                    is_active INTEGER NOT NULL DEFAULT 1,
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

                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'planning',
                    paused INTEGER NOT NULL DEFAULT 0,
                    plan TEXT NOT NULL DEFAULT '[]',
                    done_steps TEXT NOT NULL DEFAULT '[]',
                    current_step TEXT,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS task_transition_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                    from_state TEXT NOT NULL,
                    to_state TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'advance',
                    applied_by TEXT NOT NULL DEFAULT 'agent',
                    note TEXT,
                    created_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS task_machine_settings (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    invariant_ids TEXT NOT NULL DEFAULT '[]'
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_chat_id ON tasks(chat_id);
                CREATE INDEX IF NOT EXISTS idx_task_transition_log_task_id ON task_transition_log(task_id);
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
            # "День 14" — набор id инвариантов, выбранных для агента/чата из
            # общего справочника (см. `Invariant`), хранится как JSON-массив
            # строк в одной колонке — по аналогии с `stop_sequences` в
            # Settings, а не отдельной таблицей связей (множественный выбор
            # без порядка и без дополнительных атрибутов самой связи не
            # оправдывает отдельную M2M-таблицу).
            self._ensure_column(conn, "agents", "invariant_ids", "TEXT")
            self._ensure_column(conn, "chats", "invariant_ids", "TEXT")
            self._migrate_profiles_to_global_catalog(conn)
            self._migrate_task_catalog_to_code(conn)
            self._seed_task_state_machine_rule_invariants(conn)

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

    def _migrate_task_catalog_to_code(self, conn: sqlite3.Connection) -> None:
        """"Работу с задачами требуется переделать" (новое ТЗ) — раньше
        состояния/действия/машины состояний были справочниками в БД
        (task_states/task_actions/task_state_machines/task_machine_states/
        task_transitions), а `tasks` ссылались на них (`machine_id`,
        `current_state_id`). Теперь машина состояний в коде (см.
        `task_state_machine.py`), а `tasks` хранят состояние строкой и явные
        `plan`/`done_steps` (см. `models.Task`). Если обнаружена база,
        развёрнутая ДО этого изменения, — переносим существующие задачи и
        историю переходов на новую схему по лучшим усилиям (состояние — по
        `system_name` старого справочника, с откатом на "planning", если имя
        не совпадает ни с одним новым состоянием; статус паузы — по kind
        последнего перехода из истории), а затем удаляем старые справочники
        целиком. Явного действия "Отклонить" в новой модели нет — задачи,
        ранее отклонённые (kind=DECLINE), переносятся с тем состоянием, в
        которое их перевело это действие, с пометкой об этом в примечании
        последнего перехода. Идемпотентно: если старых справочников уже нет
        (свежая база или уже мигрировано), ничего не делает."""
        tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "task_states" not in tables and "task_actions" not in tables and "task_state_machines" not in tables:
            return
        conn.execute("PRAGMA foreign_keys = OFF")
        try:
            tasks_cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()} if "tasks" in tables else set()
            if "machine_id" in tasks_cols:
                state_names = {
                    r["id"]: r["system_name"] for r in conn.execute("SELECT id, system_name FROM task_states").fetchall()
                } if "task_states" in tables else {}
                action_kinds = {
                    r["id"]: (r["kind"] or "NORMAL") for r in conn.execute("SELECT id, kind FROM task_actions").fetchall()
                } if "task_actions" in tables else {}
                valid_states = {s.value for s in TaskStateEnum}

                def _resolve_state(state_id: Optional[str]) -> str:
                    name = state_names.get(state_id) if state_id else None
                    return name if name in valid_states else TaskStateEnum.PLANNING.value

                old_tasks = conn.execute(
                    "SELECT id, chat_id, title, current_state_id, current_step, created_at, updated_at FROM tasks"
                ).fetchall()
                logs_by_task: Dict[str, List[sqlite3.Row]] = {}
                if "task_transition_log" in tables:
                    for row in conn.execute(
                        "SELECT task_id, action_id, from_state_id, to_state_id, applied_by, note, created_at "
                        "FROM task_transition_log ORDER BY task_id, created_at ASC, id ASC"
                    ).fetchall():
                        logs_by_task.setdefault(row["task_id"], []).append(row)

                conn.executescript(
                    """
                    CREATE TABLE tasks__migrating_new (
                        id TEXT PRIMARY KEY,
                        chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                        title TEXT NOT NULL,
                        state TEXT NOT NULL DEFAULT 'planning',
                        paused INTEGER NOT NULL DEFAULT 0,
                        plan TEXT NOT NULL DEFAULT '[]',
                        done_steps TEXT NOT NULL DEFAULT '[]',
                        current_step TEXT,
                        created_at INTEGER NOT NULL,
                        updated_at INTEGER NOT NULL
                    );
                    CREATE TABLE task_transition_log__migrating_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                        from_state TEXT NOT NULL,
                        to_state TEXT NOT NULL,
                        kind TEXT NOT NULL DEFAULT 'advance',
                        applied_by TEXT NOT NULL DEFAULT 'agent',
                        note TEXT,
                        created_at INTEGER NOT NULL
                    );
                    """
                )
                kind_map = {"NORMAL": "advance", "PAUSE": "pause", "RESUME": "advance", "DECLINE": "advance"}
                for old in old_tasks:
                    logs = logs_by_task.get(old["id"], [])
                    paused = bool(logs) and action_kinds.get(logs[-1]["action_id"], "NORMAL") == "PAUSE"
                    conn.execute(
                        "INSERT INTO tasks__migrating_new "
                        "(id, chat_id, title, state, paused, plan, done_steps, current_step, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, '[]', '[]', ?, ?, ?)",
                        (old["id"], old["chat_id"], old["title"], _resolve_state(old["current_state_id"]),
                         int(paused), old["current_step"], old["created_at"], old["updated_at"]),
                    )
                    for log in logs:
                        kind = action_kinds.get(log["action_id"], "NORMAL")
                        note = log["note"]
                        if kind == "DECLINE":
                            note = ((note + " ") if note else "") + (
                                "(перенесено при переходе на новую модель задач: "
                                "раньше здесь было действие «Отклонить», которого больше нет)"
                            )
                        conn.execute(
                            "INSERT INTO task_transition_log__migrating_new "
                            "(task_id, from_state, to_state, kind, applied_by, note, created_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (old["id"], _resolve_state(log["from_state_id"]), _resolve_state(log["to_state_id"]),
                             kind_map.get(kind, "advance"), log["applied_by"], note, log["created_at"]),
                        )
                conn.executescript(
                    """
                    DROP TABLE IF EXISTS task_transition_log;
                    DROP TABLE tasks;
                    ALTER TABLE tasks__migrating_new RENAME TO tasks;
                    ALTER TABLE task_transition_log__migrating_new RENAME TO task_transition_log;
                    CREATE INDEX IF NOT EXISTS idx_tasks_chat_id ON tasks(chat_id);
                    CREATE INDEX IF NOT EXISTS idx_task_transition_log_task_id ON task_transition_log(task_id);
                    """
                )
            # Старые общие справочники (состояния/действия/машины/их состав)
            # больше не нужны ни в каком виде — машина состояний теперь в коде.
            conn.executescript(
                """
                DROP TABLE IF EXISTS task_machine_states;
                DROP TABLE IF EXISTS task_transitions;
                DROP TABLE IF EXISTS task_state_machines;
                DROP TABLE IF EXISTS task_actions;
                DROP TABLE IF EXISTS task_states;
                """
            )
        finally:
            conn.execute("PRAGMA foreign_keys = ON")

    def _seed_task_state_machine_rule_invariants(self, conn: sqlite3.Connection) -> None:
        """"Работу с задачами требуется переделать" (новое ТЗ, п.5) — пять
        дефолтных инвариантов категории "Правило стейт-машины", сидируются
        один раз каждый (идемпотентно — проверка по точному совпадению
        `title` среди уже существующих инвариантов этой категории).
        Пользователь может отредактировать или удалить любой из них как
        обычный инвариант — повторный запуск не восстанавливает удалённые."""
        existing_titles = {
            row["title"] for row in conn.execute(
                "SELECT title FROM invariants WHERE kind = ?", ("state_machine_rule",)
            ).fetchall()
        }
        now = int(time.time())
        for title in _DEFAULT_STATE_MACHINE_RULE_INVARIANT_TITLES:
            if title in existing_titles:
                continue
            conn.execute(
                "INSERT INTO invariants (id, title, rule_text, kind, is_active, created_at, updated_at) "
                "VALUES (?, ?, ?, 'state_machine_rule', 1, ?, ?)",
                (str(uuid.uuid4()), title, title, now, now),
            )

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

    @staticmethod
    def _row_invariant_ids(row: sqlite3.Row) -> List[str]:
        # Обратная совместимость: строки, созданные до появления этой
        # колонки (или до первого выбора инвариантов), не имеют значения —
        # "инварианты не выбраны", а не ошибка.
        if "invariant_ids" not in row.keys() or not row["invariant_ids"]:
            return []
        return json.loads(row["invariant_ids"])

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
            if row is None:
                return None
            return Agent(
                id=row["id"], name=row["name"], created_at=row["created_at"],
                updated_at=row["updated_at"], settings=_row_to_settings(row),
                default_profile_id=self._agent_default_profile_id(row),
                invariant_ids=self._row_invariant_ids(row),
            )

    def list_agents(self) -> List[Agent]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM agents ORDER BY updated_at DESC").fetchall()
            return [
                Agent(id=r["id"], name=r["name"], created_at=r["created_at"],
                      updated_at=r["updated_at"], settings=_row_to_settings(r),
                      default_profile_id=self._agent_default_profile_id(r),
                      invariant_ids=self._row_invariant_ids(r))
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

    def set_agent_invariants(self, agent_id: str, invariant_ids: List[str]) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE agents SET invariant_ids = ?, updated_at = ? WHERE id = ?",
                (json.dumps(invariant_ids, ensure_ascii=False), int(time.time()), agent_id),
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
                invariant_ids=self._row_invariant_ids(row),
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
                     active_profile_id=r["active_profile_id"] if "active_profile_id" in r.keys() else None,
                     invariant_ids=self._row_invariant_ids(r))
                for r in rows
            ]

    def set_chat_active_profile(self, chat_id: str, profile_id: Optional[str]) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE chats SET active_profile_id = ?, updated_at = ? WHERE id = ?",
                (profile_id, int(time.time()), chat_id),
            )

    def set_chat_invariants(self, chat_id: str, invariant_ids: List[str]) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE chats SET invariant_ids = ?, updated_at = ? WHERE id = ?",
                (json.dumps(invariant_ids, ensure_ascii=False), int(time.time()), chat_id),
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
                    format, branch, facts, task_events, is_task_manager_step, mcp_events)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    message.chat_id, message.role, message.content, message.created_at,
                    message.reasoning_content, int(message.is_summary),
                    message.duration_ms, message.total_tokens,
                    message.prompt_tokens, message.completion_tokens,
                    message.format, message.branch, message.facts, message.task_events,
                    int(message.is_task_manager_step), message.mcp_events,
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
            task_events=row["task_events"] if "task_events" in keys else None,
            is_task_manager_step=bool(row["is_task_manager_step"]) if "is_task_manager_step" in keys and row["is_task_manager_step"] is not None else False,
            mcp_events=row["mcp_events"] if "mcp_events" in keys else None,
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

    # ---- инварианты ("День 14", общий справочник для ВСЕХ агентов) ---------

    @staticmethod
    def _row_to_invariant(row: sqlite3.Row) -> Invariant:
        return Invariant(
            id=row["id"], title=row["title"], rule_text=row["rule_text"], kind=row["kind"],
            is_active=bool(row["is_active"]), created_at=row["created_at"], updated_at=row["updated_at"],
        )

    def create_invariant(
        self, title: str, rule_text: str, kind: Optional[str] = None, is_active: bool = True,
    ) -> Invariant:
        invariant_id = str(uuid.uuid4())
        now = int(time.time())
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO invariants (id, title, rule_text, kind, is_active, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (invariant_id, title, rule_text, kind, int(is_active), now, now),
            )
        return Invariant(
            id=invariant_id, title=title, rule_text=rule_text, kind=kind, is_active=is_active,
            created_at=now, updated_at=now,
        )

    def get_invariant(self, invariant_id: str) -> Optional[Invariant]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM invariants WHERE id = ?", (invariant_id,)).fetchone()
            return self._row_to_invariant(row) if row is not None else None

    def list_invariants(self) -> List[Invariant]:
        """Общий справочник — единый список для всех агентов/чатов (та же
        идея, что и `list_profiles`: инвариант описывается один раз и
        подключается множественным выбором в настройках любого агента/чата)."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM invariants ORDER BY created_at ASC").fetchall()
            return [self._row_to_invariant(r) for r in rows]

    def update_invariant(self, invariant_id: str, fields: Dict[str, Any]) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{k} = ?" for k in fields.keys())
        values = list(fields.values())
        with self._connect() as conn:
            conn.execute(
                f"UPDATE invariants SET {assignments}, updated_at = ? WHERE id = ?",
                values + [int(time.time()), invariant_id],
            )

    def delete_invariant(self, invariant_id: str) -> None:
        """В отличие от `active_profile_id` (одиночная FK-ссылка с ON DELETE
        SET NULL), выбор инвариантов хранится JSON-массивом в обычной TEXT-
        колонке — SQLite не может снять ссылку автоматически, поэтому здесь
        же вычищаем удалённый id из invariant_ids ВСЕХ агентов и чатов, где
        он был выбран, а также из настройки машины состояний задач
        (`task_machine_settings`, см. `set_task_machine_invariant_ids`)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM invariants WHERE id = ?", (invariant_id,))
            now = int(time.time())
            for table in ("agents", "chats"):
                rows = conn.execute(
                    f"SELECT id, invariant_ids FROM {table} WHERE invariant_ids IS NOT NULL"
                ).fetchall()
                for row in rows:
                    try:
                        ids = json.loads(row["invariant_ids"]) if row["invariant_ids"] else []
                    except (ValueError, TypeError):
                        continue
                    if invariant_id in ids:
                        ids = [i for i in ids if i != invariant_id]
                        conn.execute(
                            f"UPDATE {table} SET invariant_ids = ?, updated_at = ? WHERE id = ?",
                            (json.dumps(ids, ensure_ascii=False), now, row["id"]),
                        )
            machine_row = conn.execute("SELECT invariant_ids FROM task_machine_settings WHERE id = 1").fetchone()
            if machine_row is not None and machine_row["invariant_ids"]:
                try:
                    ids = json.loads(machine_row["invariant_ids"])
                except (ValueError, TypeError):
                    ids = []
                if invariant_id in ids:
                    ids = [i for i in ids if i != invariant_id]
                    conn.execute(
                        "UPDATE task_machine_settings SET invariant_ids = ? WHERE id = 1",
                        (json.dumps(ids, ensure_ascii=False),),
                    )

    # ---- задачи (tasks, область видимости — чат) -----------------------------
    # Машина состояний — в коде (`task_state_machine.py`), см. модуль-докстринг
    # блока dataclass'ов `Task`/`TaskTransitionLog` в `models.py`.

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> Task:
        return Task(
            id=row["id"], chat_id=row["chat_id"], title=row["title"], state=row["state"],
            paused=bool(row["paused"]),
            plan=json.loads(row["plan"]) if row["plan"] else [],
            done_steps=json.loads(row["done_steps"]) if row["done_steps"] else [],
            current_step=row["current_step"],
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    def create_task(self, chat_id: str, title: str, plan: Optional[List[str]] = None) -> Task:
        task_id = str(uuid.uuid4())
        now = int(time.time())
        plan = plan or []
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO tasks (id, chat_id, title, state, paused, plan, done_steps, current_step, created_at, updated_at) "
                "VALUES (?, ?, ?, 'planning', 0, ?, '[]', NULL, ?, ?)",
                (task_id, chat_id, title, json.dumps(plan, ensure_ascii=False), now, now),
            )
        return Task(id=task_id, chat_id=chat_id, title=title, state="planning", plan=plan, created_at=now, updated_at=now)

    def get_task(self, task_id: str) -> Optional[Task]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
            return self._row_to_task(row) if row is not None else None

    def list_tasks_for_chat(self, chat_id: str) -> List[Task]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE chat_id = ? ORDER BY created_at ASC", (chat_id,)
            ).fetchall()
            return [self._row_to_task(r) for r in rows]

    def list_tasks_for_agent(self, agent_id: str) -> List[Task]:
        """Задачи по ВСЕМ чатам агента сразу — агрегированный блок "Задачи" на
        карточке агента (см. итоговую концепцию, п.1 замечаний пользователя)."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT tasks.* FROM tasks
                   JOIN chats ON chats.id = tasks.chat_id
                   WHERE chats.agent_id = ?
                   ORDER BY tasks.created_at ASC""",
                (agent_id,),
            ).fetchall()
            return [self._row_to_task(r) for r in rows]

    def update_task_progress(
        self, task_id: str, state: Optional[str] = None, current_step: Optional[str] = None,
        plan: Optional[List[str]] = None, done_steps: Optional[List[str]] = None,
        _current_step_set: bool = True,
    ) -> None:
        """Частичное обновление прогресса задачи — любой из параметров,
        оставленный `None` (кроме `_current_step_set=False`, см. ниже),
        просто не меняется. `current_step` обрабатывается отдельным флагом
        [_current_step_set], а не проверкой на `None`, — модель имеет право
        явно ОЧИСТИТЬ текущий шаг (`current_step=None`), а не только задать
        новый."""
        assignments: List[str] = []
        values: List[Any] = []
        if state is not None:
            assignments.append("state = ?")
            values.append(state)
        if _current_step_set:
            assignments.append("current_step = ?")
            values.append(current_step)
        if plan is not None:
            assignments.append("plan = ?")
            values.append(json.dumps(plan, ensure_ascii=False))
        if done_steps is not None:
            assignments.append("done_steps = ?")
            values.append(json.dumps(done_steps, ensure_ascii=False))
        if not assignments:
            return
        assignments.append("updated_at = ?")
        values.append(int(time.time()))
        values.append(task_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE tasks SET {', '.join(assignments)} WHERE id = ?", values)

    def set_task_paused(self, task_id: str, paused: bool) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE tasks SET paused = ?, updated_at = ? WHERE id = ?",
                (int(paused), int(time.time()), task_id),
            )

    def delete_task(self, task_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))

    # ---- журнал переходов задачи (task_transition_log) -----------------------

    @staticmethod
    def _row_to_task_transition_log(row: sqlite3.Row) -> TaskTransitionLog:
        return TaskTransitionLog(
            id=row["id"], task_id=row["task_id"], from_state=row["from_state"],
            to_state=row["to_state"], kind=row["kind"] or "advance",
            applied_by=row["applied_by"] or "agent", note=row["note"], created_at=row["created_at"],
        )

    def add_task_transition_log(
        self, task_id: str, from_state: str, to_state: str, kind: str,
        applied_by: str, note: Optional[str],
    ) -> TaskTransitionLog:
        now = int(time.time())
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO task_transition_log (task_id, from_state, to_state, kind, applied_by, note, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (task_id, from_state, to_state, kind, applied_by, note, now),
            )
            log_id = cur.lastrowid
        return TaskTransitionLog(
            id=log_id, task_id=task_id, from_state=from_state, to_state=to_state,
            kind=kind, applied_by=applied_by, note=note, created_at=now,
        )

    def list_task_transition_log(self, task_id: str) -> List[TaskTransitionLog]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM task_transition_log WHERE task_id = ? ORDER BY created_at ASC, id ASC", (task_id,)
            ).fetchall()
            return [self._row_to_task_transition_log(r) for r in rows]

    # ---- настройка машины состояний задач (task_machine_settings) ------------
    # "Работу с задачами требуется переделать" (новое ТЗ, п.4) — список
    # инвариантов категории "Правило стейт-машины" (`state_machine_rule`),
    # привязанных к (единственной, заданной в коде) машине состояний. Ровно
    # одна строка на всю систему — по аналогии с `default_settings`, но
    # отдельной таблицей, чтобы не путать с настройками нового агента.

    def get_task_machine_invariant_ids(self) -> List[str]:
        with self._connect() as conn:
            row = conn.execute("SELECT invariant_ids FROM task_machine_settings WHERE id = 1").fetchone()
            if row is None or not row["invariant_ids"]:
                return []
            return json.loads(row["invariant_ids"])

    def set_task_machine_invariant_ids(self, invariant_ids: List[str]) -> None:
        payload = json.dumps(invariant_ids, ensure_ascii=False)
        with self._connect() as conn:
            existing = conn.execute("SELECT id FROM task_machine_settings WHERE id = 1").fetchone()
            if existing is None:
                conn.execute("INSERT INTO task_machine_settings (id, invariant_ids) VALUES (1, ?)", (payload,))
            else:
                conn.execute("UPDATE task_machine_settings SET invariant_ids = ? WHERE id = 1", (payload,))

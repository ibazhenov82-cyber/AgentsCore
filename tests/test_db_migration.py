"""
Проверяет защитную миграцию `Database._ensure_column`/`_init_schema`:
если сервис запускают поверх файла БД, созданного предыдущей версией (без
новых колонок вроде `summary_system_prompt`, `context_strategy`,
`context_strategy_limit`, `extraction_system_prompt`), новый код должен:

1) не падать при открытии такой базы;
2) сохранить уже накопленные данные (имя агента, старые настройки);
3) вернуть для новых полей осмысленные значения по умолчанию (а не None
   там, где Settings ожидает строку/число);
4) позволить обновить и надёжно сохранить именно эти новые поля — так,
   чтобы значение пережило "перезапуск" (открытие того же файла новым
   объектом Database/Repository).

Это отдельный тест-файл, а не часть test_repository.py, потому что тут
намеренно создаётся файл БД вручную, через голый sqlite3, а не через
Database() — чтобы смоделировать по-настоящему "старую" схему.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
import uuid

from agents_core.db import Database
from agents_core.repository import Repository
from agents_core.models import ModelInfo

from tests.test_repository import FakeCatalog, FakeProvider, FakeRegistry, TEST_MODEL_ID


# Заведомо старая схема — только то подмножество колонок settings, которое
# существовало ДО появления полей суммаризации/стратегий контекста.
_OLD_SETTINGS_COLUMNS = [
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
    ("stop_sequences", "TEXT"),
    ("tool_choice", "TEXT"),
    ("tools_json", "TEXT"),
    ("summary_prompt", "TEXT"),
    ("autosummary", "TEXT"),
    ("autosummary_by_messages", "INTEGER"),
    ("autosummary_by_tokens", "INTEGER"),
    # Намеренно ОТСУТСТВУЮТ: summary_system_prompt, context_strategy,
    # context_strategy_limit, extraction_system_prompt,
    # include_usage_in_stream, logprobs, frequency_penalty, presence_penalty.
]

_OLD_DEFAULT_SETTINGS_COLUMNS = [
    ("model", "TEXT"),
    ("system_prompt", "TEXT"),
    ("temperature", "REAL"),
    ("top_p", "REAL"),
    ("seed", "INTEGER"),
    ("stream", "INTEGER"),
    ("thinking_enabled", "INTEGER"),
    ("reasoning_effort", "TEXT"),
    ("summary_prompt", "TEXT"),
    # Намеренно ОТСУТСТВУЕТ: summary_system_prompt, include_usage_in_stream.
]


def _create_old_schema_db(path: str, agent_id: str) -> None:
    settings_cols_sql = ",\n".join(f"{name} {sql_type}" for name, sql_type in _OLD_SETTINGS_COLUMNS)
    default_cols_sql = ",\n".join(f"{name} {sql_type}" for name, sql_type in _OLD_DEFAULT_SETTINGS_COLUMNS)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            f"""
            CREATE TABLE default_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                {default_cols_sql}
            );

            CREATE TABLE agents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                {settings_cols_sql}
            );

            CREATE TABLE chats (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                {settings_cols_sql}
            );

            CREATE TABLE messages (
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
                completion_tokens INTEGER
            );

            CREATE TABLE chat_branches (
                chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                number INTEGER NOT NULL,
                name TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                PRIMARY KEY (chat_id, number)
            );
            """
        )
        now = 1_700_000_000
        conn.execute(
            "INSERT INTO default_settings (id, model, system_prompt, temperature, top_p, seed, "
            "stream, thinking_enabled, reasoning_effort, summary_prompt) VALUES "
            "(1, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                TEST_MODEL_ID, "старый системный промпт по умолчанию", 0.7, 1.0, None,
                1, 0, None, "старый шаблон суммаризации по умолчанию",
            ),
        )
        conn.execute(
            "INSERT INTO agents (id, name, created_at, updated_at, model, system_prompt, "
            "temperature, top_p, seed, stream, thinking_enabled, reasoning_effort, max_tokens, "
            "json_mode, stop_sequences, tool_choice, tools_json, summary_prompt, autosummary, "
            "autosummary_by_messages, autosummary_by_tokens) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                agent_id, "Старый агент", now, now, TEST_MODEL_ID, "старый системный промпт",
                0.7, 1.0, None, 1, 0, None, None,
                0, "[]", None, "", "старый шаблон суммаризации", "off",
                None, None,
            ),
        )
        conn.commit()
    finally:
        conn.close()


class DatabaseMigrationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(db_fd)
        os.unlink(self.db_path)
        self.agent_id = str(uuid.uuid4())
        _create_old_schema_db(self.db_path, self.agent_id)

    def tearDown(self) -> None:
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def _make_repo(self) -> Repository:
        db = Database(self.db_path)  # запускает _init_schema() -> защитную миграцию
        provider = FakeProvider()
        registry = FakeRegistry(provider)
        model = ModelInfo(
            id=TEST_MODEL_ID, provider="fake", model_id="test-model", display_name="Test model",
            is_local=True, context_window=1200, max_input_tokens=1000,
            max_output_tokens=256, supports_thinking=True, supports_tools=True,
            supports_json_mode=True, supports_logprobs=False,
        )
        catalog = FakeCatalog({TEST_MODEL_ID: model})
        return Repository(db, registry, catalog)

    def test_opening_old_schema_db_does_not_crash_and_adds_columns(self):
        # Сам факт успешного создания Database() на старом файле — уже
        # проверка того, что _ensure_column не падает на ALTER TABLE.
        repo = self._make_repo()
        agent = repo.get_agent(self.agent_id)
        self.assertEqual(agent.name, "Старый агент")
        self.assertEqual(agent.settings.system_prompt, "старый системный промпт")
        self.assertEqual(agent.settings.summary_prompt, "старый шаблон суммаризации")

    def test_new_fields_get_code_defaults_not_none_after_migration(self):
        repo = self._make_repo()
        agent = repo.get_agent(self.agent_id)
        # summary_system_prompt/extraction_system_prompt должны получить
        # непустые дефолты из кода (Settings.__dataclass_fields__), а не
        # None/пустую строку — иначе именно это и объясняет жалобу
        # пользователя "поле не предзаполнено".
        self.assertTrue(agent.settings.summary_system_prompt)
        self.assertGreater(len(agent.settings.summary_system_prompt), 100)
        self.assertTrue(agent.settings.extraction_system_prompt)
        # context_strategy/context_strategy_limit просто отсутствовали —
        # ожидаем None (осмысленный "не задано"), а не падение при чтении.
        self.assertIsNone(agent.settings.context_strategy)
        self.assertIsNone(agent.settings.context_strategy_limit)

    def test_updating_new_field_persists_across_reopen(self):
        repo = self._make_repo()
        repo.update_agent_settings(self.agent_id, {
            "summary_system_prompt": "МОЙ КАСТОМНЫЙ ПРОМПТ СУММАРИЗАЦИИ",
            "context_strategy": "sliding_window",
            "context_strategy_limit": 7,
        })

        # "Перезапуск сервера": открываем тот же файл заново, новым
        # объектом Database/Repository, как это происходит при реальном
        # рестарте процесса.
        repo2 = self._make_repo()
        reloaded = repo2.get_agent(self.agent_id)
        self.assertEqual(reloaded.settings.summary_system_prompt, "МОЙ КАСТОМНЫЙ ПРОМПТ СУММАРИЗАЦИИ")
        self.assertEqual(reloaded.settings.context_strategy, "sliding_window")
        self.assertEqual(reloaded.settings.context_strategy_limit, 7)

    def test_new_memory_type_flags_default_to_backward_compatible_values(self):
        # working_memory_enabled/long_term_memory_enabled должны стать True
        # после миграции СТАРОЙ базы (эти слои у существующих агентов и так
        # были всегда включены — миграция не должна их внезапно "выключить"
        # из-за NULL после ALTER TABLE), а новые расширенные типы — False,
        # как и для только что созданных агентов.
        repo = self._make_repo()
        agent = repo.get_agent(self.agent_id)
        self.assertTrue(agent.settings.working_memory_enabled)
        self.assertTrue(agent.settings.long_term_memory_enabled)
        self.assertFalse(agent.settings.episodic_memory_enabled)
        self.assertFalse(agent.settings.semantic_memory_enabled)
        self.assertFalse(agent.settings.procedural_memory_enabled)

    def test_default_settings_migration_prefills_summary_system_prompt(self):
        # Строка default_settings уже существовала (старая схема, без
        # summary_system_prompt) — после миграции колонка должна
        # появиться и вернуть непустой дефолт, а не None/ошибку.
        repo = self._make_repo()
        defaults = repo.get_default_settings()
        self.assertEqual(defaults.summary_prompt, "старый шаблон суммаризации по умолчанию")
        self.assertTrue(defaults.summary_system_prompt)
        self.assertGreater(len(defaults.summary_system_prompt), 100)

    def test_updating_default_settings_new_field_persists_across_reopen(self):
        repo = self._make_repo()
        repo.update_default_settings({"summary_system_prompt": "ДЕФОЛТНЫЙ КАСТОМНЫЙ ПРОМПТ"})
        repo2 = self._make_repo()
        reloaded = repo2.get_default_settings()
        self.assertEqual(reloaded.summary_system_prompt, "ДЕФОЛТНЫЙ КАСТОМНЫЙ ПРОМПТ")


def _create_old_schema_db_with_agent_scoped_profile(path: str, agent_id: str, other_agent_id: str, chat_id: str, profile_id: str) -> None:
    """Строит файл БД по схеме ДО перехода профилей на общий справочник:
    `profiles.agent_id NOT NULL REFERENCES agents(id) ON DELETE CASCADE` и
    `chats.active_profile_id`, уже указывающий на такой профиль — именно
    такую базу должна уметь мигрировать `Database._migrate_profiles_to_global_catalog`
    без потери данных и, что важнее всего, без поломки FK на chats (см.
    комментарий в db.py про подводный камень с ALTER TABLE RENAME)."""
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        settings_cols_sql = ",\n".join(f"{name} {sql_type}" for name, sql_type in _OLD_SETTINGS_COLUMNS)
        default_cols_sql = ",\n".join(f"{name} {sql_type}" for name, sql_type in _OLD_DEFAULT_SETTINGS_COLUMNS)
        conn.executescript(
            f"""
            CREATE TABLE default_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                {default_cols_sql}
            );

            CREATE TABLE agents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                {settings_cols_sql}
            );

            CREATE TABLE profiles (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
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
            CREATE INDEX idx_profiles_agent_id ON profiles(agent_id);

            CREATE TABLE chats (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                active_profile_id TEXT REFERENCES profiles(id) ON DELETE SET NULL,
                {settings_cols_sql}
            );

            CREATE TABLE messages (
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
                completion_tokens INTEGER
            );

            CREATE TABLE chat_branches (
                chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                number INTEGER NOT NULL,
                name TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                PRIMARY KEY (chat_id, number)
            );
            """
        )
        now = 1_700_000_000
        conn.execute(
            "INSERT INTO default_settings (id, model, system_prompt, temperature, top_p, seed, "
            "stream, thinking_enabled, reasoning_effort, summary_prompt) VALUES "
            "(1, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (TEST_MODEL_ID, "старый системный промпт по умолчанию", 0.7, 1.0, None, 1, 0, None, "старый шаблон"),
        )
        for aid, aname in ((agent_id, "Агент 1"), (other_agent_id, "Агент 2")):
            conn.execute(
                "INSERT INTO agents (id, name, created_at, updated_at, model, system_prompt, "
                "temperature, top_p, seed, stream, thinking_enabled, reasoning_effort, max_tokens, "
                "json_mode, stop_sequences, tool_choice, tools_json, summary_prompt, autosummary, "
                "autosummary_by_messages, autosummary_by_tokens) VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (aid, aname, now, now, TEST_MODEL_ID, "системный промпт", 0.7, 1.0, None, 1, 0, None, None,
                 0, "[]", None, "", "шаблон суммаризации", "off", None, None),
            )
        conn.execute(
            "INSERT INTO profiles (id, agent_id, name, style, format, constraints, skills_json, "
            "orchestration_prompt, is_default, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (profile_id, agent_id, "Старый профиль", "дружелюбно", None, None, "[]", None, 0, now, now),
        )
        conn.execute(
            "INSERT INTO chats (id, agent_id, title, created_at, updated_at, active_profile_id, model, "
            "system_prompt, temperature, top_p, seed, stream, thinking_enabled, reasoning_effort, max_tokens, "
            "json_mode, stop_sequences, tool_choice, tools_json, summary_prompt, autosummary, "
            "autosummary_by_messages, autosummary_by_tokens) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (chat_id, agent_id, "Чат", now, now, profile_id, TEST_MODEL_ID, "системный промпт", 0.7, 1.0, None,
             1, 0, None, None, 0, "[]", None, "", "шаблон суммаризации", "off", None, None),
        )
        conn.commit()
    finally:
        conn.close()


class ProfilesGlobalCatalogMigrationTestCase(unittest.TestCase):
    """Профили перестали быть привязаны к одному агенту (общий справочник —
    см. замечание пользователя): `profiles.agent_id NOT NULL REFERENCES
    agents(id) ON DELETE CASCADE` убран из схемы. Это первая миграция в
    проекте, которая не просто добавляет колонку, а пересобирает таблицу —
    поэтому отдельно проверяем, что после неё FK `chats.active_profile_id`
    по-прежнему разрешается (см. предупреждение в db.py про RENAME TABLE)."""

    def setUp(self) -> None:
        db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(db_fd)
        os.unlink(self.db_path)
        self.agent_id = str(uuid.uuid4())
        self.other_agent_id = str(uuid.uuid4())
        self.chat_id = str(uuid.uuid4())
        self.profile_id = str(uuid.uuid4())
        _create_old_schema_db_with_agent_scoped_profile(
            self.db_path, self.agent_id, self.other_agent_id, self.chat_id, self.profile_id,
        )

    def tearDown(self) -> None:
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def _make_repo(self) -> Repository:
        db = Database(self.db_path)
        provider = FakeProvider()
        registry = FakeRegistry(provider)
        model = ModelInfo(
            id=TEST_MODEL_ID, provider="fake", model_id="test-model", display_name="Test model",
            is_local=True, context_window=1200, max_input_tokens=1000,
            max_output_tokens=256, supports_thinking=True, supports_tools=True,
            supports_json_mode=True, supports_logprobs=False,
        )
        catalog = FakeCatalog({TEST_MODEL_ID: model})
        return Repository(db, registry, catalog)

    def test_migration_does_not_crash_and_drops_agent_id_column(self):
        Database(self.db_path)  # запускает _migrate_profiles_to_global_catalog()
        conn = sqlite3.connect(self.db_path)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(profiles)").fetchall()}
        conn.close()
        self.assertNotIn("agent_id", cols)
        self.assertIn("name", cols)

    def test_existing_profile_data_preserved(self):
        repo = self._make_repo()
        profile = repo.get_profile(self.profile_id)
        self.assertEqual(profile.name, "Старый профиль")
        self.assertEqual(profile.style, "дружелюбно")

    def test_chat_active_profile_fk_still_resolves_after_migration(self):
        # Критическая проверка: до фикса порядка операций в миграции
        # (см. комментарий в db.py) ALTER TABLE ... RENAME переписывал текст
        # FK в chats на имя временной таблицы, и chat.active_profile_id
        # переставал резолвиться в реальный профиль после пересборки.
        repo = self._make_repo()
        chat = repo.get_chat(self.chat_id)
        self.assertEqual(chat.active_profile_id, self.profile_id)
        profile = repo.get_profile(chat.active_profile_id)
        self.assertEqual(profile.name, "Старый профиль")

    def test_profile_catalog_global_after_migration(self):
        # Профиль, "принадлежавший" одному агенту в старой схеме, теперь
        # подключается и к чату ДРУГОГО агента — подтверждает, что миграция
        # не просто не сломалась, а реально дала общий справочник.
        repo = self._make_repo()
        other_chat = repo.create_chat(self.other_agent_id, "Чат другого агента")
        updated = repo.set_chat_active_profile(other_chat.id, self.profile_id)
        self.assertEqual(updated.active_profile_id, self.profile_id)
        self.assertIn(self.profile_id, {p.id for p in repo.list_profiles()})

    def test_new_profile_created_after_migration_has_no_agent_id_attribute(self):
        repo = self._make_repo()
        profile = repo.create_profile("Новый общий профиль")
        self.assertFalse(hasattr(profile, "agent_id"))
        self.assertIn(profile.id, {p.id for p in repo.list_profiles()})


class TaskCatalogToCodeMigrationTestCase(unittest.TestCase):
    """"Работу с задачами требуется переделать" (новое ТЗ) — раньше
    состояния/действия/машины состояний задач были справочниками в БД
    (task_states/task_actions/task_state_machines/task_machine_states/
    task_transitions), а `tasks` ссылались на них (`machine_id`,
    `current_state_id`). `Database._migrate_task_catalog_to_code` должна
    перенести существующие задачи/историю на новую схему (state — строкой,
    paused — по последнему переходу) без потери данных, а затем удалить
    старые справочники целиком. Строит старую схему поверх СВЕЖЕЙ БД новой
    версии (проще, чем вручную собирать весь старый agents/chats DDL) —
    создаёт агента/чат обычным способом, затем подменяет только
    task-таблицы на старый вид сырым sqlite3."""

    def setUp(self) -> None:
        db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(db_fd)
        os.unlink(self.db_path)

    def tearDown(self) -> None:
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def _make_repo(self) -> Repository:
        db = Database(self.db_path)
        registry = FakeRegistry(FakeProvider())
        model = ModelInfo(
            id=TEST_MODEL_ID, provider="fake", model_id="test-model", display_name="Test model",
            is_local=True, context_window=1200, max_input_tokens=1000, max_output_tokens=256,
            supports_thinking=True, supports_tools=True, supports_json_mode=True, supports_logprobs=False,
        )
        return Repository(db, registry, FakeCatalog({TEST_MODEL_ID: model}))

    def _downgrade_to_old_task_schema(self, agent_id: str, chat_id: str) -> dict:
        """Заменяет новые task-таблицы (только что созданные `Database` по
        актуальной схеме) на старые, с тремя задачами: (1) активная, этап
        "execution" (последний переход — NORMAL, значит НЕ на паузе), (2) с
        состоянием, чьё `system_name` в новой машине не существует (проверка
        отката на "planning"), последний переход — PAUSE, (3) "отклонённая"
        через больше не существующее действие DECLINE — проверка переноса с
        пометкой в примечании. Возвращает id трёх задач."""
        now = 1_700_000_000
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.executescript(
                """
                DROP TABLE IF EXISTS task_transition_log;
                DROP TABLE IF EXISTS tasks;
                DROP TABLE IF EXISTS task_machine_settings;

                CREATE TABLE task_states (
                    id TEXT PRIMARY KEY, system_name TEXT NOT NULL, display_name TEXT NOT NULL,
                    description TEXT, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
                );
                CREATE TABLE task_actions (
                    id TEXT PRIMARY KEY, system_name TEXT NOT NULL, display_name TEXT NOT NULL,
                    kind TEXT NOT NULL, description TEXT, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
                );
                CREATE TABLE task_state_machines (
                    id TEXT PRIMARY KEY, system_name TEXT NOT NULL, display_name TEXT NOT NULL,
                    description TEXT, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
                );
                CREATE TABLE tasks (
                    id TEXT PRIMARY KEY, chat_id TEXT NOT NULL, machine_id TEXT NOT NULL,
                    title TEXT NOT NULL, current_state_id TEXT NOT NULL, current_step TEXT,
                    created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
                );
                CREATE TABLE task_transition_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
                    action_id TEXT NOT NULL, from_state_id TEXT NOT NULL, to_state_id TEXT NOT NULL,
                    applied_by TEXT NOT NULL, note TEXT, created_at INTEGER NOT NULL
                );
                """
            )
            conn.executemany(
                "INSERT INTO task_states (id, system_name, display_name, description, created_at, updated_at) "
                "VALUES (?, ?, ?, NULL, ?, ?)",
                [
                    ("s_planning", "planning", "Планирование", now, now),
                    ("s_execution", "execution", "Выполнение", now, now),
                    ("s_validation", "validation", "Проверка", now, now),
                    ("s_done", "done", "Готово", now, now),
                    ("s_weird", "weird_custom_state", "Кастомное", now, now),
                ],
            )
            conn.executemany(
                "INSERT INTO task_actions (id, system_name, display_name, kind, description, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, NULL, ?, ?)",
                [
                    ("a_normal", "advance", "Продолжить", "NORMAL", now, now),
                    ("a_pause", "pause", "Пауза", "PAUSE", now, now),
                    ("a_decline", "decline", "Отклонить", "DECLINE", now, now),
                ],
            )
            conn.execute(
                "INSERT INTO task_state_machines (id, system_name, display_name, description, created_at, updated_at) "
                "VALUES ('m1', 'default', 'Стандартный процесс', NULL, ?, ?)", (now, now),
            )

            task_active, task_fallback, task_declined = "t_active", "t_fallback", "t_declined"
            conn.executemany(
                "INSERT INTO tasks (id, chat_id, machine_id, title, current_state_id, current_step, "
                "created_at, updated_at) VALUES (?, ?, 'm1', ?, ?, ?, ?, ?)",
                [
                    (task_active, chat_id, "Активная задача", "s_execution", None, now, now),
                    (task_fallback, chat_id, "Задача с кастомным состоянием", "s_weird", "шаг X", now, now),
                    (task_declined, chat_id, "Отклонённая задача", "s_done", None, now, now),
                ],
            )
            conn.executemany(
                "INSERT INTO task_transition_log (task_id, action_id, from_state_id, to_state_id, "
                "applied_by, note, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (task_active, "a_pause", "s_planning", "s_planning", "system", None, now),
                    (task_active, "a_normal", "s_planning", "s_execution", "agent", None, now + 1),
                    (task_fallback, "a_pause", "s_weird", "s_weird", "manual", "ушёл на встречу", now),
                    (task_declined, "a_decline", "s_planning", "s_done", "agent", "не актуально", now),
                ],
            )
            conn.commit()
        finally:
            conn.close()
        return {"active": task_active, "fallback": task_fallback, "declined": task_declined}

    def test_migration_preserves_tasks_and_maps_state_and_pause_flag(self):
        repo = self._make_repo()
        agent = repo.create_agent("Агент", model=TEST_MODEL_ID)
        chat = repo.create_chat(agent.id, "Чат")
        ids = self._downgrade_to_old_task_schema(agent.id, chat.id)

        # Переоткрытие БД запускает `_migrate_task_catalog_to_code`.
        repo2 = self._make_repo()

        active = repo2.get_task(ids["active"])
        self.assertEqual(active["task"].state, "execution")
        self.assertEqual(active["status"], "active")  # последний переход NORMAL -> не на паузе

        fallback = repo2.get_task(ids["fallback"])
        # "weird_custom_state" не входит в новую машину состояний -> откат на "planning".
        self.assertEqual(fallback["task"].state, "planning")
        self.assertEqual(fallback["status"], "paused")  # последний переход PAUSE
        self.assertEqual(fallback["task"].current_step, "шаг X")

        declined = repo2.get_task(ids["declined"])
        self.assertEqual(declined["task"].state, "done")
        self.assertEqual(declined["status"], "done")
        history = declined["history"]
        self.assertEqual(history[-1]["kind"], "advance")  # DECLINE -> advance (см. kind_map)
        self.assertIn("Отклонить", history[-1]["note"])  # пометка о перенесённом действии сохранена
        self.assertIn("не актуально", history[-1]["note"])  # исходное примечание не потеряно

    def test_migration_drops_old_catalog_tables(self):
        repo = self._make_repo()
        agent = repo.create_agent("Агент", model=TEST_MODEL_ID)
        chat = repo.create_chat(agent.id, "Чат")
        self._downgrade_to_old_task_schema(agent.id, chat.id)

        self._make_repo()  # переоткрытие запускает миграцию

        conn = sqlite3.connect(self.db_path)
        try:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        finally:
            conn.close()
        for old_table in ("task_states", "task_actions", "task_state_machines", "task_machine_states", "task_transitions"):
            self.assertNotIn(old_table, tables)
        self.assertIn("tasks", tables)
        self.assertIn("task_transition_log", tables)

    def test_migration_is_idempotent_on_reopen(self):
        repo = self._make_repo()
        agent = repo.create_agent("Агент", model=TEST_MODEL_ID)
        chat = repo.create_chat(agent.id, "Чат")
        ids = self._downgrade_to_old_task_schema(agent.id, chat.id)
        self._make_repo()  # первая миграция

        repo3 = self._make_repo()  # повторное открытие — уже новая схема, миграция не запускается
        active = repo3.get_task(ids["active"])
        self.assertEqual(active["task"].state, "execution")


if __name__ == "__main__":
    unittest.main()

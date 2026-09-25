"""
agents_core.config
===================

Настройки уровня сервиса AgentsCore: адрес/порт HTTP-сервера, путь к БД,
а также список сконфигурированных провайдеров LLM (облачных и локальных)
и моделей, которые нужно опросить/зарегистрировать при старте.

Всё считывается один раз из переменных окружения либо файла `.env` рядом с
местом запуска процесса (перезапуск сервиса нужен, чтобы подхватить
изменения — это не "горячая" конфигурация).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


def _load_dotenv(path: str = ".env") -> None:
    """Минимальный загрузчик `.env`: строки вида KEY=VALUE, комментарии
    через '#', уже существующие переменные окружения имеют приоритет."""
    file = Path(path)
    if not file.exists():
        return
    for raw_line in file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


#: "Работу с задачами требуется переделать" (ТЗ по замене system prompt
#: "Менеджера задач") — системный prompt, подмешиваемый, пока у чата
#: включена настройка `task_tracking_enabled`, ЗАПОЛНЯЕМЫЙ ШАБЛОН, а не
#: готовый текст: `<task_states></task_states>` подставляется описанием
#: состояний машины прямо из кода (`task_state_machine.format_states_for_prompt`),
#: `<task_state_machine_invariants></task_state_machine_invariants>` —
#: текстами инвариантов категории "Правило стейт-машины", привязанных к
#: машине состояний (настройка `task_state_machine_invariants`, экран
#: "Модели состояний задач"), каждый на отдельной строке. См.
#: `Repository._build_task_tracking_prompt`. Значение по умолчанию — ровно
#: текст шаблона из ТЗ; переопределяется целиком переменной окружения
#: `TASK_TRACKING_PROMPT_TEMLATE` (можно отредактировать перед запуском
#: сервиса — см. `.env.example`; т.к. простой загрузчик `.env` не понимает
#: многострочные значения, перевод строки в переопределении нужно писать
#: как литеральные символы `\n` — они разворачиваются обратно при чтении).
DEFAULT_TASK_TRACKING_PROMPT_TEMPLATE = """Если в переписке появляется задача/задачи возвращай со следующими параметрами:
task - Название задачи,
state - Этап конечного автомата - TaskState,
step - Целое от 1 до 4, в соответствии с перечнем TaskState,
total - Целое число, всего шагов (4 в соответствии с TaskState),
plan - Утверждённый план, список ["", ..],
done - Что уже сделано, список ["", ..],
current - Что делаем сейчас.
TaskState: <task_states></task_states>
В дальнейшем при работе в этом диалоге соблюдай следующие правила:
[TASK STATE MACHINE INVARIANTS]
<task_state_machine_invariants></task_state_machine_invariants>
При ответе на запрос, предлагай информацию по задаче (в описанном выше формате), промежуточный или конечный результат (если задача выполнена) и в соответствии с требованиями правил (TASK STATE MACHINE INVARIANTS) запрашивай подтверждение пользователем на продолжение работы."""


@dataclass
class ProviderModelEntry:
    """Одна модель, объявленная в .env для конкретного провайдера."""

    provider: str
    model_id: str

    @property
    def id(self) -> str:
        # Составной идентификатор модели, используемый во всём HTTP API
        # (например "ollama:qwen3:0.6b" или "deepseek:deepseek-v4-flash").
        return f"{self.provider}:{self.model_id}"


class AgentConfig:
    """Читается один раз при импорте модуля."""

    HOST: str = os.environ.get("AGENT_HOST", "0.0.0.0")
    PORT: int = int(os.environ.get("AGENT_PORT", "8000"))
    DB_PATH: str = os.environ.get("AGENT_DB_PATH", "agents_core.db")

    # Список активных провайдеров, например "deepseek,ollama". Провайдер,
    # не перечисленный здесь, не опрашивается и его модели не регистрируются,
    # даже если для него заданы прочие переменные окружения.
    PROVIDERS: List[str] = _split_csv(os.environ.get("PROVIDERS", "deepseek,ollama"))

    DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    DEEPSEEK_MODELS: List[str] = _split_csv(
        os.environ.get(
            "DEEPSEEK_MODELS",
            "deepseek-v4-flash,deepseek-v4-pro,deepseek-v4-flash-vision-exp",
        )
    )

    # Реестр скиллов, доступных для привязки к профилю через API/приложение
    # (см. `agents_core.skills.registry`) — JSON-массив описаний функций в
    # формате OpenAI function-tool, тот же формат, что и `Profile.skills_json`.
    # Регистрация в этой переменной делает скилл ВИДИМЫМ и ВЫБИРАЕМЫМ для
    # профиля; она не заменяет реализацию обработчика в коде
    # (`Repository._TOOL_HANDLERS`) — это отдельный, более крупный шаг.
    REGISTERED_SKILLS_JSON: str = os.environ.get("AGENT_REGISTERED_SKILLS", "").strip()

    OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    # По умолчанию предлагается qwen3:0.6b — компактная модель, которую
    # реально запустить локально без мощного GPU; ниже перечислены и другие
    # размеры семейства qwen3 для локального запуска через Ollama.
    OLLAMA_MODELS: List[str] = _split_csv(
        os.environ.get(
            "OLLAMA_MODELS",
            "qwen3:0.6b,qwen3:1.7b,qwen3:4b,qwen3:8b,qwen3:14b,qwen3:30b-a3b",
        )
    )

    @classmethod
    def configured_models(cls) -> List[ProviderModelEntry]:
        """Полный список моделей, объявленных в .env для всех активных
        провайдеров — используется при старте сервиса для наполнения каталога
        моделей (см. `agents_core.repository.ModelCatalog`)."""
        entries: List[ProviderModelEntry] = []
        if "deepseek" in cls.PROVIDERS:
            for model_id in cls.DEEPSEEK_MODELS:
                entries.append(ProviderModelEntry("deepseek", model_id))
        if "ollama" in cls.PROVIDERS:
            for model_id in cls.OLLAMA_MODELS:
                entries.append(ProviderModelEntry("ollama", model_id))
        return entries

    # --- "Менеджер задач": system prompt по шаблону ---------------------
    # Точное имя переменной — TASK_TRACKING_PROMPT_TEMLATE (см. ТЗ) —
    # сохранено буквально, чтобы совпадать с тем, что реально будет в .env.
    # "\n" в значении переменной окружения разворачивается в перевод
    # строки — см. докстринг DEFAULT_TASK_TRACKING_PROMPT_TEMPLATE выше.
    TASK_TRACKING_PROMPT_TEMLATE: str = (
        os.environ.get("TASK_TRACKING_PROMPT_TEMLATE", "").replace("\\n", "\n").strip()
        or DEFAULT_TASK_TRACKING_PROMPT_TEMPLATE
    )

    # --- Логирование ---------------------------------------------------
    # Уровень логирования REST-запросов (agents_core.access) и вызовов
    # внешних API LLM (agents_core.llm.*) — см. agents_core.logging_setup.
    LOG_LEVEL: str = os.environ.get("AGENT_LOG_LEVEL", "INFO").strip().upper() or "INFO"
    # Путь к файлу лога В ДОПОЛНЕНИЕ к stderr; пусто — лог только в stderr.
    LOG_FILE: str = os.environ.get("AGENT_LOG_FILE", "").strip()
    # Сколько символов тела запроса/ответа или JSON-параметров вызова
    # показывать в одной строке лога, остальное усекается.
    LOG_BODY_LIMIT: int = int(os.environ.get("AGENT_LOG_BODY_LIMIT", "2000") or "2000")

    # --- MCP-сервер (третий, отдельно разворачиваемый компонент) --------
    # AgentsCore подключается к нему как MCP-клиент (Streamable HTTP) для
    # получения списка дополнительных инструментов и их вызова —
    # см. `agents_core.mcp_client.MCPClient`. Выключено по умолчанию:
    # существующие развёртывания без MCP-сервера не затрагиваются.
    MCP_ENABLED: bool = os.environ.get("MCP_ENABLED", "false").strip().lower() in ("1", "true", "yes")
    # Например "http://mcp:8001/mcp" (тот же адрес, что слушает mcp_server).
    MCP_SERVER_URL: str = os.environ.get("MCP_SERVER_URL", "").strip()
    # Несколько MCP-серверов: "tools=http://localhost:8001/mcp,scheduler=http://localhost:8002/mcp".
    # Если задано — используется вместо MCP_SERVER_URL.
    MCP_SERVERS: str = os.environ.get("MCP_SERVERS", "").strip()
    # Заготовка под будущую аутентификацию к MCP-серверу — пока нигде не
    # используется (см. mcp_server/.env.example, тот же принцип).
    MCP_API_KEY: str = os.environ.get("MCP_API_KEY", "").strip()
    # Таймаут запросов к MCP-серверу (секунды). С запасом: цепочка
    # инструментов (поиск, загрузка страниц, ответ модели) идёт минуту и больше.
    MCP_REQUEST_TIMEOUT: float = float(os.environ.get("MCP_REQUEST_TIMEOUT", "180") or "180")

    # --- Асинхронные запуски (ТЗ «асинхронные ответы», раздел 2.4) ---------
    # Сколько ответов модели выполняется одновременно (в разных чатах; в
    # одном чате — всегда не больше одного).
    RUN_WORKERS: int = int(os.environ.get("AGENT_RUN_WORKERS", "4") or "4")
    # Предельное время одного запуска, секунды — затем он завершается с ошибкой.
    RUN_TIMEOUT: float = float(os.environ.get("AGENT_RUN_TIMEOUT", "600") or "600")
    # Сколько секунд лента событий завершённого запуска хранится в памяти.
    RUN_EVENTS_TTL: float = float(os.environ.get("AGENT_RUN_EVENTS_TTL", "600") or "600")
    # Сколько дней хранятся записи о завершённых запусках.
    RUN_RETENTION_DAYS: int = int(os.environ.get("AGENT_RUN_RETENTION_DAYS", "30") or "30")

    @classmethod
    def is_deepseek_configured(cls) -> bool:
        return bool(cls.DEEPSEEK_API_KEY)

    @classmethod
    def mcp_servers(cls) -> list:
        """Список `(имя, адрес)` подключаемых MCP-серверов: из `MCP_SERVERS`,
        иначе единственный `MCP_SERVER_URL` под именем "tools"."""
        from .mcp_client import parse_mcp_servers

        if cls.MCP_SERVERS:
            return parse_mcp_servers(cls.MCP_SERVERS)
        return [("tools", cls.MCP_SERVER_URL)] if cls.MCP_SERVER_URL else []

    @classmethod
    def is_mcp_configured(cls) -> bool:
        return cls.MCP_ENABLED and bool(cls.mcp_servers())

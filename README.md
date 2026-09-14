# AgentsCore

AgentsCore — самостоятельный HTTP-сервис для работы с LLM-агентами, на
FastAPI. Сервис хранит всё сам и открывает управление через REST API с
документацией Swagger UI.

Главный объект системы — **агент**: сущность верхнего уровня, у которой
есть своя модель, свой провайдер (облачный DeepSeek или локальный, через
Ollama) и свой набор параметров запроса; по сути это «старший чат». Агент
владеет произвольным числом **чатов** (связь один-ко-многим). Каждый чат
при создании наследует текущие настройки своего агента и может
переопределить часть из них — кроме модели, она зафиксирована на уровне
агента и на экране чата только отображается.

Сервис владеет:

- внутренней базой данных SQLite: агенты (со встроенными настройками), их
  чаты (тоже со встроенными настройками), сообщения чатов;
- каталогом моделей — при старте формируется на основе `.env` и кэшируется
  в БД и в оперативной памяти (см. раздел «Каталог моделей»);
- ключами/адресами подключения ко всем провайдерам LLM — они никогда не
  покидают процесс сервиса;
- оркестрацией обращений к LLM: сборкой запроса из настроек чата, выбором
  провайдера, потоковой и блокирующей отправкой, подсчётом токенов,
  суммаризацией истории диалога (вручную и автоматически).

Поддерживаемые провайдеры:

- **DeepSeek API** — облачные модели (`deepseek-v4-flash`, `deepseek-v4-pro`,
  `deepseek-v4-flash-vision-exp` и другие — список задаётся в `.env`, не
  зашит в код).
- **Ollama API** (https://docs.ollama.com/api/) — локальный запуск моделей,
  в первую очередь семейства **Qwen3** любого размера (`qwen3:0.6b`,
  `qwen3:1.7b`, `qwen3:4b`, `qwen3:8b`, `qwen3:14b`, `qwen3:30b-a3b` и т. д.).

## Установка и запуск

```bash
pip install -r requirements.txt

cp .env.example .env
# отредактируйте .env: ключ DeepSeek, адрес Ollama, список моделей

python -m agents_core.main
# [agents_core] Swagger UI: http://0.0.0.0:8000/docs
```

Swagger UI доступен на `/docs`, альтернативная документация ReDoc — на
`/redoc`, "сырая" OpenAPI-схема — на `/openapi.json`.

Конфигурация (`.env`, см. `.env.example`):

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `AGENT_HOST` / `AGENT_PORT` | `0.0.0.0` / `8000` | адрес HTTP-сервера |
| `AGENT_DB_PATH` | `agents_core.db` | путь к файлу SQLite |
| `PROVIDERS` | `deepseek,ollama` | какие провайдеры активны |
| `DEEPSEEK_API_KEY` | — | ключ DeepSeek API |
| `DEEPSEEK_MODELS` | `deepseek-v4-flash,deepseek-v4-pro,deepseek-v4-flash-vision-exp` | облачные модели DeepSeek |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | адрес запущенного Ollama-сервера |
| `OLLAMA_MODELS` | `qwen3:0.6b,qwen3:1.7b,qwen3:4b,qwen3:8b,qwen3:14b,qwen3:30b-a3b` | локальные модели через Ollama |

Сервис нормально запускается и без настроенного ключа DeepSeek — работа с
агентами/чатами (создание, настройки, история) не требует ключа ни от
одного провайдера; ключ/доступность провайдера нужны только в момент
фактической отправки сообщения этой моделью.

## Каталог моделей

При каждом старте сервис сравнивает список моделей, объявленных в `.env`
(`DEEPSEEK_MODELS` + `OLLAMA_MODELS`), с тем, что уже сохранено в БД:

- если списки совпадают — данные о моделях (лимиты токенов, поддержка
  режима размышлений и т. д.) просто загружаются из БД в память, провайдеры
  повторно не опрашиваются;
- если списки различаются — для новых моделей выполняется опрос
  провайдера: Ollama раскрывает часть характеристик через `POST
  /api/show` (размер контекстного окна и т. п.), DeepSeek не отдаёт эти
  данные через API — для него используются встроенные справочные значения.

`GET /models` всегда отвечает из памяти, без обращения к провайдерам.
Каждая модель получает признак `is_local` (`true` для моделей, запущенных
через Ollama) и составной идентификатор `provider:model_id`, например
`ollama:qwen3:0.6b` или `deepseek:deepseek-v4-flash` — именно этот
идентификатор используется в поле `model` настроек агента/чата и в пути
`GET /model/{id}/health`.

## Настройки: по умолчанию / агента / чата

Настройки по умолчанию — единственный ресурс, применяемый **при создании
нового агента**; при создании чата настройки берутся не из них, а из
текущих настроек его агента.

Настройки чата — это тот же набор полей, что и у агента, только `model`
там не редактируется (значение показывается для информации и жёстко
фиксировано агентом-владельцем).

| Группа | Поле (`sys_name`) | Только у агента/чата |
|---|---|---|
| Модель | `model` | да (в настройках по умолчанию — стартовая модель нового агента) |
| Модель | `system_prompt`, `temperature`, `top_p`, `seed` | нет |
| Параметры ответа | `stream`, `thinking_enabled`, `reasoning_effort` | нет |
| Параметры ответа | `max_tokens`, `json_mode`, `stop_sequences` | только агент/чат |
| Инструменты | `tool_choice`, `tools_json` | только агент/чат |
| Суммаризация запросов | `summary_prompt` | нет |
| Суммаризация запросов | `autosummary`, `autosummary_by_messages`, `autosummary_by_tokens` | только агент/чат |
| Дополнительно | `include_usage_in_stream` | нет |
| Дополнительно | `logprobs`, `frequency_penalty`, `presence_penalty` | только агент/чат |

`autosummary` принимает значения `"off"` (по умолчанию, автосуммаризация
выключена), `"messages"` (порог — число пользовательских сообщений с
последней суммаризации, `autosummary_by_messages`, по умолчанию 10) или
`"tokens"` (порог — текущий размер контекста чата в токенах,
`autosummary_by_tokens`, по умолчанию 50000).

## Суммаризация истории чата

Суммаризация — это отдельный запрос к той же модели, что использует чат:
ей передаётся системный промпт, вся текущая история (или, если чат уже
суммаризировался раньше, — только последнее summary-сообщение и всё, что
было после него) и, последним сообщением, текст из настройки
`summary_prompt` (по умолчанию «Сформируй суммарный prompt на основе
истории чата»). Ответ модели сохраняется в чат как **новое сообщение
пользователя** с признаком `is_summary = true` и с этого момента
используется вместо всей предыдущей переписки как отправная точка
контекста для следующих сообщений — до тех пор, пока не будет создано
новое summary.

Суммаризация (и ручная — по эндпоинту `/chats/{id}/summarize`, и
автоматическая) недоступна, если текущий размер контекста чата уже
превышает максимум входных токенов модели или заданный в настройках
`max_tokens` — в этом случае эндпоинт вернёт `409`.

## Подсчёт токенов и заполнение контекстного окна

Для каждого чата (`GET /chats/{id}` и `GET /chats/{id}` в составе `GET
/agents`) считаются агрегаты:

- `prompt_tokens` / `completion_tokens` / `total_tokens` — сумма входящих,
  исходящих и общих токенов по всем ответам ассистента в истории чата
  (точные значения из ответа провайдера);
- `current_context_tokens` — оценка размера контекста, который уйдёт в
  следующем запросе (после последнего summary, если он есть);
- `max_input_tokens` — лимит входных токенов модели, заданной для агента
  этого чата (из каталога моделей);
- `context_fill_ratio` — `current_context_tokens / max_input_tokens`,
  готовое значение для индикатора заполненности контекста;
- `can_summarize` — доступна ли сейчас суммаризация (см. выше).

## HTTP API

Полное описание каждого эндпоинта, схемы тел запросов/ответов и примеры —
в Swagger UI (`/docs`). Ниже — сводная таблица и curl-примеры.

Соглашение об именовании: `chats`/`agents` (множественное число) — только
для операций со списком; создание одного ресурса — через единственное
число: `POST /agent`, `POST /agents/{id}/chat`.

| Метод | Путь | Описание |
|---|---|---|
| GET | `/health` | статус сервиса |
| GET | `/models` | каталог моделей (из памяти) |
| GET | `/model/{id}/health` | проверка связи с провайдером конкретной модели |
| GET | `/settings/default` | настройки по умолчанию |
| PUT | `/settings/default` | изменить настройки по умолчанию (частично) |
| POST | `/settings/default/reset` | сбросить настройки по умолчанию к встроенным |
| GET | `/agents` | список агентов вместе с их чатами (для главного экрана) |
| POST | `/agent` | создать агента |
| GET | `/agents/{id}` | детали агента |
| PATCH | `/agents/{id}` | переименовать агента |
| DELETE | `/agents/{id}` | удалить агента (каскадно — его чаты и сообщения) |
| GET | `/agents/{id}/settings` | настройки агента |
| PUT | `/agents/{id}/settings` | обновить настройки агента (частично; включая модель) |
| POST | `/agents/{id}/chat` | создать чат внутри агента |
| GET | `/chats` | список чатов (опционально `?agent_id=`) |
| GET | `/chats/{id}` | детали чата: настройки + агрегаты по токенам |
| PATCH | `/chats/{id}` | переименовать чат |
| DELETE | `/chats/{id}` | удалить чат |
| POST | `/chats/{id}/copy` | скопировать чат (настройки + вся история) |
| GET | `/chats/{id}/settings` | настройки чата |
| PUT | `/chats/{id}/settings` | обновить настройки чата (частично; без модели) |
| POST | `/chats/{id}/summarize` | суммаризировать историю чата вручную |
| GET | `/chats/{id}/messages` | история сообщений |
| DELETE | `/chats/{id}/messages` | очистить всю историю |
| DELETE | `/chats/{id}/messages/{message_id}` | удалить одно сообщение |
| POST | `/chats/{id}/messages/bulk-delete` | удалить несколько выбранных сообщений |
| POST | `/chats/{id}/messages` | отправить сообщение, блокирующий режим |
| POST | `/chats/{id}/messages/stream` | отправить сообщение, потоковый режим (SSE) |

Любой ответ с ошибкой — `{"error": "..."}` с одним из статусов: `400`
(некорректные входные данные, включая неизвестное имя поля настроек или
попытку изменить модель у чата), `404` (нет такого агента/чата/модели),
`409` (суммаризация недоступна — превышены лимиты токенов), `502`
(провайдер LLM вернул ошибку), `503` (провайдер не сконфигурирован,
например не задан `DEEPSEEK_API_KEY`).

## Примеры вызовов через curl

Все примеры предполагают, что сервис запущен локально на порту `8000`.

### Проверка состояния и моделей

```bash
curl http://localhost:8000/health

curl http://localhost:8000/models

# id модели — как в ответе /models, например "ollama:qwen3:0.6b"
curl http://localhost:8000/model/ollama:qwen3:0.6b/health
```

### Настройки по умолчанию

```bash
curl http://localhost:8000/settings/default

curl -X PUT http://localhost:8000/settings/default \
  -H "Content-Type: application/json" \
  -d '{"model": "ollama:qwen3:0.6b", "temperature": 0.7}'

curl -X POST http://localhost:8000/settings/default/reset
```

### Агенты

```bash
# Создать агента (настройки копируются из настроек по умолчанию)
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"name": "Домашний помощник", "model": "ollama:qwen3:0.6b"}'

# Список агентов вместе с их чатами (главный экран)
curl http://localhost:8000/agents

curl http://localhost:8000/agents/AGENT_ID

curl -X PATCH http://localhost:8000/agents/AGENT_ID \
  -H "Content-Type: application/json" \
  -d '{"name": "Новое имя"}'

curl http://localhost:8000/agents/AGENT_ID/settings

curl -X PUT http://localhost:8000/agents/AGENT_ID/settings \
  -H "Content-Type: application/json" \
  -d '{"thinking_enabled": true, "reasoning_effort": "high", "autosummary": "tokens", "autosummary_by_tokens": 20000}'

curl -X DELETE http://localhost:8000/agents/AGENT_ID
```

### Чаты

```bash
curl -X POST http://localhost:8000/agents/AGENT_ID/chat \
  -H "Content-Type: application/json" \
  -d '{"title": "Первый чат"}'

curl http://localhost:8000/chats
curl "http://localhost:8000/chats?agent_id=AGENT_ID"
curl http://localhost:8000/chats/CHAT_ID

curl -X PATCH http://localhost:8000/chats/CHAT_ID \
  -H "Content-Type: application/json" \
  -d '{"title": "Новое название"}'

curl -X POST http://localhost:8000/chats/CHAT_ID/copy \
  -H "Content-Type: application/json" \
  -d '{"title": "Первый чат (копия)"}'

curl http://localhost:8000/chats/CHAT_ID/settings

curl -X PUT http://localhost:8000/chats/CHAT_ID/settings \
  -H "Content-Type: application/json" \
  -d '{"temperature": 0.5, "max_tokens": 2048}'

curl -X DELETE http://localhost:8000/chats/CHAT_ID
```

### Сообщения

```bash
curl http://localhost:8000/chats/CHAT_ID/messages

# Отправка, блокирующий режим
curl -X POST http://localhost:8000/chats/CHAT_ID/messages \
  -H "Content-Type: application/json" \
  -d '{"text": "Привет! Расскажи в двух словах, кто ты."}'

# Отправка, потоковый режим (SSE); -N отключает буферизацию curl
curl -N -X POST http://localhost:8000/chats/CHAT_ID/messages/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Напиши короткое стихотворение про осень."}'

# Удалить одно сообщение
curl -X DELETE http://localhost:8000/chats/CHAT_ID/messages/42

# Удалить несколько выбранных сообщений
curl -X POST http://localhost:8000/chats/CHAT_ID/messages/bulk-delete \
  -H "Content-Type: application/json" \
  -d '{"ids": [42, 43, 44]}'

# Очистить всю историю чата
curl -X DELETE http://localhost:8000/chats/CHAT_ID/messages

# Суммаризировать историю чата вручную
curl -X POST http://localhost:8000/chats/CHAT_ID/summarize
```

События потока (`POST .../messages/stream`) — по одному JSON-объекту на
строку `data:`:

```
data: {"type": "delta", "content": "Ли", "reasoning_content": null}

data: {"type": "delta", "content": "стья", "reasoning_content": null}

data: {"type": "done", "message": {"id": 4, "chat_id": "CHAT_ID", "role": "assistant", "content": "Листья...", "is_summary": false, "duration_ms": 950, "total_tokens": 42, ...}}
```

Если во время генерации произойдёт ошибка, вместо `"done"` придёт
`{"type": "error", "message": "..."}`.

## Структура пакета

- `providers/` — адаптеры провайдеров LLM (`deepseek.py`, `ollama.py`) за
  общим интерфейсом `base.BaseProvider`; `deepseek_client.py` — низкоуровневый
  клиент DeepSeek API, ничего не знающий об агентах/чатах.
- `models.py` — доменные dataclass'ы (`Agent`, `Chat`, `Message`,
  `ModelInfo`, `Settings`, `DefaultSettings`) и метаданные полей настроек
  для форм интерфейса (`AGENT_SETTINGS_FIELDS`, `DEFAULT_SETTINGS_FIELDS`).
- `config.py` — настройки уровня сервиса и провайдеров, считываются из
  окружения/`.env` при старте.
- `db.py` — хранение в SQLite.
- `catalog.py` — наполнение и кэширование каталога моделей.
- `repository.py` — вся бизнес-логика: агенты, чаты, сообщения, токены,
  суммаризация.
- `schemas.py` — Pydantic-модели запросов/ответов HTTP API.
- `api/` — маршруты FastAPI, тонкая обёртка над `Repository`.
- `main.py` — точка входа `python -m agents_core.main`.

## Тесты

Бизнес-логика (`Repository`, `Database`, преобразование в Pydantic-схемы)
тестируется полностью офлайн — настоящие DeepSeek/Ollama не вызываются,
вместо них подставляется предсказуемый `FakeProvider`; эти тесты не
зависят от FastAPI и работают при наличии только `requests`/`pydantic`.
Тесты самого HTTP-слоя (`tests/test_api.py`, через `fastapi.testclient`)
дополнительно требуют пакет `fastapi` из `requirements.txt` — если его нет,
они автоматически пропускаются, а не падают.

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

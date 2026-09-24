"""
Точка входа: `python -m agents_core.main`.

Запускает Uvicorn с приложением из `agents_core.api.app.create_app()`.
Адрес/порт берутся из `AgentConfig` (переменные окружения `AGENT_HOST`/
`AGENT_PORT`, см. `.env.example`).
"""

from __future__ import annotations

import sys

import uvicorn

from .api.app import create_app
from .config import AgentConfig
from .mcp_client import normalize_mcp_base_url

app = create_app()


def main() -> None:
    print(f"[agents_core] DeepSeek: {'настроен' if AgentConfig.is_deepseek_configured() else 'ключ не задан'}", file=sys.stderr)
    print(f"[agents_core] Ollama: {AgentConfig.OLLAMA_BASE_URL}", file=sys.stderr)
    # Печатаем УЖЕ нормализованный адрес (с домысленным /mcp, если его не
    # было в MCP_SERVER_URL) — то, что реально будет использовано клиентом
    # (см. MCPClient/normalize_mcp_base_url), а не сырое значение из .env,
    # чтобы при диагностике не приходилось гадать, откуда взялся 404.
    mcp_status = (
        f"подключён к {normalize_mcp_base_url(AgentConfig.MCP_SERVER_URL)}"
        if AgentConfig.is_mcp_configured()
        else "выключен"
    )
    print(f"[agents_core] MCP-сервер: {mcp_status}", file=sys.stderr)
    print(f"[agents_core] Swagger UI: http://{AgentConfig.HOST}:{AgentConfig.PORT}/docs", file=sys.stderr)
    uvicorn.run(app, host=AgentConfig.HOST, port=AgentConfig.PORT)


if __name__ == "__main__":
    main()

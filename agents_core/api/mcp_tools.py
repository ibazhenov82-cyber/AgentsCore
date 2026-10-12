"""Список инструментов подключённых MCP-серверов (`GET /mcp/tools`)."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends

from ..repository import Repository
from ..schemas import McpToolOut
from .deps import get_repository

router = APIRouter(tags=["MCP"])


@router.get(
    "/mcp/tools",
    response_model=List[McpToolOut],
    summary="Инструменты для выбора в настройках (MCP и «Доступ к agents core»)",
    description=(
        "Инструменты всех MCP-серверов из `MCP_SERVERS` (или `MCP_SERVER_URL`) и встроенные инструменты "
        "AgentsCore группы «Доступ к agents core» (`read_chat` — «Чтение чата») — для выбора в настройках "
        "агента/чата. Встроенные выполняются самим AgentsCore и есть всегда, даже без MCP."
    ),
)
def list_mcp_tools(repo: Repository = Depends(get_repository)) -> List[McpToolOut]:
    return [McpToolOut(**t) for t in repo.list_mcp_tools()]

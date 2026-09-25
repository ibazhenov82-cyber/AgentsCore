"""Зависимости FastAPI, общие для всех роутеров."""

from __future__ import annotations

from fastapi import Request

from ..repository import Repository
from ..runs import RunManager


def get_repository(request: Request) -> Repository:
    return request.app.state.repo


def get_run_manager(request: Request) -> RunManager:
    return request.app.state.run_manager

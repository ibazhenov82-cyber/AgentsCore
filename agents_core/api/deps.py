"""Зависимости FastAPI, общие для всех роутеров."""

from __future__ import annotations

from fastapi import Request

from ..repository import Repository


def get_repository(request: Request) -> Repository:
    return request.app.state.repo

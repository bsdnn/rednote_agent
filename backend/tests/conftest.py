"""Shared pytest fixtures for rednote_agent test suite."""
from __future__ import annotations

import asyncio
from collections import defaultdict
from pathlib import Path

import pytest


class MockDeepSeekClient:
    """Programmable stub for DeepSeekClient.chat().

    Use `set_responses([...])` with a queue of dicts (matching DeepSeek
    response shape) or Exception instances (which will be raised).
    """

    def __init__(self):
        self.responses: list = []
        self.call_count: int = 0
        self.calls_received: list[tuple[list, list | None]] = []

    def set_responses(self, responses: list) -> None:
        self.responses = responses
        self.call_count = 0
        self.calls_received = []

    async def chat(self, messages: list[dict], tools: list | None = None) -> dict:
        self.calls_received.append((messages, tools))
        if self.call_count >= len(self.responses):
            raise RuntimeError(
                f"MockDeepSeekClient: unexpected chat call #{self.call_count + 1} "
                f"(only {len(self.responses)} responses queued)"
            )
        resp = self.responses[self.call_count]
        self.call_count += 1
        if isinstance(resp, Exception):
            raise resp
        return resp


@pytest.fixture
def mock_deepseek_client(monkeypatch):
    """Provides a MockDeepSeekClient and patches agent.get_client to return it."""
    from backend.core import agent

    client = MockDeepSeekClient()
    monkeypatch.setattr(agent, "get_client", lambda: client)
    return client


@pytest.fixture
async def tmp_memory_db(monkeypatch, tmp_path):
    """Redirect memory_service to use a fresh temp SQLite DB."""
    from backend.services import memory_service

    db_path = tmp_path / "test_memory.db"
    monkeypatch.setattr(memory_service, "_DB_PATH", db_path)
    await memory_service.init_db()
    yield db_path
    # Windows file-lock tolerance: best-effort delete
    try:
        if db_path.exists():
            db_path.unlink()
    except PermissionError:
        pass


@pytest.fixture
def reset_rate_limit_dict(monkeypatch):
    """Replace the module-level rate-limit dict with a fresh one per test."""
    from backend.api import dependencies

    fresh: dict[str, list[float]] = defaultdict(list)
    monkeypatch.setattr(dependencies, "_request_times", fresh)
    return fresh

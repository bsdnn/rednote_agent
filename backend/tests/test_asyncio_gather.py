"""Q14 — asyncio.gather and wait_for semantics relevant to agent.py.

agent.py:232 uses `asyncio.gather(*[_exec_tool(...)])` with default
return_exceptions=False. We document the consequence: one failing tool
crashes the entire request. Fix would be return_exceptions=True with
per-tool result handling.
"""
from __future__ import annotations

import asyncio

import pytest


async def test_gather_propagates_first_exception():
    """PROVE: default gather raises if any task fails. agent.py uses
    the default, so any tool exception kills the whole agent loop.
    Recommended fix: gather(..., return_exceptions=True) and handle
    each result individually.
    """
    async def ok(label: str) -> str:
        await asyncio.sleep(0.05)
        return label

    async def fail() -> None:
        await asyncio.sleep(0.01)
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await asyncio.gather(ok("a"), fail(), ok("c"))


async def test_wait_for_cancels_only_self():
    """PASS: wait_for cancels the wrapped coroutine on timeout, but
    sibling tasks under gather continue. _exec_tool's per-tool
    wait_for is correctly isolated.
    """
    completed: list[str] = []

    async def slow_tool() -> str:
        try:
            await asyncio.sleep(0.5)
            completed.append("slow")
            return "slow-done"
        except asyncio.CancelledError:
            completed.append("slow-cancelled")
            raise

    async def fast_tool(label: str) -> str:
        await asyncio.sleep(0.05)
        completed.append(label)
        return f"{label}-done"

    async def _exec(coro, timeout):
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            return "TIMEOUT"

    results = await asyncio.gather(
        _exec(slow_tool(), 0.1),
        _exec(fast_tool("a"), 0.5),
        _exec(fast_tool("b"), 0.5),
    )
    assert results[0] == "TIMEOUT"
    assert results[1] == "a-done"
    assert results[2] == "b-done"
    assert "a" in completed and "b" in completed

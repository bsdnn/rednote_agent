"""Q18.1 — main agent loop event sequence and tool-call handling."""
from __future__ import annotations

import asyncio
import json
import time

import pytest

from backend.core import agent
from backend.models.request import GenerateRequest, Tone
from backend.services import tools_registry


def _chat(content: str | None, tool_calls: list | None = None) -> dict:
    msg: dict = {"content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5}}


def _tool_call(call_id: str, name: str, args: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _draft_json() -> str:
    return json.dumps({"title": "T✨", "body": "B" * 30,
                       "hashtags": ["#a", "#b"], "emojis": ["✨"]})


def _critique(score: int = 9) -> str:
    return json.dumps({
        "virality_score": score, "tone_match_score": score,
        "accuracy_score": score, "min_score": score,
        "issues": [], "suggestions": [],
    })


async def _collect(req: GenerateRequest) -> list[dict]:
    events = []
    async for ev in agent.generate_rednote(req, enable_planning=False):
        events.append(ev)
    return events


async def test_event_sequence_on_happy_path(mock_deepseek_client, monkeypatch):
    """PASS: tool round → final JSON → reflection (pass) → complete + token_usage."""
    # Stub the tool to avoid hitting the real DB/network
    async def fake_recall(user_id: str) -> str:
        return "(test history)"

    monkeypatch.setitem(
        tools_registry.AVAILABLE_TOOLS, "recall_user_history", fake_recall
    )

    mock_deepseek_client.set_responses([
        _chat(None, [_tool_call("c1", "recall_user_history", {"user_id": "u"})]),
        _chat(_draft_json()),
        _chat(_critique(9)),
    ])

    req = GenerateRequest(query="happy path", tone=Tone.lively, max_iterations=5)
    events = await _collect(req)
    names = [e["event"] for e in events]

    # Required ordered subsequence
    assert "agent_thinking" in names
    assert "tool_result" in names
    assert "complete" in names
    assert "token_usage" in names
    assert names.index("tool_result") < names.index("complete")
    assert names.index("complete") < names.index("token_usage")


async def test_tool_calls_executed_in_parallel(mock_deepseek_client, monkeypatch):
    """PASS: two 0.5s-sleeping tools dispatched together finish in <0.8s
    (not ~1.0s as serial would).
    """
    async def slow_a(query: str) -> str:
        await asyncio.sleep(0.5)
        return "A"

    async def slow_b(query: str) -> str:
        await asyncio.sleep(0.5)
        return "B"

    monkeypatch.setitem(tools_registry.AVAILABLE_TOOLS, "query_product_database", slow_a)
    monkeypatch.setitem(tools_registry.AVAILABLE_TOOLS, "search_web", slow_b)

    mock_deepseek_client.set_responses([
        _chat(None, [
            _tool_call("c1", "query_product_database", {"query": "x"}),
            _tool_call("c2", "search_web", {"query": "y"}),
        ]),
        _chat(_draft_json()),
        _chat(_critique(9)),
    ])

    req = GenerateRequest(query="parallel", tone=Tone.lively, max_iterations=5)
    t0 = time.monotonic()
    await _collect(req)
    elapsed = time.monotonic() - t0

    assert elapsed < 0.9, f"Tools ran serially? elapsed={elapsed:.2f}s"


async def test_tool_cache_hits_within_request(mock_deepseek_client, monkeypatch):
    """PASS: when LLM requests the same (name, args) twice in one request,
    the second call hits the per-request _tool_cache and the real tool
    function is invoked only once.
    """
    call_counter = {"n": 0}

    async def counted(query: str) -> str:
        call_counter["n"] += 1
        return f"result-{call_counter['n']}"

    monkeypatch.setitem(tools_registry.AVAILABLE_TOOLS, "query_product_database", counted)

    mock_deepseek_client.set_responses([
        # Iter 1: one call
        _chat(None, [_tool_call("c1", "query_product_database", {"query": "same"})]),
        # Iter 2: same (name, args) — should hit cache
        _chat(None, [_tool_call("c2", "query_product_database", {"query": "same"})]),
        _chat(_draft_json()),
        _chat(_critique(9)),
    ])

    req = GenerateRequest(query="cache", tone=Tone.lively, max_iterations=5)
    events = await _collect(req)

    assert call_counter["n"] == 1, (
        f"Expected real tool to run once (cache hit on 2nd call); got {call_counter['n']}"
    )
    cached_events = [
        e for e in events
        if e["event"] == "tool_result" and e["data"].get("cached") is True
    ]
    assert len(cached_events) == 1, "Expected exactly one cached=True tool_result event"


async def test_json_retry_on_malformed_output(mock_deepseek_client, monkeypatch):
    """PASS: when LLM returns non-JSON, the agent issues a correction
    prompt and accepts a clean JSON on retry.
    """
    mock_deepseek_client.set_responses([
        _chat("this is not JSON"),
        _chat(_draft_json()),
        _chat(_critique(9)),
    ])

    req = GenerateRequest(query="retry", tone=Tone.lively, max_iterations=5)
    events = await _collect(req)

    thinking_steps = [e["data"].get("step", "") for e in events
                      if e["event"] == "agent_thinking"]
    assert any("修正" in s for s in thinking_steps), (
        f"Expected a JSON-correction notice in agent_thinking events; got {thinking_steps}"
    )
    assert any(e["event"] == "complete" for e in events)

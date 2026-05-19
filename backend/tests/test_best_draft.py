"""Q18.1 — best_draft selection logic in _agent_loop.

PROVE test documents the bug discovered in conversation review:
when MAX_REFLECTIONS is exhausted, a fresh post-budget draft is
silently discarded in favor of an older best_draft, regardless of
whether the fresh draft might be better. agent.py:312.
"""
from __future__ import annotations

import json

import pytest

from backend.core import agent
from backend.core.config import settings
from backend.models.request import GenerateRequest, Tone


def _msg(content: str, tool_calls: list | None = None) -> dict:
    """Build a chat-completion response dict."""
    message: dict = {"content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
        message["content"] = None
    return {
        "choices": [{"message": message}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10},
    }


def _draft(title: str) -> str:
    return json.dumps({
        "title": title,
        "body": "正文" * 20,
        "hashtags": ["#a", "#b", "#c"],
        "emojis": ["✨"],
    }, ensure_ascii=False)


def _critique(score: int) -> str:
    return json.dumps({
        "virality_score": score,
        "tone_match_score": score,
        "accuracy_score": score,
        "min_score": score,
        "issues": ["issue1"] if score < 7 else [],
        "suggestions": ["suggest1"] if score < 7 else [],
    })


async def _collect(req: GenerateRequest) -> list[dict]:
    events = []
    async for ev in agent.generate_rednote(req, enable_planning=False):
        events.append(ev)
    return events


async def test_best_draft_tracks_highest_score(mock_deepseek_client, monkeypatch):
    """PASS: among drafts scored 6, 8, 7, the agent should pick the 8-er.
    But 8 >= REFLECTION_MIN_SCORE (7) → the agent returns immediately on draft 2.
    So we test that draft 2 (the 8) is what comes out.
    """
    monkeypatch.setattr(settings, "MAX_REFLECTIONS", 3)
    monkeypatch.setattr(settings, "REFLECTION_MIN_SCORE", 9)  # force rewrites

    mock_deepseek_client.set_responses([
        _msg(_draft("稿1")), _msg(_critique(6)),       # iter 1 -> revise
        _msg(_draft("稿2")), _msg(_critique(8)),       # iter 2 -> revise (8 < 9)
        _msg(_draft("稿3")), _msg(_critique(7)),       # iter 3 -> revise (7 < 9)
        # Reflection count == 3 == MAX, next draft will skip reflection
        _msg(_draft("稿4")),                           # iter 4 -> returns best_draft
    ])

    req = GenerateRequest(query="test", tone=Tone.lively, max_iterations=8)
    events = await _collect(req)

    complete = next(e for e in events if e["event"] == "complete")
    # Best score was 8 → draft 2
    assert complete["data"]["title"] == "稿2", (
        f"Expected highest-scored draft (稿2, score 8); got {complete['data']['title']}"
    )


async def test_third_draft_swallowed_by_best_draft(mock_deepseek_client, monkeypatch):
    """PROVE the bug: when MAX_REFLECTIONS is hit, the post-budget draft
    is dropped and best_draft (from an earlier round) is returned, even
    if the fresh draft incorporated the prior critiques.
    Recommended fix:  give the final draft one scoring pass, OR
    require the current result to lose on score before falling back.
    """
    monkeypatch.setattr(settings, "MAX_REFLECTIONS", 2)
    monkeypatch.setattr(settings, "REFLECTION_MIN_SCORE", 9)  # force both reflections to revise

    mock_deepseek_client.set_responses([
        _msg(_draft("稿1")), _msg(_critique(6)),   # iter 1 -> revise
        _msg(_draft("稿2")), _msg(_critique(6)),   # iter 2 -> revise; best_draft = 稿1 (first 6)
        _msg(_draft("稿3")),                       # iter 3 -> reflection skipped, returns best_draft
    ])

    req = GenerateRequest(query="test", tone=Tone.lively, max_iterations=5)
    events = await _collect(req)

    complete = next(e for e in events if e["event"] == "complete")
    final_title = complete["data"]["title"]

    # PROVE: the post-budget revision (稿3) is silently dropped.
    assert final_title != "稿3", (
        "Expected bug: third draft swallowed by best_draft fallback. "
        f"If you see 稿3, the bug has been fixed — update this test accordingly."
    )
    # best_draft tracking favors the first occurrence at equal score
    assert final_title == "稿1", f"Expected best_draft to be 稿1, got {final_title}"


async def test_returns_unscored_when_no_best_exists(mock_deepseek_client, monkeypatch):
    """PASS: if every reflection raises, best_draft stays None and the
    code falls back to the current (unscored) result.
    """
    monkeypatch.setattr(settings, "MAX_REFLECTIONS", 2)

    mock_deepseek_client.set_responses([
        _msg(_draft("稿1")), RuntimeError("reflection blew up"),
    ])

    req = GenerateRequest(query="test", tone=Tone.lively, max_iterations=5)
    events = await _collect(req)

    complete = next(e for e in events if e["event"] == "complete")
    assert complete["data"]["title"] == "稿1"

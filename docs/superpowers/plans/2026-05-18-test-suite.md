# Test Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 17-case offline pytest suite (6 PROVE-type bug-documenting cases) plus a paired A/B experiment script that quantifies whether `_planning_phase` is worth its cost. Final deliverable is `docs/testing/2026-05-18-results.md` summarizing all findings.

**Architecture:** Offline tests live in `backend/tests/` and run via `pytest`. The A/B experiment is a standalone script in `scripts/` that imports `generate_rednote` directly, runs each query under both arms (planning on/off), and emits an incremental JSONL log + auto-generated markdown report. The only production code change is one new keyword parameter on `generate_rednote`.

**Tech Stack:** pytest, pytest-asyncio, pytest-timeout, asyncio, monkeypatch, existing FastAPI/aiosqlite/FAISS stack.

**Spec:** `docs/superpowers/specs/2026-05-18-test-suite-design.md`

---

## File Structure

**Create:**
- `backend/requirements-dev.txt` — pytest + plugins
- `backend/tests/__init__.py` — empty marker
- `backend/tests/conftest.py` — shared fixtures: `mock_deepseek_client`, `tmp_memory_db`, `reset_rate_limit_dict`
- `backend/tests/test_memory_security.py`
- `backend/tests/test_rate_limit.py`
- `backend/tests/test_rag_stability.py`
- `backend/tests/test_asyncio_gather.py`
- `backend/tests/test_best_draft.py`
- `backend/tests/test_agent_flow.py`
- `pytest.ini` — async mode + path config
- `scripts/__init__.py` — empty marker
- `scripts/ab_queries.json` — 30 hand-written queries
- `scripts/ab_planning_experiment.py` — A/B runner
- `docs/testing/2026-05-18-results.md` — final report

**Modify:**
- `backend/core/agent.py` — add `enable_planning: bool = True` keyword parameter to `generate_rednote`

---

## Task 1: Dev dependencies, pytest config, and conftest fixtures

**Files:**
- Create: `backend/requirements-dev.txt`
- Create: `pytest.ini`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/conftest.py`

- [ ] **Step 1: Create dev requirements file**

Create `backend/requirements-dev.txt`:
```
pytest>=8.0
pytest-asyncio>=0.23
pytest-timeout>=2.2
```

- [ ] **Step 2: Install dev dependencies**

Run:
```powershell
pip install -r backend/requirements-dev.txt
```
Expected: pytest, pytest-asyncio, pytest-timeout installed without conflict.

- [ ] **Step 3: Create pytest.ini at repo root**

Create `pytest.ini`:
```ini
[pytest]
asyncio_mode = auto
asyncio_default_fixture_loop_scope = function
timeout = 30
testpaths = backend/tests
python_files = test_*.py
pythonpath = .
```

- [ ] **Step 4: Create empty tests package marker**

Create `backend/tests/__init__.py` (empty file).

- [ ] **Step 5: Create conftest.py with shared fixtures**

Create `backend/tests/conftest.py`:
```python
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
```

- [ ] **Step 6: Smoke-test pytest is wired up**

Run:
```powershell
pytest backend/tests --collect-only
```
Expected: `no tests ran` or `collected 0 items` (no errors).

- [ ] **Step 7: Commit**

```powershell
git add backend/requirements-dev.txt pytest.ini backend/tests/__init__.py backend/tests/conftest.py
git commit -m "test: bootstrap pytest infrastructure with shared fixtures"
```

---

## Task 2: Add `enable_planning` parameter to `generate_rednote`

**Files:**
- Modify: `backend/core/agent.py:83-126` (signature + planning gate)

This is the ONLY production code change in this plan. Default `True` preserves existing behavior; the A/B script will pass `False` to skip planning.

- [ ] **Step 1: Update `generate_rednote` signature and gate the planning call**

Edit `backend/core/agent.py` — change the function signature and wrap the planning call:

Find:
```python
async def generate_rednote(request: GenerateRequest) -> AsyncIterator[dict]:
    client = get_client()

    # Phase 1: Planning
    yield {"event": "agent_thinking", "data": {"step": "制定任务计划中...", "iteration": 0}}
    plan: dict | None = None
    try:
        plan = await _planning_phase(
            client, request.query, str(request.tone.value), request.persona, request.user_id
        )
        yield {"event": "agent_plan", "data": plan}
        logger.info("Plan created: %s", plan.get("goal"))
    except Exception as e:
        logger.warning("Planning phase failed: %s — continuing without plan", e)
```

Replace with:
```python
async def generate_rednote(
    request: GenerateRequest, *, enable_planning: bool = True
) -> AsyncIterator[dict]:
    client = get_client()

    # Phase 1: Planning (skippable for A/B experiment)
    plan: dict | None = None
    if enable_planning:
        yield {"event": "agent_thinking", "data": {"step": "制定任务计划中...", "iteration": 0}}
        try:
            plan = await _planning_phase(
                client, request.query, str(request.tone.value), request.persona, request.user_id
            )
            yield {"event": "agent_plan", "data": plan}
            logger.info("Plan created: %s", plan.get("goal"))
        except Exception as e:
            logger.warning("Planning phase failed: %s — continuing without plan", e)
```

- [ ] **Step 2: Verify caller in api/routes.py still works (no change needed)**

Run:
```powershell
python -c "from backend.core.agent import generate_rednote; import inspect; print(inspect.signature(generate_rednote))"
```
Expected: `(request: backend.models.request.GenerateRequest, *, enable_planning: bool = True) -> ...`

- [ ] **Step 3: Commit**

```powershell
git add backend/core/agent.py
git commit -m "feat(agent): add enable_planning keyword (default True) for A/B experiment"
```

---

## Task 3: `test_memory_security.py` — SQL safety +越权读取

**Files:**
- Create: `backend/tests/test_memory_security.py`

3 cases: 2 PASS (parameterized SQL, injection neutralized) + 1 PROVE (越权读取).

- [ ] **Step 1: Write the test file**

Create `backend/tests/test_memory_security.py`:
```python
"""Q15 — Memory service security properties.

PASS tests confirm safe defaults. PROVE tests document the
unauthenticated-user-id flaw: anyone can read any history if they
know/guess the id. No authentication layer exists.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from backend.services import memory_service


MEMORY_SERVICE_SOURCE = Path(memory_service.__file__).read_text(encoding="utf-8")


def test_all_sql_uses_placeholders():
    """PASS: static scan — no f-string SQL, no % concatenation in execute()."""
    # Match db.execute("...{var}...") with f-string interpolation
    fstring_sql = re.search(
        r"""execute\(\s*f["']""", MEMORY_SERVICE_SOURCE
    )
    assert fstring_sql is None, (
        f"Found f-string in execute() — likely SQL injection: {fstring_sql.group(0)}"
    )

    # Match db.execute("..." % var) percent-formatting
    percent_sql = re.search(
        r"""execute\(\s*["'][^"']*["']\s*%\s*""", MEMORY_SERVICE_SOURCE
    )
    assert percent_sql is None, (
        f"Found %-formatted SQL in execute() — likely SQL injection: {percent_sql.group(0)}"
    )

    # Sanity: at least one parameterized execute exists with the ? placeholder
    assert "?" in MEMORY_SERVICE_SOURCE, "Expected parameterized queries with '?'"


async def test_sql_injection_payload_is_neutralized(tmp_memory_db):
    """PASS: a classic injection string is treated as a literal user_id."""
    payload = "' OR 1=1 --"
    # First, seed an unrelated user so the DB isn't empty
    await memory_service.save_copy_result("victim", "正常查询", "正常标题")

    # Inject — if the SQL were vulnerable, this would return victim's row
    result = await memory_service.recall_user_history(payload)
    assert "暂无历史记录" in result, (
        f"Injection payload returned data — SQL is NOT parameterized: {result!r}"
    )


async def test_arbitrary_user_id_reads_anyone_history(tmp_memory_db):
    """PROVE: any client that knows a user_id can read that user's history.
    There is no authentication. This is a real authorization vulnerability
    in production; here it is documented as expected current behavior.
    Recommended fix: server-issued signed session ids, or HMAC on user_id.
    """
    # Alice saves a private note
    await memory_service.save_copy_result("alice", "私密查询", "私密标题")

    # Mallory just guesses alice's id (or sniffs it from a previous request)
    stolen = await memory_service.recall_user_history("alice")

    assert "私密查询" in stolen, "Expected to read alice's history with no auth"
    assert "私密标题" in stolen, "Expected to read alice's title with no auth"
```

- [ ] **Step 2: Run the tests**

Run:
```powershell
pytest backend/tests/test_memory_security.py -v
```
Expected: 3 passed.

- [ ] **Step 3: Commit**

```powershell
git add backend/tests/test_memory_security.py
git commit -m "test: Q15 memory security (params + unauthenticated user_id PROVE)"
```

---

## Task 4: `test_rate_limit.py` — in-memory limit holes

**Files:**
- Create: `backend/tests/test_rate_limit.py`

3 cases: 1 PASS (window counting) + 2 PROVE (no IP eviction, multi-instance bypass).

- [ ] **Step 1: Write the test file**

Create `backend/tests/test_rate_limit.py`:
```python
"""Q17.1 — In-memory rate limiter limitations.

The current implementation in api/dependencies.py keys per client IP in
a module-level defaultdict. Two known holes are documented as PROVE
tests: stale IP keys never get evicted (memory leak), and a multi-pod
deployment splits the count per process (effective limit is N × pods).
"""
from __future__ import annotations

from collections import defaultdict
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from backend.api import dependencies
from backend.core.config import settings


def _make_request(ip: str = "1.2.3.4"):
    """Build a minimal fake Request object with .client.host."""
    req = MagicMock()
    req.client.host = ip
    return req


async def test_window_correctly_counts_in_window_requests(reset_rate_limit_dict):
    """PASS: N requests succeed, (N+1)-th raises 429."""
    req = _make_request()
    for _ in range(settings.RATE_LIMIT_PER_MINUTE):
        await dependencies.rate_limit(req)

    with pytest.raises(HTTPException) as exc_info:
        await dependencies.rate_limit(req)
    assert exc_info.value.status_code == 429


async def test_ip_key_never_evicted(monkeypatch, reset_rate_limit_dict):
    """PROVE: stale timestamps are dropped from the per-IP list, but the
    IP key itself stays in the dict forever. With 1M unique IPs over a
    day, this leaks roughly 1M dict entries.
    Recommended fix: periodic sweep, or use Redis with TTL.
    """
    import backend.api.dependencies as deps

    # Simulate 1000 unique IPs each making one request "at time 0"
    monkeypatch.setattr(deps, "time", lambda: 0.0)
    for i in range(1000):
        await deps.rate_limit(_make_request(f"10.0.{i // 256}.{i % 256}"))

    assert len(reset_rate_limit_dict) == 1000, "expected 1000 IPs seeded"

    # Advance time past the 60s window — all timestamps now stale
    monkeypatch.setattr(deps, "time", lambda: 9999.0)
    # Touch one more IP to trigger the per-IP filter (which only filters that key)
    await deps.rate_limit(_make_request("99.99.99.99"))

    # PROVE: even though every original timestamp is stale, the 1000
    # original IP keys are still sitting in the dict consuming memory.
    assert len(reset_rate_limit_dict) == 1001, (
        f"Expected leak: 1000 stale keys + 1 fresh = 1001. "
        f"Got {len(reset_rate_limit_dict)}"
    )


async def test_multi_instance_simulated(monkeypatch):
    """PROVE: with 2 pods behind a round-robin LB, the same IP can
    effectively double its rate. Each pod's dict only sees half the traffic.
    Recommended fix: external counter (Redis INCR + EXPIRE) or gateway layer.
    """
    import backend.api.dependencies as deps

    pod_a: dict[str, list[float]] = defaultdict(list)
    pod_b: dict[str, list[float]] = defaultdict(list)

    req = _make_request("attacker.ip")
    limit = settings.RATE_LIMIT_PER_MINUTE

    # Attacker hits pod A up to the limit — fine
    monkeypatch.setattr(deps, "_request_times", pod_a)
    for _ in range(limit):
        await deps.rate_limit(req)

    # ...then hits pod B up to the limit — also fine, no 429
    monkeypatch.setattr(deps, "_request_times", pod_b)
    for _ in range(limit):
        await deps.rate_limit(req)

    # PROVE: attacker pushed 2x the nominal limit without a single 429.
    assert len(pod_a["attacker.ip"]) == limit
    assert len(pod_b["attacker.ip"]) == limit
```

- [ ] **Step 2: Run the tests**

Run:
```powershell
pytest backend/tests/test_rate_limit.py -v
```
Expected: 3 passed.

- [ ] **Step 3: Commit**

```powershell
git add backend/tests/test_rate_limit.py
git commit -m "test: Q17.1 rate-limit window + IP-leak PROVE + multi-pod PROVE"
```

---

## Task 5: `test_rag_stability.py` — RAG determinism + LRU semantics

**Files:**
- Create: `backend/tests/test_rag_stability.py`

2 cases: 1 PASS (deterministic top-3) + 1 PROVE (LRU is exact-match, not semantic).

- [ ] **Step 1: Write the test file**

Create `backend/tests/test_rag_stability.py`:
```python
"""Q18.1 / Q13.1 — RAG determinism and LRU cache key semantics.

The LRU cache uses (query, top_k) as the key, which is exact-match.
A common confusion is to assume semantic-similarity caching: that is
documented here as a PROVE test.
"""
from __future__ import annotations

import pytest

from backend.services import rag_service


async def test_top3_deterministic_for_same_query():
    """PASS: with embeddings fixed, the same query returns the same top-K."""
    rag_service._sync_query.cache_clear()
    first = await rag_service.query_product_database("敏感肌补水")
    repeats = [await rag_service.query_product_database("敏感肌补水") for _ in range(4)]
    for r in repeats:
        assert r == first, "RAG top-K drifted across identical queries"


def test_lru_does_not_match_paraphrase():
    """PROVE: lru_cache keys are hashed args tuples — '敏感肌补水' and
    '敏感皮补水' are different strings, so the second is a cache miss.
    This contradicts the (wrong) intuition that LRU does semantic match.
    """
    rag_service._sync_query.cache_clear()

    rag_service._sync_query("敏感肌补水")
    misses_after_first = rag_service._sync_query.cache_info().misses
    assert misses_after_first == 1

    rag_service._sync_query("敏感皮补水")
    misses_after_second = rag_service._sync_query.cache_info().misses

    assert misses_after_second == 2, (
        "PROVE: paraphrased query was a cache miss — lru_cache is exact-match only. "
        f"misses went {misses_after_first} -> {misses_after_second}"
    )
```

- [ ] **Step 2: Run the tests**

Run:
```powershell
pytest backend/tests/test_rag_stability.py -v
```
Expected: 2 passed. First test may take a few seconds (loads embedding model on first import).

- [ ] **Step 3: Commit**

```powershell
git add backend/tests/test_rag_stability.py
git commit -m "test: Q18.1 RAG determinism + LRU paraphrase PROVE"
```

---

## Task 6: `test_asyncio_gather.py` — gather/wait_for semantics

**Files:**
- Create: `backend/tests/test_asyncio_gather.py`

2 cases: 1 PROVE (default gather propagates first exception, which agent.py:232 inherits) + 1 PASS (wait_for cancels only its own task).

- [ ] **Step 1: Write the test file**

Create `backend/tests/test_asyncio_gather.py`:
```python
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
```

- [ ] **Step 2: Run the tests**

Run:
```powershell
pytest backend/tests/test_asyncio_gather.py -v
```
Expected: 2 passed.

- [ ] **Step 3: Commit**

```powershell
git add backend/tests/test_asyncio_gather.py
git commit -m "test: Q14 gather propagation PROVE + wait_for isolation PASS"
```

---

## Task 7: `test_best_draft.py` — reflection fallback logic

**Files:**
- Create: `backend/tests/test_best_draft.py`

3 cases: 1 PASS (best tracked correctly) + 1 PROVE (third draft swallowed) + 1 PASS (no best → fall back to current).

- [ ] **Step 1: Write the test file**

Create `backend/tests/test_best_draft.py`:
```python
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
```

- [ ] **Step 2: Run the tests**

Run:
```powershell
pytest backend/tests/test_best_draft.py -v
```
Expected: 3 passed.

- [ ] **Step 3: Commit**

```powershell
git add backend/tests/test_best_draft.py
git commit -m "test: Q18.1 best_draft tracking + third-draft swallowed PROVE"
```

---

## Task 8: `test_agent_flow.py` — main loop behavior

**Files:**
- Create: `backend/tests/test_agent_flow.py`

4 cases: event sequence, parallel tool exec, per-request tool cache, JSON retry.

- [ ] **Step 1: Write the test file**

Create `backend/tests/test_agent_flow.py`:
```python
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
```

- [ ] **Step 2: Run the tests**

Run:
```powershell
pytest backend/tests/test_agent_flow.py -v
```
Expected: 4 passed.

- [ ] **Step 3: Run the entire test suite to confirm no regressions**

Run:
```powershell
pytest backend/tests -v
```
Expected: 17 passed.

- [ ] **Step 4: Commit**

```powershell
git add backend/tests/test_agent_flow.py
git commit -m "test: Q18.1 agent main flow — sequence, parallelism, cache, JSON retry"
```

---

## Task 9: A/B query set + experiment script

**Files:**
- Create: `scripts/__init__.py` (empty)
- Create: `scripts/ab_queries.json`
- Create: `scripts/ab_planning_experiment.py`

- [ ] **Step 1: Create scripts package marker**

Create `scripts/__init__.py` (empty file).

- [ ] **Step 2: Create the query set**

Create `scripts/ab_queries.json` with 30 hand-picked Chinese skincare queries balanced across 6 categories (5 each):
```json
{
  "version": 1,
  "categories": ["补水", "控油", "抗老", "防晒", "美白", "敏感修复"],
  "queries": [
    {"id": 1,  "category": "补水",     "query": "干皮秋冬深层补水的精华推荐"},
    {"id": 2,  "category": "补水",     "query": "学生党平价补水面膜测评"},
    {"id": 3,  "category": "补水",     "query": "熬夜后急救补水方案"},
    {"id": 4,  "category": "补水",     "query": "敏感肌可用的玻尿酸产品"},
    {"id": 5,  "category": "补水",     "query": "空调房通勤族保湿喷雾推荐"},

    {"id": 6,  "category": "控油",     "query": "夏季油皮控油散粉测评"},
    {"id": 7,  "category": "控油",     "query": "T区出油严重洗面奶推荐"},
    {"id": 8,  "category": "控油",     "query": "混油皮控油不闷痘的乳液"},
    {"id": 9,  "category": "控油",     "query": "学生党控油爆款合集"},
    {"id": 10, "category": "控油",     "query": "运动健身后控油急救"},

    {"id": 11, "category": "抗老",     "query": "25+早C晚A入门方案"},
    {"id": 12, "category": "抗老",     "query": "眼周细纹抗老精华推荐"},
    {"id": 13, "category": "抗老",     "query": "胜肽抗老面霜真实测评"},
    {"id": 14, "category": "抗老",     "query": "30+熟龄肌抗氧化护肤"},
    {"id": 15, "category": "抗老",     "query": "学生党预防型抗老入门"},

    {"id": 16, "category": "防晒",     "query": "敏感肌物理防晒推荐"},
    {"id": 17, "category": "防晒",     "query": "通勤党不假白防晒霜"},
    {"id": 18, "category": "防晒",     "query": "户外旅行高倍防晒"},
    {"id": 19, "category": "防晒",     "query": "油皮控油防晒一次满足"},
    {"id": 20, "category": "防晒",     "query": "孕妇可用的温和防晒"},

    {"id": 21, "category": "美白",     "query": "377美白精华使用心得"},
    {"id": 22, "category": "美白",     "query": "熬夜暗沉急救美白"},
    {"id": 23, "category": "美白",     "query": "晒后修护美白方案"},
    {"id": 24, "category": "美白",     "query": "美白淡斑早期入门"},
    {"id": 25, "category": "美白",     "query": "全身美白乳液推荐"},

    {"id": 26, "category": "敏感修复", "query": "屏障受损泛红修护乳"},
    {"id": 27, "category": "敏感修复", "query": "刷酸翻车后的舒缓方案"},
    {"id": 28, "category": "敏感修复", "query": "换季敏感肌急救面膜"},
    {"id": 29, "category": "敏感修复", "query": "B5修护精华种草"},
    {"id": 30, "category": "敏感修复", "query": "屏障修复温和洁面"}
  ]
}
```

- [ ] **Step 3: Create the experiment runner**

Create `scripts/ab_planning_experiment.py`:
```python
"""A/B experiment: does _planning_phase improve agent output enough
to justify its extra cost?

Paired design: each query is run once with planning ON (arm A) and
once with planning OFF (arm B), in randomized order. Captures:
- reflection scores (post-hoc, applied to final draft)
- token usage (prompt + completion + total)
- iteration count and wall-clock time

Incremental JSONL output makes the run resumable.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.core.agent import generate_rednote, _reflection_phase
from backend.models.request import GenerateRequest, Tone
from backend.models.response import GenerateResponse
from backend.services.deepseek_client import get_client

QUERY_FILE = ROOT / "scripts" / "ab_queries.json"
DEFAULT_OUT_JSONL = ROOT / "scripts" / "ab_results.jsonl"
DEFAULT_REPORT_MD = ROOT / "docs" / "testing" / "2026-05-18-ab-report.md"


def _empty_record(query_id: int, query: str, arm: str) -> dict:
    return {
        "query_id": query_id,
        "query": query,
        "arm": arm,
        "success": False,
        "error": None,
        "reflection_min_score": None,
        "reflection_virality": None,
        "reflection_tone": None,
        "reflection_accuracy": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "iterations": 0,
        "wall_clock_seconds": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def _run_one(query_id: int, query: str, arm: str) -> dict:
    record = _empty_record(query_id, query, arm)
    enable_planning = (arm == "A")
    final_draft: dict | None = None

    t0 = time.monotonic()
    try:
        req = GenerateRequest(query=query, tone=Tone.lively, max_iterations=5)
        async for ev in generate_rednote(req, enable_planning=enable_planning):
            if ev["event"] == "agent_thinking":
                it = ev["data"].get("iteration")
                if isinstance(it, int):
                    record["iterations"] = max(record["iterations"], it)
            elif ev["event"] == "complete":
                final_draft = ev["data"]
                record["success"] = True
            elif ev["event"] == "token_usage":
                record["prompt_tokens"] = ev["data"]["prompt_tokens"]
                record["completion_tokens"] = ev["data"]["completion_tokens"]
                record["total_tokens"] = ev["data"]["total_tokens"]
            elif ev["event"] == "error":
                record["error"] = ev["data"].get("message")
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"

    record["wall_clock_seconds"] = round(time.monotonic() - t0, 3)

    if final_draft is not None:
        try:
            draft_obj = GenerateResponse.model_validate(final_draft)
            critique = await _reflection_phase(get_client(), draft_obj)
            record["reflection_min_score"] = critique.get("min_score")
            record["reflection_virality"] = critique.get("virality_score")
            record["reflection_tone"] = critique.get("tone_match_score")
            record["reflection_accuracy"] = critique.get("accuracy_score")
        except Exception as e:
            record["error"] = (record["error"] or "") + f" | post-hoc reflection: {e}"

    return record


def _load_done_pairs(jsonl_path: Path) -> set[tuple[int, str]]:
    if not jsonl_path.exists():
        return set()
    done: set[tuple[int, str]] = set()
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("success"):
                    done.add((rec["query_id"], rec["arm"]))
            except json.JSONDecodeError:
                continue
    return done


def _build_run_plan(queries: list[dict], n: int, done: set[tuple[int, str]],
                    seed: int) -> list[tuple[int, str, str]]:
    """Return [(query_id, query, arm)] in randomized within-pair order."""
    rng = random.Random(seed)
    plan: list[tuple[int, str, str]] = []
    for q in queries[:n]:
        arms = ["A", "B"]
        rng.shuffle(arms)
        for arm in arms:
            if (q["id"], arm) not in done:
                plan.append((q["id"], q["query"], arm))
    return plan


async def _run_experiment(n: int, jsonl_out: Path, seed: int) -> list[dict]:
    queries = json.loads(QUERY_FILE.read_text(encoding="utf-8"))["queries"]
    done = _load_done_pairs(jsonl_out)
    plan = _build_run_plan(queries, n, done, seed)

    print(f"[ab] {len(plan)} runs to execute "
          f"({len(done)} already done, {n * 2 - len(done) - len(plan)} skipped)")
    print(f"[ab] streaming results to {jsonl_out}")

    jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []

    for i, (qid, qtext, arm) in enumerate(plan, 1):
        print(f"[ab] {i}/{len(plan)}  qid={qid:>2}  arm={arm}  '{qtext[:30]}...'", flush=True)
        rec = await _run_one(qid, qtext, arm)
        with jsonl_out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        all_records.append(rec)
        status = "ok " if rec["success"] else "FAIL"
        score = rec["reflection_min_score"]
        tokens = rec["total_tokens"]
        print(f"        -> {status} score={score} tokens={tokens} "
              f"iter={rec['iterations']} t={rec['wall_clock_seconds']}s")

    return all_records


def _summarize(jsonl_path: Path) -> dict:
    """Read all records and compute paired comparison summary."""
    records: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    by_pair: dict[int, dict[str, dict]] = {}
    for r in records:
        by_pair.setdefault(r["query_id"], {})[r["arm"]] = r

    paired = [(qid, p["A"], p["B"]) for qid, p in by_pair.items()
              if "A" in p and "B" in p and p["A"]["success"] and p["B"]["success"]]

    def arm_stat(field: str, arm: str) -> tuple[float | None, float | None]:
        vals = [r.get(field) for r in records if r["arm"] == arm]
        vals = [v for v in vals if v is not None]
        if not vals:
            return (None, None)
        return (round(mean(vals), 3), round(stdev(vals), 3) if len(vals) > 1 else 0.0)

    summary = {
        "total_records": len(records),
        "paired_count": len(paired),
        "successes_A": sum(1 for r in records if r["arm"] == "A" and r["success"]),
        "successes_B": sum(1 for r in records if r["arm"] == "B" and r["success"]),
        "failures": [r for r in records if not r["success"]],
    }

    for field in ("reflection_min_score", "total_tokens", "iterations",
                  "wall_clock_seconds"):
        for arm in ("A", "B"):
            m, sd = arm_stat(field, arm)
            summary[f"{field}_{arm}_mean"] = m
            summary[f"{field}_{arm}_stdev"] = sd

    def paired_diff(field: str) -> dict:
        diffs = [(qid, a[field], b[field], a[field] - b[field])
                 for qid, a, b in paired
                 if a[field] is not None and b[field] is not None]
        if not diffs:
            return {"n": 0, "mean_diff": None, "a_wins": 0, "ties": 0, "b_wins": 0}
        return {
            "n": len(diffs),
            "mean_diff": round(mean(d[3] for d in diffs), 3),
            "a_wins": sum(1 for d in diffs if d[3] > 0),
            "ties":   sum(1 for d in diffs if d[3] == 0),
            "b_wins": sum(1 for d in diffs if d[3] < 0),
            "per_query": diffs,
        }

    summary["paired_score"]      = paired_diff("reflection_min_score")
    summary["paired_tokens"]     = paired_diff("total_tokens")
    summary["paired_iterations"] = paired_diff("iterations")
    summary["paired_wall_clock"] = paired_diff("wall_clock_seconds")
    return summary


def _write_report(summary: dict, report_path: Path, n: int) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(v):
        return "—" if v is None else v

    lines: list[str] = []
    lines.append(f"# A/B 实验报告：Planning 阶段是否值得？")
    lines.append("")
    lines.append(f"**生成时间：** {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"**设计：** 配对设计，N={n} 条 query，每条在 A（planning=ON）"
                 f"和 B（planning=OFF）各跑一次。")
    lines.append(f"**实际数据：** 总记录 {summary['total_records']}，"
                 f"成功配对 {summary['paired_count']}，"
                 f"A 成功 {summary['successes_A']}，B 成功 {summary['successes_B']}。")
    lines.append("")
    lines.append("## 汇总指标")
    lines.append("")
    lines.append("| 指标 | A (planning) mean ± sd | B (no planning) mean ± sd |")
    lines.append("|---|---|---|")
    for field, label in [
        ("reflection_min_score", "Reflection 最低分"),
        ("total_tokens", "总 tokens"),
        ("iterations", "迭代数"),
        ("wall_clock_seconds", "墙钟秒数"),
    ]:
        lines.append(
            f"| {label} | {fmt(summary[f'{field}_A_mean'])} ± "
            f"{fmt(summary[f'{field}_A_stdev'])} | "
            f"{fmt(summary[f'{field}_B_mean'])} ± "
            f"{fmt(summary[f'{field}_B_stdev'])} |"
        )
    lines.append("")
    lines.append("## 配对差值（A − B，正数 = A 更高）")
    lines.append("")
    lines.append("| 指标 | n | 配对差均值 | A 胜 | 平 | B 胜 |")
    lines.append("|---|---|---|---|---|---|")
    for key, label in [
        ("paired_score", "Reflection 最低分"),
        ("paired_tokens", "总 tokens"),
        ("paired_iterations", "迭代数"),
        ("paired_wall_clock", "墙钟秒数"),
    ]:
        d = summary[key]
        lines.append(
            f"| {label} | {d['n']} | {fmt(d['mean_diff'])} | "
            f"{d['a_wins']} | {d['ties']} | {d['b_wins']} |"
        )
    lines.append("")

    failures = summary["failures"]
    if failures:
        lines.append("## 失败案例")
        lines.append("")
        for f in failures:
            lines.append(f"- qid={f['query_id']} arm={f['arm']} — {f.get('error')}")
        lines.append("")

    lines.append("## 诚实结论段")
    lines.append("")
    lines.append("**已知局限：**")
    lines.append("")
    lines.append("- **Self-eval bias：** 评分用的是同一个 DeepSeek 模型做 reflection，"
                 "存在自评偏高的可能。本报告不宣称\"质量\"差异，只宣称\"系统自评分\"差异。")
    lines.append(f"- **样本量 {n} 条 query** 不足以做严格的显著性检验。本报告只报"
                 "趋势、配对差均值、胜率，不报 p-value。")
    lines.append("- **单评估源：** 没有引入独立 judge 模型，因此无法消除评估偏差。")
    lines.append("- **Query 集人工编写：** 不能外推到所有用户场景。")
    lines.append("")
    lines.append("**怎么看这份数据：**")
    lines.append("")
    lines.append("- 如果 A 路 reflection 分均值 > B 路且配对胜率 > 60% → planning 有正向效果，"
                 "值得保留；")
    lines.append("- 如果 A 路总 tokens 显著高于 B 路且 reflection 分无优势 → planning 是纯"
                 "成本，建议移除；")
    lines.append("- 如果两路指标接近 → planning 价值未证，下一步应该用更大样本 + "
                 "外部 judge 重测。")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ab] report written to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Planning A/B experiment.")
    parser.add_argument("--n", type=int, default=30,
                        help="Number of queries (default 30)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use only 3 queries; skip cost check")
    parser.add_argument("--confirm", action="store_true",
                        help="Required if --n > 10")
    parser.add_argument("--resume", action="store_true",
                        help="Skip query/arm pairs already present in --out")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_JSONL,
                        help="JSONL output file")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_MD,
                        help="Markdown report file")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for arm-order shuffling")
    args = parser.parse_args()

    if args.dry_run:
        n = 3
    else:
        n = args.n
        if n > 10 and not args.confirm:
            print(f"refusing to run {n} queries × 2 arms without --confirm "
                  f"(this will cost real DeepSeek tokens, est. ${n * 0.06:.2f}-${n * 0.10:.2f})")
            sys.exit(1)

    if not args.resume and args.out.exists():
        args.out.unlink()

    print(f"[ab] starting: n={n}, planning A on, B off, paired")
    print(f"[ab] estimated cost: ${n * 0.06:.2f}-${n * 0.10:.2f} USD")
    asyncio.run(_run_experiment(n, args.out, args.seed))
    summary = _summarize(args.out)
    _write_report(summary, args.report, n)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Sanity-check the script's argparse**

Run:
```powershell
python scripts/ab_planning_experiment.py --help
```
Expected: usage block listing all flags.

- [ ] **Step 5: Commit**

```powershell
git add scripts/__init__.py scripts/ab_queries.json scripts/ab_planning_experiment.py
git commit -m "feat(scripts): A/B planning experiment runner + 30-query set"
```

---

## Task 10: Dry-run the A/B script

**Files:**
- Output: `scripts/ab_results.jsonl` (gitignored or kept as reference)
- Output: `docs/testing/2026-05-18-ab-report.md` (partial — dry-run only)

- [ ] **Step 1: Confirm DEEPSEEK_API_KEY is set**

Run:
```powershell
$env:DEEPSEEK_API_KEY
```
If empty: stop. Get the key from the project's `.env` (or set it in env) before continuing. The dry-run will fail otherwise.

- [ ] **Step 2: Execute dry-run (3 queries × 2 arms = 6 runs, ~$0.20)**

Run:
```powershell
python scripts/ab_planning_experiment.py --dry-run --out scripts/ab_dry.jsonl --report docs/testing/2026-05-18-ab-dryrun.md
```
Expected:
- 6 progress lines, each `ok score=X tokens=Y iter=Z t=Ns`
- Final `[ab] report written to ...`

If any run fails with API error: investigate the API connectivity / key before proceeding to the full run.

- [ ] **Step 3: Eyeball the dry-run report**

Read `docs/testing/2026-05-18-ab-dryrun.md`. Confirm:
- Tables render
- Numbers look plausible (e.g. reflection scores in 6–10 range, tokens in 1000–10000 range)
- 失败案例 section either empty or lists honest errors

- [ ] **Step 4: Delete dry-run artifacts (we'll regenerate in the full run)**

Run:
```powershell
Remove-Item scripts/ab_dry.jsonl, docs/testing/2026-05-18-ab-dryrun.md
```

- [ ] **Step 5: Commit (no artifacts to commit; just a checkpoint message)**

Run:
```powershell
git status
```
If clean: no commit needed. Proceed.

---

## Task 11: Full A/B run

**Files:**
- Output: `scripts/ab_results.jsonl`
- Output: `docs/testing/2026-05-18-ab-report.md`

- [ ] **Step 1: Execute full run (30 queries × 2 arms = 60 runs, ~$1.80–3.00)**

Run:
```powershell
python scripts/ab_planning_experiment.py --n 30 --confirm
```
Expected: ~15-30 minutes wall-clock. 60 progress lines. Final report file written.

If interrupted (Ctrl+C or crash):

Run:
```powershell
python scripts/ab_planning_experiment.py --n 30 --confirm --resume
```
to continue from where it stopped.

- [ ] **Step 2: Verify the artifacts**

Run:
```powershell
(Get-Content scripts/ab_results.jsonl | Measure-Object -Line).Lines
```
Expected: 60 (or close; failed runs are still logged).

Open `docs/testing/2026-05-18-ab-report.md` and confirm:
- All 4 summary metrics populated
- Paired-diff table has n=30 (or close)
- 诚实结论段 present

- [ ] **Step 3: Commit results and report**

```powershell
git add scripts/ab_results.jsonl docs/testing/2026-05-18-ab-report.md
git commit -m "experiment: full A/B planning run (n=30 paired), report + raw jsonl"
```

---

## Task 12: Final consolidated results document

**Files:**
- Create: `docs/testing/2026-05-18-results.md`

- [ ] **Step 1: Run the full pytest suite one last time to capture fresh pass/fail**

Run:
```powershell
pytest backend/tests -v --tb=short | Out-File -Encoding utf8 scripts/pytest_output.txt
```
Expected: 17 passed.

- [ ] **Step 2: Read the A/B report so its numbers are ready to summarize**

Read `docs/testing/2026-05-18-ab-report.md`. Note the four key numbers (mean reflection score A vs B, mean tokens A vs B, A-wins count, B-wins count).

- [ ] **Step 3: Write the consolidated final results doc**

Create `docs/testing/2026-05-18-results.md`. Use this exact structure (fill `<…>` placeholders with real numbers from steps 1 and 2):

```markdown
# rednote_agent 测试结果报告

**日期：** 2026-05-18
**Spec：** `docs/superpowers/specs/2026-05-18-test-suite-design.md`
**Plan：** `docs/superpowers/plans/2026-05-18-test-suite.md`

## 概览

- **离线测试：** 17 用例，全部 pass。其中 6 个 PROVE 类用例"通过"实际表示
  现有 buggy 行为已被记录在案。
- **A/B 实验：** 30 对（60 次完整 agent 调用）配对设计，planning ON vs OFF。

## 离线测试结果（pytest）

| 文件 | 用例数 | PASS | PROVE | 备注 |
|---|---|---|---|---|
| test_agent_flow.py | 4 | 4 | 0 | 事件序列 / 并行 / 缓存 / JSON 重试 |
| test_best_draft.py | 3 | 2 | 1 | 第 3 稿被 best_draft 吃掉 |
| test_rag_stability.py | 2 | 1 | 1 | LRU 不语义匹配 |
| test_asyncio_gather.py | 2 | 1 | 1 | 默认 gather 异常冲掘全路 |
| test_memory_security.py | 3 | 2 | 1 | user_id 越权读取 |
| test_rate_limit.py | 3 | 1 | 2 | IP key 不回收；多 pod 失效 |

完整 pytest 输出见 `scripts/pytest_output.txt`。

## 发现的 Bug 与建议修复

### Bug 1：第 3 次草稿被 best_draft 吃掉（agent.py:312）

**症状：** 当 `_reflection_count` 已达 `MAX_REFLECTIONS` 时，下一次草稿被解析、
被验证，然后被 `final_result = best_draft if best_draft is not None else result`
丢弃，返回的是更早评分最高的稿。
**触发证明：** `test_best_draft.py::test_third_draft_swallowed_by_best_draft`
**建议修复：** 二选一
- A) 给最后一次草稿一次反思机会：
  `if _reflection_count < settings.MAX_REFLECTIONS or best_draft is None:`
- B) 在比较条件里实际比对分数：
  `final_result = result if score_of_current >= best_score else best_draft`

### Bug 2：asyncio.gather 默认 propagation（agent.py:232）

**症状：** 5 个 tool 并行，1 个抛错，整个请求废。
**触发证明：** `test_asyncio_gather.py::test_gather_propagates_first_exception`
**建议修复：** `await asyncio.gather(..., return_exceptions=True)`，然后逐个判
断 result 是否是 Exception 实例并降级处理。

### Bug 3：限流 IP key 内存泄漏（dependencies.py:11-27）

**症状：** 即便所有时间戳过期，IP 字典 key 也永远不删。100 万 IP = 100 万 key
驻留。
**触发证明：** `test_rate_limit.py::test_ip_key_never_evicted`
**建议修复：** 周期性 sweep 删空列表的 key；或改 Redis + EXPIRE。

### Bug 4：限流多 pod 失效（dependencies.py 设计层面）

**症状：** 进程内 dict 在多实例部署下被分片，攻击者每加一个 pod 多一倍可用配额。
**触发证明：** `test_rate_limit.py::test_multi_instance_simulated`
**建议修复：** Redis INCR + EXPIRE，或在网关层（nginx / API gateway）做限流。

### Bug 5（安全）：user_id 无鉴权（memory_service.py + models/request.py:20）

**症状：** 客户端任意传 user_id 即可读他人历史。
**触发证明：** `test_memory_security.py::test_arbitrary_user_id_reads_anyone_history`
**简历表述修正：** "persistent per-user memory" → "session-scoped memory
keyed by client-provided identifier"。
**建议修复（最小成本）：** 服务端签名 cookie 或 HMAC(user_id)。

### Bug 6（认知）：LRU cache 不语义匹配（rag_service.py:40）

**症状：** 同义改写不命中。`"敏感肌补水"` vs `"敏感皮补水"` 是两次 miss。
**触发证明：** `test_rag_stability.py::test_lru_does_not_match_paraphrase`
**建议修复：** 如需语义缓存，引入 embedding-based cache（向量近邻 + 阈值）。
但实际场景下命中率本身就低，**最简单的修复是删掉 RAG LRU，仅保留 agent.py 内
的 per-request _tool_cache**——两层 cache 本来就冗余。

## A/B 实验结果

完整报告：`docs/testing/2026-05-18-ab-report.md`

**关键数字：**

- A 路 reflection 最低分均值：`<填入>`，B 路：`<填入>`，差值：`<填入>`
- A 路总 tokens 均值：`<填入>`，B 路：`<填入>`，差值：`<填入>`
- A 路迭代数均值：`<填入>`，B 路：`<填入>`
- 30 对里 A 优于 B 的数量：`<填入>`，B 优于 A：`<填入>`，平局：`<填入>`

**结论（一句话）：** `<根据数据三选一>`
- 若 A 反思分显著高且 tokens 增幅可接受 → planning 有价值，保留。
- 若 A 反思分无优势但 tokens 多 ~30% → planning 是纯成本，建议移除。
- 若双方接近 → planning 价值未证，需要更大样本 + 独立 judge 重测。

**已知局限：**

- Self-eval bias（同一模型评分）
- N=30 不足以做严格显著性检验
- 单评估源、无独立 judge
- Query 集人工编写，不能外推

## 复盘：从这次测试得到的工程教训

1. **best_draft 的 fallback 比较条件没想清楚。** 注释写 "fall back to current"
   但代码实际反过来。新人很容易掉这个坑——comment 和 code 漂移。
2. **测试是验证设计假设的最便宜手段。** Q19.1 的 A/B 实验只花了 $X 就让"planning
   该不该留"这个问题从感觉变成数字。
3. **PROVE 类测试是面试财富。** 把 bug 写成 xfail 风格的"故意通过"测试，
   能在面试里直接说"我发现的、我证明的、我会怎么修"——比口头描述强 10 倍。

## 复现方法

```powershell
# 离线测试
pip install -r backend/requirements-dev.txt
pytest backend/tests -v

# A/B 实验（约 $2 USD）
python scripts/ab_planning_experiment.py --n 30 --confirm

# 仅 dry-run 验证管道（约 $0.20）
python scripts/ab_planning_experiment.py --dry-run
```
```

- [ ] **Step 4: Fill in the `<…>` placeholders with real numbers**

Open `docs/testing/2026-05-18-results.md`. Replace every `<填入>` and the one-line conclusion `<根据数据三选一>` based on what `docs/testing/2026-05-18-ab-report.md` actually shows.

- [ ] **Step 5: Verify no placeholders remain**

Run:
```powershell
Select-String -Path docs/testing/2026-05-18-results.md -Pattern "<填入>|<根据数据"
```
Expected: no matches.

- [ ] **Step 6: Commit**

```powershell
git add docs/testing/2026-05-18-results.md scripts/pytest_output.txt
git commit -m "docs: consolidated test + A/B results for 2026-05-18 run"
```

---

## Done

After Task 12, the deliverables are:

- **Code:** `backend/tests/*.py` (6 files, 17 tests) + `scripts/ab_planning_experiment.py` + `scripts/ab_queries.json` + one keyword-arg added to `generate_rednote`.
- **Docs:** `docs/superpowers/specs/2026-05-18-test-suite-design.md` (already exists), `docs/superpowers/plans/2026-05-18-test-suite.md` (this file), `docs/testing/2026-05-18-ab-report.md` (auto-generated), `docs/testing/2026-05-18-results.md` (consolidated final report).
- **Raw data:** `scripts/ab_results.jsonl` (60 records), `scripts/pytest_output.txt`.

Confirm everything is committed:

```powershell
git log --oneline -15
git status
```

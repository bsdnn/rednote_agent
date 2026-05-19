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

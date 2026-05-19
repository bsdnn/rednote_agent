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

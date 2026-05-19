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

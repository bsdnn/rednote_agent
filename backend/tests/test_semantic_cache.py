import asyncio
import time
import numpy as np
import pytest
from backend.services.rag.cache.semantic_cache import SemanticCache


@pytest.fixture
def aligned_embed():
    """补水 / 保湿 → near-identical vectors; 面膜 → orthogonal."""
    map_ = {
        "补水产品": np.array([1.0, 0.0, 0.0], dtype="float32"),
        "保湿产品": np.array([0.98, 0.198, 0.0], dtype="float32"),  # cosine ~0.98
        "面膜推荐": np.array([0.0, 1.0, 0.0], dtype="float32"),
    }
    def _e(text):
        if text in map_:
            return map_[text]
        # Distinct near-orthogonal vectors per unmapped key (each character
        # contributes on its own dimension), so LRU eviction is not masked
        # by semantic-hit collisions on a shared fallback vector.
        dim = 64
        v = np.zeros(dim, dtype="float32")
        for i, ch in enumerate(text):
            v[ord(ch) % dim] += 1.0
            v[(ord(ch) * 31 + i) % dim] += 0.5
        return v
    return _e


def _run(coro): return asyncio.get_event_loop().run_until_complete(coro)


def test_exact_hit(aligned_embed):
    c = SemanticCache(embedding_fn=aligned_embed, threshold=0.9)
    _run(c.set("补水产品", "RESULT_A"))
    assert _run(c.get("补水产品")) == "RESULT_A"
    assert c.stats()["exact_hits"] == 1


def test_semantic_hit_flips_v1_PROVE_bug(aligned_embed):
    # v1 PROVE: test_lru_paraphrase_is_miss — this is the v2 PASS counterpart
    c = SemanticCache(embedding_fn=aligned_embed, threshold=0.9)
    _run(c.set("补水产品", "RESULT_A"))
    assert _run(c.get("保湿产品")) == "RESULT_A"
    assert c.stats()["semantic_hits"] == 1


def test_below_threshold_is_miss(aligned_embed):
    c = SemanticCache(embedding_fn=aligned_embed, threshold=0.9)
    _run(c.set("补水产品", "RESULT_A"))
    assert _run(c.get("面膜推荐")) is None
    assert c.stats()["misses"] == 1


def test_ttl_eviction(aligned_embed):
    c = SemanticCache(embedding_fn=aligned_embed, threshold=0.9, ttl_seconds=0)
    _run(c.set("补水产品", "RESULT_A"))
    time.sleep(0.01)
    assert _run(c.get("补水产品")) is None
    assert c.stats()["evictions_ttl"] >= 1


def test_lru_eviction(aligned_embed):
    c = SemanticCache(embedding_fn=aligned_embed, threshold=0.9, max_size=2)
    _run(c.set("a", "1"))
    _run(c.set("b", "2"))
    _run(c.set("c", "3"))  # should evict "a"
    assert _run(c.get("a")) is None
    assert c.stats()["evictions_lru"] >= 1

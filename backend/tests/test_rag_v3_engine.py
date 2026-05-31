"""Integration tests for the LlamaIndex RAG v3 engine.

These tests are SLOW (~30-60s for first run due to model load).
They build a tiny in-process index from synthetic docs and verify
shape — node count, doc_id format, score monotonicity — NOT exact text.
"""
import pytest
import json
from pathlib import Path

from backend.models.persona import UserPersona


@pytest.fixture(scope="module")
def tiny_corpus(tmp_path_factory):
    """5-doc fixture: 2 products, 1 ingredient, 2 posts."""
    d = tmp_path_factory.mktemp("tiny_corpus")
    (d / "corpus").mkdir()

    (d / "products.json").write_text(json.dumps([
        {"id": 1, "name": "美白精华A", "category": "brightening",
         "description": "含烟酰胺，专门控油提亮",
         "key_ingredients": ["niacinamide"], "suitable_skin_types": ["oily"],
         "price_tier": "mid-range", "selling_points": ["美白"]},
        {"id": 2, "name": "保湿乳液B", "category": "hydrating",
         "description": "玻尿酸三重补水",
         "key_ingredients": ["hyaluronic_acid"], "suitable_skin_types": ["dry"],
         "price_tier": "budget", "selling_points": ["保湿"]},
    ], ensure_ascii=False), encoding="utf-8")

    (d / "corpus" / "ingredients.json").write_text(json.dumps([
        {"slug": "niacinamide", "name": "烟酰胺",
         "content": "烟酰胺是经典的美白成分，常见浓度2-5%。",
         "effects": ["美白", "控油"], "concerns": [], "age_groups": ["25-30"]},
    ], ensure_ascii=False), encoding="utf-8")

    (d / "corpus" / "posts.json").write_text(json.dumps([
        {"post_id": 1, "title": "油皮控油精华推荐",
         "content": "上班族下午脸油到不行，试试含烟酰胺的精华，一周见效！",
         "tone": "活泼甜美", "linked_products": ["美白精华A"]},
        {"post_id": 2, "title": "干皮急救面霜",
         "content": "干皮姐妹必看，含玻尿酸的保湿乳真的好用。",
         "tone": "温柔治愈", "linked_products": ["保湿乳液B"]},
    ], ensure_ascii=False), encoding="utf-8")
    return d


@pytest.fixture(scope="module")
def patched_engine(tiny_corpus):
    """Build an engine pointing at the tiny corpus instead of real data."""
    from backend.services.rag import engine as engine_mod
    original_data_dir = engine_mod._data_dir
    engine_mod._data_dir = lambda: tiny_corpus
    engine_mod._engine = None  # force rebuild against tiny corpus
    e = engine_mod.get_engine()
    yield e
    # Restore so subsequent test modules don't inherit the patched corpus
    engine_mod._data_dir = original_data_dir
    engine_mod._engine = None


@pytest.mark.asyncio
async def test_query_returns_source_nodes(patched_engine):
    resp = await patched_engine.aquery("含烟酰胺的精华")
    assert len(resp.source_nodes) > 0
    assert len(resp.source_nodes) <= 9   # reranker top_n=9


@pytest.mark.asyncio
async def test_node_ids_match_doc_id_prefix(patched_engine):
    resp = await patched_engine.aquery("烟酰胺")
    for nws in resp.source_nodes:
        assert "::" in nws.node.node_id
        prefix = nws.node.node_id.split("::")[0]
        assert prefix.startswith(("prod_", "ing_", "post_"))


@pytest.mark.asyncio
async def test_scores_are_monotonic(patched_engine):
    resp = await patched_engine.aquery("美白")
    scores = [nws.score for nws in resp.source_nodes if nws.score is not None]
    assert scores == sorted(scores, reverse=True), "results should be sorted by score desc"


@pytest.mark.asyncio
async def test_persona_filter_drops_mismatched_product(patched_engine):
    """With persona skin_type=dry, the 'oily-only' product should NOT appear in top 3."""
    from backend.services.rag._context import current_persona
    persona = UserPersona(skin_type="dry", budget="mid-range")
    tok = current_persona.set(persona)
    try:
        resp = await patched_engine.aquery("精华")
    finally:
        current_persona.reset(tok)
    ids = [nws.node.node_id.split("::")[0] for nws in resp.source_nodes[:3]]
    # prod_1 is for "oily" — should be filtered out
    assert "prod_1" not in ids


@pytest.mark.asyncio
async def test_no_persona_does_not_filter(patched_engine):
    """Without persona, products from any skin_type can appear."""
    from backend.services.rag._context import current_persona
    # Ensure no leftover persona
    tok = current_persona.set(None)
    try:
        resp = await patched_engine.aquery("精华推荐")
    finally:
        current_persona.reset(tok)
    assert len(resp.source_nodes) >= 1

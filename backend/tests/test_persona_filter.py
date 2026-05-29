import pytest
from backend.models.persona import UserPersona
from backend.services.rag.corpus.schema import Chunk
from backend.services.rag.retrievers.vector import RetrievalHit
from backend.services.rag.persona_filter import hard_filter, soft_boost, BUDGET_TIER_RULES


def _product(cid, skin_types, price_tier, effects=None, age_groups=None):
    return Chunk(chunk_id=cid, doc_id=cid, doc_type="product", text="",
                 metadata={"doc_type": "product", "source": "internal",
                           "suitable_skin_types": skin_types, "price_tier": price_tier,
                           "effects": effects or [], "age_groups": age_groups or []})


def _knowledge(cid, doc_type, effects=None, age_groups=None):
    return Chunk(cid, cid, doc_type, "",
                 {"doc_type": doc_type, "source": "synthesized",
                  "effects": effects or [], "age_groups": age_groups or []})


def test_no_persona_returns_chunks_unchanged():
    chunks = [_product("c1", ["oily"], "budget")]
    assert hard_filter(chunks, None) == chunks


def test_skin_type_mismatch_filters_product():
    persona = UserPersona(skin_type="dry")
    chunks = [_product("c1", ["oily"], "budget"), _product("c2", ["dry", "normal"], "budget")]
    out = hard_filter(chunks, persona)
    assert [c.chunk_id for c in out] == ["c2"]


def test_budget_tier_inclusion_mid_range_accepts_budget():
    persona = UserPersona(budget="mid-range")
    chunks = [_product("c1", ["oily"], "budget"), _product("c2", ["oily"], "luxury")]
    out = hard_filter(chunks, persona)
    assert [c.chunk_id for c in out] == ["c1"]


def test_budget_luxury_accepts_all():
    assert BUDGET_TIER_RULES["luxury"] == {"budget", "mid-range", "luxury"}


def test_ingredient_and_post_never_filtered():
    persona = UserPersona(skin_type="sensitive", budget="budget")
    chunks = [_knowledge("ing_1", "ingredient"), _knowledge("post_1", "post")]
    assert hard_filter(chunks, persona) == chunks


def test_empty_after_filter_fallback_drops_skin_type():
    persona = UserPersona(skin_type="sensitive", budget="mid-range")
    chunks = [_product("c1", ["oily"], "budget"), _product("c2", ["dry"], "mid-range")]
    # nothing matches sensitive → fallback drops skin_type, keeps budget
    out = hard_filter(chunks, persona)
    assert len(out) > 0
    # both c1 (budget) and c2 (mid-range) survive budget filter
    assert {c.chunk_id for c in out} == {"c1", "c2"}


def test_soft_boost_adds_for_preference_overlap():
    persona = UserPersona(preferences=["美白", "保湿"])
    chunk = _product("c1", ["oily"], "budget", effects=["美白"])
    boosted = soft_boost(1.0, chunk, persona, per_match=0.05)
    assert boosted == pytest.approx(1.05)


def test_soft_boost_age_group_match():
    persona = UserPersona(age_group="25-30")
    chunk = _product("c1", ["oily"], "budget", age_groups=["25-30", "31-40"])
    boosted = soft_boost(1.0, chunk, persona, per_match=0.05)
    assert boosted == pytest.approx(1.05)

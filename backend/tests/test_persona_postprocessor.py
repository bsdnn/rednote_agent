import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from backend.models.persona import UserPersona
from backend.services.rag._context import current_persona
from backend.services.rag.persona_postprocessor import PersonaNodePostprocessor


def _node(doc_id, doc_type, *, suitable_skin_types=None, price_tier=None,
          effects=None, age_groups=None, score=1.0):
    n = TextNode(
        id_=f"{doc_id}::0",
        text="",
        metadata={
            "doc_type": doc_type,
            "suitable_skin_types": suitable_skin_types or [],
            "price_tier": price_tier or "mid-range",
            "effects": effects or [],
            "age_groups": age_groups or [],
        },
    )
    return NodeWithScore(node=n, score=score)


def test_no_persona_returns_nodes_unchanged():
    pp = PersonaNodePostprocessor()
    nodes = [_node("prod_1", "product")]
    out = pp._postprocess_nodes(nodes, query_bundle=None)
    assert out == nodes


def test_hard_filter_drops_mismatched_skin_type():
    pp = PersonaNodePostprocessor()
    nodes = [
        _node("prod_1", "product", suitable_skin_types=["oily"]),
        _node("prod_2", "product", suitable_skin_types=["dry"]),
        _node("ing_1",  "ingredient"),
    ]
    tok = current_persona.set(UserPersona(skin_type="dry"))
    try:
        out = pp._postprocess_nodes(nodes, query_bundle=None)
    finally:
        current_persona.reset(tok)
    ids = [n.node.id_ for n in out]
    assert "prod_2::0" in ids
    assert "ing_1::0" in ids
    assert "prod_1::0" not in ids


def test_hard_filter_drops_unaffordable_products():
    pp = PersonaNodePostprocessor()
    nodes = [
        _node("prod_1", "product", suitable_skin_types=["oily"], price_tier="budget"),
        _node("prod_2", "product", suitable_skin_types=["oily"], price_tier="luxury"),
    ]
    tok = current_persona.set(UserPersona(skin_type="oily", budget="mid-range"))
    try:
        out = pp._postprocess_nodes(nodes, query_bundle=None)
    finally:
        current_persona.reset(tok)
    ids = [n.node.id_ for n in out]
    assert "prod_1::0" in ids
    assert "prod_2::0" not in ids


def test_empty_products_fallback_drops_skin_type():
    pp = PersonaNodePostprocessor()
    nodes = [
        _node("prod_1", "product", suitable_skin_types=["oily"], price_tier="mid-range"),
        _node("prod_2", "product", suitable_skin_types=["dry"],  price_tier="mid-range"),
    ]
    tok = current_persona.set(UserPersona(skin_type="sensitive", budget="mid-range"))
    try:
        out = pp._postprocess_nodes(nodes, query_bundle=None)
    finally:
        current_persona.reset(tok)
    assert len(out) == 2


def test_soft_boost_for_preferences_overlap_and_resort():
    pp = PersonaNodePostprocessor(per_match_boost=0.1)
    n1 = _node("prod_1", "product",
               suitable_skin_types=["oily"], effects=[], age_groups=[], score=1.0)
    n2 = _node("prod_2", "product",
               suitable_skin_types=["oily"], effects=["美白", "保湿"],
               age_groups=["25-30"], score=0.7)
    persona = UserPersona(skin_type="oily", preferences=["美白", "保湿"], age_group="25-30")
    tok = current_persona.set(persona)
    try:
        out = pp._postprocess_nodes([n1, n2], query_bundle=None)
    finally:
        current_persona.reset(tok)
    assert out[0].node.id_ == "prod_2::0"


def test_ingredient_and_post_never_filtered():
    pp = PersonaNodePostprocessor()
    nodes = [
        _node("ing_1",  "ingredient"),
        _node("post_1", "post"),
    ]
    tok = current_persona.set(UserPersona(skin_type="sensitive", budget="budget"))
    try:
        out = pp._postprocess_nodes(nodes, query_bundle=None)
    finally:
        current_persona.reset(tok)
    assert len(out) == 2

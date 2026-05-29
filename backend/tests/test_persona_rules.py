from backend.services.rag.persona_rules import (
    BUDGET_TIER_RULES, passes_skin_type, passes_budget,
)


def test_budget_rules():
    assert BUDGET_TIER_RULES["budget"] == {"budget"}
    assert BUDGET_TIER_RULES["mid-range"] == {"budget", "mid-range"}
    assert BUDGET_TIER_RULES["luxury"] == {"budget", "mid-range", "luxury"}


def test_passes_skin_type_with_dict():
    assert passes_skin_type({"suitable_skin_types": ["oily"]}, "oily")
    assert not passes_skin_type({"suitable_skin_types": ["dry"]}, "oily")
    assert passes_skin_type({"suitable_skin_types": ["all"]}, "oily")
    assert passes_skin_type({"suitable_skin_types": []}, "oily")
    assert passes_skin_type({}, "oily")
    assert passes_skin_type({"suitable_skin_types": ["dry"]}, None)


def test_passes_budget_with_dict():
    assert passes_budget({"price_tier": "budget"}, "mid-range")
    assert passes_budget({"price_tier": "mid-range"}, "mid-range")
    assert not passes_budget({"price_tier": "luxury"}, "mid-range")
    assert passes_budget({"price_tier": "luxury"}, "luxury")
    assert passes_budget({}, "mid-range")
    assert passes_budget({"price_tier": "luxury"}, None)

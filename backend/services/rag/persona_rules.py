"""Pure functions for persona-based filtering and boost computation.

These take a `metadata: dict` (not a domain object) so they work equally
with our legacy `Chunk` dataclass and LlamaIndex `TextNode.metadata`.
"""
from typing import Optional


BUDGET_TIER_RULES: dict[str, set[str]] = {
    "budget":    {"budget"},
    "mid-range": {"budget", "mid-range"},
    "luxury":    {"budget", "mid-range", "luxury"},
}


def passes_skin_type(metadata: dict, skin_type: Optional[str]) -> bool:
    if not skin_type:
        return True
    types = metadata.get("suitable_skin_types") or []
    if not types:
        return True
    return skin_type in types or "all" in types


def passes_budget(metadata: dict, budget: Optional[str]) -> bool:
    if not budget:
        return True
    tier = metadata.get("price_tier")
    if not tier:
        return True
    allowed = BUDGET_TIER_RULES.get(budget, {tier})
    return tier in allowed

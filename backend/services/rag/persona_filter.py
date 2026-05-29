import logging
from backend.models.persona import UserPersona
from .corpus.schema import Chunk
from .retrievers.vector import RetrievalHit

logger = logging.getLogger(__name__)

BUDGET_TIER_RULES: dict[str, set[str]] = {
    "budget":    {"budget"},
    "mid-range": {"budget", "mid-range"},
    "luxury":    {"budget", "mid-range", "luxury"},
}


def _passes_skin_type(chunk: Chunk, skin_type: str | None) -> bool:
    if not skin_type:
        return True
    types = chunk.metadata.get("suitable_skin_types") or []
    # empty types = unspecified, don't penalize
    if not types:
        return True
    return skin_type in types or "all" in types


def _passes_budget(chunk: Chunk, budget: str | None) -> bool:
    if not budget:
        return True
    tier = chunk.metadata.get("price_tier")
    if not tier:
        return True
    allowed = BUDGET_TIER_RULES.get(budget, {tier})
    return tier in allowed


def hard_filter(chunks: list[Chunk], persona: UserPersona | None) -> list[Chunk]:
    """Hard-filter products only; ingredient/post always survive.

    Fallback: if skin_type filter empties product candidates, drop it and keep only budget filter.
    """
    if persona is None:
        return chunks

    def _filter(skip_skin_type: bool) -> list[Chunk]:
        out = []
        for c in chunks:
            if c.doc_type != "product":
                out.append(c); continue
            if not skip_skin_type and not _passes_skin_type(c, persona.skin_type):
                continue
            if not _passes_budget(c, persona.budget):
                continue
            out.append(c)
        return out

    primary = _filter(skip_skin_type=False)
    product_count = sum(1 for c in primary if c.doc_type == "product")
    if product_count == 0 and persona.skin_type:
        logger.warning("persona hard_filter emptied products; dropping skin_type=%s", persona.skin_type)
        return _filter(skip_skin_type=True)
    return primary


def soft_boost(score: float, chunk: Chunk, persona: UserPersona, per_match: float = 0.05) -> float:
    if persona is None:
        return score
    boost = 0.0
    age_groups = chunk.metadata.get("age_groups") or []
    if persona.age_group and persona.age_group in age_groups:
        boost += per_match
    effects = set(chunk.metadata.get("effects") or [])
    overlap = set(persona.preferences or []) & effects
    boost += per_match * len(overlap)
    return score + boost


def apply_soft_boost(hits: list[RetrievalHit], persona: UserPersona | None, per_match: float) -> list[RetrievalHit]:
    if persona is None:
        return hits
    return sorted(
        [RetrievalHit(chunk=h.chunk, score=soft_boost(h.score, h.chunk, persona, per_match)) for h in hits],
        key=lambda h: h.score, reverse=True,
    )

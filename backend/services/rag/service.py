"""Public RAG entrypoint — wraps LlamaIndex engine with persona injection + semantic cache.

The previous v1/v2 dispatcher logic is gone — v3 is the only implementation.
"""
import logging
from typing import Optional
import numpy as np

from backend.core.config import settings
from backend.models.persona import UserPersona

from ._context import current_persona
from .engine import get_engine, get_embed_model
from .cache.semantic_cache import SemanticCache

logger = logging.getLogger(__name__)


def _embed_for_cache(text: str) -> np.ndarray:
    """Embed a single string for the semantic cache (reuses BGE-M3 from engine)."""
    return np.asarray(get_embed_model().get_text_embedding(text), dtype="float32")


_cache = SemanticCache(
    embedding_fn=_embed_for_cache,
    threshold=settings.SEMANTIC_CACHE_THRESHOLD,
    max_size=settings.SEMANTIC_CACHE_MAX_SIZE,
    ttl_seconds=settings.SEMANTIC_CACHE_TTL_SECONDS,
)


def _cache_key(query: str, persona: Optional[UserPersona]) -> str:
    if persona is None:
        return query
    return (
        f"{query}||{persona.skin_type}|{persona.budget}|{persona.age_group}|"
        f"{','.join(sorted(persona.preferences or []))}"
    )


def _format_top(nodes_with_scores, k: int = 3) -> str:
    if not nodes_with_scores:
        return "数据库中未找到匹配产品。"
    parts = []
    for rank, nws in enumerate(nodes_with_scores[:k], 1):
        n = nws.node
        meta = n.metadata
        # node_id is like "prod_3::0"; the parent doc_id is the prefix before "::"
        doc_id = n.node_id.split("::")[0]
        ingredients = "、".join(meta.get("key_ingredients", [])[:4])
        skin_types = "、".join(meta.get("suitable_skin_types", []))
        selling = "、".join(meta.get("selling_points", []))
        preview = n.get_content()[:120].replace("\n", " ")
        parts.append(
            f"[匹配{rank}] [{doc_id}] {preview}\n"
            f"类型: {meta.get('doc_type', '')} | 成分: {ingredients} "
            f"| 适合: {skin_types} | 卖点: {selling}"
        )
    return "\n\n".join(parts)


async def query_product_database(
    query: str,
    persona: Optional[UserPersona] = None,
) -> str:
    """Public API consumed by `backend/services/tools_registry.py`."""
    key = _cache_key(query, persona)
    cached = await _cache.get(key)
    if cached is not None:
        return cached

    engine = get_engine()
    token = current_persona.set(persona)
    try:
        response = await engine.aquery(query)
    finally:
        current_persona.reset(token)

    result = _format_top(response.source_nodes, k=settings.RAG_TOP_K)
    await _cache.set(key, result)
    return result


def cache_stats() -> dict:
    return _cache.stats()

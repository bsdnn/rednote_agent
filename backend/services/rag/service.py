"""Public RAG entrypoint with v1/v2 dispatch.

`tools_registry.query_product_database` imports from here.
"""
import logging
from pathlib import Path
import numpy as np

from backend.core.config import settings
from backend.models.persona import UserPersona

logger = logging.getLogger(__name__)

_v2_pipeline = None  # lazily built


def _data_dir() -> Path:
    return Path(__file__).parent.parent.parent / "data"


def _build_v2():
    global _v2_pipeline
    if _v2_pipeline is not None:
        return _v2_pipeline

    from sentence_transformers import SentenceTransformer
    from .corpus.loader import load_all
    from .pipeline import RAGv2Pipeline
    from .retrievers.reranker import get_cross_encoder

    data = _data_dir()
    docs = load_all(
        products_path=data / "products.json",
        ingredients_path=data / "corpus" / "ingredients.json",
        posts_path=data / "corpus" / "posts.json",
    )
    logger.info("RAG v2: loaded %d documents", len(docs))

    embed_model = SentenceTransformer("shibing624/text2vec-base-chinese")
    cross = get_cross_encoder(settings.RERANKER_MODEL)

    def embed_batch(texts):
        return np.asarray(embed_model.encode(list(texts)), dtype="float32")

    def embed_single(text):
        return np.asarray(embed_model.encode([text])[0], dtype="float32")

    _v2_pipeline = RAGv2Pipeline(
        docs=docs,
        embed_fn_batch=embed_batch,
        embed_fn_single=embed_single,
        cross_encoder=cross,
        hybrid_topk=settings.RAG_HYBRID_TOPK,
        final_topk=settings.RAG_TOP_K,
        cache_threshold=settings.SEMANTIC_CACHE_THRESHOLD,
        cache_max_size=settings.SEMANTIC_CACHE_MAX_SIZE,
        cache_ttl=settings.SEMANTIC_CACHE_TTL_SECONDS,
        soft_boost_per_match=settings.PERSONA_SOFT_BOOST_PER_MATCH,
    )
    return _v2_pipeline


async def query_product_database(query: str, persona: UserPersona | None = None) -> str:
    if settings.RAG_VERSION == "v2":
        pipeline = _build_v2()
        return await pipeline.query(query, persona=persona)
    # v1 fallback — persona is ignored (legacy)
    from backend.services.rag_service import query_product_database as v1_query
    return await v1_query(query)


def cache_stats() -> dict:
    if _v2_pipeline is None:
        return {"status": "v2 not initialized"}
    return _v2_pipeline.cache_stats()

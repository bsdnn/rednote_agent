import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

from backend.models.persona import UserPersona
from .corpus.schema import Document, Chunk
from .corpus.chunker import chunk_all
from .retrievers.vector import VectorRetriever, RetrievalHit
from .retrievers.bm25 import BM25Retriever
from .retrievers.hybrid import rrf_fuse
from .retrievers.reranker import CrossEncoderReranker
from .persona_filter import hard_filter, apply_soft_boost
from .cache.semantic_cache import SemanticCache

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=4)


def _format_hits(hits: list[RetrievalHit]) -> str:
    if not hits:
        return "数据库中未找到匹配产品。"
    parts = []
    for rank, h in enumerate(hits, 1):
        meta = h.chunk.metadata
        ingredients = "、".join(meta.get("key_ingredients", [])[:4])
        skin_types = "、".join(meta.get("suitable_skin_types", []))
        selling = "、".join(meta.get("selling_points", []))
        parts.append(
            f"[匹配{rank}] {h.chunk.text[:80]}\n"
            f"类型: {h.chunk.doc_type} | 成分: {ingredients} | 适合: {skin_types} | 卖点: {selling}"
        )
    return "\n\n".join(parts)


class RAGv2Pipeline:
    def __init__(
        self,
        docs: list[Document],
        embed_fn_batch: Callable[[list[str]], np.ndarray],
        embed_fn_single: Callable[[str], np.ndarray],
        cross_encoder,
        hybrid_topk: int = 20,
        final_topk: int = 3,
        cache_threshold: float = 0.92,
        cache_max_size: int = 256,
        cache_ttl: int = 3600,
        soft_boost_per_match: float = 0.05,
    ):
        chunks: list[Chunk] = chunk_all(docs)
        self._vector = VectorRetriever(embed_fn=embed_fn_batch)
        self._vector.build(chunks)
        self._bm25 = BM25Retriever()
        self._bm25.build(chunks)
        self._reranker = CrossEncoderReranker(model=cross_encoder)
        self._cache = SemanticCache(
            embedding_fn=embed_fn_single,
            threshold=cache_threshold,
            max_size=cache_max_size,
            ttl_seconds=cache_ttl,
        )
        self._all_chunks = chunks
        self._hybrid_topk = hybrid_topk
        self._final_topk = final_topk
        self._soft_boost_per_match = soft_boost_per_match

    async def query(self, query: str, persona: UserPersona | None = None) -> str:
        # cache key includes persona to avoid cross-persona leak
        cache_key = self._cache_key(query, persona)
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return cached

        # filter candidate pool by persona hard rules
        filtered = hard_filter(self._all_chunks, persona)
        if not filtered:
            result = "数据库中未找到匹配产品。"
            await self._cache.set(cache_key, result)
            return result

        # parallel retrieval on filtered pool
        loop = asyncio.get_event_loop()
        # rebuild lightweight indices restricted to filtered chunks per query is too expensive;
        # instead retrieve from full index then drop filtered-out hits
        filtered_ids = {c.chunk_id for c in filtered}

        def _vec():
            hits = self._vector.retrieve(query, top_k=self._hybrid_topk * 2)
            return [h for h in hits if h.chunk.chunk_id in filtered_ids][: self._hybrid_topk]

        def _bm():
            hits = self._bm25.retrieve(query, top_k=self._hybrid_topk * 2)
            return [h for h in hits if h.chunk.chunk_id in filtered_ids][: self._hybrid_topk]

        vec_hits, bm_hits = await asyncio.gather(
            loop.run_in_executor(_executor, _vec),
            loop.run_in_executor(_executor, _bm),
        )

        # RRF fusion
        candidates = rrf_fuse([vec_hits, bm_hits], top_k=self._hybrid_topk)
        if not candidates:
            result = "数据库中未找到匹配产品。"
            await self._cache.set(cache_key, result)
            return result

        # cross-encoder rerank
        def _rerank():
            return self._reranker.rerank(query, candidates, top_k=self._final_topk * 3)
        reranked = await loop.run_in_executor(_executor, _rerank)

        # persona soft boost (re-sort)
        boosted = apply_soft_boost(reranked, persona, per_match=self._soft_boost_per_match)
        final = boosted[: self._final_topk]

        result = _format_hits(final)
        await self._cache.set(cache_key, result)
        return result

    def _cache_key(self, query: str, persona: UserPersona | None) -> str:
        if persona is None:
            return query
        return f"{query}||{persona.skin_type}|{persona.budget}|{persona.age_group}|{','.join(sorted(persona.preferences or []))}"

    def cache_stats(self) -> dict:
        return self._cache.stats()

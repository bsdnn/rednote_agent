"""5 ablation configurations for RAG v2.

Each config returns a query-callable: `async def(query, persona) -> str`.
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Callable, Awaitable

import numpy as np

from backend.models.persona import UserPersona
from backend.services.rag.corpus.loader import load_all
from backend.services.rag.pipeline import RAGv2Pipeline
from backend.services.rag.retrievers.reranker import get_cross_encoder
from backend.services.rag.retrievers.vector import VectorRetriever
from backend.services.rag.retrievers.bm25 import BM25Retriever
from backend.services.rag.retrievers.hybrid import rrf_fuse
from backend.services.rag.corpus.chunker import chunk_all
from sentence_transformers import SentenceTransformer

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "backend" / "data"


@dataclass
class ConfigSpec:
    name: str
    use_bm25: bool
    use_rerank: bool
    use_semantic_cache: bool
    use_persona: bool
    description: str


CONFIG_SPECS = [
    ConfigSpec("C0", False, False, False, False, "baseline (v1 equivalent: vector only, no rerank, lru-style cache)"),
    ConfigSpec("C1", True,  False, False, False, "+ BM25/RRF hybrid"),
    ConfigSpec("C2", True,  True,  False, False, "+ cross-encoder rerank"),
    ConfigSpec("C3", True,  True,  True,  False, "+ semantic cache"),
    ConfigSpec("C4", True,  True,  True,  True,  "full v2 (+ persona-aware)"),
]


def _load_resources():
    docs = load_all(
        products_path=DATA / "products.json",
        ingredients_path=DATA / "corpus" / "ingredients.json",
        posts_path=DATA / "corpus" / "posts.json",
    )
    embed_model = SentenceTransformer("shibing624/text2vec-base-chinese")
    def embed_batch(texts): return np.asarray(embed_model.encode(list(texts)), dtype="float32")
    def embed_single(text): return np.asarray(embed_model.encode([text])[0], dtype="float32")
    cross = get_cross_encoder("BAAI/bge-reranker-base")
    return docs, embed_batch, embed_single, cross


def build_query_fn(spec: ConfigSpec) -> Callable[[str, UserPersona | None], Awaitable[str]]:
    docs, embed_batch, embed_single, cross = _load_resources()

    if not spec.use_bm25 and not spec.use_rerank and not spec.use_semantic_cache:
        # C0: vector only — use a stripped-down pipeline
        chunks = chunk_all(docs)
        retr = VectorRetriever(embed_fn=embed_batch); retr.build(chunks)

        async def _q(query: str, persona):
            hits = retr.retrieve(query, top_k=3)
            return "\n".join(h.chunk.text[:120] for h in hits) or "数据库中未找到匹配产品。"
        return _q

    # C1..C4: re-use RAGv2Pipeline with toggles
    # For simplicity, the pipeline always has BM25 + rerank built; toggles short-circuit at query time.
    # Here we synthesize using direct retriever combinations to keep ablations clean.
    chunks = chunk_all(docs)
    vector = VectorRetriever(embed_fn=embed_batch); vector.build(chunks)
    bm25 = BM25Retriever(); bm25.build(chunks)

    from backend.services.rag.retrievers.reranker import CrossEncoderReranker
    from backend.services.rag.cache.semantic_cache import SemanticCache
    from backend.services.rag.persona_filter import hard_filter, apply_soft_boost

    reranker = CrossEncoderReranker(model=cross) if spec.use_rerank else None
    cache = SemanticCache(embedding_fn=embed_single, threshold=0.92) if spec.use_semantic_cache else None

    async def _q(query: str, persona):
        cache_key = f"{query}||{persona.skin_type if persona else 'none'}" if spec.use_persona else query
        if cache:
            cached = await cache.get(cache_key)
            if cached is not None: return cached

        candidate_pool = hard_filter(chunks, persona) if spec.use_persona else chunks
        filtered_ids = {c.chunk_id for c in candidate_pool}
        if not candidate_pool:
            result = "数据库中未找到匹配产品。"
            if cache: await cache.set(cache_key, result)
            return result

        v_hits = [h for h in vector.retrieve(query, top_k=40) if h.chunk.chunk_id in filtered_ids][:20]
        b_hits = [h for h in bm25.retrieve(query, top_k=40) if h.chunk.chunk_id in filtered_ids][:20] if spec.use_bm25 else []
        candidates = rrf_fuse([v_hits, b_hits], top_k=20) if spec.use_bm25 else v_hits[:20]

        if reranker:
            candidates = reranker.rerank(query, candidates, top_k=10)
        if spec.use_persona:
            candidates = apply_soft_boost(candidates, persona, per_match=0.05)
        final = candidates[:3]
        result = "\n".join(h.chunk.text[:120] for h in final) or "数据库中未找到匹配产品。"
        if cache: await cache.set(cache_key, result)
        return result

    return _q

"""5 ablation configurations for RAG v3 (LlamaIndex-based).

Each config builds its own query callable:
    async def(query: str, persona: UserPersona | None) -> str
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

import faiss
import jieba
import numpy as np

from llama_index.core import Settings as LISettings, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

from backend.core.config import settings
from backend.models.persona import UserPersona
from backend.services.rag._context import current_persona
from backend.services.rag.corpus.loader import load_all_as_li
from backend.services.rag.persona_postprocessor import PersonaNodePostprocessor
from backend.services.rag.cache.semantic_cache import SemanticCache

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "backend" / "data"

BGE_M3_DIM = 1024


@dataclass
class ConfigSpec:
    name: str
    description: str
    use_bm25: bool
    use_rerank: bool
    use_cache: bool
    use_persona: bool


CONFIG_SPECS = [
    ConfigSpec("C0", "vector only (LI baseline)",
               use_bm25=False, use_rerank=False, use_cache=False, use_persona=False),
    ConfigSpec("C1", "+ BM25/RRF",
               use_bm25=True,  use_rerank=False, use_cache=False, use_persona=False),
    ConfigSpec("C2", "+ cross-encoder rerank",
               use_bm25=True,  use_rerank=True,  use_cache=False, use_persona=False),
    ConfigSpec("C3", "+ semantic cache",
               use_bm25=True,  use_rerank=True,  use_cache=True,  use_persona=False),
    ConfigSpec("C4", "+ persona-aware (full v3)",
               use_bm25=True,  use_rerank=True,  use_cache=True,  use_persona=True),
]


# Cache heavy resources across configs to avoid re-downloading models
_resource_cache: dict = {}


def _load_resources():
    """Load BGE-M3 embed, reranker, docs, nodes — once across all configs."""
    if "loaded" in _resource_cache:
        return _resource_cache

    LISettings.llm = OpenAILike(
        model="deepseek-chat",
        api_base="https://api.deepseek.com/v1",
        api_key=settings.DEEPSEEK_API_KEY,
        is_chat_model=True,
    )

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    LISettings.embed_model = embed_model

    docs = load_all_as_li(
        products_path=DATA / "products.json",
        ingredients_path=DATA / "corpus" / "ingredients.json",
        posts_path=DATA / "corpus" / "posts.json",
    )
    splitter = SentenceSplitter(
        chunk_size=350, chunk_overlap=50,
        paragraph_separator="\n\n",
        secondary_chunking_regex=r"(?<=[。!?!?])",
    )
    nodes: list[TextNode] = []
    for doc in docs:
        if doc.metadata.get("doc_type") == "post":
            chunks = splitter.get_nodes_from_documents([doc])
            for idx, chunk in enumerate(chunks):
                chunk.id_ = f"{doc.doc_id}::{idx}"
                chunk.excluded_embed_metadata_keys = list(doc.excluded_embed_metadata_keys or [])
                chunk.excluded_llm_metadata_keys = list(doc.excluded_llm_metadata_keys or [])
            nodes.extend(chunks)
        else:
            n = TextNode(id_=f"{doc.doc_id}::0", text=doc.text, metadata=dict(doc.metadata))
            n.excluded_embed_metadata_keys = list(doc.excluded_embed_metadata_keys or [])
            nodes.append(n)

    faiss_index = faiss.IndexFlatIP(BGE_M3_DIM)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_ctx)

    reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-v2-m3", top_n=9)

    _resource_cache.update({
        "loaded": True,
        "embed_model": embed_model,
        "docs": docs,
        "nodes": nodes,
        "index": index,
        "reranker": reranker,
    })
    return _resource_cache


def _build_engine(spec: ConfigSpec) -> RetrieverQueryEngine:
    res = _load_resources()
    nodes = res["nodes"]
    index = res["index"]

    vector_r = VectorIndexRetriever(index=index, similarity_top_k=20)

    if spec.use_bm25:
        bm25_r = BM25Retriever.from_defaults(
            nodes=nodes, similarity_top_k=20, tokenizer=jieba.lcut,
        )
        retriever = QueryFusionRetriever(
            retrievers=[vector_r, bm25_r],
            mode="reciprocal_rerank",
            num_queries=1,
            similarity_top_k=20,
            use_async=True,
        )
    else:
        retriever = vector_r

    postprocessors = []
    if spec.use_rerank:
        postprocessors.append(res["reranker"])
    if spec.use_persona:
        postprocessors.append(
            PersonaNodePostprocessor(per_match_boost=settings.PERSONA_SOFT_BOOST_PER_MATCH)
        )

    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=postprocessors,
        response_mode="no_text",
    )


def _format_nodes(nodes_with_scores, k: int = 3) -> str:
    if not nodes_with_scores:
        return "数据库中未找到匹配产品。"
    parts = []
    for rank, nws in enumerate(nodes_with_scores[:k], 1):
        n = nws.node
        doc_id = n.node_id.split("::")[0]
        preview = n.get_content()[:120].replace("\n", " ")
        parts.append(f"[匹配{rank}] [{doc_id}] {preview}")
    return "\n".join(parts)


def build_query_fn(spec: ConfigSpec) -> Callable[[str, UserPersona | None], Awaitable[str]]:
    """Return an async (query, persona) -> formatted-context-str."""
    engine = _build_engine(spec)

    if spec.use_cache:
        embed_model = _resource_cache["embed_model"]
        def _embed_for_cache(text: str) -> np.ndarray:
            return np.asarray(embed_model.get_text_embedding(text), dtype="float32")
        cache = SemanticCache(
            embedding_fn=_embed_for_cache,
            threshold=settings.SEMANTIC_CACHE_THRESHOLD,
            max_size=settings.SEMANTIC_CACHE_MAX_SIZE,
            ttl_seconds=settings.SEMANTIC_CACHE_TTL_SECONDS,
        )
    else:
        cache = None

    async def _q(query: str, persona: UserPersona | None) -> str:
        key = (f"{query}||{persona.skin_type}|{persona.budget}"
               if persona and spec.use_persona else query)
        if cache is not None:
            hit = await cache.get(key)
            if hit is not None:
                return hit

        tok = current_persona.set(persona if spec.use_persona else None)
        try:
            response = await engine.aquery(query)
        finally:
            current_persona.reset(tok)

        result = _format_nodes(response.source_nodes, k=3)
        if cache is not None:
            await cache.set(key, result)
        return result

    return _q

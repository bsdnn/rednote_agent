"""LlamaIndex RetrieverQueryEngine factory + singleton.

Builds:
  QueryFusionRetriever(
    retrievers=[VectorIndexRetriever (BGE-M3 dense), BM25Retriever (jieba)],
    mode="reciprocal_rerank",
  )
  -> SentenceTransformerRerank (bge-reranker-v2-m3)
  -> PersonaNodePostprocessor (reads ContextVar)
  -> response.source_nodes

response_mode="no_text" means LlamaIndex does NOT invoke an LLM to synthesize
text -- we only need the retrieved nodes; the agent layer (backend/core/agent.py)
does its own generation.
"""
import logging
from pathlib import Path

import faiss
import jieba

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
from .corpus.loader import load_all_as_li
from .persona_postprocessor import PersonaNodePostprocessor

logger = logging.getLogger(__name__)

BGE_M3_DIM = 1024

_engine: RetrieverQueryEngine | None = None
_embed_model: HuggingFaceEmbedding | None = None  # cached for reuse in semantic cache


def _data_dir() -> Path:
    return Path(__file__).parent.parent.parent / "data"


def get_embed_model() -> HuggingFaceEmbedding:
    """Return the singleton BGE-M3 embedding model.

    Shared between the LlamaIndex engine and the SemanticCache so we only
    load weights once.
    """
    global _embed_model
    if _embed_model is None:
        logger.info("Loading BAAI/bge-m3 embedding model (~2.2GB first time)...")
        _embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    return _embed_model


def get_engine() -> RetrieverQueryEngine:
    """Build the singleton query engine. First call loads models + builds indices."""
    global _engine
    if _engine is not None:
        return _engine

    LISettings.embed_model = get_embed_model()
    LISettings.llm = OpenAILike(
        model="deepseek-chat",
        api_base="https://api.deepseek.com/v1",
        api_key=settings.DEEPSEEK_API_KEY,
        is_chat_model=True,
    )

    data = _data_dir()
    docs = load_all_as_li(
        products_path=data / "products.json",
        ingredients_path=data / "corpus" / "ingredients.json",
        posts_path=data / "corpus" / "posts.json",
    )
    logger.info("RAG v3: loaded %d documents", len(docs))

    splitter = SentenceSplitter(
        chunk_size=350,
        chunk_overlap=50,
        paragraph_separator="\n\n",
        secondary_chunking_regex=r"(?<=[。!?!?])",
    )
    nodes: list[TextNode] = []
    for doc in docs:
        if doc.metadata.get("doc_type") == "post":
            chunks = splitter.get_nodes_from_documents([doc])
            # Re-id with deterministic "{doc_id}::{idx}" — SentenceSplitter assigns
            # UUIDs by default, which breaks the downstream doc_id-prefix convention
            # used by service._format_top and eval's _extract_doc_ids regex.
            for idx, chunk in enumerate(chunks):
                chunk.id_ = f"{doc.doc_id}::{idx}"
                # SentenceSplitter copies metadata but may not copy the
                # excluded-keys config — set explicitly to keep title-aware embedding.
                chunk.excluded_embed_metadata_keys = list(doc.excluded_embed_metadata_keys or [])
                chunk.excluded_llm_metadata_keys = list(doc.excluded_llm_metadata_keys or [])
            nodes.extend(chunks)
        else:
            n = TextNode(
                id_=f"{doc.doc_id}::0",
                text=doc.text,
                metadata=dict(doc.metadata),
            )
            n.excluded_embed_metadata_keys = list(doc.excluded_embed_metadata_keys or [])
            n.excluded_llm_metadata_keys = list(doc.excluded_llm_metadata_keys or [])
            nodes.append(n)
    logger.info("RAG v3: produced %d nodes after chunking", len(nodes))

    # FAISS dense index (cosine via normalized inner product)
    faiss_index = faiss.IndexFlatIP(BGE_M3_DIM)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_ctx)

    vector_r = VectorIndexRetriever(index=index, similarity_top_k=20)
    bm25_r = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=20,
        tokenizer=jieba.lcut,
    )

    fusion_r = QueryFusionRetriever(
        retrievers=[vector_r, bm25_r],
        mode="reciprocal_rerank",
        num_queries=1,
        similarity_top_k=20,
        use_async=True,
    )

    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-v2-m3",
        top_n=9,
    )
    persona_pp = PersonaNodePostprocessor(
        per_match_boost=settings.PERSONA_SOFT_BOOST_PER_MATCH,
    )

    _engine = RetrieverQueryEngine.from_args(
        retriever=fusion_r,
        node_postprocessors=[reranker, persona_pp],
        response_mode="no_text",
    )
    logger.info("RAG v3 engine ready")
    return _engine

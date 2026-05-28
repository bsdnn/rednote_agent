import pytest
from backend.services.rag.corpus.schema import Chunk
from backend.services.rag.retrievers.bm25 import BM25Retriever, tokenize_zh


@pytest.fixture
def chunks():
    return [
        Chunk(chunk_id="c1", doc_id="d1", doc_type="product",
              text="含烟酰胺的美白精华液", metadata={}),
        Chunk(chunk_id="c2", doc_id="d2", doc_type="product",
              text="温和保湿乳液 适合敏感肌", metadata={}),
        Chunk(chunk_id="c3", doc_id="d3", doc_type="ingredient",
              text="烟酰胺是经典的美白成分", metadata={}),
    ]


def test_tokenize_zh_splits_chinese():
    toks = tokenize_zh("含烟酰胺的精华")
    assert "烟酰胺" in toks
    assert "精华" in toks


def test_bm25_exact_ingredient_match_ranks_first(chunks):
    r = BM25Retriever()
    r.build(chunks)
    hits = r.retrieve("烟酰胺", top_k=3)
    # both c1 and c3 mention 烟酰胺; either is fine to top
    assert hits[0].chunk.chunk_id in ("c1", "c3")
    assert hits[0].score > 0


def test_bm25_empty_query_returns_empty(chunks):
    r = BM25Retriever()
    r.build(chunks)
    assert r.retrieve("", top_k=3) == []


def test_bm25_empty_index_raises():
    r = BM25Retriever()
    with pytest.raises(RuntimeError):
        r.retrieve("x", top_k=3)

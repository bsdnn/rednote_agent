import numpy as np
import pytest
from backend.services.rag.corpus.schema import Chunk
from backend.services.rag.retrievers.vector import VectorRetriever


@pytest.fixture
def fake_embed():
    # deterministic 4-dim embedding based on first char codepoint
    def _e(texts):
        out = []
        for t in texts:
            base = ord(t[0]) if t else 0
            out.append([base % 7, base % 11, base % 13, base % 17])
        return np.array(out, dtype="float32")
    return _e


@pytest.fixture
def chunks():
    return [
        Chunk(chunk_id="c1", doc_id="d1", doc_type="product", text="苹果",
              metadata={"doc_type": "product", "source": "internal"}),
        Chunk(chunk_id="c2", doc_id="d2", doc_type="product", text="香蕉",
              metadata={"doc_type": "product", "source": "internal"}),
        Chunk(chunk_id="c3", doc_id="d3", doc_type="post", text="梨",
              metadata={"doc_type": "post", "source": "synthesized"}),
    ]


def test_retrieve_returns_top_k(fake_embed, chunks):
    r = VectorRetriever(embed_fn=fake_embed)
    r.build(chunks)
    results = r.retrieve("苹果", top_k=2)
    assert len(results) == 2
    assert all(hasattr(r_, "chunk") and hasattr(r_, "score") for r_ in results)


def test_retrieve_query_matches_self_first(fake_embed, chunks):
    r = VectorRetriever(embed_fn=fake_embed)
    r.build(chunks)
    results = r.retrieve("苹果", top_k=1)
    assert results[0].chunk.text == "苹果"


def test_empty_index_raises(fake_embed):
    r = VectorRetriever(embed_fn=fake_embed)
    with pytest.raises(RuntimeError):
        r.retrieve("any", top_k=3)

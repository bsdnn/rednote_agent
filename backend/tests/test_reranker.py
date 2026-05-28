import pytest
from backend.services.rag.corpus.schema import Chunk
from backend.services.rag.retrievers.vector import RetrievalHit
from backend.services.rag.retrievers.reranker import CrossEncoderReranker


@pytest.fixture
def fake_cross_encoder():
    """Pretend model: score = 1 / (1 + abs(len(q) - len(t)))"""
    class _CE:
        def predict(self, pairs):
            return [1.0 / (1.0 + abs(len(q) - len(t))) for q, t in pairs]
    return _CE()


def _hit(cid, text, score=0.0):
    return RetrievalHit(chunk=Chunk(chunk_id=cid, doc_id=cid, doc_type="product", text=text, metadata={}), score=score)


def test_rerank_reorders_by_cross_encoder_score(fake_cross_encoder):
    r = CrossEncoderReranker(model=fake_cross_encoder)
    candidates = [_hit("c1", "x"), _hit("c2", "xxxx"), _hit("c3", "xx")]
    out = r.rerank("xx", candidates, top_k=2)
    # query "xx" is length 2 → "xx" (c3) wins
    assert out[0].chunk.chunk_id == "c3"
    assert len(out) == 2


def test_rerank_empty_input_returns_empty(fake_cross_encoder):
    r = CrossEncoderReranker(model=fake_cross_encoder)
    assert r.rerank("x", [], top_k=3) == []

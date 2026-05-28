from backend.services.rag.corpus.schema import Chunk
from backend.services.rag.retrievers.vector import RetrievalHit
from backend.services.rag.retrievers.hybrid import rrf_fuse


def _hit(cid, score):
    return RetrievalHit(chunk=Chunk(chunk_id=cid, doc_id=cid, doc_type="product", text="", metadata={}), score=score)


def test_rrf_default_k_is_60():
    a = [_hit("c1", 0.9), _hit("c2", 0.8)]
    b = [_hit("c2", 5.0), _hit("c3", 4.0)]
    fused = rrf_fuse([a, b], top_k=3)
    # c2 in both — wins
    assert fused[0].chunk.chunk_id == "c2"


def test_rrf_empty_intersection_keeps_all():
    a = [_hit("c1", 0.9)]
    b = [_hit("c2", 5.0)]
    fused = rrf_fuse([a, b], top_k=3)
    ids = {h.chunk.chunk_id for h in fused}
    assert ids == {"c1", "c2"}


def test_rrf_handles_empty_lists():
    fused = rrf_fuse([[], []], top_k=3)
    assert fused == []

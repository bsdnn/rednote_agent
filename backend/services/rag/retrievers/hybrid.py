from .vector import RetrievalHit


def rrf_fuse(rank_lists: list[list[RetrievalHit]], top_k: int, k: int = 60) -> list[RetrievalHit]:
    """Reciprocal Rank Fusion. Each list is assumed already sorted desc by relevance."""
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievalHit] = {}
    for hits in rank_lists:
        for rank, hit in enumerate(hits, start=1):
            cid = hit.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            # keep the first-seen hit reference; score is overwritten with fused score below
            chunks.setdefault(cid, hit)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [RetrievalHit(chunk=chunks[cid].chunk, score=s) for cid, s in fused]

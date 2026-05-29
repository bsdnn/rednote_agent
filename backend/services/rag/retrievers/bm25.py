import jieba
from rank_bm25 import BM25Okapi

from ..corpus.schema import Chunk
from .vector import RetrievalHit


def tokenize_zh(text: str) -> list[str]:
    # jieba.lcut returns a list; filter whitespace tokens
    return [t for t in jieba.lcut(text) if t.strip()]


class BM25Retriever:
    def __init__(self):
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            self._chunks, self._bm25 = [], None
            return
        self._chunks = list(chunks)
        corpus_tokens = [tokenize_zh(c.text) for c in self._chunks]
        # rank_bm25 requires non-empty corpus; if any doc is empty, use a sentinel
        corpus_tokens = [toks or ["_empty_"] for toks in corpus_tokens]
        self._bm25 = BM25Okapi(corpus_tokens)

    def retrieve(self, query: str, top_k: int) -> list[RetrievalHit]:
        if self._bm25 is None:
            raise RuntimeError("BM25Retriever.build() must be called first")
        toks = tokenize_zh(query)
        if not toks:
            return []
        scores = self._bm25.get_scores(toks)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [RetrievalHit(chunk=self._chunks[i], score=float(scores[i])) for i in order if scores[i] > 0]

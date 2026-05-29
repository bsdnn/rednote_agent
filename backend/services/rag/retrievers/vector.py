from dataclasses import dataclass
from typing import Callable
import numpy as np
import faiss

from ..corpus.schema import Chunk


@dataclass
class RetrievalHit:
    chunk: Chunk
    score: float  # higher = better (cosine sim)


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class VectorRetriever:
    def __init__(self, embed_fn: Callable[[list[str]], np.ndarray]):
        self._embed = embed_fn
        self._chunks: list[Chunk] = []
        self._index: faiss.Index | None = None

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            self._chunks, self._index = [], None
            return
        self._chunks = list(chunks)
        vecs = _normalize(np.asarray(self._embed([c.text for c in self._chunks]), dtype="float32"))
        self._index = faiss.IndexFlatIP(vecs.shape[1])  # inner product on normalized = cosine
        self._index.add(vecs)

    def retrieve(self, query: str, top_k: int) -> list[RetrievalHit]:
        if self._index is None:
            raise RuntimeError("VectorRetriever.build() must be called first")
        qv = _normalize(np.asarray(self._embed([query]), dtype="float32"))
        scores, idxs = self._index.search(qv, min(top_k, len(self._chunks)))
        return [
            RetrievalHit(chunk=self._chunks[i], score=float(s))
            for s, i in zip(scores[0], idxs[0]) if i != -1
        ]

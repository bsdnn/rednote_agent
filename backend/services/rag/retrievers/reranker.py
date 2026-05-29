import logging
from typing import Protocol

from sentence_transformers import CrossEncoder

from .vector import RetrievalHit

logger = logging.getLogger(__name__)


class _CrossEncoderLike(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]: ...


_model_cache: dict[str, CrossEncoder] = {}


def get_cross_encoder(model_name: str) -> CrossEncoder:
    if model_name not in _model_cache:
        logger.info("Loading cross-encoder %s...", model_name)
        _model_cache[model_name] = CrossEncoder(model_name)
    return _model_cache[model_name]


class CrossEncoderReranker:
    def __init__(self, model: _CrossEncoderLike):
        self._model = model

    def rerank(self, query: str, candidates: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        if not candidates:
            return []
        pairs = [(query, h.chunk.text) for h in candidates]
        scores = self._model.predict(pairs)
        scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [RetrievalHit(chunk=h.chunk, score=float(s)) for s, h in scored[:top_k]]

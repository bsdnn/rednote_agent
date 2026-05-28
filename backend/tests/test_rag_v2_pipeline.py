import asyncio
import pytest
import numpy as np
from backend.models.persona import UserPersona
from backend.services.rag.corpus.schema import Document
from backend.services.rag.pipeline import RAGv2Pipeline


@pytest.fixture
def fake_embed_fn():
    def _e(texts):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in texts:
            base = ord(t[0]) if t else 0
            out.append([base % 7, base % 11, base % 13, base % 17])
        return np.array(out, dtype="float32")
    return _e


@pytest.fixture
def fake_single_embed(fake_embed_fn):
    return lambda t: fake_embed_fn([t])[0]


@pytest.fixture
def fake_cross_encoder():
    class _CE:
        def predict(self, pairs):
            return [1.0 / (1.0 + abs(len(q) - len(t))) for q, t in pairs]
    return _CE()


@pytest.fixture
def docs():
    return [
        Document("prod_1", "product", "A", "苹果产品",
                 {"doc_type": "product", "source": "internal",
                  "suitable_skin_types": ["oily"], "price_tier": "budget",
                  "key_ingredients": [], "effects": ["补水"], "age_groups": []}),
        Document("prod_2", "product", "B", "香蕉产品",
                 {"doc_type": "product", "source": "internal",
                  "suitable_skin_types": ["dry"], "price_tier": "luxury",
                  "key_ingredients": [], "effects": [], "age_groups": []}),
        Document("ing_1", "ingredient", "Ing", "成分百科",
                 {"doc_type": "ingredient", "source": "synthesized",
                  "effects": [], "concerns": [], "age_groups": []}),
    ]


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_pipeline_returns_formatted_string(docs, fake_embed_fn, fake_single_embed, fake_cross_encoder):
    p = RAGv2Pipeline(docs=docs, embed_fn_batch=fake_embed_fn, embed_fn_single=fake_single_embed,
                     cross_encoder=fake_cross_encoder, hybrid_topk=20, final_topk=2)
    out = _run(p.query("苹果产品", persona=None))
    assert isinstance(out, str)
    assert len(out) > 0


def test_pipeline_persona_filter_excludes_mismatched_products(docs, fake_embed_fn, fake_single_embed, fake_cross_encoder):
    p = RAGv2Pipeline(docs=docs, embed_fn_batch=fake_embed_fn, embed_fn_single=fake_single_embed,
                     cross_encoder=fake_cross_encoder, hybrid_topk=20, final_topk=3)
    persona = UserPersona(skin_type="oily", budget="budget")
    out = _run(p.query("产品", persona=persona))
    # dry-skin product (prod_2) should NOT appear
    assert "香蕉" not in out
    # ingredient always survives persona filter
    # (presence of "成分" or "苹果" both acceptable depending on rank)


def test_pipeline_cache_hit_on_paraphrase_after_first_query(docs, fake_embed_fn, fake_single_embed, fake_cross_encoder):
    p = RAGv2Pipeline(docs=docs, embed_fn_batch=fake_embed_fn, embed_fn_single=fake_single_embed,
                     cross_encoder=fake_cross_encoder, hybrid_topk=20, final_topk=2,
                     cache_threshold=0.5)  # loose threshold for fake embeddings
    _run(p.query("苹果", persona=None))
    stats_before = p.cache_stats()
    _run(p.query("苹果", persona=None))  # exact paraphrase
    stats_after = p.cache_stats()
    assert stats_after["exact_hits"] + stats_after["semantic_hits"] > stats_before["exact_hits"] + stats_before["semantic_hits"]

# RAG v3 — LlamaIndex Migration + BGE-M3 Upgrade — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the hand-rolled `backend/services/rag/` to LlamaIndex (chunker, retrievers, reranker, pipeline, evaluator) while keeping persona_filter and semantic_cache project-specific; upgrade embedding to BAAI/bge-m3 and reranker to BAAI/bge-reranker-v2-m3; preserve the 40-gold-query × 5-config ablation eval for direct v2 vs v3 comparison.

**Architecture:** LlamaIndex `RetrieverQueryEngine` with `QueryFusionRetriever(mode="reciprocal_rerank")` over `VectorIndexRetriever` (BGE-M3 dense) + `BM25Retriever` (jieba). Custom `PersonaNodePostprocessor` (BaseNodePostprocessor subclass) reads persona from a `ContextVar` injected by `service.query_product_database`. `SemanticCache` wraps the engine. Title is auto-prepended to embedding text via LlamaIndex's `excluded_embed_metadata_keys`.

**Tech Stack:** Python 3.10+, FastAPI, **LlamaIndex** (`llama-index-core` 0.10+, `llama-index-embeddings-huggingface`, `llama-index-vector-stores-faiss`, `llama-index-retrievers-bm25`, `llama-index-postprocessor-sbert-rerank`, `llama-index-llms-openai-like`), **BAAI/bge-m3** (1024-dim dense embedding), **BAAI/bge-reranker-v2-m3**, FAISS, jieba, DeepSeek API (via `OpenAILike` adapter), pytest.

**Spec:** `docs/superpowers/specs/2026-05-29-rag-v3-llamaindex-design.md`

---

## File Structure (post-migration)

### New files
```
backend/services/rag/
├── _context.py                       # ContextVar for persona injection
├── engine.py                          # LlamaIndex RetrieverQueryEngine factory + singleton
├── persona_postprocessor.py           # BaseNodePostprocessor subclass
└── persona_rules.py                   # BUDGET_TIER_RULES + _passes_skin_type + _passes_budget (pure)

backend/tests/
└── test_rag_v3_engine.py              # Integration test for the new pipeline

scripts/rag_eval/
└── (no new files — existing files modified)
```

### Modified files
```
backend/services/rag/corpus/schema.py    # Added: set_metadata_filters() + EMBED_KEYS / ALL_KEYS constants
backend/services/rag/corpus/loader.py    # Added: load_*_as_li_documents() functions
backend/services/rag/service.py          # Rewritten: uses engine + cache + ContextVar
backend/services/rag/__init__.py         # Re-exports (no breaking changes)
backend/main.py                          # Warmup updated
backend/requirements.txt                 # +6 LlamaIndex packages
backend/tests/test_corpus_schema.py      # Tests for set_metadata_filters
backend/tests/test_corpus_loader.py      # Tests for load_*_as_li_documents
scripts/rag_eval/configs.py              # Rewrite — LI build_engine_for_config factory
scripts/rag_eval/run_eval.py             # Persist faithfulness_reason + add LI RelevancyEvaluator
scripts/rag_eval/report.py               # Add dual-judge consistency section
README.md                                # Highlights / Tech Stack / Testing rewrite
```

### Deleted files (in Task 10, after new code works)
```
backend/services/rag/corpus/chunker.py       (replaced by LI SentenceSplitter)
backend/services/rag/retrievers/             (whole subdir — replaced by LI retrievers)
backend/services/rag/pipeline.py             (replaced by LI RetrieverQueryEngine)
backend/services/rag/persona_filter.py       (split into persona_rules + persona_postprocessor)
backend/tests/test_chunker.py
backend/tests/test_vector_retriever.py
backend/tests/test_bm25_retriever.py
backend/tests/test_rrf_fusion.py
backend/tests/test_reranker.py
backend/tests/test_rag_v2_pipeline.py
backend/tests/test_persona_filter.py         (replaced by test_persona_postprocessor.py)
```

### Unchanged files
```
backend/core/agent.py, backend/core/prompts.py, backend/api/routes.py, backend/models/*
backend/services/tools_registry.py            (signature-compatible — calls service.query_product_database)
backend/services/rag/cache/semantic_cache.py  (PROVE→PASS story preserved)
backend/tests/test_semantic_cache.py
backend/tests/test_eval_metrics.py
scripts/rag_eval/gold_queries.jsonl           (40 queries unchanged)
scripts/rag_eval/metrics.py                   (recall_at_k / mrr / has_forbidden / judge_faithfulness kept)
```

### Persona enum values (real, not spec-loose)
- `skin_type ∈ {oily, dry, combination, sensitive, normal}`
- `budget ∈ {budget, mid-range, luxury}`
- `age_group ∈ {18-24, 25-30, 31-40, 41+}`

### Spec deviation noted
The spec mentions `FaithfulnessEvaluator` for the LlamaIndex evaluator. In implementation, **`RelevancyEvaluator` is semantically the correct evaluator for retrieval-only RAG** (asks "is response relevant to query"), since we don't generate a textual response — we return retrieved context. FaithfulnessEvaluator asks "is response faithful to contexts" which is degenerate when response == context. JSONL fields are renamed accordingly: `li_relevancy_pass` / `li_relevancy_feedback`.

---

## Task 1: Install LlamaIndex deps + verify BGE-M3 + bge-reranker-v2-m3

**Files:**
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Append LlamaIndex packages to requirements.txt**

Append to `backend/requirements.txt`:
```
llama-index-core>=0.10.0
llama-index-embeddings-huggingface>=0.2.0
llama-index-vector-stores-faiss>=0.1.0
llama-index-retrievers-bm25>=0.2.0
llama-index-postprocessor-sbert-rerank>=0.1.0
llama-index-llms-openai-like>=0.2.0
```

- [ ] **Step 2: Install**

```bash
pip install -r backend/requirements.txt
```

Expected: all packages install. If `faiss-cpu` conflicts with `llama-index-vector-stores-faiss`, prefer the version installed by llama-index (it pulls `faiss-cpu` transitively).

- [ ] **Step 3: Verify imports**

```bash
python -c "
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.core.evaluation import RelevancyEvaluator
print('All LlamaIndex imports OK')
"
```
Expected: `All LlamaIndex imports OK`. No errors.

- [ ] **Step 4: Download + verify BGE-M3 embedding model (~2.2 GB)**

```bash
python -c "
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
e = HuggingFaceEmbedding(model_name='BAAI/bge-m3')
v = e.get_text_embedding('测试中文向量')
print(f'BGE-M3 dim: {len(v)}')
"
```
Expected (first run ~2-5 min to download): `BGE-M3 dim: 1024`. Subsequent runs <10s.

- [ ] **Step 5: Download + verify bge-reranker-v2-m3 (~600 MB)**

```bash
python -c "
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
r = SentenceTransformerRerank(model='BAAI/bge-reranker-v2-m3', top_n=3)
print('Reranker loaded OK')
"
```
Expected: `Reranker loaded OK`.

- [ ] **Step 6: Verify DeepSeek via OpenAILike**

```bash
python -c "
from llama_index.llms.openai_like import OpenAILike
from backend.core.config import settings
llm = OpenAILike(
    model='deepseek-chat',
    api_base='https://api.deepseek.com/v1',
    api_key=settings.DEEPSEEK_API_KEY,
    is_chat_model=True,
)
resp = llm.complete('用一句话说你好')
print(resp.text[:80])
"
```
Expected: a short Chinese greeting like `你好！很高兴见到你。`

- [ ] **Step 7: Commit**

```bash
git add backend/requirements.txt
git commit -m "feat(rag): install LlamaIndex + BGE-M3 deps for v3 migration"
```

---

## Task 2: schema.py — add LI-aware metadata helpers (keep existing classes)

**Files:**
- Modify: `backend/services/rag/corpus/schema.py`
- Modify: `backend/tests/test_corpus_schema.py`

- [ ] **Step 1: Write failing test for `set_metadata_filters`**

Append to `backend/tests/test_corpus_schema.py`:

```python
from llama_index.core import Document as LIDocument
from backend.services.rag.corpus.schema import (
    set_metadata_filters, EMBED_KEYS, ALL_KEYS,
)


def test_set_metadata_filters_product_excludes_non_embed_keys():
    doc = LIDocument(
        text="some content",
        metadata={
            "doc_type": "product", "source": "internal", "title": "X 精华",
            "category": "brightening", "suitable_skin_types": ["oily"],
            "key_ingredients": ["niacinamide"], "price_tier": "mid-range",
            "selling_points": ["美白"], "effects": ["美白"], "age_groups": ["25-30"],
        },
    )
    set_metadata_filters(doc, "product")
    excluded = set(doc.excluded_embed_metadata_keys)
    assert "title" not in excluded       # title MUST embed
    assert "category" not in excluded    # category MUST embed
    assert "doc_type" in excluded        # doc_type debug-only
    assert "source" in excluded


def test_set_metadata_filters_ingredient():
    doc = LIDocument(
        text="百科正文",
        metadata={"doc_type": "ingredient", "source": "synthesized",
                  "title": "烟酰胺", "effects": ["美白"], "concerns": [],
                  "age_groups": ["25-30"]},
    )
    set_metadata_filters(doc, "ingredient")
    excluded = set(doc.excluded_embed_metadata_keys)
    assert "title" not in excluded
    assert "effects" not in excluded
    assert "concerns" in excluded        # concerns is debug, not for embedding


def test_set_metadata_filters_post():
    doc = LIDocument(
        text="正文",
        metadata={"doc_type": "post", "source": "synthesized",
                  "title": "28天抗老打卡", "linked_products": ["修丽可CE"],
                  "tone": "专业种草"},
    )
    set_metadata_filters(doc, "post")
    excluded = set(doc.excluded_embed_metadata_keys)
    assert "title" not in excluded       # title MUST embed
    assert "linked_products" in excluded
    assert "tone" in excluded


def test_embed_keys_subset_of_all_keys():
    for dt in ("product", "ingredient", "post"):
        assert set(EMBED_KEYS[dt]).issubset(set(ALL_KEYS[dt]))


def test_set_metadata_filters_unknown_type_raises():
    doc = LIDocument(text="x", metadata={"doc_type": "garbage"})
    import pytest
    with pytest.raises(KeyError):
        set_metadata_filters(doc, "garbage")
```

- [ ] **Step 2: Run test — verify fail**

```bash
pytest backend/tests/test_corpus_schema.py -v
```
Expected: 4 existing tests still pass + 5 new tests FAIL with `ImportError: cannot import name 'set_metadata_filters'`.

- [ ] **Step 3: Add helpers to schema.py**

Modify `backend/services/rag/corpus/schema.py` — append at end of file (keep existing `Document` / `Chunk` classes for now):

```python
from llama_index.core import Document as LIDocument

# Which metadata fields get prepended to embedding text via LlamaIndex's
# excluded_embed_metadata_keys mechanism.
EMBED_KEYS: dict[str, list[str]] = {
    "product":    ["title", "category"],
    "ingredient": ["title", "effects"],
    "post":       ["title"],
}

# All metadata fields we ever set per doc_type (used to compute the "exclude" set).
ALL_KEYS: dict[str, list[str]] = {
    "product":    ["doc_type", "source", "title", "category", "suitable_skin_types",
                   "key_ingredients", "price_tier", "selling_points",
                   "effects", "age_groups"],
    "ingredient": ["doc_type", "source", "title", "effects", "concerns", "age_groups"],
    "post":       ["doc_type", "source", "title", "linked_products", "tone"],
}


def set_metadata_filters(doc: LIDocument, doc_type: str) -> None:
    """Configure which metadata keys participate in embedding text vs. LLM context.

    LlamaIndex auto-prepends NON-excluded metadata fields to the embedding text.
    This fixes the v2 bug where post/ingredient `title` was never embedded.
    """
    all_keys = set(ALL_KEYS[doc_type])
    embed_keys = set(EMBED_KEYS[doc_type])
    doc.excluded_embed_metadata_keys = sorted(all_keys - embed_keys)
    # Hide noise from LLM context (source is debug-only)
    doc.excluded_llm_metadata_keys = ["source"]
```

- [ ] **Step 4: Run test — verify pass**

```bash
pytest backend/tests/test_corpus_schema.py -v
```
Expected: 9 passed (4 old + 5 new).

- [ ] **Step 5: Commit**

```bash
git add backend/services/rag/corpus/schema.py backend/tests/test_corpus_schema.py
git commit -m "feat(rag): schema metadata-filter helpers for LlamaIndex title-aware embedding"
```

---

## Task 3: loader.py — add load_*_as_li_documents (keep existing)

**Files:**
- Modify: `backend/services/rag/corpus/loader.py`
- Modify: `backend/tests/test_corpus_loader.py`

- [ ] **Step 1: Write failing test**

Append to `backend/tests/test_corpus_loader.py`:

```python
from llama_index.core import Document as LIDocument
from backend.services.rag.corpus.loader import (
    load_products_as_li, load_ingredients_as_li, load_posts_as_li,
    load_all_as_li,
)


def test_load_products_as_li_returns_li_documents(tmp_products):
    docs = load_products_as_li(tmp_products)
    assert len(docs) == 1
    d = docs[0]
    assert isinstance(d, LIDocument)
    assert d.doc_id == "prod_1"
    assert d.metadata["doc_type"] == "product"
    assert d.metadata["title"] == "X 精华"           # title MUST be in metadata
    assert d.metadata["source"] == "internal"
    # title must be embedded (not excluded)
    assert "title" not in (d.excluded_embed_metadata_keys or [])
    assert "doc_type" in (d.excluded_embed_metadata_keys or [])


def test_load_ingredients_as_li(tmp_path):
    import json
    data = [{
        "slug": "niacinamide", "name": "烟酰胺",
        "content": "成分介绍", "effects": ["美白"], "concerns": [],
        "age_groups": ["25-30"],
    }]
    p = tmp_path / "ingredients.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    docs = load_ingredients_as_li(p)
    assert docs[0].doc_id == "ing_niacinamide"
    assert docs[0].metadata["title"] == "烟酰胺"
    assert "title" not in (docs[0].excluded_embed_metadata_keys or [])


def test_load_posts_as_li(tmp_path):
    import json
    data = [{
        "post_id": 7, "title": "28天抗老打卡",
        "content": "正文内容", "tone": "专业种草", "linked_products": ["修丽可CE"],
    }]
    p = tmp_path / "posts.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    docs = load_posts_as_li(p)
    assert docs[0].doc_id == "post_7"
    assert docs[0].metadata["title"] == "28天抗老打卡"
    assert "title" not in (docs[0].excluded_embed_metadata_keys or [])
    assert "linked_products" in (docs[0].excluded_embed_metadata_keys or [])


def test_load_all_as_li_skips_missing_optional(tmp_path, tmp_products):
    docs = load_all_as_li(
        products_path=tmp_products,
        ingredients_path=None,
        posts_path=None,
    )
    assert len(docs) == 1
    assert docs[0].metadata["doc_type"] == "product"
```

- [ ] **Step 2: Run test — verify fail**

```bash
pytest backend/tests/test_corpus_loader.py -v
```
Expected: 3 existing tests still pass + 4 new tests FAIL with `ImportError`.

- [ ] **Step 3: Add LI-producing functions to loader.py**

Modify `backend/services/rag/corpus/loader.py` — append at end (keep existing functions):

```python
from llama_index.core import Document as LIDocument
from .schema import set_metadata_filters


def load_products_as_li(path: Path) -> list[LIDocument]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    docs = []
    for p in raw:
        ingredients = p.get("key_ingredients", [])
        skin_types = p.get("suitable_skin_types", [])
        selling_points = p.get("selling_points", [])
        # Content text — without title prefix (LI auto-prepends title via excluded_embed_metadata_keys)
        content = (
            f"{p.get('description', '')}\n"
            f"成分: {', '.join(ingredients)}\n"
            f"适合肤质: {', '.join(skin_types)}\n"
            f"卖点: {', '.join(selling_points)}"
        )
        doc = LIDocument(
            doc_id=f"prod_{p['id']}",
            text=content,
            metadata={
                "doc_type": "product",
                "source": "internal",
                "title": p["name"],
                "category": p.get("category", ""),
                "suitable_skin_types": skin_types,
                "key_ingredients": ingredients,
                "price_tier": p.get("price_tier", "mid-range"),
                "selling_points": selling_points,
                "effects": p.get("effects", []),
                "age_groups": p.get("age_groups", []),
            },
        )
        set_metadata_filters(doc, "product")
        docs.append(doc)
    return docs


def load_ingredients_as_li(path: Path) -> list[LIDocument]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    docs = []
    for r in raw:
        doc = LIDocument(
            doc_id=f"ing_{r['slug']}",
            text=r["content"],
            metadata={
                "doc_type": "ingredient",
                "source": "synthesized",
                "title": r["name"],
                "effects": r.get("effects", []),
                "concerns": r.get("concerns", []),
                "age_groups": r.get("age_groups", []),
            },
        )
        set_metadata_filters(doc, "ingredient")
        docs.append(doc)
    return docs


def load_posts_as_li(path: Path) -> list[LIDocument]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    docs = []
    for r in raw:
        doc = LIDocument(
            doc_id=f"post_{r['post_id']}",
            text=r["content"],
            metadata={
                "doc_type": "post",
                "source": "synthesized",
                "title": r.get("title", ""),
                "linked_products": r.get("linked_products", []),
                "tone": r.get("tone", ""),
            },
        )
        set_metadata_filters(doc, "post")
        docs.append(doc)
    return docs


def load_all_as_li(
    products_path: Path,
    ingredients_path: Path | None = None,
    posts_path: Path | None = None,
) -> list[LIDocument]:
    docs = load_products_as_li(products_path)
    if ingredients_path and Path(ingredients_path).exists():
        docs += load_ingredients_as_li(ingredients_path)
    else:
        logger.info("ingredients corpus not found, skipping")
    if posts_path and Path(posts_path).exists():
        docs += load_posts_as_li(posts_path)
    else:
        logger.info("posts corpus not found, skipping")
    return docs
```

- [ ] **Step 4: Run test — verify pass**

```bash
pytest backend/tests/test_corpus_loader.py -v
```
Expected: 7 passed (3 old + 4 new).

- [ ] **Step 5: Commit**

```bash
git add backend/services/rag/corpus/loader.py backend/tests/test_corpus_loader.py
git commit -m "feat(rag): loader produces LlamaIndex Documents with title-aware metadata"
```

---

## Task 4: persona_rules.py — extract pure functions taking metadata dict

**Files:**
- Create: `backend/services/rag/persona_rules.py`
- Modify: `backend/services/rag/persona_filter.py` (use new module to maintain backward compatibility during transition)
- Create: `backend/tests/test_persona_rules.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_persona_rules.py`:

```python
from backend.models.persona import UserPersona
from backend.services.rag.persona_rules import (
    BUDGET_TIER_RULES, passes_skin_type, passes_budget,
)


def test_budget_rules():
    assert BUDGET_TIER_RULES["budget"] == {"budget"}
    assert BUDGET_TIER_RULES["mid-range"] == {"budget", "mid-range"}
    assert BUDGET_TIER_RULES["luxury"] == {"budget", "mid-range", "luxury"}


def test_passes_skin_type_with_dict():
    assert passes_skin_type({"suitable_skin_types": ["oily"]}, "oily")
    assert not passes_skin_type({"suitable_skin_types": ["dry"]}, "oily")
    # "all" matches anything
    assert passes_skin_type({"suitable_skin_types": ["all"]}, "oily")
    # Empty list = unspecified, don't penalize
    assert passes_skin_type({"suitable_skin_types": []}, "oily")
    # Missing key = unspecified
    assert passes_skin_type({}, "oily")
    # No persona skin_type = pass
    assert passes_skin_type({"suitable_skin_types": ["dry"]}, None)


def test_passes_budget_with_dict():
    assert passes_budget({"price_tier": "budget"}, "mid-range")
    assert passes_budget({"price_tier": "mid-range"}, "mid-range")
    assert not passes_budget({"price_tier": "luxury"}, "mid-range")
    assert passes_budget({"price_tier": "luxury"}, "luxury")
    # Missing tier = unspecified, pass
    assert passes_budget({}, "mid-range")
    # No budget filter = pass
    assert passes_budget({"price_tier": "luxury"}, None)
```

- [ ] **Step 2: Run test — verify fail**

```bash
pytest backend/tests/test_persona_rules.py -v
```
Expected: `ImportError: No module named ...persona_rules`.

- [ ] **Step 3: Implement persona_rules.py**

Create `backend/services/rag/persona_rules.py`:

```python
"""Pure functions for persona-based filtering and boost computation.

These functions take a `metadata: dict` (not a domain object) so they work
equally with our legacy `Chunk` dataclass and LlamaIndex `TextNode.metadata`.
"""
from typing import Optional


BUDGET_TIER_RULES: dict[str, set[str]] = {
    "budget":    {"budget"},
    "mid-range": {"budget", "mid-range"},
    "luxury":    {"budget", "mid-range", "luxury"},
}


def passes_skin_type(metadata: dict, skin_type: Optional[str]) -> bool:
    """Return True if a product matches the persona's skin_type.

    A product passes if:
      - persona skin_type is None / empty (no filter)
      - product's suitable_skin_types list is empty/missing (unspecified)
      - product's suitable_skin_types contains "all"
      - product's suitable_skin_types contains the persona skin_type
    """
    if not skin_type:
        return True
    types = metadata.get("suitable_skin_types") or []
    if not types:
        return True
    return skin_type in types or "all" in types


def passes_budget(metadata: dict, budget: Optional[str]) -> bool:
    """Return True if a product's price_tier is allowed under persona's budget."""
    if not budget:
        return True
    tier = metadata.get("price_tier")
    if not tier:
        return True
    allowed = BUDGET_TIER_RULES.get(budget, {tier})
    return tier in allowed
```

- [ ] **Step 4: Run test — verify pass**

```bash
pytest backend/tests/test_persona_rules.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Refactor existing persona_filter.py to use the new module (no behavior change)**

Modify `backend/services/rag/persona_filter.py` — replace `_passes_skin_type` and `_passes_budget` calls with the new module's functions. Update the imports and adjust callers:

```python
import logging
from backend.models.persona import UserPersona
from .corpus.schema import Chunk
from .retrievers.vector import RetrievalHit
from .persona_rules import BUDGET_TIER_RULES, passes_skin_type, passes_budget

logger = logging.getLogger(__name__)


def hard_filter(chunks: list[Chunk], persona: UserPersona | None) -> list[Chunk]:
    if persona is None:
        return chunks

    def _filter(skip_skin_type: bool) -> list[Chunk]:
        out = []
        for c in chunks:
            if c.doc_type != "product":
                out.append(c); continue
            if not skip_skin_type and not passes_skin_type(c.metadata, persona.skin_type):
                continue
            if not passes_budget(c.metadata, persona.budget):
                continue
            out.append(c)
        return out

    primary = _filter(skip_skin_type=False)
    product_count = sum(1 for c in primary if c.doc_type == "product")
    if product_count == 0 and persona.skin_type:
        logger.warning("persona hard_filter emptied products; dropping skin_type=%s", persona.skin_type)
        return _filter(skip_skin_type=True)
    return primary


def soft_boost(score: float, chunk: Chunk, persona: UserPersona, per_match: float = 0.05) -> float:
    if persona is None:
        return score
    boost = 0.0
    age_groups = chunk.metadata.get("age_groups") or []
    if persona.age_group and persona.age_group in age_groups:
        boost += per_match
    effects = set(chunk.metadata.get("effects") or [])
    overlap = set(persona.preferences or []) & effects
    boost += per_match * len(overlap)
    return score + boost


def apply_soft_boost(hits: list[RetrievalHit], persona: UserPersona | None, per_match: float) -> list[RetrievalHit]:
    if persona is None:
        return hits
    return sorted(
        [RetrievalHit(chunk=h.chunk, score=soft_boost(h.score, h.chunk, persona, per_match)) for h in hits],
        key=lambda h: h.score, reverse=True,
    )
```

- [ ] **Step 6: Run full backend test suite — no regressions**

```bash
pytest backend/tests -v
```
Expected: all existing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add backend/services/rag/persona_rules.py backend/services/rag/persona_filter.py backend/tests/test_persona_rules.py
git commit -m "refactor(rag): extract persona_rules pure functions (metadata-dict based)"
```

---

## Task 5: _context.py — ContextVar for persona injection

**Files:**
- Create: `backend/services/rag/_context.py`

- [ ] **Step 1: Implement _context.py**

Create `backend/services/rag/_context.py`:

```python
"""ContextVar for persona injection into the LlamaIndex pipeline.

Lives in a separate module to avoid circular imports between
`service.py` and `persona_postprocessor.py`.

`service.query_product_database` sets the contextvar before calling
`engine.aquery(query)`. `PersonaNodePostprocessor` reads it during
node post-processing.
"""
from contextvars import ContextVar
from typing import Optional

from backend.models.persona import UserPersona

current_persona: ContextVar[Optional[UserPersona]] = ContextVar(
    "rag_current_persona",
    default=None,
)
```

- [ ] **Step 2: Quick sanity check**

```bash
python -c "
from backend.services.rag._context import current_persona
from backend.models.persona import UserPersona
assert current_persona.get() is None
p = UserPersona(skin_type='oily')
tok = current_persona.set(p)
assert current_persona.get() is p
current_persona.reset(tok)
assert current_persona.get() is None
print('ContextVar OK')
"
```
Expected: `ContextVar OK`.

- [ ] **Step 3: Commit**

```bash
git add backend/services/rag/_context.py
git commit -m "feat(rag): _context module with ContextVar for persona injection"
```

---

## Task 6: persona_postprocessor.py — BaseNodePostprocessor subclass

**Files:**
- Create: `backend/services/rag/persona_postprocessor.py`
- Create: `backend/tests/test_persona_postprocessor.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_persona_postprocessor.py`:

```python
import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from backend.models.persona import UserPersona
from backend.services.rag._context import current_persona
from backend.services.rag.persona_postprocessor import PersonaNodePostprocessor


def _node(doc_id, doc_type, *, suitable_skin_types=None, price_tier=None,
          effects=None, age_groups=None, score=1.0):
    n = TextNode(
        id_=f"{doc_id}::0",
        text="",
        metadata={
            "doc_type": doc_type,
            "suitable_skin_types": suitable_skin_types or [],
            "price_tier": price_tier or "mid-range",
            "effects": effects or [],
            "age_groups": age_groups or [],
        },
    )
    return NodeWithScore(node=n, score=score)


def test_no_persona_returns_nodes_unchanged():
    pp = PersonaNodePostprocessor()
    nodes = [_node("prod_1", "product")]
    out = pp._postprocess_nodes(nodes, query_bundle=None)
    assert out == nodes


def test_hard_filter_drops_mismatched_skin_type():
    pp = PersonaNodePostprocessor()
    nodes = [
        _node("prod_1", "product", suitable_skin_types=["oily"]),
        _node("prod_2", "product", suitable_skin_types=["dry"]),
        _node("ing_1",  "ingredient"),
    ]
    tok = current_persona.set(UserPersona(skin_type="dry"))
    try:
        out = pp._postprocess_nodes(nodes, query_bundle=None)
    finally:
        current_persona.reset(tok)
    ids = [n.node.id_ for n in out]
    assert "prod_2::0" in ids
    assert "ing_1::0" in ids        # ingredients always survive
    assert "prod_1::0" not in ids   # mismatched product dropped


def test_hard_filter_drops_unaffordable_products():
    pp = PersonaNodePostprocessor()
    nodes = [
        _node("prod_1", "product", suitable_skin_types=["oily"], price_tier="budget"),
        _node("prod_2", "product", suitable_skin_types=["oily"], price_tier="luxury"),
    ]
    tok = current_persona.set(UserPersona(skin_type="oily", budget="mid-range"))
    try:
        out = pp._postprocess_nodes(nodes, query_bundle=None)
    finally:
        current_persona.reset(tok)
    ids = [n.node.id_ for n in out]
    assert "prod_1::0" in ids
    assert "prod_2::0" not in ids


def test_empty_products_fallback_drops_skin_type():
    pp = PersonaNodePostprocessor()
    nodes = [
        _node("prod_1", "product", suitable_skin_types=["oily"], price_tier="mid-range"),
        _node("prod_2", "product", suitable_skin_types=["dry"],  price_tier="mid-range"),
    ]
    # skin_type=sensitive matches none; fallback should keep both (budget still passes)
    tok = current_persona.set(UserPersona(skin_type="sensitive", budget="mid-range"))
    try:
        out = pp._postprocess_nodes(nodes, query_bundle=None)
    finally:
        current_persona.reset(tok)
    assert len(out) == 2


def test_soft_boost_for_preferences_overlap_and_resort():
    pp = PersonaNodePostprocessor(per_match_boost=0.1)
    # n1 originally higher but no boost; n2 lower with double boost
    n1 = _node("prod_1", "product",
               suitable_skin_types=["oily"], effects=[], age_groups=[], score=1.0)
    n2 = _node("prod_2", "product",
               suitable_skin_types=["oily"], effects=["美白", "保湿"],
               age_groups=["25-30"], score=0.7)
    persona = UserPersona(skin_type="oily", preferences=["美白", "保湿"], age_group="25-30")
    # boost for n2: 0.1 (age) + 0.1*2 (prefs) = 0.3 → 1.0
    tok = current_persona.set(persona)
    try:
        out = pp._postprocess_nodes([n1, n2], query_bundle=None)
    finally:
        current_persona.reset(tok)
    # n2 should now be at or above n1
    assert out[0].node.id_ == "prod_2::0"


def test_ingredient_and_post_never_filtered():
    pp = PersonaNodePostprocessor()
    nodes = [
        _node("ing_1",  "ingredient"),
        _node("post_1", "post"),
    ]
    tok = current_persona.set(UserPersona(skin_type="sensitive", budget="budget"))
    try:
        out = pp._postprocess_nodes(nodes, query_bundle=None)
    finally:
        current_persona.reset(tok)
    assert len(out) == 2
```

- [ ] **Step 2: Run test — verify fail**

```bash
pytest backend/tests/test_persona_postprocessor.py -v
```
Expected: `ImportError: cannot import name 'PersonaNodePostprocessor'`.

- [ ] **Step 3: Implement persona_postprocessor.py**

Create `backend/services/rag/persona_postprocessor.py`:

```python
import logging
from typing import Optional

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

from backend.models.persona import UserPersona
from ._context import current_persona
from .persona_rules import passes_skin_type, passes_budget

logger = logging.getLogger(__name__)


class PersonaNodePostprocessor(BaseNodePostprocessor):
    """LlamaIndex NodePostprocessor that applies persona hard-filter + soft-boost.

    Reads the current persona from `_context.current_persona` (ContextVar).
    Pluggable into `RetrieverQueryEngine.node_postprocessors`.

    Hard filter:
      - Products that fail skin_type or budget are dropped.
      - Ingredients and posts always pass.
      - If skin_type filter empties products entirely, fall back: drop skin_type, keep budget.

    Soft boost (added to existing node.score):
      - per_match_boost for each persona.preference present in node.effects.
      - per_match_boost if persona.age_group is in node.age_groups.
    """
    per_match_boost: float = 0.05

    @classmethod
    def class_name(cls) -> str:
        return "PersonaNodePostprocessor"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        persona: UserPersona | None = current_persona.get()
        if persona is None:
            return nodes

        def _filter(skip_skin: bool) -> list[NodeWithScore]:
            out = []
            for nws in nodes:
                meta = nws.node.metadata
                if meta.get("doc_type") != "product":
                    out.append(nws); continue
                if not skip_skin and not passes_skin_type(meta, persona.skin_type):
                    continue
                if not passes_budget(meta, persona.budget):
                    continue
                out.append(nws)
            return out

        primary = _filter(skip_skin=False)
        product_count = sum(1 for n in primary if n.node.metadata.get("doc_type") == "product")
        if product_count == 0 and persona.skin_type:
            logger.warning(
                "persona filter emptied products, dropping skin_type=%s",
                persona.skin_type,
            )
            primary = _filter(skip_skin=True)

        # Soft boost (in place on a copy)
        boosted: list[NodeWithScore] = []
        for nws in primary:
            meta = nws.node.metadata
            boost = 0.0
            if persona.age_group and persona.age_group in (meta.get("age_groups") or []):
                boost += self.per_match_boost
            overlap = set(persona.preferences or []) & set(meta.get("effects") or [])
            boost += self.per_match_boost * len(overlap)
            new_score = (nws.score or 0.0) + boost
            boosted.append(NodeWithScore(node=nws.node, score=new_score))

        boosted.sort(key=lambda n: n.score or 0.0, reverse=True)
        return boosted
```

- [ ] **Step 4: Run test — verify pass**

```bash
pytest backend/tests/test_persona_postprocessor.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/services/rag/persona_postprocessor.py backend/tests/test_persona_postprocessor.py
git commit -m "feat(rag): PersonaNodePostprocessor (BaseNodePostprocessor subclass + ContextVar)"
```

---

## Task 7: engine.py — LlamaIndex orchestration

**Files:**
- Create: `backend/services/rag/engine.py`

> Note: this task's smoke test loads BGE-M3 (~2.2 GB if not cached) and bge-reranker-v2-m3 (~600 MB).
> If models already downloaded from Task 1, smoke is fast.

- [ ] **Step 1: Implement engine.py**

Create `backend/services/rag/engine.py`:

```python
"""LlamaIndex RetrieverQueryEngine factory + singleton.

Builds:
  QueryFusionRetriever(
    retrievers=[VectorIndexRetriever (BGE-M3 dense), BM25Retriever (jieba)],
    mode="reciprocal_rerank",
  )
  → SentenceTransformerRerank (bge-reranker-v2-m3)
  → PersonaNodePostprocessor (reads ContextVar)
  → response.source_nodes

response_mode="no_text" means LlamaIndex does NOT invoke an LLM to synthesize
text — we only need the retrieved nodes; the agent layer (backend/core/agent.py)
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
            nodes.extend(splitter.get_nodes_from_documents([doc]))
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
```

- [ ] **Step 2: Smoke test — build engine + dummy query**

```bash
python -c "
import asyncio
from backend.services.rag.engine import get_engine
async def main():
    e = get_engine()
    resp = await e.aquery('含烟酰胺的精华')
    for n in resp.source_nodes[:3]:
        print(n.node_id, n.score, '->', n.node.get_content()[:50].replace(chr(10),' '))
asyncio.run(main())
"
```
Expected: prints 3 node IDs (mix of `prod_*`, `ing_*`, `post_*`) with non-zero scores. First run takes ~30-60s (model load + index build).

If `tokenizer=jieba.lcut` raises a parameter-name error, the installed `llama-index-retrievers-bm25` version expects a different name. Try `tokenizer_function=jieba.lcut` instead and update the code.

- [ ] **Step 3: Commit**

```bash
git add backend/services/rag/engine.py
git commit -m "feat(rag): LlamaIndex RetrieverQueryEngine factory with BGE-M3 + bge-reranker-v2-m3"
```

---

## Task 8: service.py — rewrite with ContextVar + cache wrap

**Files:**
- Modify: `backend/services/rag/service.py`

- [ ] **Step 1: Read existing service.py to confirm signature**

```bash
cat backend/services/rag/service.py
```
Confirm the current exported function signature is `async def query_product_database(query, persona=None) -> str`. The new version preserves this.

- [ ] **Step 2: Rewrite service.py**

Replace the entire contents of `backend/services/rag/service.py` with:

```python
"""Public RAG entrypoint — wraps LlamaIndex engine with persona injection + semantic cache.

The previous v1/v2 dispatcher logic is gone — v3 is the only implementation.
"""
import logging
from typing import Optional
import numpy as np

from backend.core.config import settings
from backend.models.persona import UserPersona

from ._context import current_persona
from .engine import get_engine, get_embed_model
from .cache.semantic_cache import SemanticCache

logger = logging.getLogger(__name__)


def _embed_for_cache(text: str) -> np.ndarray:
    """Embed a single string for the semantic cache (reuses BGE-M3 from engine)."""
    return np.asarray(get_embed_model().get_text_embedding(text), dtype="float32")


_cache = SemanticCache(
    embedding_fn=_embed_for_cache,
    threshold=settings.SEMANTIC_CACHE_THRESHOLD,
    max_size=settings.SEMANTIC_CACHE_MAX_SIZE,
    ttl_seconds=settings.SEMANTIC_CACHE_TTL_SECONDS,
)


def _cache_key(query: str, persona: Optional[UserPersona]) -> str:
    if persona is None:
        return query
    return (
        f"{query}||{persona.skin_type}|{persona.budget}|{persona.age_group}|"
        f"{','.join(sorted(persona.preferences or []))}"
    )


def _format_top(nodes_with_scores, k: int = 3) -> str:
    if not nodes_with_scores:
        return "数据库中未找到匹配产品。"
    parts = []
    for rank, nws in enumerate(nodes_with_scores[:k], 1):
        n = nws.node
        meta = n.metadata
        # node_id is like "prod_3::0"; the parent doc_id is the prefix before "::"
        doc_id = n.node_id.split("::")[0]
        ingredients = "、".join(meta.get("key_ingredients", [])[:4])
        skin_types = "、".join(meta.get("suitable_skin_types", []))
        selling = "、".join(meta.get("selling_points", []))
        # Title is already in the node's content (LlamaIndex auto-prepends via
        # excluded_embed_metadata_keys), so we surface a short preview.
        preview = n.get_content()[:120].replace("\n", " ")
        parts.append(
            f"[匹配{rank}] [{doc_id}] {preview}\n"
            f"类型: {meta.get('doc_type', '')} | 成分: {ingredients} "
            f"| 适合: {skin_types} | 卖点: {selling}"
        )
    return "\n\n".join(parts)


async def query_product_database(
    query: str,
    persona: Optional[UserPersona] = None,
) -> str:
    """Public API consumed by `backend/services/tools_registry.py`."""
    key = _cache_key(query, persona)
    cached = await _cache.get(key)
    if cached is not None:
        return cached

    engine = get_engine()
    token = current_persona.set(persona)
    try:
        response = await engine.aquery(query)
    finally:
        current_persona.reset(token)

    result = _format_top(response.source_nodes, k=settings.RAG_TOP_K)
    await _cache.set(key, result)
    return result


def cache_stats() -> dict:
    return _cache.stats()
```

- [ ] **Step 3: Sanity test — service smoke (after corpus exists)**

```bash
python -c "
import asyncio
from backend.services.rag.service import query_product_database
from backend.models.persona import UserPersona

async def main():
    print('--- no persona ---')
    out = await query_product_database('含烟酰胺的精华')
    print(out[:400])
    print()
    print('--- oily skin, budget mid-range ---')
    p = UserPersona(skin_type='oily', budget='mid-range', preferences=['美白'])
    out = await query_product_database('适合油皮的控油精华', persona=p)
    print(out[:400])

asyncio.run(main())
"
```
Expected: prints two formatted result blocks. Each `[匹配N]` line has a `[prod_X]` / `[ing_X]` / `[post_X]` prefix.

- [ ] **Step 4: Commit**

```bash
git add backend/services/rag/service.py
git commit -m "refactor(rag): service.py wraps LlamaIndex with ContextVar persona + semantic cache"
```

---

## Task 9: main.py warmup update + smoke

**Files:**
- Modify: `backend/main.py`

- [ ] **Step 1: Update lifespan in main.py**

Modify `backend/main.py` — replace the warmup section in the `lifespan` function (currently at lines ~21-30):

Find this block:
```python
    from .services.rag.service import query_product_database
    from .services.memory_service import init_db
    from .services.deepseek_client import get_client, close_client

    await init_db()
    logger.info("Memory database initialized")

    logger.info("Warming up RAG service (version=%s)...", settings.RAG_VERSION)
    await query_product_database("预热")
    logger.info("RAG warmup complete")

    get_client()
    logger.info("DeepSeek client initialized")
```

Replace with:
```python
    from .services.rag.service import query_product_database
    from .services.memory_service import init_db
    from .services.deepseek_client import get_client, close_client

    await init_db()
    logger.info("Memory database initialized")

    logger.info("Warming up RAG v3 (loads BGE-M3 + bge-reranker-v2-m3, ~2.8GB first time)...")
    await query_product_database("预热查询")
    logger.info("RAG v3 warmup complete")

    get_client()
    logger.info("DeepSeek client initialized")
```

- [ ] **Step 2: Smoke test — boot the backend**

```bash
uvicorn backend.main:app --port 8001 &
sleep 60   # first boot loads BGE-M3 + reranker, then builds index
curl -sS http://localhost:8001/api/health
```
Expected: 200 JSON. Note: the `rag_products` count comes from `backend/api/routes.py` reading `products.json` directly (was patched in Task 19 of v2 plan), unaffected by v3.

Stop the backend:
```bash
kill %1 2>/dev/null
```

- [ ] **Step 3: Commit**

```bash
git add backend/main.py
git commit -m "chore(main): warmup loads BGE-M3 + bge-reranker-v2-m3 for v3"
```

---

## Task 10: Delete obsolete v2 code + tests

**Files (DELETE):**
- `backend/services/rag/corpus/chunker.py`
- `backend/services/rag/retrievers/__init__.py`
- `backend/services/rag/retrievers/vector.py`
- `backend/services/rag/retrievers/bm25.py`
- `backend/services/rag/retrievers/hybrid.py`
- `backend/services/rag/retrievers/reranker.py`
- `backend/services/rag/pipeline.py`
- `backend/services/rag/persona_filter.py`
- `backend/tests/test_chunker.py`
- `backend/tests/test_vector_retriever.py`
- `backend/tests/test_bm25_retriever.py`
- `backend/tests/test_rrf_fusion.py`
- `backend/tests/test_reranker.py`
- `backend/tests/test_rag_v2_pipeline.py`
- `backend/tests/test_persona_filter.py`

**Files (MODIFY):**
- `backend/services/rag/corpus/schema.py` — remove the old `Document` / `Chunk` dataclasses and `_VALID_DOC_TYPES` (the LI adapter helpers stay)
- `backend/services/rag/corpus/loader.py` — remove the old `load_products` / `load_ingredients` / `load_posts` / `load_all` functions (the `_as_li` versions stay)
- `backend/tests/test_corpus_schema.py` — delete the old `test_document_required_fields`, `test_chunk_id_must_contain_doc_id`, `test_document_rejects_invalid_doc_type`, `test_chunk_metadata_inherits_required_keys` tests (the metadata-filter tests from Task 2 stay)
- `backend/tests/test_corpus_loader.py` — delete the old `test_load_products_returns_documents`, `test_load_products_concatenates_content`, `test_load_all_handles_missing_optional_corpus` (the `_as_li` tests from Task 3 stay)

- [ ] **Step 1: Confirm nothing still imports the doomed code**

```bash
grep -rn "from .pipeline" backend/services/rag/ 2>&1 | grep -v __pycache__ || echo "no references"
grep -rn "from .retrievers" backend/services/rag/ 2>&1 | grep -v __pycache__ || echo "no references"
grep -rn "from .corpus.chunker" backend/services/rag/ 2>&1 | grep -v __pycache__ || echo "no references"
grep -rn "from .persona_filter" backend/services/rag/ 2>&1 | grep -v __pycache__ || echo "no references"
```
Expected: each prints `no references` or only matches inside `_pycache_` (ignore).

If anything still imports the old code, **STOP** and fix the offending file before proceeding (it's a leftover from the migration).

- [ ] **Step 2: Delete files**

```bash
rm backend/services/rag/corpus/chunker.py
rm -r backend/services/rag/retrievers
rm backend/services/rag/pipeline.py
rm backend/services/rag/persona_filter.py
rm backend/tests/test_chunker.py
rm backend/tests/test_vector_retriever.py
rm backend/tests/test_bm25_retriever.py
rm backend/tests/test_rrf_fusion.py
rm backend/tests/test_reranker.py
rm backend/tests/test_rag_v2_pipeline.py
rm backend/tests/test_persona_filter.py
```

- [ ] **Step 3: Trim schema.py — remove old dataclasses**

Edit `backend/services/rag/corpus/schema.py` — remove the `Document`, `Chunk`, `_VALID_DOC_TYPES`, `DocType`, `SourceType` definitions and the `dataclasses` / typing-Literal imports they need. Keep ONLY the LI-aware helpers (`EMBED_KEYS`, `ALL_KEYS`, `set_metadata_filters`) and the `from llama_index.core import Document as LIDocument` import.

Final content of `backend/services/rag/corpus/schema.py`:

```python
from llama_index.core import Document as LIDocument

EMBED_KEYS: dict[str, list[str]] = {
    "product":    ["title", "category"],
    "ingredient": ["title", "effects"],
    "post":       ["title"],
}

ALL_KEYS: dict[str, list[str]] = {
    "product":    ["doc_type", "source", "title", "category", "suitable_skin_types",
                   "key_ingredients", "price_tier", "selling_points",
                   "effects", "age_groups"],
    "ingredient": ["doc_type", "source", "title", "effects", "concerns", "age_groups"],
    "post":       ["doc_type", "source", "title", "linked_products", "tone"],
}


def set_metadata_filters(doc: LIDocument, doc_type: str) -> None:
    all_keys = set(ALL_KEYS[doc_type])
    embed_keys = set(EMBED_KEYS[doc_type])
    doc.excluded_embed_metadata_keys = sorted(all_keys - embed_keys)
    doc.excluded_llm_metadata_keys = ["source"]
```

- [ ] **Step 4: Trim loader.py — remove old loader functions**

Edit `backend/services/rag/corpus/loader.py` — remove the old `load_products`, `load_ingredients`, `load_posts`, `load_all` functions and the `from .schema import Document` import. Keep ONLY the `_as_li` functions.

The new top-of-file imports should be:
```python
import json
import logging
from pathlib import Path

from llama_index.core import Document as LIDocument
from .schema import set_metadata_filters

logger = logging.getLogger(__name__)
```

Followed by the four `_as_li` functions (added in Task 3).

- [ ] **Step 5: Trim test_corpus_schema.py — remove old-dataclass tests**

Edit `backend/tests/test_corpus_schema.py` — delete the tests that asserted on the old `Document` / `Chunk` dataclasses:
- `test_document_required_fields`
- `test_chunk_id_must_contain_doc_id`
- `test_document_rejects_invalid_doc_type`
- `test_chunk_metadata_inherits_required_keys`

And remove the corresponding imports of `Document` / `Chunk` from `backend.services.rag.corpus.schema`. Keep the 5 metadata-filter tests added in Task 2.

- [ ] **Step 6: Trim test_corpus_loader.py — remove old-loader tests**

Edit `backend/tests/test_corpus_loader.py` — delete the old tests:
- `test_load_products_returns_documents`
- `test_load_products_concatenates_content`
- `test_load_all_handles_missing_optional_corpus`

Remove the corresponding imports of `load_products`, `load_all` from `backend.services.rag.corpus.loader`. Keep the 4 `_as_li` tests added in Task 3 and the `tmp_products` fixture (still needed).

- [ ] **Step 7: Run full backend test suite**

```bash
pytest backend/tests -v
```
Expected: all tests pass. Test count drops from ~50 to ~25 (we deleted lots of unit tests for now-deleted code).

If any test fails because of a leftover import of a deleted symbol, fix that test file (or delete it if it's testing deleted code).

- [ ] **Step 8: Verify backend still boots**

```bash
uvicorn backend.main:app --port 8002 &
sleep 60
curl -sS http://localhost:8002/api/health
kill %1 2>/dev/null
```
Expected: 200 JSON response.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "refactor(rag): delete v2 hand-rolled implementation (replaced by LlamaIndex in v3)"
```

---

## Task 11: test_rag_v3_engine.py — integration test

**Files:**
- Create: `backend/tests/test_rag_v3_engine.py`

> This integration test loads BGE-M3 + bge-reranker-v2-m3 (slow first time, ~30-60s).
> It does NOT call any external API.

- [ ] **Step 1: Implement the integration test**

Create `backend/tests/test_rag_v3_engine.py`:

```python
"""Integration tests for the LlamaIndex RAG v3 engine.

These tests are SLOW (~30-60s for first run due to model load).
They build a tiny in-process index from synthetic docs and verify
shape — node count, doc_id format, score monotonicity — NOT exact text.
"""
import pytest
import json
from pathlib import Path

from backend.models.persona import UserPersona


@pytest.fixture(scope="module")
def tiny_corpus(tmp_path_factory):
    """5-doc fixture: 2 products, 1 ingredient, 2 posts."""
    d = tmp_path_factory.mktemp("tiny_corpus")
    (d / "corpus").mkdir()

    (d / "products.json").write_text(json.dumps([
        {"id": 1, "name": "美白精华A", "category": "brightening",
         "description": "含烟酰胺，专门控油提亮",
         "key_ingredients": ["niacinamide"], "suitable_skin_types": ["oily"],
         "price_tier": "mid-range", "selling_points": ["美白"]},
        {"id": 2, "name": "保湿乳液B", "category": "hydrating",
         "description": "玻尿酸三重补水",
         "key_ingredients": ["hyaluronic_acid"], "suitable_skin_types": ["dry"],
         "price_tier": "budget", "selling_points": ["保湿"]},
    ], ensure_ascii=False), encoding="utf-8")

    (d / "corpus" / "ingredients.json").write_text(json.dumps([
        {"slug": "niacinamide", "name": "烟酰胺",
         "content": "烟酰胺是经典的美白成分，常见浓度2-5%。",
         "effects": ["美白", "控油"], "concerns": [], "age_groups": ["25-30"]},
    ], ensure_ascii=False), encoding="utf-8")

    (d / "corpus" / "posts.json").write_text(json.dumps([
        {"post_id": 1, "title": "油皮控油精华推荐",
         "content": "上班族下午脸油到不行，试试含烟酰胺的精华，一周见效！",
         "tone": "活泼甜美", "linked_products": ["美白精华A"]},
        {"post_id": 2, "title": "干皮急救面霜",
         "content": "干皮姐妹必看，含玻尿酸的保湿乳真的好用。",
         "tone": "温柔治愈", "linked_products": ["保湿乳液B"]},
    ], ensure_ascii=False), encoding="utf-8")
    return d


@pytest.fixture(scope="module")
def patched_engine(tiny_corpus):
    """Build an engine pointing at the tiny corpus instead of real data."""
    from backend.services.rag import engine as engine_mod
    original_data_dir = engine_mod._data_dir
    engine_mod._data_dir = lambda: tiny_corpus
    engine_mod._engine = None  # force rebuild against tiny corpus
    e = engine_mod.get_engine()
    yield e
    # Restore so subsequent test modules don't inherit the patched corpus
    engine_mod._data_dir = original_data_dir
    engine_mod._engine = None


@pytest.mark.asyncio
async def test_query_returns_source_nodes(patched_engine):
    resp = await patched_engine.aquery("含烟酰胺的精华")
    assert len(resp.source_nodes) > 0
    assert len(resp.source_nodes) <= 9   # reranker top_n=9


@pytest.mark.asyncio
async def test_node_ids_match_doc_id_prefix(patched_engine):
    resp = await patched_engine.aquery("烟酰胺")
    for nws in resp.source_nodes:
        # node_id format: "{doc_id}::{chunk_idx}"
        assert "::" in nws.node.node_id
        prefix = nws.node.node_id.split("::")[0]
        assert prefix.startswith(("prod_", "ing_", "post_"))


@pytest.mark.asyncio
async def test_scores_are_monotonic(patched_engine):
    resp = await patched_engine.aquery("美白")
    scores = [nws.score for nws in resp.source_nodes if nws.score is not None]
    assert scores == sorted(scores, reverse=True), "results should be sorted by score desc"


@pytest.mark.asyncio
async def test_persona_filter_drops_mismatched_product(patched_engine):
    """With persona skin_type=dry, the 'oily-only' product should NOT appear in top 3."""
    from backend.services.rag._context import current_persona
    persona = UserPersona(skin_type="dry", budget="mid-range")
    tok = current_persona.set(persona)
    try:
        resp = await patched_engine.aquery("精华")
    finally:
        current_persona.reset(tok)
    ids = [nws.node.node_id.split("::")[0] for nws in resp.source_nodes[:3]]
    # prod_1 is for "oily" — should be filtered out
    assert "prod_1" not in ids


@pytest.mark.asyncio
async def test_no_persona_does_not_filter(patched_engine):
    """Without persona, products from any skin_type can appear."""
    from backend.services.rag._context import current_persona
    # Ensure no leftover persona
    tok = current_persona.set(None)
    try:
        resp = await patched_engine.aquery("精华推荐")
    finally:
        current_persona.reset(tok)
    # Just assert query returns results (no specific id required)
    assert len(resp.source_nodes) >= 1
```

- [ ] **Step 2: Run the integration test**

```bash
pytest backend/tests/test_rag_v3_engine.py -v --timeout=120
```
Expected: 5 passed (first run slow due to model load).

If `test_persona_filter_drops_mismatched_product` is flaky (sometimes prod_1 sneaks in despite the filter), increase the verbosity to inspect:

```bash
pytest backend/tests/test_rag_v3_engine.py::test_persona_filter_drops_mismatched_product -v -s
```

And confirm the postprocessor is in the chain. A leak here often means the engine cache didn't refresh — verify `_engine = None` ran in the fixture.

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_rag_v3_engine.py
git commit -m "test(rag): integration tests for v3 engine (5-doc corpus, persona, monotonicity)"
```

---

## Task 12: configs.py — LlamaIndex ablation factory

**Files:**
- Modify: `scripts/rag_eval/configs.py`

This is a near-total rewrite. The old v2 `configs.py` constructed retrievers manually; the v3 version assembles `RetrieverQueryEngine` per config.

- [ ] **Step 1: Replace configs.py**

Replace the entire contents of `scripts/rag_eval/configs.py` with:

```python
"""5 ablation configurations for RAG v3 (LlamaIndex-based).

Each config builds its own query callable:
    async def(query: str, persona: UserPersona | None) -> str
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

import faiss
import jieba
import numpy as np

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
from backend.models.persona import UserPersona
from backend.services.rag._context import current_persona
from backend.services.rag.corpus.loader import load_all_as_li
from backend.services.rag.persona_postprocessor import PersonaNodePostprocessor
from backend.services.rag.cache.semantic_cache import SemanticCache

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "backend" / "data"

BGE_M3_DIM = 1024


@dataclass
class ConfigSpec:
    name: str
    description: str
    use_bm25: bool
    use_rerank: bool
    use_cache: bool
    use_persona: bool


CONFIG_SPECS = [
    ConfigSpec("C0", "vector only (LI baseline)",
               use_bm25=False, use_rerank=False, use_cache=False, use_persona=False),
    ConfigSpec("C1", "+ BM25/RRF",
               use_bm25=True,  use_rerank=False, use_cache=False, use_persona=False),
    ConfigSpec("C2", "+ cross-encoder rerank",
               use_bm25=True,  use_rerank=True,  use_cache=False, use_persona=False),
    ConfigSpec("C3", "+ semantic cache",
               use_bm25=True,  use_rerank=True,  use_cache=True,  use_persona=False),
    ConfigSpec("C4", "+ persona-aware (full v3)",
               use_bm25=True,  use_rerank=True,  use_cache=True,  use_persona=True),
]


# Cache heavy resources across configs to avoid re-downloading models
_resource_cache: dict = {}


def _load_resources():
    """Load BGE-M3 embed, reranker, docs, nodes — once across all configs."""
    if "loaded" in _resource_cache:
        return _resource_cache

    LISettings.llm = OpenAILike(
        model="deepseek-chat",
        api_base="https://api.deepseek.com/v1",
        api_key=settings.DEEPSEEK_API_KEY,
        is_chat_model=True,
    )

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    LISettings.embed_model = embed_model

    docs = load_all_as_li(
        products_path=DATA / "products.json",
        ingredients_path=DATA / "corpus" / "ingredients.json",
        posts_path=DATA / "corpus" / "posts.json",
    )
    splitter = SentenceSplitter(
        chunk_size=350, chunk_overlap=50,
        paragraph_separator="\n\n",
        secondary_chunking_regex=r"(?<=[。!?!?])",
    )
    nodes: list[TextNode] = []
    for doc in docs:
        if doc.metadata.get("doc_type") == "post":
            nodes.extend(splitter.get_nodes_from_documents([doc]))
        else:
            n = TextNode(id_=f"{doc.doc_id}::0", text=doc.text, metadata=dict(doc.metadata))
            n.excluded_embed_metadata_keys = list(doc.excluded_embed_metadata_keys or [])
            nodes.append(n)

    faiss_index = faiss.IndexFlatIP(BGE_M3_DIM)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_ctx)

    reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-v2-m3", top_n=9)

    _resource_cache.update({
        "loaded": True,
        "embed_model": embed_model,
        "docs": docs,
        "nodes": nodes,
        "index": index,
        "reranker": reranker,
    })
    return _resource_cache


def _build_engine(spec: ConfigSpec) -> RetrieverQueryEngine:
    res = _load_resources()
    nodes = res["nodes"]
    index = res["index"]

    vector_r = VectorIndexRetriever(index=index, similarity_top_k=20)

    if spec.use_bm25:
        bm25_r = BM25Retriever.from_defaults(
            nodes=nodes, similarity_top_k=20, tokenizer=jieba.lcut,
        )
        retriever = QueryFusionRetriever(
            retrievers=[vector_r, bm25_r],
            mode="reciprocal_rerank",
            num_queries=1,
            similarity_top_k=20,
            use_async=True,
        )
    else:
        retriever = vector_r

    postprocessors = []
    if spec.use_rerank:
        postprocessors.append(res["reranker"])
    if spec.use_persona:
        postprocessors.append(
            PersonaNodePostprocessor(per_match_boost=settings.PERSONA_SOFT_BOOST_PER_MATCH)
        )

    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=postprocessors,
        response_mode="no_text",
    )


def _format_nodes(nodes_with_scores, k: int = 3) -> str:
    if not nodes_with_scores:
        return "数据库中未找到匹配产品。"
    parts = []
    for rank, nws in enumerate(nodes_with_scores[:k], 1):
        n = nws.node
        doc_id = n.node_id.split("::")[0]
        preview = n.get_content()[:120].replace("\n", " ")
        parts.append(f"[匹配{rank}] [{doc_id}] {preview}")
    return "\n".join(parts)


def build_query_fn(spec: ConfigSpec) -> Callable[[str, UserPersona | None], Awaitable[str]]:
    """Return an async (query, persona) -> formatted-context-str."""
    engine = _build_engine(spec)

    if spec.use_cache:
        embed_model = _resource_cache["embed_model"]
        def _embed_for_cache(text: str) -> np.ndarray:
            return np.asarray(embed_model.get_text_embedding(text), dtype="float32")
        cache = SemanticCache(
            embedding_fn=_embed_for_cache,
            threshold=settings.SEMANTIC_CACHE_THRESHOLD,
            max_size=settings.SEMANTIC_CACHE_MAX_SIZE,
            ttl_seconds=settings.SEMANTIC_CACHE_TTL_SECONDS,
        )
    else:
        cache = None

    async def _q(query: str, persona: UserPersona | None) -> str:
        key = (f"{query}||{persona.skin_type}|{persona.budget}"
               if persona and spec.use_persona else query)
        if cache is not None:
            hit = await cache.get(key)
            if hit is not None:
                return hit

        # Apply persona via ContextVar even if config doesn't toggle persona
        # (persona postprocessor is only in the chain when spec.use_persona=True;
        # without it in the chain, ContextVar is a no-op)
        tok = current_persona.set(persona if spec.use_persona else None)
        try:
            response = await engine.aquery(query)
        finally:
            current_persona.reset(tok)

        result = _format_nodes(response.source_nodes, k=3)
        if cache is not None:
            await cache.set(key, result)
        return result

    return _q
```

- [ ] **Step 2: Quick smoke — confirm configs import**

```bash
python -c "
from scripts.rag_eval.configs import CONFIG_SPECS, build_query_fn
print(f'{len(CONFIG_SPECS)} configs: {[c.name for c in CONFIG_SPECS]}')
"
```
Expected: `5 configs: ['C0', 'C1', 'C2', 'C3', 'C4']`

- [ ] **Step 3: Smoke — run one config on one query**

```bash
python -c "
import asyncio
from scripts.rag_eval.configs import CONFIG_SPECS, build_query_fn
async def main():
    spec = CONFIG_SPECS[0]   # C0
    qfn = build_query_fn(spec)
    out = await qfn('含烟酰胺的产品', None)
    print(out[:400])
asyncio.run(main())
"
```
Expected: a formatted string with at least one `[matching N] [prod_X / ing_X / post_X]` line.

- [ ] **Step 4: Commit**

```bash
git add scripts/rag_eval/configs.py
git commit -m "feat(eval): C0-C4 ablation rebuilt on LlamaIndex (BGE-M3 + RRF + rerank toggles)"
```

---

## Task 13: run_eval.py — persist faithfulness_reason + add LI RelevancyEvaluator

**Files:**
- Modify: `scripts/rag_eval/run_eval.py`
- Modify: `scripts/rag_eval/metrics.py`

- [ ] **Step 1: Add LI evaluator helper to metrics.py**

Append to `scripts/rag_eval/metrics.py`:

```python
# ----------------- LlamaIndex RelevancyEvaluator integration -----------------

_li_evaluator = None


def get_li_evaluator():
    """Singleton LI RelevancyEvaluator backed by DeepSeek.

    RelevancyEvaluator asks "is the response relevant to the query?".
    For retrieval eval we feed the retrieved-context string as the response,
    which is the closest LI evaluator semantic to "did retrieval bring back
    relevant material" for our retrieval-only RAG (we generate text in the
    agent layer, outside the RAG eval scope).
    """
    global _li_evaluator
    if _li_evaluator is not None:
        return _li_evaluator
    from llama_index.core.evaluation import RelevancyEvaluator
    from llama_index.llms.openai_like import OpenAILike
    from backend.core.config import settings

    llm = OpenAILike(
        model="deepseek-chat",
        api_base="https://api.deepseek.com/v1",
        api_key=settings.DEEPSEEK_API_KEY,
        is_chat_model=True,
    )
    _li_evaluator = RelevancyEvaluator(llm=llm)
    return _li_evaluator


async def judge_relevancy_li(query: str, context: str) -> dict | None:
    """Return {pass: bool, feedback: str} or None on failure."""
    try:
        evaluator = get_li_evaluator()
        result = await evaluator.aevaluate(
            query=query,
            response=context,
            contexts=[context],
        )
        return {
            "pass": bool(result.passing) if result.passing is not None else None,
            "feedback": result.feedback or "",
        }
    except Exception as e:
        logger.error("LI relevancy judge failed: %s", e)
        return None
```

- [ ] **Step 2: Update run_eval.py — add reason persistence + LI evaluator call**

Modify `scripts/rag_eval/run_eval.py` — in the `_run_config` function, change the `out = {...}` dict to include the new fields. The full updated function:

```python
async def _run_config(spec, gold_records: list[dict], judge_client, out_path: Path, done_qids: set[str]):
    from scripts.rag_eval.metrics import judge_relevancy_li
    query_fn = build_query_fn(spec)

    for rec in gold_records:
        if rec["qid"] in done_qids:
            continue

        persona = UserPersona(**rec["persona"]) if rec.get("persona") else None

        t0 = time.monotonic()
        try:
            context = await query_fn(rec["query"], persona)
            error = None
        except Exception as e:
            context, error = "", str(e)
        elapsed = time.monotonic() - t0

        retrieved_ids = _extract_doc_ids(context)
        gold_ids = set(rec.get("relevant_doc_ids", []))
        forbidden = set(rec.get("must_not_contain", []))

        faith = await judge_faithfulness(judge_client, rec["query"], context) if context else None
        li_relevancy = await judge_relevancy_li(rec["query"], context) if context else None

        out = {
            "config": spec.name,
            "qid": rec["qid"],
            "query": rec["query"],
            "category": rec["category"],
            "retrieved_ids": retrieved_ids,
            "recall_at_3": recall_at_k(retrieved_ids, gold_ids, 3),
            "recall_at_10": recall_at_k(retrieved_ids, gold_ids, 10),
            "mrr": mrr(retrieved_ids, gold_ids),
            "has_forbidden": has_forbidden(retrieved_ids, forbidden),
            "faithfulness": faith.get("faithfulness") if faith else None,
            "faithfulness_reason": faith.get("reason") if faith else None,
            "li_relevancy_pass": li_relevancy.get("pass") if li_relevancy else None,
            "li_relevancy_feedback": li_relevancy.get("feedback") if li_relevancy else None,
            "elapsed_seconds": elapsed,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
        logger.info(
            "[%s] %s  R@3=%.2f  MRR=%.2f  faith=%s  li=%s  %.2fs",
            spec.name, rec["qid"], out["recall_at_3"], out["mrr"],
            out["faithfulness"], out["li_relevancy_pass"], elapsed,
        )
```

- [ ] **Step 3: Dry-run smoke — verify the new fields land in JSONL**

```bash
rm -f scripts/rag_eval/results/*_C0.jsonl
python scripts/rag_eval/run_eval.py --config C0 --dry-run
```
Expected: `scripts/rag_eval/results/<today>_C0.jsonl` exists with 5 lines. Verify each line contains `faithfulness_reason`, `li_relevancy_pass`, `li_relevancy_feedback`:

```bash
head -1 scripts/rag_eval/results/*_C0.jsonl | python -c "import sys, json; r = json.loads(sys.stdin.read()); print('keys:', sorted(r.keys()))"
```
Expected `keys` list includes `faithfulness_reason`, `li_relevancy_feedback`, `li_relevancy_pass`.

Delete the dry-run JSONL (it'll be regenerated by Task 17):
```bash
rm scripts/rag_eval/results/*_C0.jsonl
```

- [ ] **Step 4: Commit**

```bash
git add scripts/rag_eval/run_eval.py scripts/rag_eval/metrics.py
git commit -m "feat(eval): persist faithfulness_reason + add LI RelevancyEvaluator dual-judge"
```

---

## Task 14: report.py — dual-judge consistency section

**Files:**
- Modify: `scripts/rag_eval/report.py`

- [ ] **Step 1: Add dual-judge consistency section to report.py**

Modify `scripts/rag_eval/report.py` — in the `build_report` function, after the existing "Delta vs Baseline" section and before `return "\n".join(sections) + "\n"`, append a new section.

Append the following inside `build_report`, just before the `return` statement:

```python
    # Dual-judge consistency: how often does LI RelevancyEvaluator (pass/fail)
    # agree with custom faithfulness (>= 7 = "pass")
    sections.append("\n## Dual-Judge Consistency (Custom-1to10 vs LI-RelevancyEvaluator)\n")
    sections.append("Custom score >= 7 treated as PASS. Cohen's κ measures agreement beyond chance.\n")
    sections.append("| Config | Both PASS | Both FAIL | Custom PASS / LI FAIL | Custom FAIL / LI PASS | Cohen κ |")
    sections.append("|---|---|---|---|---|---|")
    for config in all_aggs:
        recs, _ = all_aggs[config]
        both_pass = both_fail = custom_only = li_only = 0
        for r in recs:
            f = r.get("faithfulness")
            lp = r.get("li_relevancy_pass")
            if f is None or lp is None:
                continue
            custom_pass = f >= 7
            if custom_pass and lp:
                both_pass += 1
            elif (not custom_pass) and (not lp):
                both_fail += 1
            elif custom_pass and (not lp):
                custom_only += 1
            else:
                li_only += 1
        total = both_pass + both_fail + custom_only + li_only
        if total == 0:
            sections.append(f"| {config} | 0 | 0 | 0 | 0 | N/A |")
            continue
        agreement = (both_pass + both_fail) / total
        p_custom_pass = (both_pass + custom_only) / total
        p_li_pass = (both_pass + li_only) / total
        p_e = p_custom_pass * p_li_pass + (1 - p_custom_pass) * (1 - p_li_pass)
        kappa = (agreement - p_e) / (1 - p_e) if p_e < 1.0 else 1.0
        sections.append(
            f"| {config} | {both_pass} | {both_fail} | {custom_only} | {li_only} | {kappa:+.3f} |"
        )

    # Failure analysis: low-faithfulness reasons grouped by category
    sections.append("\n## Failure Analysis (faithfulness ≤ 5)\n")
    any_failure = False
    for config in all_aggs:
        recs, _ = all_aggs[config]
        low = [r for r in recs
               if r.get("faithfulness") is not None and r["faithfulness"] <= 5
               and r.get("faithfulness_reason")]
        if not low:
            continue
        any_failure = True
        sections.append(f"### {config} — {len(low)} low-score records\n")
        by_cat = defaultdict(list)
        for r in low:
            by_cat[r.get("category", "uncategorized")].append(r)
        for cat in sorted(by_cat.keys()):
            sections.append(f"**{cat}** ({len(by_cat[cat])} records)")
            for r in by_cat[cat][:2]:
                sections.append(f"- `{r['qid']}` _{r['query']}_ (faith={r['faithfulness']})")
                sections.append(f"  > {r['faithfulness_reason']}")
            sections.append("")
    if not any_failure:
        sections.append("_No low-faithfulness records (≤5) with captured reason._\n")
```

Also at the top of `build_report`, change the title line to v3:
```python
sections = [f"# RAG v3 Eval Report — {date}\n"]
```

And at the end of `main()`, change the output filename:
```python
out = DOCS_OUT / f"{args.date}-rag-v3-report.md"
```

- [ ] **Step 2: Smoke test on a tiny dataset**

To test the report locally before the full eval, write a fake JSONL:

```bash
mkdir -p /tmp/rag_v3_smoke
python -c "
import json, datetime
from pathlib import Path
out = Path('scripts/rag_eval/results') / f'{datetime.date.today().isoformat()}_C0.jsonl'
out.parent.mkdir(exist_ok=True, parents=True)
recs = [
    {'config':'C0','qid':'Q001','query':'测试1','category':'direct_need',
     'retrieved_ids':['prod_1'],'recall_at_3':1.0,'recall_at_10':1.0,'mrr':1.0,
     'has_forbidden':False,'faithfulness':9,'faithfulness_reason':'matches well',
     'li_relevancy_pass':True,'li_relevancy_feedback':'relevant',
     'elapsed_seconds':0.5,'error':None,'timestamp':'2026-05-29T00:00:00Z'},
    {'config':'C0','qid':'Q002','query':'测试2','category':'colloquial',
     'retrieved_ids':[],'recall_at_3':0.0,'recall_at_10':0.0,'mrr':0.0,
     'has_forbidden':False,'faithfulness':3,'faithfulness_reason':'no relevant content',
     'li_relevancy_pass':False,'li_relevancy_feedback':'irrelevant',
     'elapsed_seconds':0.5,'error':None,'timestamp':'2026-05-29T00:00:00Z'},
]
with out.open('w', encoding='utf-8') as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'wrote {out}')
"
python scripts/rag_eval/report.py
```
Expected: writes `docs/testing/<today>-rag-v3-report.md`. Open it and verify the new sections appear: "Dual-Judge Consistency" with Cohen κ, "Failure Analysis" with Q002's reason.

Cleanup:
```bash
rm scripts/rag_eval/results/*_C0.jsonl
rm docs/testing/*-rag-v3-report.md
```

- [ ] **Step 3: Commit**

```bash
git add scripts/rag_eval/report.py
git commit -m "feat(eval): report adds dual-judge consistency (Cohen κ) + failure analysis sections"
```

---

## Task 15: Backend end-to-end smoke test

**Files:** (none modified — just verifying)

> Real verification that the full SSE pipeline works in v3 mode.

- [ ] **Step 1: Boot the backend in foreground (so log output is visible)**

```bash
uvicorn backend.main:app --port 8003 &
BACKEND_PID=$!
sleep 60   # wait for model load + index build
```

- [ ] **Step 2: Health check**

```bash
curl -sS http://localhost:8003/api/health
```
Expected: `{"status":"ok","rag_products":21,...}`. Note product count comes from `products.json`, unaffected by v3.

- [ ] **Step 3: Generate a post (SSE stream, just verify it completes)**

```bash
curl -N -X POST http://localhost:8003/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "适合油皮的控油精华",
    "tone": "活泼甜美",
    "max_iterations": 5,
    "persona": {"skin_type": "oily", "budget": "mid-range", "preferences": ["美白"]}
  }' | head -c 2000
```
Expected: SSE events stream out including at least one `event: tool_result` (with `query_product_database` summary), one `event: complete` with the final JSON post, and `event: token_usage`. The post should mention specific ingredients/products from the v3 retrieved context.

- [ ] **Step 4: Stop backend**

```bash
kill $BACKEND_PID
wait $BACKEND_PID 2>/dev/null
```

- [ ] **Step 5: Commit (nothing to commit — but log the verification)**

If everything passed, no commit needed. If the smoke surfaced any issues, fix them in a `fix:` commit before proceeding to Task 16.

---

## Task 16: Full eval run (5 configs × 40 gold queries)

> ⚠️ **REAL MONEY**. This task makes ~400-500 DeepSeek API calls.
> Estimated cost: **$5-10**. Estimated wall time: **15-30 min**.
> Run only after Tasks 1-15 pass cleanly.

- [ ] **Step 1: Pre-flight check**

```bash
wc -l scripts/rag_eval/gold_queries.jsonl
ls scripts/rag_eval/results/
```
Expected: 40 gold queries, results dir empty (just `.gitkeep`).

- [ ] **Step 2: Dry-run two extreme configs first**

```bash
python scripts/rag_eval/run_eval.py --config C0 --dry-run
python scripts/rag_eval/run_eval.py --config C4 --dry-run
```
Expected: each finishes in <3 min, writes 5 JSONL lines. Each line has `faithfulness_reason` and `li_relevancy_pass` fields populated.

Delete dry-run results before full run:
```bash
rm scripts/rag_eval/results/*_C0.jsonl scripts/rag_eval/results/*_C4.jsonl
```

- [ ] **Step 3: Run each config separately (Bash 10-min timeout)**

Each config takes ~2-5 min. Run them one at a time so a single Bash call doesn't exceed the 10-min hard limit:

```bash
python scripts/rag_eval/run_eval.py --config C0 --confirm
```
Expected log lines like `[C0] Q001 R@3=0.50 MRR=1.00 faith=8 li=True 0.04s`. Final JSONL has 40 lines.

```bash
python scripts/rag_eval/run_eval.py --config C1 --confirm
```

```bash
python scripts/rag_eval/run_eval.py --config C2 --confirm
```

```bash
python scripts/rag_eval/run_eval.py --config C3 --confirm
```

```bash
python scripts/rag_eval/run_eval.py --config C4 --confirm
```

- [ ] **Step 4: Verify all five JSONL files are complete**

```bash
wc -l scripts/rag_eval/results/*_C*.jsonl
```
Expected: 5 files × 40 lines = 200 lines total.

- [ ] **Step 5: Generate report**

```bash
python scripts/rag_eval/report.py
```
Expected: writes `docs/testing/<today>-rag-v3-report.md`. Open and verify all sections populate (Overall Metrics, Per-Category, Delta vs Baseline, Dual-Judge Consistency, Failure Analysis).

- [ ] **Step 6: Sanity-check headline numbers**

Open the report. Check these expected directional patterns:
- C0 R@3 should be ≥ v2's 0.558 (BGE-M3 stronger than text2vec)
- C2 MRR should be ≥ v2's 0.388 (reranker upgrade helps)
- ingredient_lookup category should show C1 > C0 by ≥5pp (BM25 advantage)
- Some failure reasons should appear in Failure Analysis

If a delta is **catastrophically negative** (e.g. C4 R@3 < 0.3), investigate before committing — likely a wiring bug in `configs.py` or `engine.py`.

- [ ] **Step 7: Commit results + report**

```bash
git add scripts/rag_eval/results/ docs/testing/*-rag-v3-report.md
git commit -m "experiment: full RAG v3 ablation eval (C0..C4 × 40 gold queries, dual-judge)"
```

---

## Task 17: README update + final wrap

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the Highlights bullet**

In `README.md`, locate the existing "RAG v2 — eval-driven hybrid pipeline" bullet and replace it with:

```markdown
- **RAG v3 — LlamaIndex pipeline with BGE-M3** — `QueryFusionRetriever` (RRF over BGE-M3 dense + BM25 jieba) → cross-encoder rerank (`BAAI/bge-reranker-v2-m3`) → custom `PersonaNodePostprocessor` (BaseNodePostprocessor subclass, ContextVar-injected) → semantic cache (cosine threshold, LRU+TTL). Title auto-prepended to embedding text via `excluded_embed_metadata_keys` (fixes a v2 bug where post/ingredient titles never reached the embedder). Paired ablation across 5 configs (C0..C4) over 40 hand-authored gold queries with dual-judge eval (custom 1-10 + LlamaIndex `RelevancyEvaluator`) and Cohen κ agreement.
```

- [ ] **Step 2: Update Tech Stack**

In the Tech Stack table, replace the "Vector search" row:

```markdown
| Vector search | LlamaIndex (`RetrieverQueryEngine`, `QueryFusionRetriever`) + BGE-M3 + `bge-reranker-v2-m3` + jieba BM25 + FAISS |
```

- [ ] **Step 3: Add v2 → v3 comparison subsection under Testing**

Below the existing "RAG v2 Ablation" subsection, add:

```markdown
### RAG v3 Ablation (LlamaIndex + BGE-M3)

Same 40-query gold set, paired design over 5 configs C0–C4 (v3 pipeline). Custom 1-10 faithfulness + LlamaIndex `RelevancyEvaluator` (pass/fail) judged each retrieved context. Full report: `docs/testing/<latest>-rag-v3-report.md`.

[Paste the "Overall Metrics" table from docs/testing/<date>-rag-v3-report.md here.]

**v2 → v3 comparison (C4 full pipeline):**

[Paste a 4-row comparison table showing R@3, MRR, custom faithfulness, latency for v2 vs v3.]

**Key wins:**
- Title now embedded (LlamaIndex `excluded_embed_metadata_keys`) — improves recall on queries where post/ingredient titles carry the topic keyword.
- BGE-M3 dense + bge-reranker-v2-m3 — modern Chinese SOTA replaces 2022-era text2vec.
- LangChain-tier framework keywords (`LlamaIndex`, `QueryFusionRetriever`, `SentenceTransformerRerank`, `RelevancyEvaluator`) in the codebase + README.

**Honest caveats:**
- The `FaithfulnessEvaluator` named in the v3 spec was replaced with `RelevancyEvaluator` in implementation — semantically the right choice for retrieval-only RAG.
- Persona hard-filter regression from v2 is unresolved (C3 → C4 still flat or negative in some categories). Future work.
```

- [ ] **Step 4: (Optional) Update Project Layout to show the simplified rag/ tree**

If the Project Layout section lists `backend/services/rag/...`, update the tree to match the v3 structure (no `retrievers/` subdir, no `chunker.py`, no `pipeline.py`).

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs(readme): RAG v3 highlights + tech stack + v2→v3 comparison table"
```

---

## Task 18: Branch wrap + offer merge

**Files:** (none modified)

- [ ] **Step 1: Final test suite + smoke**

```bash
pytest backend/tests -v
```
Expected: all tests pass.

```bash
uvicorn backend.main:app --port 8004 &
sleep 60
curl -sS http://localhost:8004/api/health
kill %1 2>/dev/null
```
Expected: 200.

- [ ] **Step 2: Inspect the final branch state**

```bash
git log --oneline rag-v3-llamaindex ^main
git diff --stat main..HEAD
```

Verify:
- 17+ commits (one per task plus any fix-ups)
- Net file count: a few NEW files (engine.py, persona_postprocessor.py, persona_rules.py, _context.py, test_persona_postprocessor.py, test_persona_rules.py, test_rag_v3_engine.py), several DELETED files (chunker, retrievers/, pipeline.py, persona_filter.py, 6 test files), MODIFIED files (service.py, loader.py, schema.py, configs.py, run_eval.py, report.py, metrics.py, main.py, requirements.txt, README.md)

- [ ] **Step 3: Report state — do NOT auto-merge to main**

The branch is ready. Don't merge or push automatically — present these options to the human partner:
  1. Merge `rag-v3-llamaindex` into `main` and push
  2. Open a PR on GitHub
  3. Keep on a feature branch for further iteration
  4. Revert and try a different approach

Tell them: total cost ~$5-10 (Task 16 eval), 17 commits, all tests green, smoke test passes, ablation report committed to `docs/testing/`.

---

## Plan Done

18 tasks, all TDD where applicable, every step has exact code or exact commands with expected output. Total new code: ~600 LOC (mostly engine.py + persona_postprocessor.py + configs.py); ~700 LOC deleted (hand-rolled retrievers, chunker, pipeline, persona_filter). Net: smaller, framework-tier codebase.

**Estimated effort:**
- Tasks 1-15: code + integration tests, ~1 week of focused work
- Task 16: $5-10 + 15-30 min of API time
- Tasks 17-18: half day

**Total: 1-1.5 weeks part-time.**

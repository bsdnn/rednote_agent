# RAG v3 — LlamaIndex Migration + BGE-M3 Upgrade

**Date:** 2026-05-29
**Status:** draft (awaiting user approval)
**Parent spec:** `docs/superpowers/specs/2026-05-28-rag-v2-design.md`
**Branch:** `rag-v3-llamaindex` (off `main` at commit `27ac520`)

---

## 背景

v2 把 hybrid + rerank + cache + persona + eval 全部**手搓**(~600 行 `backend/services/rag/`)。每个组件都有 LlamaIndex 现成实现,但 v2 spec 当时**没主动考虑"用框架 vs 手搓"**这一层架构选型。复盘下来三个硬伤:

1. **简历关键词差一档**:"自研 Python RAG" vs "LlamaIndex hybrid + rerank pipeline" — 面试 JD 80% 写 LangChain/LlamaIndex 经验
2. **手搓 chunker 有 bug**:正则 `(?<=[。!?])` 在 "0.22%" 这种数字+标点上被骗;LlamaIndex `SentenceSplitter` 不会
3. **关键的 title 字段从未进 embedding**:`load_posts` 把标题放 `Document.title`,但 chunker 只 embed `Document.content` — query 命中标题里的关键词时被严重低估

同时 v2 用的模型是 2022 年的 `text2vec-base-chinese`,中文 SOTA 已迭代到 BGE-M3(BAAI 2024)。

## v3 目标

1. **架构级目标:Hybrid 迁移**:LlamaIndex 接手所有"框架更胜任"的组件;**保留** persona_filter(业务逻辑独有)和 semantic_cache(PROVE→PASS 叙事不能拆)
2. **模型升级**:`text2vec-base-chinese` → **BAAI/bge-m3**;`bge-reranker-base` → **BAAI/bge-reranker-v2-m3**
3. **顺手修 v2 bug**:用 LlamaIndex 内建 `excluded_embed_metadata_keys` 让 title 自动进 embedding
4. **Eval 直接可比**:同 40 query × 5 config ablation,新增 LlamaIndex `FaithfulnessEvaluator` 平行评估
5. **代码净减 ~350 行**:删 `retrievers/`、`chunker.py`、`pipeline.py`,新增 ~250 行编排

## Out of Scope(明确不在本 spec)

- **LangGraph 迁移 agent**(P2 项目,独立 brainstorm)
- **CRAG adaptive flow / web 入库**(P1 项目,独立 brainstorm)
- **DeepSeek native CustomLLM**(替换 OpenAILike adapter)
- **BGE-M3 sparse + multi-vector**(只用 dense 输出)
- **Custom Chinese FaithfulnessEvaluator prompt**(override LI 英文默认)
- **RAGAS 框架做平行三角验证**

---

## 关键决策汇总

| 决策点 | 选择 |
|---|---|
| 迁移范围 | Hybrid — LlamaIndex 接手框架胜任处,保留 persona + cache |
| 模型升级 | BGE-M3 (dense 1024d) + bge-reranker-v2-m3 |
| 版本共存 | v3 直接覆盖 v2,无 feature flag;eval 重跑做对比 |
| Persona 适配 | 自定义 `PersonaNodePostprocessor(BaseNodePostprocessor)`,通过 `ContextVar` 注入 |
| Cache 位置 | 最外层 wrap `service.query_product_database`,不混入 LlamaIndex pipeline |
| LLM 接入 | `OpenAILike(api_base="https://api.deepseek.com/v1")` 一行接入 |
| LlamaIndex 生成 | `response_mode="no_text"` — 只要检索结果,不让 LI 调 LLM 生成文本 |
| Evaluator | 自定义 1-10 评分 **保留** + LI `FaithfulnessEvaluator` **新增**,JSONL 都记录 |

---

## 1. 架构总览 + 模块结构

### 新模块结构

```
backend/services/rag/
├── __init__.py
├── service.py                    # 公开 API: query_product_database(query, persona)
├── engine.py                     # LlamaIndex QueryEngine 单例 (lazy build)
├── persona_postprocessor.py      # 自定义 BaseNodePostprocessor (hard + soft)
├── persona_rules.py              # BUDGET_TIER_RULES / _passes_skin_type / _passes_budget 纯函数
├── cache/
│   ├── __init__.py
│   └── semantic_cache.py         # 保留 — 对 query_engine 调用做 wrap
└── corpus/
    ├── __init__.py
    ├── schema.py                 # adapter: 旧 Document/Chunk → LlamaIndex Document
    └── loader.py                 # 加载 JSON → list[LlamaIndex Document]

# 删除(被 LlamaIndex 替代):
# - backend/services/rag/corpus/chunker.py
# - backend/services/rag/retrievers/ 整个子目录
# - backend/services/rag/pipeline.py
```

### 组件映射表(逐项对照)

| v2 手搓 | v3 LlamaIndex |
|---|---|
| `Document` / `Chunk` dataclass | `llama_index.core.schema.Document` / `TextNode` |
| `load_products` / `load_ingredients` / `load_posts` | 保留函数名,产出 LlamaIndex `Document` |
| `chunk_document` (char/200/50) | `llama_index.core.node_parser.SentenceSplitter(chunk_size=350, chunk_overlap=50, paragraph_separator="\n\n", secondary_chunking_regex=r"(?<=[。!?!?])")` |
| `VectorRetriever` (FAISS IndexFlatIP) | `VectorStoreIndex` over `FaissVectorStore` + `VectorIndexRetriever(similarity_top_k=20)` |
| `BM25Retriever` (rank_bm25 + jieba) | `llama_index.retrievers.bm25.BM25Retriever.from_defaults(tokenizer=jieba.lcut, similarity_top_k=20)` |
| `rrf_fuse` (k=60) | `QueryFusionRetriever(retrievers=[vec, bm25], mode="reciprocal_rerank", num_queries=1, similarity_top_k=20, use_async=True)` |
| `CrossEncoderReranker` (bge-reranker-base) | `SentenceTransformerRerank(model="BAAI/bge-reranker-v2-m3", top_n=9)` |
| `SemanticCache` | **保留**(wrap query_engine) |
| `persona_filter.hard_filter` + `soft_boost` | **改成** `PersonaNodePostprocessor(BaseNodePostprocessor)` 自定义类 |
| `RAGv2Pipeline.query()` | `RetrieverQueryEngine.from_args(retriever=fusion_r, node_postprocessors=[reranker, persona_pp], response_mode="no_text")` |
| `judge_faithfulness` (1-10) | **保留** + 平行新增 `llama_index.core.evaluation.FaithfulnessEvaluator(llm=deepseek_llm)` |
| `recall_at_k` / `mrr` | **保留**(LI 的 `RetrieverEvaluator` 接口不顺,自己算更轻) |

### DeepSeek 集成

LlamaIndex 不官方支持 DeepSeek,但 DeepSeek 兼容 OpenAI API:

```python
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(
    model="deepseek-chat",
    api_base="https://api.deepseek.com/v1",
    api_key=settings.DEEPSEEK_API_KEY,
    is_chat_model=True,
)
```

**一行接入**,无需写 CustomLLM adapter。注:`api_base` 末尾的 `v1` 是 OpenAI API 协议版本,不是模型版本;模型版本在 `model=` 参数。

---

## 2. LlamaIndex 集成细节 + Persona/Cache 适配

### 2.1 完整 query 数据流

```
service.query_product_database(query, persona)
  │
  ├─ [1] semantic_cache.get(cache_key)  ──hit──► return
  │
  ├─ [2] ContextVar 注入 persona
  │
  ├─ [3] engine.aquery(query) ──┐
  │                              │
  │  RetrieverQueryEngine        │
  │  ├─ QueryFusionRetriever     │
  │  │   ├─ VectorIndexRetriever (BGE-M3 dense)
  │  │   └─ BM25Retriever (jieba)
  │  │   mode="reciprocal_rerank" → top 20
  │  │
  │  ├─ node_postprocessors:
  │  │   ├─ [0] SentenceTransformerRerank (bge-reranker-v2-m3, top_n=9)
  │  │   └─ [1] PersonaNodePostprocessor  ← 读 ContextVar
  │  │           ├─ hard filter (product only, 空集 fallback 丢 skin_type)
  │  │           ├─ soft boost (preferences ∩ effects, per_match=0.05)
  │  │           └─ re-sort
  │  │
  │  └─ response.source_nodes ──┘
  │
  ├─ [4] response.source_nodes[:3] → format with [doc_id]
  │
  └─ [5] semantic_cache.set(cache_key, result) → return
```

### 2.2 `engine.py`(核心编排骨架)

```python
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
import faiss, jieba

from backend.core.config import settings
from .corpus.loader import load_all_as_li_documents
from .persona_postprocessor import PersonaNodePostprocessor

_engine = None
BGE_M3_DIM = 1024


def get_engine():
    global _engine
    if _engine is not None:
        return _engine

    LISettings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    LISettings.llm = OpenAILike(
        model="deepseek-chat",
        api_base="https://api.deepseek.com/v1",
        api_key=settings.DEEPSEEK_API_KEY,
        is_chat_model=True,
    )

    docs = load_all_as_li_documents()

    splitter = SentenceSplitter(
        chunk_size=350, chunk_overlap=50,
        paragraph_separator="\n\n",
        secondary_chunking_regex=r"(?<=[。!?!?])",
    )
    nodes = []
    for doc in docs:
        if doc.metadata["doc_type"] == "post":
            nodes.extend(splitter.get_nodes_from_documents([doc]))
        else:
            n = TextNode(id_=f"{doc.doc_id}::0", text=doc.text, metadata=dict(doc.metadata))
            n.excluded_embed_metadata_keys = list(doc.excluded_embed_metadata_keys)
            nodes.append(n)

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

    reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-v2-m3", top_n=9)
    persona_pp = PersonaNodePostprocessor()

    _engine = RetrieverQueryEngine.from_args(
        retriever=fusion_r,
        node_postprocessors=[reranker, persona_pp],
        response_mode="no_text",  # 只要 source_nodes,不让 LI 生成
    )
    return _engine
```

### 2.3 `persona_postprocessor.py`

```python
import logging
from typing import Optional
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

from backend.models.persona import UserPersona
from .persona_rules import _passes_skin_type, _passes_budget

logger = logging.getLogger(__name__)


class PersonaNodePostprocessor(BaseNodePostprocessor):
    per_match_boost: float = 0.05

    @classmethod
    def class_name(cls) -> str:
        return "PersonaNodePostprocessor"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        from .service import _current_persona
        persona: UserPersona | None = _current_persona.get()
        if persona is None:
            return nodes

        def _filter(skip_skin):
            out = []
            for nws in nodes:
                meta = nws.node.metadata
                if meta.get("doc_type") != "product":
                    out.append(nws); continue
                if not skip_skin and not _passes_skin_type(meta, persona.skin_type):
                    continue
                if not _passes_budget(meta, persona.budget):
                    continue
                out.append(nws)
            return out

        primary = _filter(skip_skin=False)
        product_count = sum(1 for n in primary if n.node.metadata.get("doc_type") == "product")
        if product_count == 0 and persona.skin_type:
            logger.warning("persona filter emptied products, dropping skin_type=%s", persona.skin_type)
            primary = _filter(skip_skin=True)

        for nws in primary:
            meta = nws.node.metadata
            boost = 0.0
            if persona.age_group in (meta.get("age_groups") or []):
                boost += self.per_match_boost
            overlap = set(persona.preferences or []) & set(meta.get("effects") or [])
            boost += self.per_match_boost * len(overlap)
            nws.score = (nws.score or 0.0) + boost

        primary.sort(key=lambda n: n.score or 0.0, reverse=True)
        return primary
```

### 2.4 `service.py` 的 ContextVar 注入

```python
import asyncio
from contextvars import ContextVar
from typing import Optional

from backend.models.persona import UserPersona
from .cache.semantic_cache import SemanticCache
from .engine import get_engine

_current_persona: ContextVar[Optional[UserPersona]] = ContextVar("rag_persona", default=None)

# cache 用一个轻量 BGE-M3 wrapper 算 query 向量(独立于 engine 的 HuggingFaceEmbedding 实例;
# 共享即可,内部 sentence-transformers 已做模型缓存)。
def _make_cache_embed_fn():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    import numpy as np
    return lambda text: np.asarray(embed_model.get_text_embedding(text), dtype="float32")

_cache = SemanticCache(
    embedding_fn=_make_cache_embed_fn(),
    threshold=settings.SEMANTIC_CACHE_THRESHOLD,
    max_size=settings.SEMANTIC_CACHE_MAX_SIZE,
    ttl_seconds=settings.SEMANTIC_CACHE_TTL_SECONDS,
)

def _cache_key(query: str, persona: UserPersona | None) -> str:
    if persona is None:
        return query
    return f"{query}||{persona.skin_type}|{persona.budget}|{persona.age_group}|{','.join(sorted(persona.preferences or []))}"


def _format_top(nodes_with_scores, k=3) -> str:
    parts = []
    for rank, nws in enumerate(nodes_with_scores[:k], 1):
        n = nws.node
        meta = n.metadata
        doc_id = n.node_id.split("::")[0]
        parts.append(
            f"[匹配{rank}] [{doc_id}] {n.get_content()[:120]}\n"
            f"类型: {meta.get('doc_type')} | 成分: {'、'.join(meta.get('key_ingredients',[])[:4])} "
            f"| 适合: {'、'.join(meta.get('suitable_skin_types',[]))}"
        )
    return "\n\n".join(parts) or "数据库中未找到匹配产品。"


async def query_product_database(query: str, persona: UserPersona | None = None) -> str:
    key = _cache_key(query, persona)
    cached = await _cache.get(key)
    if cached is not None:
        return cached

    engine = get_engine()
    token = _current_persona.set(persona)
    try:
        response = await engine.aquery(query)
    finally:
        _current_persona.reset(token)

    result = _format_top(response.source_nodes, k=3)
    await _cache.set(key, result)
    return result
```

### 2.5 修 v2 的 chunker bug:title 自动进 embedding

LlamaIndex 内建 `excluded_embed_metadata_keys` — 声明哪些 metadata 字段**不参与** embedding。**没排除的字段会自动拼到 text 前面再 embed**。正好解决 v2 "title 没进 embedding" 的痛点。

`loader.py` 关键改造:

```python
from llama_index.core import Document as LIDocument

def _make_post_doc(r):
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
    # title 进 embedding,其他不进
    doc.excluded_embed_metadata_keys = ["doc_type", "source", "linked_products", "tone"]
    doc.excluded_llm_metadata_keys = ["source"]
    return doc
```

`SentenceSplitter` 切完后,每个 `TextNode` **自动继承父 Document 的 metadata 配置**。chunk 1、chunk 2 都会自动带上 title — 解决了 v2 手搓版的痛点。

类似配置对 ingredient(title=成分中文名)和 product(title=产品名)。

---

## 3. Eval 连续性 + Rollout + Tests + 风险

### 3.1 Eval 连续性

- **Gold set 不动**:`scripts/rag_eval/gold_queries.jsonl` 40 条 query 原封不动,v2/v3 直接可比
- **JSONL 字段增量**(不破坏旧字段):

```json
{
  "config": "C0..C4",
  "qid": "Q001",
  "query": "...",
  "category": "...",
  "retrieved_ids": ["..."],
  "recall_at_3": 0.0,
  "recall_at_10": 0.0,
  "mrr": 0.0,
  "has_forbidden": false,
  "faithfulness": 7,                    // 自定义 1-10
  "faithfulness_reason": "...",          // 自定义 LLM 给的 reason
  "li_faithfulness_pass": true,          // NEW: LlamaIndex 二元判定
  "li_faithfulness_feedback": "...",     // NEW: LI 给的 feedback
  "elapsed_seconds": 0.0,
  "error": null,
  "timestamp": "..."
}
```

**两套 evaluator 的明确分工:**
- **自定义 `judge_faithfulness`**(`scripts/rag_eval/metrics.py` 已有):产出 1-10 整数 + reason 字符串。JSONL 字段 `faithfulness` + `faithfulness_reason`。**v3 不改 prompt,只改 run_eval.py 把 reason 真正持久化(v2 写了但被丢弃)**
- **新增 LlamaIndex `FaithfulnessEvaluator`**:产出 pass/fail + feedback。JSONL 字段 `li_faithfulness_pass` + `li_faithfulness_feedback`。使用 `OpenAILike(deepseek-chat)` 作为 judge LLM
- 每条 record 同时被两个 evaluator 评判,JSONL 一次性持久化所有字段
- 报告新增一节 **"Custom-1to10 vs LI-FaithfulnessEvaluator 一致性表"**:统计 LI=pass 时 custom score 的分布、LI=fail 时 custom score 的分布、Cohen κ 一致性

### 3.2 5-config ablation 在 LlamaIndex 下重建

```python
# scripts/rag_eval/configs.py — v3 版本

@dataclass
class ConfigSpec:
    name: str
    description: str
    use_bm25: bool        # False → vector-only
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


def build_engine_for_config(spec) -> RetrieverQueryEngine:
    """工厂:按 spec 决定哪些 retriever / postprocessor 接上."""
    ...
```

### 3.3 Rollout — 8 个独立 commit 的 phase

```
Phase 1: deps + smoke test
  - 加 llama-index-core, llama-index-embeddings-huggingface,
       llama-index-vector-stores-faiss, llama-index-retrievers-bm25,
       llama-index-postprocessor-sbert-rerank, llama-index-llms-openai-like
  - 下载 BGE-M3 (~2.2GB) + bge-reranker-v2-m3 (~600MB)
  - smoke test: 加载模型 + 一次 dummy aquery()
  - commit: "feat(rag): install LlamaIndex + BGE-M3 deps"

Phase 2: corpus/schema/loader 改造
  - schema.py: 删自定义 Document/Chunk,留 adapter 函数
  - loader.py: load_*_as_li_documents 返回 LI Document
       + 配 excluded_embed_metadata_keys 让 title 进 embedding
  - test_corpus_loader.py 重写为 LI Document 断言
  - commit: "refactor(rag): corpus → LlamaIndex Document with title-aware embedding"

Phase 3: engine.py + persona_postprocessor.py
  - 新建 engine.py(单例 RetrieverQueryEngine)
  - 新建 persona_postprocessor.py + persona_rules.py
  - 删除 retrievers/、chunker.py、pipeline.py 整个
  - commit: "feat(rag): LlamaIndex engine + persona NodePostprocessor"

Phase 4: service.py 重写
  - 重写 service.py,接 engine + cache + ContextVar
  - tools_registry.py 不动(签名兼容)
  - commit: "refactor(rag): service.py wraps LlamaIndex with ContextVar persona"

Phase 5: main.py warmup
  - lifespan 改成 warmup BGE-M3 + reranker + query 各一次
  - commit: "chore(main): warmup includes BGE-M3 + reranker"

Phase 6: tests 重写
  - 删: test_chunker.py / test_vector_retriever.py / test_bm25_retriever.py
        / test_rrf_fusion.py / test_reranker.py / test_rag_v2_pipeline.py
  - 改: test_persona_filter.py → test_persona_postprocessor.py
  - 新: test_rag_v3_engine.py(集成测试,fake corpus + 真 engine)
  - 保留: test_semantic_cache.py / test_corpus_schema.py / test_eval_metrics.py
  - commit: "test: replace unit tests with integration tests for LlamaIndex pipeline"

Phase 7: eval pipeline 更新
  - configs.py 重建 ablation,新增 build_engine_for_config 工厂
  - run_eval.py 加 faithfulness_reason 持久化 + LI FaithfulnessEvaluator 调用
  - run_eval.py 加 li_faithfulness_pass / li_faithfulness_feedback 字段
  - report.py 加 "Custom vs LI Evaluator 一致性" 章节
  - commit: "feat(eval): C0-C4 configs on LlamaIndex + LI evaluator + dual-judge report"

Phase 8: 跑 eval + 写报告 + 收尾
  - 重跑 5 config × 40 query (~$5-7, ~20min)
  - 生成 docs/testing/<date>-rag-v3-report.md
  - README Highlights / Tech Stack / Testing 章节改写
  - 加 v2 → v3 对比表
  - commit: "experiment: RAG v3 ablation + v2→v3 comparison + README update"
```

总时间估算:1-1.5 周。Phase 1-5 是代码;Phase 6 测试重写 1 天;Phase 7-8 半天 + eval 自动跑。

### 3.4 测试改造清单

| 操作 | 文件 | 理由 |
|---|---|---|
| 删 | `test_chunker.py` | SentenceSplitter 是框架,不测 |
| 删 | `test_vector_retriever.py` | VectorIndexRetriever 是框架 |
| 删 | `test_bm25_retriever.py` | BM25Retriever 是框架 |
| 删 | `test_rrf_fusion.py` | QueryFusionRetriever 是框架 |
| 删 | `test_reranker.py` | SentenceTransformerRerank 是框架 |
| 删 | `test_rag_v2_pipeline.py` | pipeline.py 删了 |
| 改 | `test_persona_filter.py` → `test_persona_postprocessor.py` | 测 `PersonaNodePostprocessor._postprocess_nodes()` 接口,fake `list[NodeWithScore]`,断言过滤 + 加分行为 |
| 改 | `test_corpus_loader.py` | 断言 LI Document.metadata + excluded_embed_metadata_keys |
| 新 | `test_rag_v3_engine.py` | 集成测试 — fake 5 个 Document → 真 engine.aquery() → 断言 source_nodes 结构合理(数量 ≤ 3、有 doc_id、score 单调降) |
| 留 | `test_semantic_cache.py` | cache 逻辑不变,PROVE 翻转还在 |
| 留 | `test_corpus_schema.py` | adapter 函数测 |
| 留 | `test_eval_metrics.py` | recall_at_k / mrr 纯函数 |

净:**6 删 + 2 改 + 1 新 + 3 留**。测试 50+ 用例 → ~25,但 signal 更高(测自己写的 + 集成,不测框架)。

### 3.5 风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| BGE-M3 下载 ~2.5GB,首次启动 1-2 分钟 | 高 | warmup 日志显式标注,文档明示 |
| ContextVar 在 engine.aquery() 内被错误 reset | 低 | service.py 用 try/finally 包,reset 不在 query 内调用 |
| QueryFusionRetriever `use_async=True` 不真并行 BM25 | 中 | 实测;若不并行,latency 增加但功能不破 |
| eval 数字反而比 v2 差 | 中 | LI 默认参数与手搓微差,先调参再下结论;最坏情况报告说 "framework defaults vs custom-tuned" |
| 测试删一堆,coverage 数字掉 | 中 | README 解释:删的都是测框架内部,留的都测业务 + 集成 |
| 中文分词:LI BM25 默认英文 tokenizer | 高 | `BM25Retriever.from_defaults(tokenizer=jieba.lcut, ...)` 显式注入 |
| Persona postprocessor 通过 ContextVar 默认 None | 中 | 单测验证;若 None 走兜底逻辑(等价 no-persona),不抛错 |
| `OpenAILike` 不支持 DeepSeek 的 JSON mode | 中 | FaithfulnessEvaluator 的 prompt 不强依赖 JSON;如果需要,继承 `OpenAILike` 加 override |

---

## 改动文件清单

### 新增

- `backend/services/rag/engine.py`
- `backend/services/rag/persona_postprocessor.py`
- `backend/services/rag/persona_rules.py`
- `backend/tests/test_rag_v3_engine.py`

### 修改

- `backend/services/rag/__init__.py`
- `backend/services/rag/service.py` — 大重写
- `backend/services/rag/corpus/schema.py` — 改为 adapter
- `backend/services/rag/corpus/loader.py` — 返回 LI Document
- `backend/core/config.py` — 加 `EMBEDDING_MODEL`、`RERANKER_MODEL` 默认值更新
- `backend/main.py` — warmup 更新
- `backend/requirements.txt` — 加 LlamaIndex 系列
- `scripts/rag_eval/configs.py` — 重写为 LI 版本
- `scripts/rag_eval/run_eval.py` — 加 LI evaluator + faithfulness_reason
- `scripts/rag_eval/report.py` — 加 dual-judge 章节
- `backend/tests/test_corpus_loader.py` — 改为 LI Document 断言
- `backend/tests/test_persona_filter.py` → 重命名为 `test_persona_postprocessor.py`
- `README.md` — Highlights / Tech Stack / Testing / 新增 v2→v3 对比表

### 删除

- `backend/services/rag/corpus/chunker.py`
- `backend/services/rag/retrievers/` 整个子目录
- `backend/services/rag/pipeline.py`
- `backend/tests/test_chunker.py`
- `backend/tests/test_vector_retriever.py`
- `backend/tests/test_bm25_retriever.py`
- `backend/tests/test_rrf_fusion.py`
- `backend/tests/test_reranker.py`
- `backend/tests/test_rag_v2_pipeline.py`

### 不动

- `backend/core/agent.py` / `backend/core/prompts.py`
- `backend/services/tools_registry.py`(签名兼容)
- `backend/api/routes.py`
- `backend/models/*`
- `backend/services/rag/cache/semantic_cache.py`(逻辑不变)
- `backend/tests/test_semantic_cache.py`
- `backend/tests/test_corpus_schema.py`
- `backend/tests/test_eval_metrics.py`
- `scripts/rag_eval/gold_queries.jsonl`(40 query 不动)
- `scripts/rag_eval/metrics.py`(recall_at_k / mrr / has_forbidden 保留)

---

## 验收标准

1. ✅ `pytest backend/tests -v` 全绿(预计 ~25 个用例)
2. ✅ `python scripts/rag_eval/run_eval.py --config C4 --dry-run` 跑通,JSONL 字段含 `faithfulness`、`faithfulness_reason`、`li_faithfulness_pass`、`li_faithfulness_feedback`
3. ✅ 完整 ablation `--config all --confirm` 跑通,5 个 JSONL × 40 行
4. ✅ `docs/testing/<date>-rag-v3-report.md` 自动生成,含 Overall / Per-Category / Delta-vs-C0 / Dual-Judge 一致性 / Failure Analysis 五节
5. ✅ 后端 `uvicorn backend.main:app` 启动 SSE 完整流程跑通(query + persona)
6. ✅ v2 vs v3 关键指标对比表写进 README:
   - 整体 R@3 / MRR / Faithfulness
   - ingredient_lookup / colloquial / persona_strong 三大类的 R@3
   - chunker bug 修复(title 进 embedding 后的具体 query 改善)
7. ✅ README "Tech Stack" 写明:**LlamaIndex** + **BAAI/bge-m3** + **bge-reranker-v2-m3** + **OpenAILike DeepSeek adapter**

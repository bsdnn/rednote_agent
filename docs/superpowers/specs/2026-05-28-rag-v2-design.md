# RAG v2 Design — Eval-Driven Retrieval Overhaul

**Date:** 2026-05-28
**Status:** draft (awaiting user review)

> Note: `2026-XX-XX` 出现在 corpus / results / report 文件名中是有意占位 — 这些产物在执行 plan 时按运行日期生成,spec 不应写死。
> Expected output 表里的 `TBD` 数字是 eval 跑完后回填的真实测量值。
**Goal:** 把 v1 的 71 行单文件 FAISS 检索升级为成体系的 hybrid + rerank + semantic cache + persona-aware 管道,用可复现的 eval 数字驱动每一步改进。叙事直接对标已有的 A/B planning 实验(`docs/testing/2026-05-18-ab-report.md`)。

---

## 背景

### v1 现状(`backend/services/rag_service.py`,71 行)

- **数据**: 21 个护肤品,单文件 `products.json`
- **Embedding**: `shibing624/text2vec-base-chinese`,把 `name + description + ingredients + skin_types` 拼成一个长字符串编码
- **索引**: `FAISS IndexFlatL2`,暴力 L2 搜索
- **检索**: top-3,无 rerank
- **缓存**: `@lru_cache(maxsize=128)`,精确字符串匹配
- **接口**: `query_product_database(query: str) -> str`,被 `tools_registry.py` 注册给 agent

### 已暴露的问题(自有测试已证明)

| 测试 | 问题 |
|---|---|
| `test_rag_stability.py::test_lru_paraphrase_is_miss` (PROVE) | `"补水产品"` 与 `"保湿产品"` 是两次独立 cache miss — lru_cache 不理解语义 |
| (无成分名 BM25 测试) | 纯 dense 检索对 `"含烟酰胺的产品"` 这种成分名 query 表现差,容易召回相似品类但不含目标成分的产品 |
| (无 persona 利用) | `UserPersona` 只参与 prompt 生成,不参与检索;敏感肌用户也可能被推到刺激性产品 |
| (无 eval) | 无法量化任何检索改动的效果,所有"改进"都是 vibes-based |

### v2 目标(KR 级)

1. **可量化**:Recall@3、MRR、semantic_hit_rate、faithfulness 四个指标,在 paired ablation 中跑出每步改动的独立 delta
2. **可解释**:每个改进对应一个面试可讲的技术决策(BM25 vs vector、RRF vs 加权和、threshold 选择、persona 双层等)
3. **不破坏**:对外接口签名向后兼容,通过 `RAG_VERSION` env var 灰度
4. **闭环 PROVE 测试**:把 `test_lru_paraphrase_is_miss` 翻转为 PASS 的 v2 版本,形成 "v1 发现 → v2 修复" 叙事

---

## 设计

### 总体原则

- **拆模块**:71 行单文件 → `rag/` 子包,每个文件单一职责,可独立单测
- **接口兼容**:`query_product_database(query, persona=None)` — 只加可选参数,`agent.py` / `tools_registry.py` 改动最小
- **Feature flag**:`RAG_VERSION=v1|v2`(默认 v1),eval 阶段两版并行,稳定后切默认值 + 删 v1 代码

---

### 1. 模块结构

```
backend/services/rag/
├── __init__.py
├── service.py                # 对外接口 query_product_database(query, persona)
├── pipeline.py               # 编排: cache → filter → hybrid → rerank → boost
├── corpus/
│   ├── schema.py             # Document / Chunk dataclass + metadata 字段
│   ├── loader.py             # 加载三种 doc_type → 统一 Document
│   └── chunker.py            # 按 doc_type 分别切块
├── retrievers/
│   ├── vector.py             # FAISS dense retriever (cosine)
│   ├── bm25.py               # rank_bm25 sparse retriever
│   ├── hybrid.py             # RRF 融合
│   └── reranker.py           # bge-reranker-base-v2 cross-encoder
├── cache/
│   └── semantic_cache.py     # embedding cache + similarity threshold
└── persona_filter.py         # persona → metadata hard filter + soft boost

backend/services/rag_service.py  # 保留为 v1 实现,通过 RAG_VERSION 切换
```

`backend/services/__init__.py` 或 `service.py` 内根据 `settings.RAG_VERSION` 选择导入 v1 或 v2 实现。

---

### 2. 语料层

#### 三种文档类型

| 类型 | 数量 | 来源 | chunking | 用途 |
|---|---|---|---|---|
| Product | 21(不变) | `backend/data/products.json` | 1 product = 1 chunk | 主体,被引用做种草 |
| Ingredient Wiki | ~50 | **LLM 合成**(DeepSeek) | 1 ingredient = 1 chunk | 支持"烟酰胺能美白吗"类知识查询 |
| RED Post / Review | 200-300 | **LLM 合成** | RecursiveCharacterTextSplitter,chunk_size=200,overlap=50 | 提供口语化文案 / 用户视角 |

总规模 ~270-370 chunks,eval 区分度足够。

#### 数据生成

- 脚本:`scripts/generate_corpus.py`
  - `python scripts/generate_corpus.py ingredients` → `backend/data/corpus/ingredients.json`
  - `python scripts/generate_corpus.py posts` → `backend/data/corpus/posts.json`
- 固定 random seed、`temperature=0.7`,DeepSeek JSON Schema 结构化输出
- **诚信标注**:所有合成 chunk 在 `metadata.source = "synthesized"`,面试主动讲
- 质量门:`scripts/spot_check_corpus.py` 随机抽 10% 给人工 review,不合格批次重生成

#### 统一 schema(`rag/corpus/schema.py`)

```python
@dataclass
class Document:
    doc_id: str
    doc_type: Literal["product", "ingredient", "post"]
    title: str
    content: str
    metadata: dict

@dataclass
class Chunk:
    chunk_id: str            # "{doc_id}::{idx}"
    doc_id: str
    doc_type: str
    text: str                # 用于 embed 和 BM25
    metadata: dict           # 继承自 doc + chunk-specific
```

#### Metadata schema

公共字段:
```python
{"doc_type": "...", "source": "internal" | "synthesized" | "scraped"}
```

按类型扩展:
```python
# product
{"category": str, "skin_types": list[str], "ingredients": list[str],
 "price_tier": "budget"|"mid"|"premium",
 "effects": list[str],          # e.g. ["美白", "保湿"] — 用于 persona.preferences 软加分
 "age_groups": list[str]}       # e.g. ["18-25", "25-30"] — 用于 persona.age_group 软加分
# ingredient
{"effects": list[str], "concerns": list[str], "age_groups": list[str]}
# post
{"linked_products": list[str], "tone": str}
```

---

### 3. 检索管道

完整 query 流程(`rag/pipeline.py`):

```
query + persona
  ├─[1] semantic cache lookup ──hit──► return cached
  │  miss
  ├─[2] persona → metadata hard filter (product only)
  ├─[3] 并行 retrieve top-20:
  │        vector (FAISS, cosine)   ─┐
  │        BM25 (rank_bm25)          ─┤  asyncio.gather + run_in_executor
  ├─[4] RRF 融合 → 候选 top-20
  ├─[5] cross-encoder rerank (BAAI/bge-reranker-base)
  │        + persona soft boost (匹配字段加分)
  ├─[6] top-3 → 格式化字符串
  └─[7] 写回 semantic cache → return
```

#### 关键决策

**并行检索**:vector 与 BM25 都是 CPU-bound,用 `asyncio.gather` + `run_in_executor` 并行,沿用 `agent.py` 现有模式。

**RRF 融合(不是加权和)**:
```python
score(doc) = Σ 1 / (k + rank_i(doc))   # k=60 行业默认
```
RRF 不需要归一化 BM25 与 cosine 量纲,scale-free。面试讲 hybrid search 的标准答案。

**Cross-encoder 模型**:`BAAI/bge-reranker-base`(中文友好,330M 参数,CPU 上 top-20 rerank ~200ms)。输入 `[query, chunk.text]` pair,输出相关性分数。

**Persona 双层**(详见 §5):硬过滤踢出不兼容产品,软加分微调 rerank 排序。

#### 模型加载与冷启动

新增 dependency 增加启动开销:
- `bge-reranker-base`: ~330MB,首次下载 1 分钟,加载 2-3 秒
- `rank_bm25`: 无模型,build index ~1 秒

启动 warmup 扩展:embedding warmup + reranker warmup + BM25 warmup,各跑一次哑 query。

#### YAGNI(明确不做)

- ❌ HyDE / multi-query expansion(留给 future work,属于 Bundle B)
- ❌ Anthropic Contextual Retrieval 的 chunk 上下文增强(同上)
- ❌ rerank 只跑一次,不做 listwise / 多轮

---

### 4. Semantic Cache

#### 接口

```python
# rag/cache/semantic_cache.py
class SemanticCache:
    def __init__(
        self,
        embedding_fn: Callable[[str], np.ndarray],
        threshold: float = 0.92,
        max_size: int = 256,
        ttl_seconds: int = 3600,
    ): ...
    async def get(self, query: str) -> str | None: ...
    async def set(self, query: str, value: str) -> None: ...
    def stats(self) -> dict: ...
```

#### 关键决策

| 决策 | 值 | 理由 |
|---|---|---|
| 相似度阈值 | 0.92(起点) | 太低(0.85)易 false positive,太高(0.98)失去意义。eval 阶段扫 [0.88, 0.90, 0.92, 0.95] 选 FP rate 最低值 |
| 查找实现 | numpy 矩阵乘(不用 FAISS) | 256 条规模 cosine ~0.5ms,FAISS 是 overkill |
| 淘汰策略 | LRU + TTL(1h) 双策略 | TTL 防止过气内容卡缓存(小红书时效性强) |
| 并发 | `asyncio.Lock` 包写,读不加锁 | 写竞态会破坏数组,读最差只是错过命中 |
| 持久化 | 不做 | 进程重启清空,简单。预留 `load_from_disk` hook 给 future work |

#### Metrics

写入日志 + `/api/health`:
```python
{"exact_hits": 12, "semantic_hits": 38, "misses": 50,
 "evictions_lru": 3, "evictions_ttl": 5, "p50_lookup_ms": 0.4}
```

`semantic_hits / total_hits` 是核心面试金句 — 直接证明语义缓存 vs lru_cache 的价值。

#### PROVE → PASS 闭环

新增 `test_semantic_cache.py::test_paraphrase_now_hits`:把 `test_rag_stability.py::test_lru_paraphrase_is_miss` 翻转为 PASS。旧 PROVE 测试**保留**,作为"v1 在 paraphrase 上确实 broken"的历史证据。

---

### 5. Persona 集成

#### 映射规则(`rag/persona_filter.py`)

```python
PERSONA_TO_FILTER = {
    # persona field: (doc metadata field, hard|soft, missing behavior)
    "skin_type":   ("metadata.skin_types",   "hard",  "skip"),
    "budget":      ("metadata.price_tier",   "hard",  "skip"),
    "age_group":   ("metadata.age_groups",   "soft",  "skip"),
    "preferences": ("metadata.effects",      "soft",  "skip"),
}

BUDGET_TIER_RULES = {
    "budget":  {"budget"},
    "mid":     {"budget", "mid"},
    "premium": {"budget", "mid", "premium"},
}
```

#### 硬过滤(只作用于 product)

```python
def hard_filter(chunks, persona):
    if persona is None: return chunks
    out = []
    for c in chunks:
        if c.doc_type != "product":      # ingredient / post 永远保留
            out.append(c); continue
        if persona.skin_type and persona.skin_type not in c.metadata["skin_types"]:
            continue
        if persona.budget and c.metadata["price_tier"] not in BUDGET_TIER_RULES[persona.budget]:
            continue
        out.append(c)
    return out
```

**为什么只过滤 product**:ingredient 和 post 是知识 / 文案,跟 persona 无强绑定 — 油皮也可以读"烟酰胺百科"。避免"过滤后候选池为空"的灾难。

#### 软加分(作用于 rerank 之后)

```python
def soft_boost(score, chunk, persona):
    boost = 0.0
    if persona.age_group in chunk.metadata.get("age_groups", []):
        boost += 0.05
    overlap = set(persona.preferences or []) & set(chunk.metadata.get("effects", []))
    boost += 0.05 * len(overlap)
    return score + boost
```

**系数 0.05**:cross-encoder 输出典型 range ~[0, 5],单字段命中加 0.05 ≈ 1%,2-3 个偏好叠加能改变 top-3 边缘排名,但不盖住 rerank 主信号。**eval 时扫超参**。

#### 边界与 fallback

| 情况 | 行为 |
|---|---|
| `persona is None` | 跳过所有 filter / boost,退化为纯检索 |
| `persona.skin_type="敏感"` 且 corpus 0 匹配 | 硬过滤返回空 → **丢弃 skin_type 过滤,仅保留 budget,日志 WARN** |
| persona 所有字段 None | 等价于 `persona=None` |
| chunk metadata 缺字段(脏数据) | 视为"未声明",不被过滤(放宽不冤枉) |

**Fallback 关键**:宁可放宽也不返回空 — 否则 agent 拿到空 context 整个流程崩。

---

### 6. Eval Pipeline

#### Gold Set(`scripts/rag_eval/gold_queries.jsonl`)

手写 60-80 条 query,按类别配额:

| 类别 | 数量 | 例 | 验证 |
|---|---|---|---|
| 直接产品需求 | 20 | "适合油皮的控油精华" | baseline 召回 |
| 同义词改写 | 10 | "补水霜" vs "保湿乳" | 语义缓存 + embedding |
| 成分名查询 | 10 | "含烟酰胺的产品" | **hybrid 必赢** |
| 长尾口语化 | 10 | "脸总是干干的发紧用啥" | embedding + rerank |
| 跨品类比较 | 5 | "精华和面膜哪个补水更快" | 多文档融合 |
| Persona 强相关 | 10 | (query="美白") + persona=敏感 | hard filter + fallback |
| Adversarial(应不召回) | 5 | "美白牙膏" | precision / FP |

Record schema:
```json
{
  "qid": "Q001",
  "query": "含烟酰胺的产品",
  "category": "ingredient_lookup",
  "persona": {"skin_type": "油性", "budget": "mid"},
  "relevant_doc_ids": ["prod_005", "prod_012", "ing_niacinamide"],
  "must_not_contain": [],
  "notes": "BM25 应该比 vector 更准"
}
```

`relevant_doc_ids` 由我们标注(语料是自己合成的,知道正确答案)。

#### 三层指标

| 层 | 指标 | 算法 |
|---|---|---|
| 检索 | Recall@3, Recall@10, MRR | gold `relevant_doc_ids` ∩ retrieved top-K |
| 缓存 | semantic_hit_rate, false_hit_rate | hits/total;hit 但答案错的比例(对比无缓存 baseline) |
| 端到端 | Faithfulness, Tone-match | LLM-as-judge,复用 reflection prompt 结构 |

**LLM-as-judge 复用**:不写新 prompt 框架,改用现有 reflection 的 1-10 分模式 + DeepSeek client + tenacity retry。

#### Ablation 矩阵(`scripts/rag_eval/configs.py`)

paired 设计,5 个 config 同一 gold set 全跑:

| Config | vector | BM25 | RRF | rerank | cache | persona |
|---|---|---|---|---|---|---|
| C0 baseline (v1) | ✓ | ✗ | ✗ | ✗ | lru | ✗ |
| C1 + BM25/RRF | ✓ | ✓ | ✓ | ✗ | lru | ✗ |
| C2 + rerank | ✓ | ✓ | ✓ | ✓ | lru | ✗ |
| C3 + sem-cache | ✓ | ✓ | ✓ | ✓ | semantic | ✗ |
| C4 full v2 | ✓ | ✓ | ✓ | ✓ | semantic | ✓ |

每步 delta 即为该改进的独立贡献。

#### 主脚本(对标 `scripts/ab_planning_experiment.py`)

```
scripts/rag_eval/
├── gold_queries.jsonl
├── configs.py
├── run_eval.py        # 主脚本(--config / --n / --confirm / --resume / --dry-run)
├── metrics.py         # Recall@K, MRR, LLM judge 调用
├── report.py          # JSONL → Markdown 自动报告
└── results/
    ├── 2026-XX-XX_C0.jsonl
    ├── ...
    └── report.md
```

CLI 风格直接照搬现有 A/B 脚本:
```bash
python scripts/rag_eval/run_eval.py --config all --n 80 --confirm
python scripts/rag_eval/run_eval.py --config C2 --resume
python scripts/rag_eval/run_eval.py --dry-run
python scripts/rag_eval/report.py
```

**断点续跑**:增量 JSONL,重启 skip 已写入 qid。

#### 预期产出表(数字占位,eval 完填)

放进 README,与 A/B planning 表并列:

| Metric | C0 (v1) | C4 (v2) | Δ |
|---|---|---|---|
| Recall@3 | TBD | TBD | TBD |
| MRR | TBD | TBD | TBD |
| Semantic cache hit rate | TBD | TBD | TBD |
| Faithfulness (1-10) | TBD | TBD | TBD |
| Avg total tokens | TBD | TBD | TBD |

#### 预算

- LLM-as-judge: 80 × 5 ≈ 400 calls ≈ $1
- 端到端 generate: 80 × 5 ≈ 400 runs ≈ $5
- 总成本 < $10,2-3 天完成

---

### 7. Testing 增量

| 文件 | 新增 case 数 | 覆盖 |
|---|---|---|
| `test_rag_v2_pipeline.py` | 4 | hybrid + filter + rerank 编排,顺序,空候选 fallback |
| `test_bm25_retriever.py` | 3 | 中文分词正确性、IDF 行为、空 query |
| `test_rrf_fusion.py` | 2 | k=60 默认、空交集 |
| `test_reranker.py` | 2 | top-20 → top-3 顺序、cuda/cpu 一致 |
| `test_semantic_cache.py` | 5 | exact hit、semantic hit、miss、TTL evict、LRU evict |
| `test_persona_filter.py` | 5 | None 退化、hard filter、budget 包含、soft boost、空集 fallback |
| `test_corpus_loader.py` | 3 | 三种 doc_type 加载、metadata schema 完整性、chunk_id 唯一性 |

保留 `test_rag_stability.py` 旧 PROVE,作为历史证据。新增"v2 修复"的 PASS 测试与之对应。

CI:单测每 commit 跑(~30s)。Eval **不进 CI**(成本),手动跑、结果 commit 到 `scripts/rag_eval/results/`。

---

## 改动文件清单

### 新增

- `backend/services/rag/` 整个子包(见 §1 结构)
- `backend/data/corpus/ingredients.json`
- `backend/data/corpus/posts.json`
- `scripts/generate_corpus.py`
- `scripts/spot_check_corpus.py`
- `scripts/rag_eval/` 整个目录(见 §6.4 结构)
- `backend/tests/test_*.py` 7 个新文件(见 §7)
- `docs/testing/2026-XX-XX-rag-v2-report.md`(eval 完成后写)

### 修改

- `backend/core/config.py` — 加 `RAG_VERSION: str = "v1"`,加 reranker 配置项
- `backend/main.py` — startup warmup 扩展(reranker + BM25)
- `backend/requirements.txt` — 加 `rank_bm25`,加 reranker 所需的 transformers 版本约束
- `.env.example` — 加 `RAG_VERSION=v1`
- `README.md` — RAG 章节重写,加 v2 ablation 结果表
- `backend/services/__init__.py` 或新建 dispatcher — 按 `RAG_VERSION` 路由 v1 / v2

### 不改

- `backend/core/agent.py`(只通过 tool 调用 RAG,签名兼容)
- `backend/core/prompts.py`
- `backend/services/tools_registry.py`(query_product_database 签名兼容)
- `backend/api/routes.py`
- 现有 `backend/services/rag_service.py` 暂保留作为 v1 实现,v2 稳定后再删

---

## Rollout

1. **Phase 1 — 基建**: 写 `rag/` 子包骨架 + corpus loader + chunker + 单元测试。`RAG_VERSION=v1` 仍是默认,不影响生产
2. **Phase 2 — 检索**: hybrid (BM25 + vector + RRF)、reranker、persona filter。每个独立 commit,每个有对应单测
3. **Phase 3 — 缓存**: semantic cache 实现 + PROVE 测试翻转
4. **Phase 4 — 数据**: 运行 `generate_corpus.py`,人工 spot-check,commit corpus JSON
5. **Phase 5 — Eval**: gold_queries 标注,跑 5 个 config,生成 report.md
6. **Phase 6 — 收尾**: README 重写,切 `RAG_VERSION=v2` 为默认,删 v1 代码

每个 phase 结束 commit,可独立验证。

---

## 风险与缓解

| 风险 | 缓解 |
|---|---|
| LLM 合成语料质量低,eval 数字失真 | spot-check 10% + adversarial query 作 sanity check |
| Reranker 增加 200ms 延迟,影响 SSE 体感 | 异步执行,与 agent 思考并行;若超过 500ms 加 fallback 跳过 rerank |
| Semantic cache threshold 选错导致 false positive 污染输出 | eval 阶段扫超参 + false_hit_rate 监控,生产可通过 env var 调整 |
| 5 config × 80 query × 端到端 generate 实际超预算 | 加 `--dry-run` + 单 config 限流;先跑 C0/C4 两个极端验证基础 delta |
| Persona hard filter 把候选过滤空导致 agent 失败 | 空集自动放宽 fallback,日志 WARN,eval 中专项覆盖 |

---

## Future Work(明确不在本 spec 范围)

- HyDE / multi-query expansion(Bundle B 内容)
- Anthropic Contextual Retrieval(Bundle B)
- Agent-RAG 协同决策,reflection 反馈触发二次检索(Bundle C)
- Redis 分布式 semantic cache
- 真实小红书数据爬取(合成数据先跑通,再考虑真实数据)

# Test Suite Design — rednote_agent

**Date:** 2026-05-18
**Author:** brainstorming session
**Status:** approved, awaiting implementation

## 背景与动机

`rednote_agent` 当前**零测试**。本设计为面试准备 + 工程严谨度补课，目标是：

1. 用回归测试**保护**几处真正有意思的设计（best_draft 降级、并行 tool 调用、参数化 SQL）
2. 用**故意失败的"证明性"测试**记录已知 bug，便于面试时讲清"我发现了 X、修法是 Y"
3. 用一次**真实 A/B 实验**给"Planning 阶段是否值得"这一句简历宣称提供量化依据

测试范围来源于 `面试拷打题库.md` 的 Q14 / Q15 / Q17.1 / Q18.1 / Q19.1。

## 非目标

- **不修任何 bug。** 本次只证明、不修复（面试叙事优先于代码改进）。
- **不引入鉴权。** user_id 越权漏洞仅做演示性证明。
- **不做统计假设检验。** A/B 样本量 30 + 自评分数指标，做 p-value 反而失实。
- **不评测生成内容质量本身。** 不引入外部 judge 模型，只用 agent 自评分数（接受 self-eval bias，在报告中显式说明）。

## 生产代码改动（最小化）

为支撑 A/B 实验，**唯一**的生产代码改动：

- `generate_rednote(request, *, enable_planning: bool = True)`：新增关键字参数，默认 True（保持现有行为）。A/B 脚本通过传 `enable_planning=False` 跳过 `_planning_phase` 调用。
- 调用方 `api/routes.py` 不需要改（用默认值即可）。

这是测试 hook，不是 bug 修复，不违反"不修 bug"非目标。

## 架构

```
backend/tests/
├── conftest.py                 # 共享 fixture
├── test_agent_flow.py          # Q18.1 主流程
├── test_best_draft.py          # Q18.1 best_draft 降级 + 第3稿被吃 bug
├── test_rag_stability.py       # Q18.1 RAG top-3 稳定 + LRU 不语义匹配
├── test_asyncio_gather.py      # Q14 gather 异常 + 超时取消
├── test_memory_security.py     # Q15 SQL 参数化 + 越权读取
└── test_rate_limit.py          # Q17.1 dict 不回收 + 多实例失效

scripts/
├── ab_queries.json             # 30 条人工编写的 query
└── ab_planning_experiment.py   # Q19.1 A/B 实验（真调 DeepSeek）

docs/
├── superpowers/specs/2026-05-18-test-suite-design.md   # 本文件
└── testing/2026-05-18-results.md                       # 跑完后的结果报告

backend/requirements-dev.txt    # pytest + pytest-asyncio + pytest-timeout
```

**离线测试**（pytest）零成本、秒级、可重复。**A/B 实验**单独脚本，必须显式 `--confirm` 才能花钱。

## 离线测试清单

每个测试在 docstring 里注明 `PASS` 或 `PROVE`。`PROVE` 表示故意断言现有（buggy）行为以记录 bug。

### `test_agent_flow.py` — Agent 主流程

| 用例 | 断言 | 类型 |
|---|---|---|
| `test_event_sequence_on_happy_path` | mock client 1 轮 tool_calls + 1 轮 JSON → 事件序列为 `agent_thinking → tool_result → agent_thinking → complete → token_usage` | PASS |
| `test_tool_calls_executed_in_parallel` | 两个 tool 各 sleep 0.5s → 总耗时 < 0.8s | PASS |
| `test_tool_cache_hits_within_request` | 同一 (name, args) 第二次走缓存，事件 `cached=True`、真实函数仅调一次 | PASS |
| `test_json_retry_on_malformed_output` | 先返回非 JSON、再返回合法 JSON → 看到修正事件 → complete | PASS |

### `test_best_draft.py` — Best-draft 降级

| 用例 | 断言 | 类型 |
|---|---|---|
| `test_best_draft_tracks_highest_score` | 反思打分 6/8/7 → final_result 是 8 分那一稿 | PASS |
| `test_third_draft_swallowed_by_best_draft` | MAX_REFLECTIONS=2、第 3 稿无评分 → 最终返回 best_draft 而**不是**第 3 稿 | **PROVE** |
| `test_returns_unscored_when_no_best_exists` | 两次反思均抛异常 → best_draft=None → 兜底返回当前 result | PASS |

### `test_rag_stability.py` — RAG 与 LRU

| 用例 | 断言 | 类型 |
|---|---|---|
| `test_top3_deterministic_for_same_query` | 同 query 跑 5 次、top-3 product id 完全一致 | PASS |
| `test_lru_does_not_match_paraphrase` | "敏感肌补水" vs "敏感皮补水" → `cache_info().misses` 两次都 +1 | **PROVE** |

### `test_asyncio_gather.py` — 并发语义

| 用例 | 断言 | 类型 |
|---|---|---|
| `test_gather_propagates_first_exception` | 3 个 task 中 1 个抛错 → 默认 gather 整组失败 | **PROVE** |
| `test_wait_for_cancels_only_self` | 1 个超时 → 其余跑完不被取消 | PASS |

### `test_memory_security.py` — 持久化安全

| 用例 | 断言 | 类型 |
|---|---|---|
| `test_all_sql_uses_placeholders` | 静态扫描 `memory_service.py`，断言无 f-string SQL / 无 `%` 拼接 | PASS |
| `test_sql_injection_payload_is_neutralized` | `user_id="' OR 1=1 --"` → 返回"暂无历史" | PASS |
| `test_arbitrary_user_id_reads_anyone_history` | 客户端伪造任意 user_id 即可读到他人记录 | **PROVE** |

### `test_rate_limit.py` — 限流

| 用例 | 断言 | 类型 |
|---|---|---|
| `test_window_correctly_counts_in_window_requests` | 10 次/分钟内、第 11 次返回 429 | PASS |
| `test_ip_key_never_evicted` | 1000 不同 IP 各 1 次 → 等待窗口过期 → dict 仍保留 1000 个 key | **PROVE** |
| `test_multi_instance_simulated` | 2 个独立 dict 模拟 2 个 pod → 同 IP 各打满 10 次都不触发 | **PROVE** |

**合计 17 个用例，6 个 PROVE。**

## A/B 实验设计（Q19.1）

### 假设

H0：Planning 阶段对 agent 输出质量（自评分数）无显著影响，且额外消耗 token / 延迟。
H1：Planning 阶段改善质量，且节省的下游 token 不足以抵消 planning call 本身的开销。

### 设计

- **配对设计**（within-subject）：30 个 query，每条在 A 路（planning=True）和 B 路（planning=False）各跑一次，**共 60 次完整 agent 调用**。
- **顺序随机化**：每个 query 内的 A/B 执行顺序随机交替，消除时段 / 缓存预热偏差。
- **控制变量**：
  - 暴露 `enable_planning: bool` 参数透传进 `generate_rednote`，生产默认仍为 True，仅测试使用此开关
  - 同一 RAG 索引、同一会话、temperature=0.7 固定
- **样本来源**：手工编写 30 条 query，分布在 6 类需求（补水/控油/抗老/防晒/美白/敏感修复）各 5 条。落盘 `scripts/ab_queries.json`。**禁止用 GPT 生成 query**——避免引入"AI 偏好措辞"偏差。

### 捕获指标

每次 run 一行 JSONL：

```
query_id, arm (A|B), success, reflection_min_score,
reflection_virality, reflection_tone, reflection_accuracy,
prompt_tokens, completion_tokens, total_tokens,
iterations, wall_clock_seconds, error
```

### 报告产出

`docs/testing/2026-05-18-results.md` 自动生成，包含：

- 配对差值表（per query 的 A−B：score、tokens）
- 汇总：mean ± stdev、配对差均值、A 胜率（A 优于 B 的 query 数 / 30）
- 失败案例清单
- **诚实结论段**：必须说明
  - self-eval bias 风险（agent 给自己打分）
  - 样本量 30 的局限（趋势 only，不是显著性）
  - 单评估源不能宣称"质量"，只能宣称"系统自评分"

### 鲁棒性

- `--dry-run`：每组 3 条（约 $0.20），先跑通管道
- `--n N`：默认 30
- `--confirm`：>10 条必须显式 flag，防误炸
- **增量 checkpoint**：每完成一次 run 立刻 append 到 JSONL，崩了 `--resume` 接着跑
- 单条 run 失败不中断实验，标 success=False 继续
- 总预估耗时与成本启动时打印，Ctrl+C 友好退出保留已有数据

## Mock 与基础设施

`conftest.py` 提供：
- `mock_deepseek_client`：可编程桩客户端，按 `side_effect` 队列返回 tool_calls / 文本 / 异常。monkeypatch `get_client`。
- `tmp_memory_db`：临时 SQLite 路径，自动 init_db + 测后清理。Windows 文件锁兼容（`try/finally`）。
- pytest-asyncio 默认 event loop。

依赖（`backend/requirements-dev.txt`）：`pytest`, `pytest-asyncio`, `pytest-timeout`

## 运行命令

```powershell
# 安装开发依赖
pip install -r backend/requirements-dev.txt

# 离线全跑
pytest backend/tests -v

# 跑指定文件
pytest backend/tests/test_best_draft.py -v

# A/B dry-run（约 $0.20）
python scripts/ab_planning_experiment.py --dry-run

# A/B 正式（约 $1.8-3.0）
python scripts/ab_planning_experiment.py --n 30 --confirm
```

## 风险与限制

| 风险 | 影响 | 缓解 |
|---|---|---|
| DeepSeek API 限流 / 抖动 | A/B 失败率升高 | 单条失败不中断、checkpoint 可断点续跑 |
| Reflection 自评 bias | A/B 结果偏向"agent 觉得自己好"而非真好 | 报告显式说明、不宣称"质量"只宣称"自评分" |
| 30 query 样本量小 | 不足以做显著性 | 报告只讲趋势与胜率，不报 p-value |
| Windows SQLite 锁延迟 | 偶发测试失败 | tmp_memory_db 用 try/finally + retry 删除 |
| 手编 query 集偏个人审美 | 实验结果不可外推 | query 集 commit，可被他人 review |

## 验收标准

- `pytest backend/tests -v` 全绿（PROVE 用例也"pass"——它们断言的就是 buggy 行为）
- `python scripts/ab_planning_experiment.py --dry-run` 跑通、生成报告骨架
- `docs/testing/2026-05-18-results.md` 写出，包含真实数据 + 诚实结论段

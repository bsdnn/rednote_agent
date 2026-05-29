# Xiaohongshu AI Copywriting Agent

> A production-grade AI agent that generates Xiaohongshu (RED) viral posts. DeepSeek LLM with function calling + FAISS RAG over a 21-product skincare catalog + SSE streaming, with a Vite + React + TypeScript frontend.

Give it a skincare query (e.g. *"fade acne scars"*) and the agent autonomously calls tools — semantic product lookup, trending-topic fetch, web search — then returns a structured post: **title · body · hashtags · emojis**.

---

## Highlights

- **Three-phase agent pipeline** — Planning → Tool-using Agent Loop → Self-Reflection. The planning phase produces a structured task plan that guides the agent; reflection scores every draft and triggers rewrites when quality drops below threshold. Best-scored draft tracking ensures the agent never regresses on rewrite.
- **Plan-and-Execute with parallel tools** — LLM-issued tool calls (RAG search, Tavily web search, trending topics) execute concurrently via `asyncio.gather`, with per-request result caching to avoid redundant API calls
- **A/B-verified planning** — a 30-query paired experiment confirmed planning raises success rate from 50% to **80%** while cutting total token cost by **40%** (5753 vs 9678) — the upfront planning call more than pays for itself by keeping the agent on track
- **RAG v2 — eval-driven hybrid pipeline** — BM25 (jieba-tokenized) + dense (text2vec-base-chinese) → RRF fusion → cross-encoder rerank (bge-reranker-base) → persona-aware filter & soft boost → semantic cache (cosine threshold, LRU+TTL). Paired ablation across 5 configs (C0..C4) over 40 hand-authored gold queries with auto-generated report; cross-encoder is the biggest ranking lever (+5pp MRR, +0.25 faithfulness) at 50× retrieval latency cost.
- **User persona conditioning** — skin type / age group / preferences / budget shape the copy toward the target reader; persistent copy history via SQLite with parameterized queries
- **Self-reflection with best-draft fallback** — every draft is scored on virality, tone match, and factual accuracy; drafts scoring below threshold trigger targeted rewrites; the highest-scored draft across all reflection rounds wins
- **Multi-turn refinement** — iterate on a generated post with follow-up instructions ("make it more professional", "add more hashtags")
- **Structured output** — Pydantic validation with a `model_validator` that auto-normalizes hashtag prefixes
- **SSE event streaming** — frontend renders planning, thinking steps, parallel tool calls, reflection progress, final result, and token usage live
- **Production-minded** — request-ID tracing, structured logging, global httpx singleton + startup warmup, in-memory rate limiting, CORS allowlist, tenacity exponential-backoff retries

---

## Tech Stack

| Layer | Tech |
|---|---|
| LLM | DeepSeek `deepseek-chat` (function calling) |
| Backend | FastAPI + uvicorn |
| HTTP client | httpx (async) + tenacity retries |
| Streaming | sse-starlette |
| Vector search | FAISS (cosine) + BM25 (rank_bm25 + jieba) + RRF + bge-reranker-base |
| Web search | Tavily API (replaced DuckDuckGo after diagnosing tool-reliability failures) |
| Web fetching | Tavily extract API with 2000-char truncation |
| Config | pydantic-settings |
| Frontend | React 19 + TypeScript |
| Build | Vite 8 |

---

## Quick Start

### Requirements

- Python 3.10+
- Node.js 18+
- A DeepSeek API key ([platform.deepseek.com](https://platform.deepseek.com))

### 1. Clone & configure

```bash
git clone https://github.com/bsdnn/rednote_agent.git
cd rednote_agent

cp .env.example .env
# edit .env, fill in your DeepSeek API key
```

`.env` example:

```env
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
DEEPSEEK_API_URL=https://api.deepseek.com/chat/completions
TAVILY_API_KEY=your-key-here
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:8000
RATE_LIMIT_PER_MINUTE=10
LOG_LEVEL=INFO
```

> Tavily provides web search (`search_web`, `get_trending_topics`) and page extraction (`fetch_webpage`). Free tier: 1000 calls/month. Register at [tavily.com](https://tavily.com) (GitHub login, no credit card).

### 2. Install dependencies

```bash
# Backend
pip install -r backend/requirements.txt

# Frontend
cd frontend && npm install
```

### 3. Run in development

**Terminal 1 — backend** (from project root)

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — frontend**

```bash
cd frontend && npm run dev
```

Open `http://localhost:5173`. Vite proxies `/api/*` to the backend on `:8000`.

### 4. Production build

```bash
cd frontend && npm run build
# build artifacts go to backend/static/, served directly by FastAPI
uvicorn backend.main:app --host 0.0.0.0 --port 8000
# visit http://localhost:8000
```

Or use the Makefile:

```bash
make install        # install both backend and frontend deps
make dev-backend    # run backend
make dev-frontend   # run frontend
make build          # build frontend, then start in prod mode
```

---

## Project Layout

```
rednote/
├── backend/
│   ├── main.py                  # FastAPI entry — request-ID middleware, startup warmup, CORS
│   ├── requirements.txt
│   ├── api/
│   │   ├── routes.py            # POST /api/generate · POST /api/refine · GET /api/health
│   │   └── dependencies.py      # rate limiter (10 req/min/IP, configurable)
│   ├── core/
│   │   ├── config.py            # pydantic-settings, reads .env
│   │   ├── logging_config.py    # structured logging + ContextVar request-ID filter
│   │   ├── agent.py             # async agent loop — tool timeouts, JSON retry, token accounting
│   │   └── prompts.py           # SYSTEM_PROMPT + build_user_message()
│   ├── services/
│   │   ├── deepseek_client.py   # global httpx singleton + tenacity retry
│   │   ├── rag/                 # RAG v2 — hybrid + rerank + persona + semantic cache
│   │   │   ├── service.py       # public entrypoint, lazily builds pipeline
│   │   │   ├── pipeline.py      # RAGv2Pipeline orchestration
│   │   │   ├── corpus/          # loader + chunker (products + ingredients + posts)
│   │   │   └── retrievers/      # vector, BM25, RRF, reranker, persona, cache
│   │   ├── search_service.py    # Tavily web search + trending topics + page extraction
│   │   └── tools_registry.py    # AVAILABLE_TOOLS + TOOLS_DEFINITION
│   ├── models/
│   │   ├── request.py           # GenerateRequest · RefineRequest · Tone enum
│   │   ├── response.py          # GenerateResponse (auto-normalizes hashtag prefixes)
│   │   └── persona.py           # UserPersona
│   ├── data/
│   │   └── products.json        # 21 skincare products (with ingredients + skin types)
│   ├── tests/                   # pytest suite (17 cases, 6 PROVE)
│   │   ├── conftest.py          # shared fixtures (mock client, temp DB, reset rate limiter)
│   │   ├── test_agent_flow.py   # main loop: event order, parallelism, cache, JSON retry
│   │   ├── test_best_draft.py   # reflection fallback + third-draft-swallowed bug
│   │   ├── test_rag_stability.py       # top-3 determinism + LRU exact-match (not semantic)
│   │   ├── test_asyncio_gather.py      # exception propagation + wait_for isolation
│   │   ├── test_memory_security.py     # parameterized SQL + unauthenticated user_id
│   │   └── test_rate_limit.py          # window count + IP leak + multi-pod bypass
│   └── static/                  # frontend build output (produced by `npm run build`)
│
├── frontend/
│   ├── vite.config.ts           # /api proxy → :8000, build output → backend/static/
│   └── src/
│       ├── App.tsx              # split layout — sticky form on the left, results on the right
│       ├── index.css            # design system — warm white, single red accent, dotted background
│       ├── types/api.ts
│       ├── hooks/
│       │   ├── useSSEGenerate.ts        # SSE state machine (fetch + ReadableStream)
│       │   ├── useTypewriter.ts         # progressive text reveal
│       │   └── useConversationHistory.ts
│       └── components/
│           ├── NavBar.tsx               # brand + API health indicator
│           ├── EmptyState.tsx
│           ├── Card.tsx
│           ├── InputForm.tsx
│           ├── ToneSelector.tsx         # 5 tone presets
│           ├── PersonaPanel.tsx         # collapsible persona panel
│           ├── LoadingState.tsx         # terminal-log style step display
│           ├── ResultCard.tsx           # typewriter reveal + token-usage badge
│           ├── CopyButton.tsx
│           └── RefinementBar.tsx
│
├── scripts/
│   ├── ab_planning_experiment.py  # A/B experiment runner (paired design, incremental JSONL, auto-report)
│   ├── ab_queries.json            # 30 hand-written queries across 6 skincare categories
│   ├── diagnose_failure.py        # single-query diagnostic for debugging agent failures
│   └── ab_results.jsonl           # full N=30 experiment raw data (60 records)
├── .env.example
├── docker-compose.yml
└── Makefile
```

---

## API

### `POST /api/generate`

Streams generation as `text/event-stream`.

**Request body:**

```json
{
  "query": "fade acne scars",
  "tone": "活泼甜美",
  "max_iterations": 5,
  "persona": {
    "skin_type": "combination",
    "age_group": "25-30",
    "preferences": ["brightening", "hydration"],
    "budget": "mid-range"
  }
}
```

`tone` values: `活泼甜美` · `专业种草` · `温柔治愈` · `幽默搞笑` · `精英范儿`
(*lively · expert recommendation · gentle · humorous · elite*)

`persona` is optional; if omitted, no persona context is injected.

**SSE event stream:**

```
event: agent_thinking
data: {"step": "Thinking (round 1)...", "iteration": 1}

event: agent_thinking
data: {"step": "Calling tool: query_product_database", "tool": "query_product_database"}

event: tool_result
data: {"tool": "query_product_database", "summary": "..."}

event: complete
data: {"title": "...", "body": "...", "hashtags": ["#tag"], "emojis": ["✨"]}

event: token_usage
data: {"prompt_tokens": 412, "completion_tokens": 187, "total_tokens": 599}

event: error
data: {"message": "...", "code": 500}
```

> The frontend uses `fetch + ReadableStream` rather than the native `EventSource` so it can send a POST body.

### `POST /api/refine`

Multi-turn refinement on a previous result; also returns SSE.

**Request body:**

```json
{
  "previous_result": { "title": "...", "body": "...", "hashtags": [], "emojis": [] },
  "refinement_instruction": "make it more professional",
  "conversation_history": [],
  "tone": "专业种草"
}
```

### `GET /api/health`

```json
{
  "status": "ok",
  "rag_products": 21,
  "uptime_seconds": 142,
  "model": "deepseek-chat"
}
```

---

## Implementation Notes

- **Fully async** — `deepseek_client.py` uses `httpx.AsyncClient`. CPU-bound calls (FAISS, sentence-transformers, Tavily search) are wrapped in `run_in_executor` so they never block the event loop.
- **Three-phase design** — Planning (structured task plan) → Agent Loop (parallel tool calls, JSON drafts) → Self-Reflection (scored critique, max 2 rewrites). Best-scored draft across all rounds wins.
- **Top-3 RAG** — embedding spans name + description + ingredients + skin types, returning more candidates broadens the agent's options on each call.
- **Parallel tool execution** — multiple LLM-issued tool calls run concurrently via `asyncio.gather`; per-request `_tool_cache` prevents duplicate API calls within a single generation.
- **Pydantic normalization** — `GenerateResponse.model_validator` ensures every hashtag starts with `#`.
- **Rate limiting** — in-memory sliding window. Fine for dev; swap for Redis in production. Known limitation: per-pod dicts don't share state in multi-instance deployments.
- **CORS** — origins come from `ALLOWED_ORIGINS` (comma-separated env var); never `*`.
- **Request IDs** — middleware assigns an 8-char hex ID per request, injects it into log records and the `X-Request-ID` response header for end-to-end tracing.
- **Retry policy** — `tenacity` retries on `httpx.TimeoutException` / `httpx.ConnectError` with exponential backoff, max 3 attempts.
- **Startup warmup** — the app issues one RAG query at boot so the first user request avoids cold-start latency.

---

## Testing

62-unit pytest suite (all passing in ~13s) + a paired A/B experiment pipeline + a RAG v2 ablation harness.

```powershell
# Install test deps
pip install -r backend/requirements-dev.txt

# Run offline tests (no API calls, ~16s)
pytest backend/tests -v

# A/B experiment — 30 query × 2 arms paired design, ~$2 DeepSeek + Tavily free tier
python scripts/ab_planning_experiment.py --n 30 --confirm --force

# Dry-run first (3 queries, ~$0.20)
python scripts/ab_planning_experiment.py --dry-run

# Diagnose a single failing query
python scripts/diagnose_failure.py 1
```

### Test coverage

| Area | File | Cases | Notes |
|---|---|---|---|
| Agent main loop | `test_agent_flow.py` | 4 | Event sequence, parallel tools (0.52s measured), per-request cache, JSON retry |
| Best-draft fallback | `test_best_draft.py` | 3 | Score tracking, third-draft-swallowed bug (PROVE), unscored fallback |
| RAG v2 pipeline | `test_rag_v2_pipeline.py` + retriever/cache/persona/chunker/loader tests | 45+ | Hybrid retrieval, RRF, cross-encoder, persona filter, semantic cache (PROVE-to-PASS flip), corpus schema |
| asyncio.gather | `test_asyncio_gather.py` | 2 | Default exception propagation (PROVE), wait_for isolation |
| Memory security | `test_memory_security.py` | 3 | Parameterized SQL (pass), SQL injection neutralized (pass), unauthenticated user_id read (PROVE) |
| Rate limiting | `test_rate_limit.py` | 3 | Window counting (pass), IP key never evicted (PROVE), multi-pod bypass (PROVE) |

6 PROVE-type tests document known bugs with suggested fixes — see `docs/testing/2026-05-18-results.md` for the full analysis.

### A/B experiment (planning ON vs OFF)

N=30 queries across 6 skincare categories, paired within-subject design.

| Metric | Planning ON (A) | Planning OFF (B) |
|---|---|---|
| Success rate | **80%** (24/30) | 50% (15/30) |
| Avg tokens | **5,753** | 9,678 |
| Avg iterations | **2.38** | 3.73 |
| Reflection score | **7.0** ± 0.78 | 6.64 ± 0.84 |

Full report: `docs/testing/2026-05-18-ab-report.md` · Raw data: `scripts/ab_results.jsonl` (60 JSONL records)

### RAG v2 Ablation (Hybrid + Rerank + Semantic Cache + Persona)

N=40 hand-authored gold queries × 5 configs, paired design.
Pipeline: vector (text2vec-base-chinese) + BM25 (jieba) → RRF fusion → cross-encoder rerank (bge-reranker-base) → persona-aware filter + soft boost → semantic cache (cosine threshold, LRU+TTL).

| Config | Recall@3 | MRR | Faithfulness | Avg sec |
|---|---|---|---|---|
| C0 baseline (vector only) | 0.558 | 0.333 | 6.50 | 0.04 |
| C1 + BM25/RRF             | 0.579 | 0.321 | 6.43 | 0.04 |
| C2 + cross-encoder rerank | 0.576 | **0.388** | **6.75** | 2.13 |
| C3 + semantic cache       | 0.576 | 0.388 | 6.65 | 2.26 |
| C4 + persona-aware        | 0.564 | 0.388 | 6.48 | 2.09 |

**Per-category Recall@3 — where each technique shines:**

| Category | C0 | C2 | Delta | Insight |
|---|---|---|---|---|
| ingredient_lookup | 0.486 | **0.587** | +10pp | BM25 wins exact ingredient-name queries |
| persona_strong    | 0.667 | **0.750** | +8pp  | Persona soft boost reorders relevant products up |
| direct_need       | 0.333 | 0.333 | 0    | Reranker neutral here; C1 gets +8pp from BM25 |
| colloquial        | 0.400 | 0.200 | **-20pp** | Cross-encoder mismatched to vague口语化 queries |
| synonym           | 0.625 | 0.625 | 0    | Vector+rerank both handle synonyms equally |

**Trade-offs:**
- Cross-encoder is the single biggest lever for ranking quality (+5pp MRR, +0.25 faithfulness) but adds ~50× retrieval latency (2.1s vs 0.04s).
- BM25 hybrid wins exactly the canonical use case (exact-name ingredient lookup) but slightly hurts cross-category queries.
- Persona hard-filter (C4) is currently over-restrictive — slight overall recall drop vs C3. Future work: soften to soft constraint when candidate pool runs thin.
- Colloquial query regression with reranker (-20pp) is a real, honest finding — bge-reranker-base isn't tuned for vague natural-language queries.

Full report: `docs/testing/2026-05-28-rag-v2-report.md` · Raw data: `scripts/rag_eval/results/*.jsonl`

---

## Notes & Caveats

- First backend boot loads `sentence-transformers` (~5–10s). This is normal.
- DeepSeek total request timeout: 60s. Per-tool call: 15s timeout (configurable via `TOOL_TIMEOUT_SECONDS`).
- Default rate limit is 10 req/min/IP; tune via `RATE_LIMIT_PER_MINUTE`.
- For production, replace the dev origins in `ALLOWED_ORIGINS` with real domains.
- Log format: `time [request_id] LEVEL module: message`.
- The three-phase pipeline (planning → loop → reflection) adds an extra LLM call at the start and up to 2 reflection cycles, but empirically reduces total token consumption by keeping the agent on track (see testing section above).

---

## License

MIT

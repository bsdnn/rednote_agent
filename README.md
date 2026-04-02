# Xiaohongshu AI Copywriting Agent

> A production-grade AI agent that generates Xiaohongshu (RED) viral posts. DeepSeek LLM with function calling + FAISS RAG over a 21-product skincare catalog + SSE streaming, with a Vite + React + TypeScript frontend.

Give it a skincare query (e.g. *"fade acne scars"*) and the agent autonomously calls tools — semantic product lookup, trending-topic fetch, web search — then returns a structured post: **title · body · hashtags · emojis**.

---

## Highlights

- **Multi-tool agent loop** — product DB semantic search, trending topics, web search; per-tool 15s timeout; up to 5 iterations
- **SSE streaming** — frontend renders thinking steps, tool calls, final result, and token usage live
- **FAISS RAG** — `text2vec-base-chinese` embeddings over name + description + ingredients + skin types; LRU-cached; top-3 retrieval
- **User persona conditioning** — skin type / age group / preferences / budget shape the copy toward the target reader
- **Multi-turn refinement** — iterate on a generated post with follow-up instructions ("make it more professional", "add more hashtags")
- **Structured output** — Pydantic validation with a `model_validator` that auto-normalizes hashtag prefixes
- **Token usage** — every generation reports prompt / completion / total tokens to the UI
- **Production-minded** — request-ID tracing, structured logging, global httpx singleton + startup warmup, in-memory rate limiting, CORS allowlist, tenacity exponential-backoff retries

---

## Tech Stack

| Layer | Tech |
|---|---|
| LLM | DeepSeek `deepseek-chat` (function calling) |
| Backend | FastAPI + uvicorn |
| HTTP client | httpx (async) + tenacity retries |
| Streaming | sse-starlette |
| Vector search | FAISS + sentence-transformers (`text2vec-base-chinese`) |
| Web search | duckduckgo-search |
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
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:8000
RATE_LIMIT_PER_MINUTE=10
LOG_LEVEL=INFO
```

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
│   │   ├── rag_service.py       # FAISS index + LRU cache + top-3 retrieval
│   │   ├── search_service.py    # DuckDuckGo search + trending-topic scrape
│   │   └── tools_registry.py    # AVAILABLE_TOOLS + TOOLS_DEFINITION
│   ├── models/
│   │   ├── request.py           # GenerateRequest · RefineRequest · Tone enum
│   │   ├── response.py          # GenerateResponse (auto-normalizes hashtag prefixes)
│   │   └── persona.py           # UserPersona
│   ├── data/
│   │   └── products.json        # 21 skincare products (with ingredients + skin types)
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

- **Fully async** — `deepseek_client.py` uses `httpx.AsyncClient`. CPU-bound calls (FAISS, sentence-transformers) are wrapped in `run_in_executor` so they never block the event loop.
- **Top-3 RAG** — embedding spans name + description + ingredients + skin types, and returning more candidates broadens the agent's options on each call.
- **Pydantic normalization** — `GenerateResponse.model_validator` ensures every hashtag starts with `#`.
- **Rate limiting** — in-memory sliding window. Fine for dev; swap for Redis in production.
- **CORS** — origins come from `ALLOWED_ORIGINS` (comma-separated env var); never `*`.
- **Request IDs** — middleware assigns an 8-char hex ID per request, injects it into log records and the `X-Request-ID` response header for end-to-end tracing.
- **Retry policy** — `tenacity` retries on `httpx.TimeoutException` / `httpx.ConnectError` with exponential backoff, max 3 attempts.
- **Startup warmup** — the app issues one RAG query at boot so the first user request avoids cold-start latency.

---

## Notes & Caveats

- First backend boot loads `sentence-transformers` (~5–10s). This is normal.
- DeepSeek total request timeout: 60s. Per-tool call: additional 15s timeout.
- Default rate limit is 10 req/min/IP; tune via `RATE_LIMIT_PER_MINUTE`.
- For production, replace the dev origins in `ALLOWED_ORIGINS` with real domains.
- Log format: `time [request_id] LEVEL module: message`.

---

## License

MIT

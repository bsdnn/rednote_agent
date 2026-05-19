# Tavily Migration Design — Replace DuckDuckGo Search Backend

**Date:** 2026-05-18  
**Status:** approved  

## 背景

诊断发现 `search_service.py` 的 DuckDuckGo（DDGS）底层是三个工具失败的根源：

- `search_web` 搜出无关 ASMR 内容
- `get_trending_topics` 频繁返回空
- `fetch_webpage` 两次返回"页面内容为空"

LLM 在收到糟糕工具结果后会尝试"重搜一次"，5 轮迭代全耗在搜索循环里，**一次都没尝试输出 JSON**。根因是工具质量，不是容忍上限。

## 设计

**核心原则：工具签名不变，只换底层实现。** `tools_registry.py`、`SYSTEM_PROMPT`、`agent.py` 全部不动。

### 工具映射

| 工具 | 当前 | 改为 |
|------|------|------|
| `search_web(query)` | `DDGS().text(query, max_results=3)` | `tavily.search(query, max_results=3, search_depth="basic")` → title + content |
| `fetch_webpage(url)` | httpx + BeautifulSoup 自抓 | `tavily.extract(urls=[url])` → 干净提取文本（保留 2000 字截断） |
| `get_trending_topics(category)` | `DDGS().text()` × 2 + 正则拆词 | `tavily.search(f"小红书 {category} 热门 2025")` → 从标题提取关键词 |

### 为什么 fetch_webpage 换 Tavily extract

诊断中的 `fetch_webpage` 两次空返回是因为 httpx 本地抓取遇到反爬 / JS 渲染 / 网络问题。Tavily 的 extract API 是服务端抓 + 内容提取，可靠性远高于本地 BS4。

### 改动文件

- `backend/services/search_service.py` —— 重写 3 个函数实现（删 DDGS、换 tavily client）
- `backend/core/config.py` —— 加 `TAVILY_API_KEY: str = ""`
- `backend/requirements.txt` —— 删 `duckduckgo-search` 和 `beautifulsoup4`，加 `tavily-python`
- `.env.example` —— 加 `TAVILY_API_KEY=your-key-here`

### 不改的文件

- `backend/core/agent.py`
- `backend/core/prompts.py`
- `backend/services/tools_registry.py`
- `backend/api/routes.py`

### 成本

Tavily free tier：1000 API calls/month。A/B 实验 60 runs × 约 3 搜索/run = 180 次，完全在免费额度内。

### 同步任务

`TAVILY_API_KEY` 需要写到 `.env`。旧 key `DEEPSEEK_API_KEY` 保持在 `.env` 中不变。

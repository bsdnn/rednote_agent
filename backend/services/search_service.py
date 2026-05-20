"""Search tools backed by Tavily API (replaces DuckDuckGo + BeautifulSoup)."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import re

from tavily import TavilyClient

from ..core.config import settings

_executor = ThreadPoolExecutor(max_workers=2)

_client = TavilyClient(api_key=settings.TAVILY_API_KEY)


def _sync_search_web(query: str) -> str:
    try:
        response = _client.search(query, max_results=3, search_depth="basic")
        results = response.get("results", [])
        if not results:
            return "No results found."
        return "\n---\n".join(
            f"Title: {r.get('title', '')}\nContent: {r.get('content', '')}"
            for r in results
        )
    except Exception as e:
        return f"Search error: {e}"


def _sync_get_trending(category: str) -> str:
    queries = [
        f"小红书 {category} 爆款 2025",
        f"小红书 {category} 种草 热门",
    ]
    all_titles: list[str] = []
    try:
        for q in queries:
            response = _client.search(q, max_results=5, search_depth="basic")
            results = response.get("results", [])
            if results:
                all_titles.extend(r.get("title", "") for r in results)
    except Exception:
        return f"暂时无法获取{category}热门趋势数据。"

    # Extract Chinese keywords (2-6 char segments)
    words: list[str] = []
    for title in all_titles:
        words.extend(re.findall(r"[一-鿿]{2,6}", title))

    counter = Counter(words)
    top = [
        w for w, c in counter.most_common(settings.TRENDING_CANDIDATE_LIMIT) if c > 1
    ][: settings.TRENDING_KEYWORD_LIMIT]

    if not top:
        return f"{category}相关热门关键词暂时获取失败。"
    return f"小红书{category}当前热门关键词：{'、'.join(top)}"


async def search_web(query: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _sync_search_web, query)


async def get_trending_topics(category: str = "skincare") -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _sync_get_trending, category)


def _sync_fetch_webpage(url: str) -> str:
    try:
        response = _client.extract(urls=[url])
        results = response.get("results", [])
        if not results:
            return "页面内容为空。"
        text = results[0].get("raw_content", "") or ""
        if not text:
            return "页面内容为空。"
        # Truncate to ~2000 chars (matching the old BS4 cutoff)
        return text[:2000]
    except Exception as e:
        return f"抓取页面失败: {e}"


async def fetch_webpage(url: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _sync_fetch_webpage, url)

import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import re

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS

from ..core.config import settings

_executor = ThreadPoolExecutor(max_workers=2)
_http_client = httpx.AsyncClient(timeout=10.0, follow_redirects=True)


def _sync_search_web(query: str) -> str:
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return "No results found."
        return "\n---\n".join(
            f"Title: {r.get('title', '')}\nSummary: {r.get('body', '')}"
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
            results = DDGS().text(q, max_results=5)
            if results:
                all_titles.extend(r.get("title", "") for r in results)
    except Exception:
        return f"暂时无法获取{category}热门趋势数据。"

    # Extract Chinese keywords (2-6 char segments)
    words: list[str] = []
    for title in all_titles:
        words.extend(re.findall(r"[\u4e00-\u9fff]{2,6}", title))

    counter = Counter(words)
    # Remove single-occurrence generic words
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


async def fetch_webpage(url: str) -> str:
    try:
        response = await _http_client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [ln for ln in text.splitlines() if len(ln) > 20]
        return "\n".join(lines)[:2000] or "页面内容为空。"
    except Exception as e:
        return f"抓取页面失败: {e}"

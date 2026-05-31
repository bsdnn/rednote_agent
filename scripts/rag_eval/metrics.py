"""RAG eval metrics: recall@k, MRR, forbidden detection, and LLM faithfulness judge."""
from __future__ import annotations
import json
import logging
from typing import Sequence

logger = logging.getLogger(__name__)


def recall_at_k(retrieved_ids: Sequence[str], gold: set[str], k: int) -> float:
    if not gold:
        return 1.0  # adversarial / off-domain queries — no gold to recall
    top_k = set(retrieved_ids[:k])
    return len(top_k & gold) / len(gold)


def mrr(retrieved_ids: Sequence[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    for rank, did in enumerate(retrieved_ids, 1):
        if did in gold:
            return 1.0 / rank
    return 0.0


def has_forbidden(retrieved_ids: Sequence[str], forbidden: set[str]) -> bool:
    return bool(set(retrieved_ids) & forbidden)


# ----------------- LLM-as-judge (faithfulness) -----------------

FAITHFULNESS_PROMPT = """你是 RAG 检索质量评估员。给定用户 query 和检索返回的 context,判断 context 与 query 的相关性。

Query: {query}
Context:
{context}

请返回严格 JSON: {{"faithfulness": <1-10 整数>, "reason": "..."}}
- 10 = context 完全相关且充分回答 query
- 5 = 部分相关或信息不足
- 1 = context 与 query 完全无关
"""


async def judge_faithfulness(client, query: str, context: str) -> dict | None:
    """Call DeepSeek to score relevance of retrieved context vs. query.

    Uses the real DeepSeekClient.chat() interface (not raw httpx). Returns parsed
    JSON dict {"faithfulness": int, "reason": str} or None on failure.
    """
    try:
        result = await client.chat(messages=[
            {"role": "user", "content": FAITHFULNESS_PROMPT.format(query=query, context=context[:1500])}
        ])
        content = result["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        logger.error("judge failed: %s", e)
        return None


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

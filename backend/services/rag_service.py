import json
import asyncio
import logging
from functools import lru_cache
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_DATA_PATH = Path(__file__).parent.parent / "data" / "products.json"
_executor = ThreadPoolExecutor(max_workers=2)

logger.info("Loading sentence embedding model...")
_embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")

logger.info("Loading product database...")
with open(_DATA_PATH, encoding="utf-8") as f:
    PRODUCTS: list[dict] = json.load(f)

_embed_texts = [
    f"{p['name']} {p['description']} "
    f"{' '.join(p.get('key_ingredients', []))} "
    f"{' '.join(p.get('suitable_skin_types', []))}"
    for p in PRODUCTS
]

logger.info("Building FAISS index for %d products...", len(PRODUCTS))
_embeddings = _embedding_model.encode(_embed_texts)
_index = faiss.IndexFlatL2(_embeddings.shape[1])
_index.add(np.array(_embeddings, dtype="float32"))
logger.info("RAG service ready.")


@lru_cache(maxsize=128)
def _sync_query(query: str, top_k: int = 3) -> str:
    vec = _embedding_model.encode([query])
    distances, indices = _index.search(np.array(vec, dtype="float32"), top_k)

    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
        if idx == -1:
            continue
        p = PRODUCTS[idx]
        ingredients = "、".join(p.get("key_ingredients", [])[:4])
        skin_types = "、".join(p.get("suitable_skin_types", []))
        selling_points = "、".join(p.get("selling_points", []))
        results.append(
            f"[匹配{rank}] {p['name']}（{p['category']}）\n"
            f"描述：{p['description']}\n"
            f"核心成分：{ingredients}\n"
            f"适合肤质：{skin_types}\n"
            f"卖点：{selling_points}"
        )

    cache_info = _sync_query.cache_info()
    logger.info("RAG query '%s' — cache hits=%d misses=%d", query, cache_info.hits, cache_info.misses)
    return "\n\n".join(results) if results else "数据库中未找到匹配产品。"


async def query_product_database(query: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _sync_query, query)

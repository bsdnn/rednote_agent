import json
import logging
from pathlib import Path

from .schema import Document

logger = logging.getLogger(__name__)


def load_products(path: Path) -> list[Document]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    docs = []
    for p in raw:
        ingredients = p.get("key_ingredients", [])
        skin_types = p.get("suitable_skin_types", [])
        selling_points = p.get("selling_points", [])
        content = (
            f"{p['name']}\n{p.get('description', '')}\n"
            f"成分: {', '.join(ingredients)}\n"
            f"适合肤质: {', '.join(skin_types)}\n"
            f"卖点: {', '.join(selling_points)}"
        )
        docs.append(Document(
            doc_id=f"prod_{p['id']}",
            doc_type="product",
            title=p["name"],
            content=content,
            metadata={
                "doc_type": "product",
                "source": "internal",
                "category": p.get("category", ""),
                "suitable_skin_types": skin_types,
                "key_ingredients": ingredients,
                "price_tier": p.get("price_tier", "mid-range"),
                "selling_points": selling_points,
                "effects": p.get("effects", []),
                "age_groups": p.get("age_groups", []),
            },
        ))
    return docs


def load_ingredients(path: Path) -> list[Document]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [
        Document(
            doc_id=f"ing_{r['slug']}",
            doc_type="ingredient",
            title=r["name"],
            content=r["content"],
            metadata={
                "doc_type": "ingredient",
                "source": "synthesized",
                "effects": r.get("effects", []),
                "concerns": r.get("concerns", []),
                "age_groups": r.get("age_groups", []),
            },
        )
        for r in raw
    ]


def load_posts(path: Path) -> list[Document]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [
        Document(
            doc_id=f"post_{r['post_id']}",
            doc_type="post",
            title=r.get("title", ""),
            content=r["content"],
            metadata={
                "doc_type": "post",
                "source": "synthesized",
                "linked_products": r.get("linked_products", []),
                "tone": r.get("tone", ""),
            },
        )
        for r in raw
    ]


def load_all(
    products_path: Path,
    ingredients_path: Path | None = None,
    posts_path: Path | None = None,
) -> list[Document]:
    docs = load_products(products_path)
    if ingredients_path and Path(ingredients_path).exists():
        docs += load_ingredients(ingredients_path)
    else:
        logger.info("ingredients corpus not found, skipping")
    if posts_path and Path(posts_path).exists():
        docs += load_posts(posts_path)
    else:
        logger.info("posts corpus not found, skipping")
    return docs

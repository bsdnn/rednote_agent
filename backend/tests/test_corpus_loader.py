import json
import pytest
from pathlib import Path

from llama_index.core import Document as LIDocument
from backend.services.rag.corpus.loader import (
    load_products_as_li, load_ingredients_as_li, load_posts_as_li,
    load_all_as_li,
)


@pytest.fixture
def tmp_products(tmp_path):
    data = [{
        "id": 1, "name": "X 精华", "category": "brightening",
        "description": "desc", "key_ingredients": ["niacinamide"],
        "suitable_skin_types": ["oily"], "price_tier": "mid-range",
        "selling_points": ["美白"]
    }]
    p = tmp_path / "products.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def test_load_products_as_li_returns_li_documents(tmp_products):
    docs = load_products_as_li(tmp_products)
    assert len(docs) == 1
    d = docs[0]
    assert isinstance(d, LIDocument)
    assert d.doc_id == "prod_1"
    assert d.metadata["doc_type"] == "product"
    assert d.metadata["title"] == "X 精华"
    assert d.metadata["source"] == "internal"
    assert "title" not in (d.excluded_embed_metadata_keys or [])
    assert "doc_type" in (d.excluded_embed_metadata_keys or [])


def test_load_ingredients_as_li(tmp_path):
    data = [{
        "slug": "niacinamide", "name": "烟酰胺",
        "content": "成分介绍", "effects": ["美白"], "concerns": [],
        "age_groups": ["25-30"],
    }]
    p = tmp_path / "ingredients.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    docs = load_ingredients_as_li(p)
    assert docs[0].doc_id == "ing_niacinamide"
    assert docs[0].metadata["title"] == "烟酰胺"
    assert "title" not in (docs[0].excluded_embed_metadata_keys or [])


def test_load_posts_as_li(tmp_path):
    data = [{
        "post_id": 7, "title": "28天抗老打卡",
        "content": "正文内容", "tone": "专业种草", "linked_products": ["修丽可CE"],
    }]
    p = tmp_path / "posts.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    docs = load_posts_as_li(p)
    assert docs[0].doc_id == "post_7"
    assert docs[0].metadata["title"] == "28天抗老打卡"
    assert "title" not in (docs[0].excluded_embed_metadata_keys or [])
    assert "linked_products" in (docs[0].excluded_embed_metadata_keys or [])


def test_load_all_as_li_skips_missing_optional(tmp_path, tmp_products):
    docs = load_all_as_li(
        products_path=tmp_products,
        ingredients_path=None,
        posts_path=None,
    )
    assert len(docs) == 1
    assert docs[0].metadata["doc_type"] == "product"

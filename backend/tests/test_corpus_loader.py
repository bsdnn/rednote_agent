import json
import pytest
from pathlib import Path
from backend.services.rag.corpus.loader import load_products, load_all


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


def test_load_products_returns_documents(tmp_products):
    docs = load_products(tmp_products)
    assert len(docs) == 1
    d = docs[0]
    assert d.doc_id == "prod_1"
    assert d.doc_type == "product"
    assert d.metadata["suitable_skin_types"] == ["oily"]
    assert d.metadata["key_ingredients"] == ["niacinamide"]
    assert d.metadata["price_tier"] == "mid-range"
    assert d.metadata["source"] == "internal"


def test_load_products_concatenates_content(tmp_products):
    docs = load_products(tmp_products)
    assert "X 精华" in docs[0].content
    assert "desc" in docs[0].content
    assert "niacinamide" in docs[0].content


def test_load_all_handles_missing_optional_corpus(tmp_path, tmp_products):
    docs = load_all(products_path=tmp_products, ingredients_path=None, posts_path=None)
    assert all(d.doc_type == "product" for d in docs)
    assert len(docs) == 1

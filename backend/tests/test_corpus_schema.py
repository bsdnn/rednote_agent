import pytest
from llama_index.core import Document as LIDocument
from backend.services.rag.corpus.schema import (
    set_metadata_filters, EMBED_KEYS, ALL_KEYS,
)


def test_set_metadata_filters_product_excludes_non_embed_keys():
    doc = LIDocument(
        text="some content",
        metadata={
            "doc_type": "product", "source": "internal", "title": "X 精华",
            "category": "brightening", "suitable_skin_types": ["oily"],
            "key_ingredients": ["niacinamide"], "price_tier": "mid-range",
            "selling_points": ["美白"], "effects": ["美白"], "age_groups": ["25-30"],
        },
    )
    set_metadata_filters(doc, "product")
    excluded = set(doc.excluded_embed_metadata_keys)
    assert "title" not in excluded       # title MUST embed
    assert "category" not in excluded    # category MUST embed
    assert "doc_type" in excluded        # doc_type debug-only
    assert "source" in excluded


def test_set_metadata_filters_ingredient():
    doc = LIDocument(
        text="百科正文",
        metadata={"doc_type": "ingredient", "source": "synthesized",
                  "title": "烟酰胺", "effects": ["美白"], "concerns": [],
                  "age_groups": ["25-30"]},
    )
    set_metadata_filters(doc, "ingredient")
    excluded = set(doc.excluded_embed_metadata_keys)
    assert "title" not in excluded
    assert "effects" not in excluded
    assert "concerns" in excluded        # concerns is debug, not for embedding


def test_set_metadata_filters_post():
    doc = LIDocument(
        text="正文",
        metadata={"doc_type": "post", "source": "synthesized",
                  "title": "28天抗老打卡", "linked_products": ["修丽可CE"],
                  "tone": "专业种草"},
    )
    set_metadata_filters(doc, "post")
    excluded = set(doc.excluded_embed_metadata_keys)
    assert "title" not in excluded       # title MUST embed
    assert "linked_products" in excluded
    assert "tone" in excluded


def test_embed_keys_subset_of_all_keys():
    for dt in ("product", "ingredient", "post"):
        assert set(EMBED_KEYS[dt]).issubset(set(ALL_KEYS[dt]))


def test_set_metadata_filters_unknown_type_raises():
    doc = LIDocument(text="x", metadata={"doc_type": "garbage"})
    with pytest.raises(KeyError):
        set_metadata_filters(doc, "garbage")

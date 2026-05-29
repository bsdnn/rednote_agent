from llama_index.core import Document as LIDocument

EMBED_KEYS: dict[str, list[str]] = {
    "product":    ["title", "category"],
    "ingredient": ["title", "effects"],
    "post":       ["title"],
}

ALL_KEYS: dict[str, list[str]] = {
    "product":    ["doc_type", "source", "title", "category", "suitable_skin_types",
                   "key_ingredients", "price_tier", "selling_points",
                   "effects", "age_groups"],
    "ingredient": ["doc_type", "source", "title", "effects", "concerns", "age_groups"],
    "post":       ["doc_type", "source", "title", "linked_products", "tone"],
}


def set_metadata_filters(doc: LIDocument, doc_type: str) -> None:
    all_keys = set(ALL_KEYS[doc_type])
    embed_keys = set(EMBED_KEYS[doc_type])
    doc.excluded_embed_metadata_keys = sorted(all_keys - embed_keys)
    doc.excluded_llm_metadata_keys = ["source"]

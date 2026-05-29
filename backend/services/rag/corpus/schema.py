from dataclasses import dataclass, field
from typing import Literal, get_args

DocType = Literal["product", "ingredient", "post"]
SourceType = Literal["internal", "synthesized", "scraped"]

_VALID_DOC_TYPES = set(get_args(DocType))


@dataclass
class Document:
    doc_id: str
    doc_type: DocType
    title: str
    content: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.doc_type not in _VALID_DOC_TYPES:
            raise ValueError(f"invalid doc_type: {self.doc_type}")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_type: DocType
    text: str
    metadata: dict = field(default_factory=dict)


# ----- LlamaIndex metadata-filter helpers (v3) -----

from llama_index.core import Document as LIDocument

# Which metadata fields get prepended to embedding text via LlamaIndex's
# excluded_embed_metadata_keys mechanism.
EMBED_KEYS: dict[str, list[str]] = {
    "product":    ["title", "category"],
    "ingredient": ["title", "effects"],
    "post":       ["title"],
}

# All metadata fields we ever set per doc_type (used to compute the "exclude" set).
ALL_KEYS: dict[str, list[str]] = {
    "product":    ["doc_type", "source", "title", "category", "suitable_skin_types",
                   "key_ingredients", "price_tier", "selling_points",
                   "effects", "age_groups"],
    "ingredient": ["doc_type", "source", "title", "effects", "concerns", "age_groups"],
    "post":       ["doc_type", "source", "title", "linked_products", "tone"],
}


def set_metadata_filters(doc: LIDocument, doc_type: str) -> None:
    """Configure which metadata keys participate in embedding text vs. LLM context.

    LlamaIndex auto-prepends NON-excluded metadata fields to the embedding text.
    This fixes the v2 bug where post/ingredient `title` was never embedded.
    """
    all_keys = set(ALL_KEYS[doc_type])
    embed_keys = set(EMBED_KEYS[doc_type])
    doc.excluded_embed_metadata_keys = sorted(all_keys - embed_keys)
    # Hide noise from LLM context (source is debug-only)
    doc.excluded_llm_metadata_keys = ["source"]

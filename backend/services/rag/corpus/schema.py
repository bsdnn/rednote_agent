from dataclasses import dataclass, field
from typing import Literal

DocType = Literal["product", "ingredient", "post"]
SourceType = Literal["internal", "synthesized", "scraped"]


@dataclass
class Document:
    doc_id: str
    doc_type: DocType
    title: str
    content: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.doc_type not in ("product", "ingredient", "post"):
            raise ValueError(f"invalid doc_type: {self.doc_type}")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_type: DocType
    text: str
    metadata: dict = field(default_factory=dict)

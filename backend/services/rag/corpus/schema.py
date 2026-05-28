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

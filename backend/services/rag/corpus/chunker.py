from .schema import Document, Chunk


def _split_with_overlap(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    out = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = min(start + chunk_size, len(text))
        out.append(text[start:end])
        if end == len(text):
            break
        start += step
    return out


def chunk_document(doc: Document, chunk_size: int = 200, overlap: int = 50) -> list[Chunk]:
    if doc.doc_type in ("product", "ingredient"):
        parts = [doc.content]
    else:  # post
        parts = _split_with_overlap(doc.content, chunk_size, overlap)
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}::{i}",
            doc_id=doc.doc_id,
            doc_type=doc.doc_type,
            text=part,
            metadata=dict(doc.metadata),
        )
        for i, part in enumerate(parts)
    ]


def chunk_all(docs: list[Document], chunk_size: int = 200, overlap: int = 50) -> list[Chunk]:
    out = []
    for d in docs:
        out.extend(chunk_document(d, chunk_size, overlap))
    return out

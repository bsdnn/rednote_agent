from backend.services.rag.corpus.schema import Document
from backend.services.rag.corpus.chunker import chunk_document


def _doc(doc_id, doc_type, content):
    return Document(doc_id=doc_id, doc_type=doc_type, title="t", content=content,
                    metadata={"doc_type": doc_type, "source": "internal"})


def test_product_chunk_is_single_passthrough():
    d = _doc("prod_1", "product", "短内容")
    chunks = chunk_document(d)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "prod_1::0"
    assert chunks[0].text == "短内容"
    assert chunks[0].metadata["doc_type"] == "product"


def test_ingredient_chunk_is_single_passthrough():
    d = _doc("ing_niacinamide", "ingredient", "成分百科介绍")
    chunks = chunk_document(d)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "ing_niacinamide::0"


def test_post_long_content_is_split_with_overlap():
    long_text = "段落一。" * 80  # ~320 chars
    d = _doc("post_42", "post", long_text)
    chunks = chunk_document(d, chunk_size=200, overlap=50)
    assert len(chunks) >= 2
    assert chunks[0].text[-20:] in chunks[1].text or chunks[1].text[:50] in chunks[0].text[-100:]
    assert all(c.metadata["doc_type"] == "post" for c in chunks)
    assert chunks[0].chunk_id == "post_42::0"
    assert chunks[1].chunk_id == "post_42::1"


def test_post_short_content_is_single_chunk():
    d = _doc("post_1", "post", "短文案")
    chunks = chunk_document(d, chunk_size=200, overlap=50)
    assert len(chunks) == 1


def test_chunk_inherits_doc_metadata():
    d = Document(doc_id="prod_5", doc_type="product", title="t", content="c",
                 metadata={"doc_type": "product", "source": "internal",
                           "suitable_skin_types": ["oily"]})
    chunks = chunk_document(d)
    assert chunks[0].metadata["suitable_skin_types"] == ["oily"]

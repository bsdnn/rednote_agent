import pytest
from backend.services.rag.corpus.schema import Document, Chunk


def test_document_required_fields():
    d = Document(doc_id="prod_1", doc_type="product", title="X", content="Y", metadata={})
    assert d.doc_id == "prod_1"
    assert d.doc_type == "product"


def test_chunk_id_must_contain_doc_id():
    c = Chunk(chunk_id="prod_1::0", doc_id="prod_1", doc_type="product", text="hi", metadata={})
    assert c.chunk_id.startswith(c.doc_id)


def test_document_rejects_invalid_doc_type():
    with pytest.raises(ValueError):
        Document(doc_id="x", doc_type="invalid", title="t", content="c", metadata={})


def test_chunk_metadata_inherits_required_keys():
    c = Chunk(chunk_id="x::0", doc_id="x", doc_type="post", text="t",
              metadata={"doc_type": "post", "source": "synthesized"})
    assert c.metadata["doc_type"] == "post"
    assert c.metadata["source"] in ("internal", "synthesized", "scraped")

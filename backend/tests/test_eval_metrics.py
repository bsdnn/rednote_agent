from scripts.rag_eval.metrics import recall_at_k, mrr, has_forbidden


def test_recall_at_k_full_hit():
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0


def test_recall_at_k_partial():
    assert recall_at_k(["a", "x"], {"a", "b"}, k=2) == 0.5


def test_recall_at_k_empty_gold_returns_one():
    # adversarial queries: no gold = vacuously satisfied
    assert recall_at_k(["a"], set(), k=3) == 1.0


def test_mrr_first_position():
    assert mrr(["a", "b"], {"a"}) == 1.0


def test_mrr_third_position():
    assert mrr(["x", "y", "a"], {"a"}) == 1 / 3


def test_mrr_miss():
    assert mrr(["x", "y"], {"a"}) == 0.0


def test_has_forbidden_detects():
    assert has_forbidden(["a", "b"], {"a"}) is True
    assert has_forbidden(["c", "d"], {"a"}) is False

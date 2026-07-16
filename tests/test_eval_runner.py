"""Smoke tests for eval_runner against the baked-in demo dataset."""

from __future__ import annotations

import pytest

from ragproof.datasets import load_demo_dataset
from ragproof.eval_runner import compare_retrievers, evaluate_retriever
from ragproof.retrievers import DenseRetriever, HybridRetriever, SparseRetriever


@pytest.fixture(scope="module")
def demo_dataset():
    return load_demo_dataset()


def test_evaluate_retriever_returns_well_formed_report(demo_dataset):
    corpus, queries = demo_dataset
    sparse = SparseRetriever(corpus)

    report = evaluate_retriever(sparse, queries, k=3, recall_k=5)

    assert set(report.keys()) == {"aggregate", "per_query"}
    agg = report["aggregate"]
    for key in ("hit_at_k", "mrr", "ndcg_at_k", "recall_at_k", "precision_at_k"):
        assert 0.0 <= agg[key] <= 1.0
    assert agg["n_queries"] == len(queries)
    assert len(report["per_query"]) == len(queries)
    assert report["per_query"][0]["query"] == queries[0]["query"]


def test_evaluate_retriever_rejects_empty_dataset():
    corpus, _ = load_demo_dataset()
    sparse = SparseRetriever(corpus)
    with pytest.raises(ValueError):
        evaluate_retriever(sparse, [], k=3)


def test_evaluate_retriever_rejects_malformed_row():
    corpus, _ = load_demo_dataset()
    sparse = SparseRetriever(corpus)
    with pytest.raises(ValueError):
        evaluate_retriever(sparse, [{"query": "no relevant ids here"}], k=3)


def test_hybrid_beats_single_methods_on_demo_dataset(demo_dataset):
    """The demo corpus is designed so fusion genuinely helps: hybrid should
    match or beat both single-method retrievers on hit@3, and strictly beat
    at least one of them. This guards against silently regressing the
    "hybrid wins" story the README relies on.
    """
    corpus, queries = demo_dataset
    sparse = SparseRetriever(corpus)
    dense = DenseRetriever(corpus, n_components=16)
    hybrid = HybridRetriever(sparse, dense, sparse_weight=0.5, dense_weight=0.5, rerank_top_n=5)

    sparse_hit = evaluate_retriever(sparse, queries, k=3)["aggregate"]["hit_at_k"]
    dense_hit = evaluate_retriever(dense, queries, k=3)["aggregate"]["hit_at_k"]
    hybrid_hit = evaluate_retriever(hybrid, queries, k=3)["aggregate"]["hit_at_k"]

    assert hybrid_hit >= sparse_hit
    assert hybrid_hit >= dense_hit
    assert hybrid_hit > min(sparse_hit, dense_hit)


def test_compare_retrievers_renders_all_names(demo_dataset):
    corpus, queries = demo_dataset
    sparse = SparseRetriever(corpus)
    dense = DenseRetriever(corpus, n_components=16)

    table = compare_retrievers({"sparse": sparse, "dense": dense}, queries, k=3, recall_k=5)

    assert "sparse" in table
    assert "dense" in table
    assert "hit@3" in table

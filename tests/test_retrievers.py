"""Unit tests for SparseRetriever, DenseRetriever, and HybridRetriever."""

from __future__ import annotations

import pytest

from ragproof.retrievers import DenseRetriever, HybridRetriever, SparseRetriever

TOY_CORPUS = {
    "a": "cats are small domesticated carnivorous mammals",
    "b": "dogs are loyal domesticated mammals often kept as pets",
    "c": "rockets use combustion to reach orbital velocity",
}


def test_sparse_retriever_ranks_exact_term_match_first():
    retriever = SparseRetriever(TOY_CORPUS)
    ranked = retriever.retrieve("rockets orbital velocity", top_k=3)
    assert ranked[0] == "c"


def test_sparse_retriever_score_covers_every_doc():
    retriever = SparseRetriever(TOY_CORPUS)
    scores = retriever.score("cats mammals")
    assert set(scores.keys()) == set(TOY_CORPUS.keys())


def test_dense_retriever_ranks_topically_similar_doc_first():
    retriever = DenseRetriever(TOY_CORPUS, n_components=2)
    ranked = retriever.retrieve("small pet mammal", top_k=3)
    assert ranked[0] in {"a", "b"}  # animal docs should beat the rocket doc


def test_dense_retriever_embed_query_is_unit_normalized():
    retriever = DenseRetriever(TOY_CORPUS, n_components=2)
    vec = retriever.embed_query("rockets")
    norm = (vec**2).sum() ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-6) or norm == pytest.approx(0.0, abs=1e-6)


def test_hybrid_retriever_requires_matching_doc_ids():
    sparse = SparseRetriever(TOY_CORPUS)
    smaller_corpus = {"a": TOY_CORPUS["a"], "b": TOY_CORPUS["b"]}
    dense = DenseRetriever(smaller_corpus, n_components=1)
    with pytest.raises(ValueError):
        HybridRetriever(sparse, dense)


def test_hybrid_retriever_returns_top_k_doc_ids():
    sparse = SparseRetriever(TOY_CORPUS)
    dense = DenseRetriever(TOY_CORPUS, n_components=2)
    hybrid = HybridRetriever(sparse, dense, rerank_top_n=2)
    ranked = hybrid.retrieve("domesticated mammals kept as pets", top_k=2)
    assert len(ranked) == 2
    assert set(ranked).issubset(TOY_CORPUS.keys())

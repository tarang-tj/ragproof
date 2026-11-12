"""Retrieval evaluation metrics.

Pure functions, no side effects. Every function takes a ranked list of
retrieved doc_ids (best first) and a collection of relevant doc_ids for a
single query, and returns a float score. Formulas follow the standard IR
definitions (Manning, Raghavan & Schutze, "Introduction to Information
Retrieval").
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def _validate(retrieved: Sequence[str], relevant: Iterable[str]) -> set:
    if retrieved is None:
        raise ValueError("retrieved list must not be None")
    relevant_set = set(relevant)
    if not relevant_set:
        raise ValueError("relevant set must not be empty")
    return relevant_set


def hit_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1.0 if at least one relevant doc appears in the top k, else 0.0.

    Also called "Success@k" in some literature.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_set = _validate(retrieved, relevant)
    top_k = retrieved[:k]
    return 1.0 if any(doc_id in relevant_set for doc_id in top_k) else 0.0


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top-k retrieved docs that are relevant."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_set = _validate(retrieved, relevant)
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / len(top_k)


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of all relevant docs that appear in the top k."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_set = _validate(retrieved, relevant)
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / len(relevant_set)


def mrr(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Reciprocal rank of the first relevant doc (1/rank), 0.0 if none found.

    Uses the full retrieved list (no k cutoff), matching the standard MRR
    definition. Callers wanting a bounded variant should slice `retrieved`
    before calling.
    """
    relevant_set = _validate(retrieved, relevant)
    for idx, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k, with binary relevance.

    DCG@k = sum_{i=1}^{k} rel_i / log2(i + 1)
    IDCG@k = DCG of the ideal ranking (all relevant docs first, up to k)
    NDCG@k = DCG@k / IDCG@k
    """
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_set = _validate(retrieved, relevant)
    top_k = retrieved[:k]

    dcg = sum(
        (1.0 if doc_id in relevant_set else 0.0) / math.log2(idx + 1)
        for idx, doc_id in enumerate(top_k, start=1)
    )

    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg

"""Run retrieval evaluation over a labeled dataset and produce reports.

A labeled dataset is a list of rows shaped like:
    {"query": str, "relevant_doc_ids": list[str]}

`evaluate_retriever` scores one retriever against the whole dataset and
returns both an aggregate report and per-query rows (useful for spotting
which specific queries a retriever fails on). `compare_retrievers` runs
several retrievers side by side and returns a table.
"""

from __future__ import annotations

from typing import Any

from ragproof.metrics import hit_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k
from ragproof.retrievers import Retriever


def evaluate_retriever(
    retriever: Retriever,
    dataset: list[dict],
    k: int = 3,
    recall_k: int | None = None,
) -> dict[str, Any]:
    """Evaluate one retriever against a labeled dataset.

    Args:
        retriever: any object implementing the Retriever protocol.
        dataset: list of {"query": str, "relevant_doc_ids": list[str]} rows.
        k: cutoff used for hit@k, precision@k, ndcg@k.
        recall_k: cutoff used for recall@k. Defaults to `k` if not given.

    Returns:
        {
          "aggregate": {"hit_at_k": float, "mrr": float, "ndcg_at_k": float,
                        "recall_at_k": float, "precision_at_k": float,
                        "n_queries": int},
          "per_query": [ {"query": str, "hit_at_k": float, ...}, ... ],
        }

    Raises:
        ValueError: if dataset is empty or a row is missing required keys.
    """
    if not dataset:
        raise ValueError("dataset must not be empty")

    recall_k = recall_k or k
    max_k = max(k, recall_k)

    per_query: list[dict[str, Any]] = []
    for row in dataset:
        if "query" not in row or "relevant_doc_ids" not in row:
            raise ValueError(f"dataset row missing required keys: {row!r}")

        query = row["query"]
        relevant = row["relevant_doc_ids"]
        ranked = retriever.retrieve(query, top_k=max(max_k, 1))

        per_query.append(
            {
                "query": query,
                "hit_at_k": hit_at_k(ranked, relevant, k),
                "mrr": mrr(ranked, relevant),
                "ndcg_at_k": ndcg_at_k(ranked, relevant, k),
                "recall_at_k": recall_at_k(ranked, relevant, recall_k),
                "precision_at_k": precision_at_k(ranked, relevant, k),
            }
        )

    n = len(per_query)
    aggregate = {
        "hit_at_k": sum(row["hit_at_k"] for row in per_query) / n,
        "mrr": sum(row["mrr"] for row in per_query) / n,
        "ndcg_at_k": sum(row["ndcg_at_k"] for row in per_query) / n,
        "recall_at_k": sum(row["recall_at_k"] for row in per_query) / n,
        "precision_at_k": sum(row["precision_at_k"] for row in per_query) / n,
        "n_queries": n,
        "k": k,
        "recall_k": recall_k,
    }

    return {"aggregate": aggregate, "per_query": per_query}


def compare_retrievers(
    retrievers: dict[str, Retriever],
    dataset: list[dict],
    k: int = 3,
    recall_k: int | None = None,
) -> str:
    """Evaluate multiple retrievers and render a side-by-side text table.

    Args:
        retrievers: mapping of display name -> Retriever instance.
        dataset: labeled dataset, see evaluate_retriever().
        k: cutoff for hit@k / ndcg@k.
        recall_k: cutoff for recall@k, defaults to k.

    Returns:
        A formatted plain-text table (str), one row per retriever.
    """
    recall_k = recall_k or k
    rows = []
    for name, retriever in retrievers.items():
        report = evaluate_retriever(retriever, dataset, k=k, recall_k=recall_k)
        rows.append((name, report["aggregate"]))

    name_width = max(len(name) for name, _ in rows) if rows else 4
    header = (
        f"{'retriever':<{name_width}}  "
        f"hit@{k:<3}mrr    ndcg@{k:<3}recall@{recall_k}"
    )
    lines = [header, "-" * len(header)]
    for name, agg in rows:
        lines.append(
            f"{name:<{name_width}}  "
            f"{agg['hit_at_k']:<7.3f}"
            f"{agg['mrr']:<7.3f}"
            f"{agg['ndcg_at_k']:<7.3f}"
            f"{agg['recall_at_k']:.3f}"
        )
    return "\n".join(lines)

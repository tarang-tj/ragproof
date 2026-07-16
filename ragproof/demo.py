"""End-to-end offline demo: `python -m ragproof.demo`.

Loads the baked-in demo dataset, evaluates SparseRetriever, DenseRetriever,
and HybridRetriever (+ lexical/dense reranker) side by side, prints a
comparison table, and prints an estimated per-query cost line. No API
keys, no network calls, no downloaded model weights -- fully deterministic.
"""

from __future__ import annotations

from ragproof.cost import DEFAULT_RATES, estimate_cost
from ragproof.datasets import load_demo_dataset
from ragproof.eval_runner import compare_retrievers
from ragproof.retrievers import DenseRetriever, HybridRetriever, SparseRetriever

K_FOR_HIT_AND_NDCG = 3
K_FOR_RECALL = 5


def build_retrievers(corpus: dict[str, str]) -> dict[str, object]:
    sparse = SparseRetriever(corpus)
    dense = DenseRetriever(corpus, n_components=16)
    hybrid = HybridRetriever(
        sparse,
        dense,
        sparse_weight=0.5,
        dense_weight=0.5,
        rerank_top_n=5,
    )
    return {
        "sparse (bm25)": sparse,
        "dense (tfidf+svd)": dense,
        "hybrid (+rerank)": hybrid,
    }


def print_cost_estimate() -> None:
    """Estimate cost for one typical RAG query: a few retrieved chunks as
    context, a short generated answer, and one query embedding call.
    """
    breakdown = estimate_cost(
        input_tokens=1200,   # ~5 retrieved chunks + system prompt + query
        output_tokens=250,   # a typical grounded answer
        embed_tokens=20,     # short user query
        rates=DEFAULT_RATES,
    )
    print(
        "\nEstimated cost per query "
        f"(input={breakdown['input_tokens']}t, output={breakdown['output_tokens']}t, "
        f"embed={breakdown['embed_tokens']}t): "
        f"${breakdown['total_usd']:.6f} "
        f"(rates: input ${DEFAULT_RATES['input_usd_per_million']}/1M, "
        f"output ${DEFAULT_RATES['output_usd_per_million']}/1M, "
        f"embed ${DEFAULT_RATES['embed_usd_per_million']}/1M -- update to current pricing)"
    )


def main() -> None:
    corpus, queries = load_demo_dataset()
    retrievers = build_retrievers(corpus)

    print(f"ragproof demo -- {len(corpus)} docs, {len(queries)} labeled queries, fully offline\n")

    table = compare_retrievers(
        retrievers,
        queries,
        k=K_FOR_HIT_AND_NDCG,
        recall_k=K_FOR_RECALL,
    )
    print(table)

    print_cost_estimate()

    print(
        "\nNote: the dense retriever is a local TF-IDF+SVD stand-in, not a "
        "real semantic embedding model. It exists so this demo runs "
        "offline. Swap in Voyage/Claude embeddings for production "
        "(see ragproof/retrievers_pgvector_claude.py)."
    )


if __name__ == "__main__":
    main()

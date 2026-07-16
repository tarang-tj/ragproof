"""Generalized BEIR benchmark engine: BM25 vs dense vs hybrid (+ optional rerank).

Works with any BEIR dataset that has a test qrels split (scifact, nfcorpus, fiqa,
...). Dense embeddings default to model2vec static vectors (CPU, keyless); pass a
sentence-transformer embedder via `embed=` for a stronger number, and an optional
cross-encoder `reranker=` to reorder the hybrid shortlist.

    python -m ragproof.benchmarks.beir scifact nfcorpus
"""
from __future__ import annotations

import csv
import json
import os
import sys
import urllib.request
import zipfile
from typing import Callable, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from ragproof.metrics import hit_at_k, mrr, ndcg_at_k, recall_at_k

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
DEFAULT_MODEL = "minishlab/potion-base-8M"

Embedder = Callable[[Sequence[str]], np.ndarray]
Reranker = Callable[[str, list], list]  # (query, [(doc_id, text)]) -> reordered doc_ids


def load_beir(name: str, data_dir: str = DATA_DIR):
    """Download + parse a BEIR dataset; return (corpus, queries, qrels) for the test split."""
    base = os.path.join(data_dir, name)
    if not os.path.exists(os.path.join(base, "corpus.jsonl")):
        os.makedirs(data_dir, exist_ok=True)
        zp = os.path.join(data_dir, f"{name}.zip")
        if not os.path.exists(zp):
            urllib.request.urlretrieve(BEIR_URL.format(name=name), zp)
        with zipfile.ZipFile(zp) as z:
            z.extractall(data_dir)
    corpus = {}
    for line in open(os.path.join(base, "corpus.jsonl"), encoding="utf-8"):
        d = json.loads(line)
        corpus[d["_id"]] = (d.get("title", "") + " " + d.get("text", "")).strip()
    all_q = {}
    for line in open(os.path.join(base, "queries.jsonl"), encoding="utf-8"):
        d = json.loads(line)
        all_q[d["_id"]] = d["text"]
    qrels: dict[str, set] = {}
    with open(os.path.join(base, "qrels", "test.tsv"), encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for qid, cid, score in reader:
            if int(score) > 0:
                qrels.setdefault(qid, set()).add(cid)
    queries = {q: all_q[q] for q in qrels if q in all_q}
    return corpus, queries, qrels


def model2vec_embedder(model: str = DEFAULT_MODEL) -> Embedder:
    """Keyless CPU embedder using model2vec static vectors (returns normalized rows)."""
    from model2vec import StaticModel

    static = StaticModel.from_pretrained(model)

    def embed(texts: Sequence[str]) -> np.ndarray:
        vecs = np.asarray(static.encode(list(texts)), dtype=np.float32)
        return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)

    return embed


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def run(dataset: str = "scifact", embed: Embedder | None = None, reranker: Reranker | None = None,
        hybrid_weight: float = 0.5, topk: int = 10, pool: int = 100, dense_label: str = "model2vec",
        quiet: bool = False) -> dict:
    """Evaluate BM25, dense, hybrid (and hybrid+rerank if `reranker`) on one dataset."""
    corpus, queries, qrels = load_beir(dataset)
    doc_ids = list(corpus)
    docs = [corpus[d] for d in doc_ids]
    bm25 = BM25Okapi([d.lower().split() for d in docs])
    embed = embed or model2vec_embedder()
    doc_emb = embed(docs)

    qids = list(queries)
    q_texts = [queries[q] for q in qids]
    q_emb = embed(q_texts)

    names = ["bm25", f"dense ({dense_label})", "hybrid"]
    if reranker:
        names.append("hybrid + rerank")
    rows = {n: [] for n in names}
    for i, qid in enumerate(qids):
        rel = qrels[qid]
        bm = bm25.get_scores(q_texts[i].lower().split())
        dn = doc_emb @ q_emb[i]
        fused = hybrid_weight * _minmax(dn) + (1 - hybrid_weight) * _minmax(bm)
        ranked = {
            "bm25": [doc_ids[j] for j in np.argsort(-bm)[:pool]],
            f"dense ({dense_label})": [doc_ids[j] for j in np.argsort(-dn)[:pool]],
            "hybrid": [doc_ids[j] for j in np.argsort(-fused)[:pool]],
        }
        if reranker:
            shortlist = [(d, corpus[d]) for d in ranked["hybrid"][:min(pool, 50)]]
            ranked["hybrid + rerank"] = reranker(q_texts[i], shortlist)
        for name in names:
            r = ranked[name]
            rows[name].append((hit_at_k(r, rel, 3), mrr(r[:topk], rel),
                               ndcg_at_k(r, rel, topk), recall_at_k(r, rel, topk)))

    means = {n: np.array(v).mean(axis=0) for n, v in rows.items()}
    if not quiet:
        print(f"\nBEIR/{dataset} -- {len(doc_ids)} docs, {len(qids)} test queries, real qrels")
        print(f"{'retriever':22} {'hit@3':>6} {'mrr@10':>7} {'ndcg@10':>8} {'recall@10':>10}")
        print("-" * 58)
        for n in names:
            m = means[n]
            print(f"{n:22} {m[0]:6.3f} {m[1]:7.3f} {m[2]:8.3f} {m[3]:10.3f}")
    return means


def run_suite(datasets: Sequence[str], embed: Embedder | None = None, dense_label: str = "model2vec"):
    """Robustness table: NDCG@10 for each retriever across several datasets."""
    results = {d: run(d, embed=embed, dense_label=dense_label, quiet=True) for d in datasets}
    retrievers = list(next(iter(results.values())))
    print(f"\nRobustness -- NDCG@10 across datasets (dense = {dense_label})")
    header = f"{'retriever':22}" + "".join(f"{d:>12}" for d in datasets)
    print(header)
    print("-" * len(header))
    for r in retrievers:
        line = f"{r:22}" + "".join(f"{results[d][r][2]:12.3f}" for d in datasets)
        print(line)
    return results


if __name__ == "__main__":
    args = sys.argv[1:] or ["scifact"]
    run_suite(args) if len(args) > 1 else run(args[0])

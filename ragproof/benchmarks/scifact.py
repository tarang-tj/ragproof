"""BEIR/scifact benchmark for ragproof: BM25 vs dense vs hybrid fusion.

Downloads the official BEIR scifact test split (5.2k docs, 300 queries with
real relevance judgments) and reports the canonical NDCG@10 plus hit@3,
MRR@10, and recall@10, all computed with ragproof.metrics.

Dense retrieval uses model2vec static embeddings (real semantic vectors, CPU
only, no GPU / no API key). Swap in Voyage or Claude embeddings via the
pgvector adapter for production. Run:

    python -m ragproof.benchmarks.scifact
"""
from __future__ import annotations

import csv
import json
import os
import urllib.request
import zipfile

import numpy as np
from rank_bm25 import BM25Okapi

from ragproof.metrics import hit_at_k, mrr, ndcg_at_k, recall_at_k

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
MODEL = "minishlab/potion-base-8M"


def _download_scifact(data_dir: str) -> str:
    base = os.path.join(data_dir, "scifact")
    if os.path.exists(os.path.join(base, "corpus.jsonl")):
        return base
    os.makedirs(data_dir, exist_ok=True)
    zp = os.path.join(data_dir, "scifact.zip")
    if not os.path.exists(zp):
        urllib.request.urlretrieve(SCIFACT_URL, zp)
    with zipfile.ZipFile(zp) as z:
        z.extractall(data_dir)
    return base


def load_scifact(data_dir: str = DATA_DIR):
    """Return (corpus, queries, qrels); queries limited to those with test qrels."""
    base = _download_scifact(data_dir)
    corpus = {}
    for line in open(os.path.join(base, "corpus.jsonl"), encoding="utf-8"):
        d = json.loads(line)
        corpus[d["_id"]] = (d.get("title", "") + " " + d.get("text", "")).strip()
    all_queries = {}
    for line in open(os.path.join(base, "queries.jsonl"), encoding="utf-8"):
        d = json.loads(line)
        all_queries[d["_id"]] = d["text"]
    qrels = {}
    with open(os.path.join(base, "qrels", "test.tsv"), encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for qid, cid, score in reader:
            if int(score) > 0:
                qrels.setdefault(qid, set()).add(cid)
    queries = {q: all_queries[q] for q in qrels if q in all_queries}
    return corpus, queries, qrels


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def run(hybrid_weight: float = 0.5, topk: int = 10, candidate_pool: int = 100):
    """Evaluate BM25, dense, and hybrid retrieval on scifact; print + return means."""
    corpus, queries, qrels = load_scifact()
    doc_ids = list(corpus.keys())
    docs = [corpus[d] for d in doc_ids]

    bm25 = BM25Okapi([d.lower().split() for d in docs])

    from model2vec import StaticModel

    model = StaticModel.from_pretrained(MODEL)
    doc_emb = np.asarray(model.encode(docs), dtype=np.float32)
    doc_emb /= np.linalg.norm(doc_emb, axis=1, keepdims=True) + 1e-9

    qids = list(queries.keys())
    q_texts = [queries[q] for q in qids]
    q_emb = np.asarray(model.encode(q_texts), dtype=np.float32)
    q_emb /= np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9

    rows = {"bm25": [], "dense (model2vec)": [], "hybrid": []}
    for i, qid in enumerate(qids):
        rel = qrels[qid]
        bm = bm25.get_scores(q_texts[i].lower().split())
        dn = doc_emb @ q_emb[i]
        fused = hybrid_weight * _minmax(dn) + (1 - hybrid_weight) * _minmax(bm)
        for name, scores in (("bm25", bm), ("dense (model2vec)", dn), ("hybrid", fused)):
            ranked = [doc_ids[j] for j in np.argsort(-scores)[:candidate_pool]]
            rows[name].append((
                hit_at_k(ranked, rel, 3),
                mrr(ranked[:topk], rel),
                ndcg_at_k(ranked, rel, topk),
                recall_at_k(ranked, rel, topk),
            ))

    print(f"BEIR/scifact -- {len(doc_ids)} docs, {len(qids)} test queries, real qrels")
    print(f"dense = model2vec {MODEL} static embeddings (CPU, no API key)\n")
    print(f"{'retriever':20} {'hit@3':>6} {'mrr@10':>7} {'ndcg@10':>8} {'recall@10':>10}")
    print("-" * 56)
    means = {}
    for name, r in rows.items():
        m = np.array(r).mean(axis=0)
        means[name] = m
        print(f"{name:20} {m[0]:6.3f} {m[1]:7.3f} {m[2]:8.3f} {m[3]:10.3f}")
    return means


if __name__ == "__main__":
    run()

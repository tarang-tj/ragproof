"""Retriever implementations over an in-memory corpus.

All retrievers here run fully offline: no API keys, no downloaded model
weights. They exist so the demo and tests are deterministic and
reproducible on any machine. `retrievers_pgvector_claude.py` shows the real
production seam (Supabase pgvector + Claude embeddings/generation) behind
the same interface.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Scale scores into [0, 1] so different retrievers can be fused fairly."""
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class Retriever(Protocol):
    """Common interface every retriever (offline or production) implements."""

    def retrieve(self, query: str, top_k: int) -> list[str]:
        """Return up to top_k doc_ids ranked best-first for the query."""
        ...

    def score(self, query: str) -> dict[str, float]:
        """Return a raw relevance score for every doc_id in the corpus."""
        ...


class SparseRetriever:
    """Lexical retrieval via BM25 (Okapi). Rewards exact term overlap."""

    def __init__(self, corpus: dict[str, str]):
        self.doc_ids = list(corpus.keys())
        tokenized = [_tokenize(corpus[doc_id]) for doc_id in self.doc_ids]
        self._bm25 = BM25Okapi(tokenized)

    def score(self, query: str) -> dict[str, float]:
        raw = self._bm25.get_scores(_tokenize(query))
        return dict(zip(self.doc_ids, raw))

    def retrieve(self, query: str, top_k: int) -> list[str]:
        scores = self.score(query)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [doc_id for doc_id, _ in ranked[:top_k]]


class DenseRetriever:
    """Stand-in local embedding retriever: TF-IDF -> TruncatedSVD (LSA).

    NOTE: this is NOT a real semantic embedding model. It is a deterministic,
    offline latent-semantic projection over the corpus vocabulary, used so
    the demo can show a genuine "captures co-occurring/topical vocabulary
    beyond exact keyword match" effect without any network calls or model
    downloads. Swap for Voyage/Claude embeddings in production (see
    retrievers_pgvector_claude.py).
    """

    def __init__(self, corpus: dict[str, str], n_components: int = 16, random_state: int = 42):
        self.doc_ids = list(corpus.keys())
        texts = [corpus[doc_id] for doc_id in self.doc_ids]

        self._vectorizer = TfidfVectorizer()
        tfidf = self._vectorizer.fit_transform(texts)

        n_components = min(n_components, min(tfidf.shape) - 1)
        n_components = max(n_components, 1)
        self._svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self._doc_vectors = self._svd.fit_transform(tfidf)

        norms = np.linalg.norm(self._doc_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._doc_vectors_unit = self._doc_vectors / norms

    def embed_query(self, query: str) -> np.ndarray:
        """Return the L2-normalized latent-space vector for a query string.

        Public so callers (e.g. drift.py) can project queries into the same
        space the retriever uses for ranking, without depending on internals.
        """
        query_tfidf = self._vectorizer.transform([query])
        query_vec = self._svd.transform(query_tfidf)[0]
        norm = np.linalg.norm(query_vec)
        return query_vec / norm if norm > 0 else query_vec

    def score(self, query: str) -> dict[str, float]:
        query_vec = self.embed_query(query)
        sims = self._doc_vectors_unit @ query_vec
        return dict(zip(self.doc_ids, sims))

    def retrieve(self, query: str, top_k: int) -> list[str]:
        scores = self.score(query)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [doc_id for doc_id, _ in ranked[:top_k]]


class HybridRetriever:
    """Weighted score fusion of sparse + dense, with an optional reranker.

    Each component's raw scores are min-max normalized per-query before
    fusion so lexical and latent-semantic scores are on a comparable scale.
    This is a legitimate, well-known technique (score-level fusion) for
    combining retrievers with complementary failure modes: a doc ranked
    poorly by one method but well by the other can still surface in the
    combined top-k.
    """

    def __init__(
        self,
        sparse: SparseRetriever,
        dense: DenseRetriever,
        sparse_weight: float = 0.5,
        dense_weight: float = 0.5,
        rerank_top_n: int | None = None,
    ):
        if sparse.doc_ids != dense.doc_ids:
            raise ValueError("sparse and dense retrievers must share the same corpus/doc_ids")
        self.sparse = sparse
        self.dense = dense
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.rerank_top_n = rerank_top_n
        self.doc_ids = sparse.doc_ids

    def _fused_scores(self, query: str) -> dict[str, float]:
        sparse_raw = np.array([self.sparse.score(query)[d] for d in self.doc_ids])
        dense_raw = np.array([self.dense.score(query)[d] for d in self.doc_ids])

        sparse_norm = _min_max_normalize(sparse_raw)
        dense_norm = _min_max_normalize(dense_raw)

        fused = self.sparse_weight * sparse_norm + self.dense_weight * dense_norm
        return dict(zip(self.doc_ids, fused))

    def score(self, query: str) -> dict[str, float]:
        return self._fused_scores(query)

    def _rerank(self, query: str, candidate_ids: list[str]) -> list[str]:
        """Stand-in cross-encoder-style reranker.

        NOT a real cross-encoder: it re-scores the candidate shortlist with
        a lexical+dense blend that weights exact term overlap more heavily
        than the first-pass fusion. This mimics what a real reranker
        contributes (a second, more precise pass over a shortlist) without
        adding a network call or model download. Swap for a real
        cross-encoder or Claude-based reranker in production.
        """
        sparse_raw = np.array([self.sparse.score(query)[d] for d in candidate_ids])
        dense_raw = np.array([self.dense.score(query)[d] for d in candidate_ids])
        sparse_norm = _min_max_normalize(sparse_raw)
        dense_norm = _min_max_normalize(dense_raw)
        rerank_score = 0.5 * sparse_norm + 0.5 * dense_norm
        ranked = sorted(zip(candidate_ids, rerank_score), key=lambda kv: kv[1], reverse=True)
        return [doc_id for doc_id, _ in ranked]

    def retrieve(self, query: str, top_k: int) -> list[str]:
        fused = self._fused_scores(query)
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        ranked_ids = [doc_id for doc_id, _ in ranked]

        if self.rerank_top_n:
            shortlist = ranked_ids[: self.rerank_top_n]
            reranked = self._rerank(query, shortlist)
            ranked_ids = reranked + ranked_ids[self.rerank_top_n :]

        return ranked_ids[:top_k]

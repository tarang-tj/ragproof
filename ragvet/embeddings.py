"""Optional real-transformer embeddings and a cross-encoder reranker.

These are the production-grade retrievers: a sentence-transformer dense embedder
(much stronger than the model2vec static default) and a real cross-encoder that
reorders a shortlist by query-document relevance. They need the heavier `[embeddings]`
extra (torch + sentence-transformers), so everything imports lazily -- the rest of
ragvet runs without them.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def sentence_transformer_embedder(model: str = "BAAI/bge-small-en-v1.5",
                                  revision: str | None = None) -> Callable[[Sequence[str]], np.ndarray]:
    """Return a normalized-embedding callable backed by a sentence-transformer model.

    `revision` pins a Hub commit, so a benchmark score belongs to one exact set of weights.
    """
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model, revision=revision)

    def embed(texts: Sequence[str]) -> np.ndarray:
        vecs = encoder.encode(list(texts), normalize_embeddings=True,
                              show_progress_bar=False, batch_size=64)
        return np.asarray(vecs, dtype=np.float32)

    return embed


def cross_encoder_reranker(model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Return a reranker: (query, [(doc_id, text), ...]) -> doc_ids reordered by relevance."""
    from sentence_transformers import CrossEncoder

    ranker = CrossEncoder(model)

    def rerank(query: str, shortlist: list) -> list:
        if not shortlist:
            return []
        scores = ranker.predict([(query, text) for _, text in shortlist])
        order = sorted(range(len(shortlist)), key=lambda i: -float(scores[i]))
        return [shortlist[i][0] for i in order]

    return rerank

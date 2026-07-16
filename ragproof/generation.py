"""Answer-quality evaluation for RAG: faithfulness and answer relevance.

Retrieval metrics only tell you whether the right context was fetched. They say
nothing about the answer the LLM then wrote. This module scores the answer:

- **Faithfulness** -- what fraction of the answer's sentences are supported by the
  retrieved context. Low faithfulness = hallucination (the model made things up).
- **Answer relevance** -- how semantically on-topic the answer is to the question.

Two backends, mirroring the retriever pattern:
- A local, keyless heuristic (default): embed answer sentences and context, score
  support by max cosine similarity. Deterministic, no GPU, no API key.
- A Claude LLM-as-judge adapter: higher-fidelity 0-1 scores with rationale. Needs
  ANTHROPIC_API_KEY and the `anthropic` package (see the [judge] extra).

All heuristic functions take an `embed` callable so they can be unit tested with a
trivial embedder and swapped to model2vec / a sentence-transformer in real use.
"""
from __future__ import annotations

import re
from typing import Callable, Sequence

import numpy as np

Embedder = Callable[[Sequence[str]], np.ndarray]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences; drop empties. Simple and dependency-free."""
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)


def faithfulness(answer: str, contexts: Sequence[str], embed: Embedder,
                 threshold: float = 0.55) -> float:
    """Fraction of answer sentences supported by >=1 retrieved context chunk.

    A sentence is "supported" if its max cosine similarity to any context chunk is
    at least `threshold`. Returns 1.0 for a fully grounded answer, ~0.0 for one made
    up out of whole cloth. Empty answer or no contexts -> 0.0.
    """
    sentences = split_sentences(answer)
    if not sentences or not contexts:
        return 0.0
    ctx = _normalize(embed(list(contexts)))
    sent = _normalize(embed(sentences))
    sims = sent @ ctx.T  # (S, C) cosine similarities
    supported = int((sims.max(axis=1) >= threshold).sum())
    return supported / len(sentences)


def answer_relevance(question: str, answer: str, embed: Embedder) -> float:
    """Cosine similarity between the question and the answer (0-1 after clamping)."""
    if not question.strip() or not answer.strip():
        return 0.0
    q = _normalize(embed([question]))[0]
    a = _normalize(embed([answer]))[0]
    return max(0.0, float(q @ a))


def evaluate_answer(question: str, answer: str, contexts: Sequence[str],
                    embed: Embedder, threshold: float = 0.55) -> dict:
    """Both answer metrics in one call. Returns {faithfulness, answer_relevance}."""
    return {
        "faithfulness": faithfulness(answer, contexts, embed, threshold),
        "answer_relevance": answer_relevance(question, answer, embed),
    }


class ClaudeJudge:
    """LLM-as-judge faithfulness / relevance via Claude.

    Higher fidelity than the embedding heuristic because it reasons about entailment,
    not just surface similarity. Requires `pip install ragproof[judge]` and
    ANTHROPIC_API_KEY. Scores are floats in [0, 1].
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: str | None = None):
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised only with the extra
            raise ImportError("ClaudeJudge needs `pip install ragproof[judge]`") from exc
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def _score(self, prompt: str) -> float:  # pragma: no cover - needs a live API key
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=8,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in msg.content if block.type == "text")
        match = re.search(r"[01](?:\.\d+)?", text)
        return float(match.group()) if match else 0.0

    def faithfulness(self, answer: str, contexts: Sequence[str]) -> float:  # pragma: no cover
        joined = "\n\n".join(contexts)
        return self._score(
            "Score 0.0-1.0 how fully the ANSWER is supported by the CONTEXT "
            "(1.0 = every claim is grounded, 0.0 = unsupported). Reply with only the number.\n\n"
            f"CONTEXT:\n{joined}\n\nANSWER:\n{answer}\n\nScore:"
        )

    def answer_relevance(self, question: str, answer: str) -> float:  # pragma: no cover
        return self._score(
            "Score 0.0-1.0 how well the ANSWER addresses the QUESTION. "
            "Reply with only the number.\n\n"
            f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nScore:"
        )

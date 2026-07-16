"""Tests for answer-quality evaluation (faithfulness, answer relevance).

Uses a trivial deterministic bag-of-words embedder so the logic is tested without
any model download or API key -- keeps CI fast and hermetic.
"""
import hashlib

import numpy as np

from ragproof.generation import (
    answer_relevance,
    evaluate_answer,
    faithfulness,
    split_sentences,
)

_DIM = 128


def toy_embed(texts):
    """Deterministic bag-of-words vectors; vocabulary overlap -> cosine similarity."""
    out = []
    for text in texts:
        vec = np.zeros(_DIM, dtype=np.float32)
        for word in text.lower().split():
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % _DIM
            vec[idx] += 1.0
        out.append(vec)
    return np.array(out, dtype=np.float32)


CONTEXT = ["Photosynthesis converts sunlight into chemical energy stored in glucose within plant cells."]


def test_split_sentences():
    assert split_sentences("One fact. Two facts! Three?") == ["One fact.", "Two facts!", "Three?"]
    assert split_sentences("   ") == []


def test_faithful_answer_scores_high():
    grounded = "Photosynthesis converts sunlight into chemical energy stored in glucose."
    assert faithfulness(grounded, CONTEXT, toy_embed) == 1.0


def test_hallucinated_answer_scores_low():
    made_up = "The stock market crashed sharply because central banks raised interest rates."
    assert faithfulness(made_up, CONTEXT, toy_embed) == 0.0


def test_partially_grounded_answer_is_between():
    mixed = ("Photosynthesis converts sunlight into chemical energy stored in glucose. "
             "It was also the best-selling album of the decade.")
    score = faithfulness(mixed, CONTEXT, toy_embed)
    assert 0.0 < score < 1.0


def test_answer_relevance_prefers_on_topic():
    question = "How does photosynthesis convert sunlight into energy?"
    on_topic = "Photosynthesis converts sunlight into chemical energy."
    off_topic = "Interest rates rose and the market crashed."
    assert answer_relevance(question, on_topic, toy_embed) > answer_relevance(question, off_topic, toy_embed)


def test_empty_inputs_are_zero():
    assert faithfulness("", CONTEXT, toy_embed) == 0.0
    assert faithfulness("something", [], toy_embed) == 0.0
    assert answer_relevance("q", "", toy_embed) == 0.0


def test_evaluate_answer_returns_both_metrics():
    result = evaluate_answer("What is photosynthesis?",
                             "Photosynthesis converts sunlight into chemical energy stored in glucose.",
                             CONTEXT, toy_embed)
    assert set(result) == {"faithfulness", "answer_relevance"}
    assert result["faithfulness"] == 1.0

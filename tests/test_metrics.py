"""Unit tests for ragproof.metrics with hand-computed expected values.

Every expected value below is derived by hand from the standard IR
formulas (see metrics.py docstrings), not copied from the implementation,
so a bug in the implementation would actually be caught.
"""

from __future__ import annotations

import math

import pytest

from ragproof.metrics import hit_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k

RETRIEVED = ["d1", "d2", "d3", "d4", "d5"]
RELEVANT = {"d2", "d4"}


class TestHitAtK:
    def test_no_hit_in_top_1(self):
        assert hit_at_k(RETRIEVED, RELEVANT, k=1) == 0.0

    def test_hit_in_top_2(self):
        assert hit_at_k(RETRIEVED, RELEVANT, k=2) == 1.0

    def test_hit_in_top_5(self):
        assert hit_at_k(RETRIEVED, RELEVANT, k=5) == 1.0

    def test_empty_relevant_raises(self):
        with pytest.raises(ValueError):
            hit_at_k(RETRIEVED, [], k=1)

    def test_nonpositive_k_raises(self):
        with pytest.raises(ValueError):
            hit_at_k(RETRIEVED, RELEVANT, k=0)


class TestPrecisionAtK:
    def test_precision_at_2(self):
        # top2 = [d1, d2] -> 1 hit / 2 = 0.5
        assert precision_at_k(RETRIEVED, RELEVANT, k=2) == pytest.approx(0.5)

    def test_precision_at_4(self):
        # top4 = [d1, d2, d3, d4] -> 2 hits / 4 = 0.5
        assert precision_at_k(RETRIEVED, RELEVANT, k=4) == pytest.approx(0.5)

    def test_precision_at_1_zero_hits(self):
        assert precision_at_k(RETRIEVED, RELEVANT, k=1) == pytest.approx(0.0)


class TestRecallAtK:
    def test_recall_at_2(self):
        # top2 = [d1, d2] -> 1 of 2 relevant found
        assert recall_at_k(RETRIEVED, RELEVANT, k=2) == pytest.approx(0.5)

    def test_recall_at_4_full(self):
        # top4 = [d1, d2, d3, d4] -> both relevant found
        assert recall_at_k(RETRIEVED, RELEVANT, k=4) == pytest.approx(1.0)

    def test_recall_at_1_zero(self):
        assert recall_at_k(RETRIEVED, RELEVANT, k=1) == pytest.approx(0.0)


class TestMRR:
    def test_first_relevant_at_rank_2(self):
        # first relevant doc (d2) is at rank 2 -> 1/2
        assert mrr(RETRIEVED, RELEVANT) == pytest.approx(0.5)

    def test_first_relevant_at_rank_1(self):
        assert mrr(["d4", "d1", "d2"], RELEVANT) == pytest.approx(1.0)

    def test_no_relevant_found_returns_zero(self):
        assert mrr(["d1", "d3", "d5"], RELEVANT) == pytest.approx(0.0)

    def test_single_relevant_doc(self):
        # relevant "b" at rank 2 -> 1/2
        assert mrr(["a", "b", "c"], ["b"]) == pytest.approx(0.5)


class TestNDCGAtK:
    def test_ndcg_at_4_two_relevant(self):
        # DCG@4 = 1/log2(3) [d2 @ rank2] + 1/log2(5) [d4 @ rank4]
        # IDCG@4 = 1/log2(2) + 1/log2(3) (ideal: both relevant docs first)
        dcg = 1 / math.log2(3) + 1 / math.log2(5)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        expected = dcg / idcg
        assert ndcg_at_k(RETRIEVED, RELEVANT, k=4) == pytest.approx(expected)
        assert ndcg_at_k(RETRIEVED, RELEVANT, k=4) == pytest.approx(0.6509209298071326)

    def test_ndcg_at_3_only_one_relevant_visible(self):
        # only d2 (rank 2) is within top3; d4 falls outside the cutoff
        dcg = 1 / math.log2(3)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)  # ideal still has 2 relevant docs
        expected = dcg / idcg
        assert ndcg_at_k(RETRIEVED, RELEVANT, k=3) == pytest.approx(expected)
        assert ndcg_at_k(RETRIEVED, RELEVANT, k=3) == pytest.approx(0.38685280723454163)

    def test_ndcg_perfect_ranking_is_one(self):
        # relevant docs occupy the top ranks exactly -> NDCG == 1.0
        assert ndcg_at_k(["d2", "d4", "d1"], RELEVANT, k=3) == pytest.approx(1.0)

    def test_ndcg_no_hits_is_zero(self):
        assert ndcg_at_k(["d1", "d3", "d5"], RELEVANT, k=3) == pytest.approx(0.0)

    def test_nonpositive_k_raises(self):
        with pytest.raises(ValueError):
            ndcg_at_k(RETRIEVED, RELEVANT, k=0)

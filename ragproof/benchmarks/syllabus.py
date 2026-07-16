"""Domain eval on the SyllabusAI problem: retrieval + faithfulness over syllabus Q&A.

SyllabusAI answers student questions from course syllabi. This is a small, hand-
labeled, representative syllabus-QA set (not the live 500-user corpus -- that has no
ground-truth labels) so ragproof can be exercised end to end on the actual domain:
retrieval metrics on real question->section labels, plus a faithfulness contrast
showing the answer-eval layer flag a hallucinated answer.

To run against the live SyllabusAI corpus, point a retriever at its Supabase/pgvector
store (see ragproof/retrievers_pgvector_claude.py) and reuse the same metrics.

    python -m ragproof.benchmarks.syllabus
"""
from __future__ import annotations

import numpy as np

from ragproof.generation import faithfulness
from ragproof.metrics import hit_at_k, mrr, ndcg_at_k, recall_at_k

# --- representative syllabus corpus (id -> section text) ---
CHUNKS = {
    "cs-grade": "CS 250 grading: 40% projects, 30% midterm, 30% final. Late work loses 10% per day up to three days.",
    "cs-exams": "CS 250 midterm is October 14 in class; the final exam is December 12 at 2pm.",
    "cs-office": "CS 250 office hours: Professor Lee, Tuesdays and Thursdays 1 to 3pm in Room 402.",
    "cs-book": "CS 250 textbook: Introduction to Algorithms (CLRS), 4th edition, required.",
    "bus-grade": "BUS 310 grade breakdown: 25% participation, 25% group project, 20% quizzes, 30% final.",
    "bus-exams": "BUS 310 has no midterm; there are five quizzes and a cumulative final on December 10.",
    "bus-attend": "BUS 310 attendance: more than three unexcused absences lowers the final grade by one letter.",
    "bus-project": "BUS 310 group project proposal is due week 4; the final presentation is in week 10.",
    "math-grade": "MATH 120 grading: weekly homework 20%, two midterms 40%, final 40%.",
    "math-exams": "MATH 120 midterm 1 is September 30, midterm 2 is November 4, final is December 8.",
    "math-calc": "MATH 120 calculators are not permitted on any exam.",
    "math-office": "MATH 120 office hours: Monday, Wednesday, Friday 10 to 11am, plus a Wednesday evening TA session.",
}

# --- labeled student questions (query -> relevant chunk id) ---
QUERIES = {
    "When is the CS 250 final exam?": "cs-exams",
    "How much is the CS 250 midterm worth?": "cs-grade",
    "What is the late work policy in CS 250?": "cs-grade",
    "Which textbook do I need for data structures?": "cs-book",
    "When are Professor Lee's office hours?": "cs-office",
    "Does the marketing class have a midterm?": "bus-exams",
    "How is participation weighted in BUS 310?": "bus-grade",
    "What happens if I miss too many marketing classes?": "bus-attend",
    "When is the marketing group presentation?": "bus-project",
    "Can I use a calculator on the calculus exam?": "math-calc",
    "When is calculus midterm 2?": "math-exams",
    "How much is homework worth in MATH 120?": "math-grade",
    "When can I get calculus help outside class?": "math-office",
    "How many quizzes does BUS 310 have?": "bus-exams",
    "What is the grade breakdown for calculus?": "math-grade",
    "Where are CS 250 office hours held?": "cs-office",
}


def run():
    ids = list(CHUNKS)
    docs = [CHUNKS[i] for i in ids]
    from ragproof.benchmarks.beir import model2vec_embedder, _minmax
    from rank_bm25 import BM25Okapi

    embed = model2vec_embedder()
    doc_emb = embed(docs)
    bm25 = BM25Okapi([d.lower().split() for d in docs])

    rows = {"bm25": [], "dense (model2vec)": [], "hybrid": []}
    for q, gold in QUERIES.items():
        rel = {gold}
        bm = bm25.get_scores(q.lower().split())
        dn = doc_emb @ embed([q])[0]
        fused = 0.5 * _minmax(dn) + 0.5 * _minmax(bm)
        for name, scores in (("bm25", bm), ("dense (model2vec)", dn), ("hybrid", fused)):
            r = [ids[j] for j in np.argsort(-scores)]
            rows[name].append((hit_at_k(r, rel, 3), mrr(r[:5], rel), ndcg_at_k(r, rel, 5), recall_at_k(r, rel, 5)))

    print(f"SyllabusAI domain eval -- {len(ids)} syllabus sections, {len(QUERIES)} labeled questions")
    print(f"{'retriever':20} {'hit@3':>6} {'mrr@5':>6} {'ndcg@5':>7} {'recall@5':>9}")
    print("-" * 52)
    for name, v in rows.items():
        m = np.array(v).mean(axis=0)
        print(f"{name:20} {m[0]:6.3f} {m[1]:6.3f} {m[2]:7.3f} {m[3]:9.3f}")

    # faithfulness contrast on a real retrieved context
    context = [CHUNKS["cs-grade"]]
    grounded = "CS 250 is graded with 40% projects, 30% midterm, and 30% final."
    hallucinated = "CS 250 is graded entirely on class attendance and one group presentation."
    print("\nAnswer faithfulness (retrieved context = CS 250 grading section):")
    print(f"  grounded answer     -> {faithfulness(grounded, context, embed):.2f}")
    print(f"  hallucinated answer -> {faithfulness(hallucinated, context, embed):.2f}")
    print("(embedding heuristic flags topically ungrounded answers; use ClaudeJudge for fine-grained facts.)")


if __name__ == "__main__":
    run()

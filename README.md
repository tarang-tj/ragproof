# ragvet

[![ci](https://github.com/tarang-tj/ragproof/actions/workflows/ci.yml/badge.svg)](https://github.com/tarang-tj/ragproof/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Evaluation and retrieval-benchmark harness for RAG systems. Prove retrieval **and** answer quality with numbers instead of vibes.

**Live:** a [monthly leaderboard](#monthly-embedding-leaderboard) of the most-downloaded small embedding models on Hugging Face, each new revision scored on BEIR/scifact by a scheduled GitHub Action.

## Why evals

Most RAG failures are retrieval failures, not generation failures: the model answers badly because the wrong chunks were fetched, not because it reasoned poorly. Most teams shipping LLM apps have no evals at all, so a retrieval regression ships silently and shows up as "the bot got worse" three weeks later with no way to diagnose it. `ragvet` scores retrieval (hit@k, MRR, nDCG, recall), scores the generated answer (faithfulness, relevance), tracks cost, and detects query drift, so you catch regressions before your users do.

## Install

Install from a clone of this repository. This project is not published on PyPI. It was renamed from `ragproof` because a `ragproof` package there is an unrelated project by another author, so `pip install ragproof` will not install this code; the package, CLI and imports are `ragvet`, and the GitHub repo keeps its old name.

```bash
git clone https://github.com/tarang-tj/ragproof.git
cd ragproof
pip install -e .                    # core (offline demo + CLI)
pip install -e ".[benchmark]"       # + model2vec for keyless BEIR benchmarks
pip install -e ".[embeddings]"      # + sentence-transformers for real dense/rerank
pip install -e ".[judge]"           # + anthropic for LLM-as-judge faithfulness
pip install -e ".[dashboard]"       # + streamlit for the drift dashboard
```

## CLI

```bash
ragvet demo                                  # offline retrieval demo, no downloads
ragvet benchmark scifact                     # BEIR benchmark, model2vec dense (keyless)
ragvet benchmark scifact --embeddings st --rerank   # bge-small + cross-encoder
ragvet suite scifact nfcorpus                # robustness table across datasets
ragvet leaderboard                           # monthly embedding leaderboard
ragvet version
```

## Retrieval quality (offline demo)

`ragvet demo` loads a baked-in 40-document corpus and 24 labeled queries and evaluates three retrievers, no keys or downloads:

```
retriever          hit@3  mrr    ndcg@3  recall@5
-------------------------------------------------
sparse (bm25)      0.917  0.849  0.860  0.958
dense (tfidf+svd)  0.958  0.861  0.886  0.958
hybrid (+rerank)   1.000  0.882  0.912  1.000
```

Hybrid wins because sparse and dense retrieval have complementary failure modes: BM25 misses paraphrased queries, dense misses exact-keyword ones, and score fusion recovers both. `tests/test_eval_runner.py::test_hybrid_beats_single_methods_on_demo_dataset` locks the claim in.

## Answer quality: faithfulness and relevance

Retrieval metrics say the right context was fetched. They say nothing about the answer the LLM then wrote. `ragvet.generation` scores that:

- **Faithfulness** — the fraction of the answer's sentences supported by the retrieved context. Low = hallucination.
- **Answer relevance** — how on-topic the answer is to the question.

Two backends: a keyless embedding heuristic (default) and a Claude LLM-as-judge (`[judge]` extra) for fine-grained factual entailment.

```python
from ragvet.generation import evaluate_answer
from ragvet.benchmarks.beir import model2vec_embedder
embed = model2vec_embedder()
evaluate_answer(question, answer, retrieved_contexts, embed)
# -> {"faithfulness": 1.0, "answer_relevance": 0.71}
```

## Benchmarks on real data

### BEIR/scifact (model2vec dense, keyless, reproduces on a laptop)

```
BEIR/scifact -- 5183 docs, 300 test queries, real qrels
retriever             hit@3  mrr@10  ndcg@10  recall@10
--------------------------------------------------------
bm25                  0.590   0.524    0.560      0.686
dense (model2vec)     0.520   0.467    0.506      0.662
hybrid                0.633   0.576    0.609      0.731
```

Hybrid fusion lifts NDCG@10 from 0.560 (BM25) to 0.609. Honest caveat: BM25 here is untuned `rank_bm25` (a tuned Anserini BM25 scores ~0.66), and model2vec static embeddings trade quality for CPU speed — the point is the relative hybrid gain and full reproducibility without a GPU.

### Stronger embeddings (bge-small + cross-encoder rerank)

`ragvet benchmark scifact --embeddings st --rerank` swaps in a real sentence-transformer (`BAAI/bge-small-en-v1.5`) and a cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`):

```
retriever               hit@3  mrr@10  ndcg@10  recall@10
----------------------------------------------------------
bm25                    0.590   0.524    0.560      0.686
dense (bge-small)       0.727   0.684    0.720      0.845
hybrid                  0.687   0.627    0.660      0.788
hybrid + rerank         0.717   0.654    0.681      0.795
```

A real embedder lifts dense NDCG@10 from 0.506 (model2vec) to **0.720** — the single biggest quality jump in this repo. The honest surprise: on scifact, dense alone beats hybrid and hybrid+rerank, because naive score fusion dilutes a strong dense retriever with a weaker BM25 signal. Fusion helps when the two retrievers are comparably strong (see scifact with model2vec, and the SyllabusAI set below); it hurts when one dominates. That is exactly the kind of thing an eval harness is for.

### Monthly embedding leaderboard

`ragvet leaderboard` asks the Hugging Face Hub for today's most-downloaded small embedding models and scores every new revision on scifact. A GitHub Action reruns it on the 15th of each month and commits the result (full table: [leaderboard/LEADERBOARD.md](leaderboard/LEADERBOARD.md)).

<!-- leaderboard:start -->
Dense retrieval on BEIR/scifact (5,183 docs, 300 test queries), no instruction prefixes, so models trained with query prefixes (e5, bge) score below their model cards. Candidates: the 8 most-downloaded sentence-transformers models on Hugging Face under 150M parameters without custom code, re-checked monthly (last check 2026-11). Each score is pinned to the linked revision. BM25 on the same set: 0.560.

| # | model | nDCG@10 | recall@10 | params | downloads rank |
|---|---|---|---|---|---|
| 1 | [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small/tree/614241f622f53c4eeff9890bdc4f31cfecc418b3) | 0.669 | 0.805 | 118M | 4 |
| 2 | [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2/tree/e8c3b32edf5434bc2275fc9bab85f82640a19130) | 0.656 | 0.790 | 109M | 3 |
| 3 | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41) | 0.645 | 0.783 | 23M | 1 |
| 4 | [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/tree/e8f8c211226b894fcb81acc59f3b34ba3efd5f42) | 0.484 | 0.616 | 118M | 2 |
| - | [datasocietyco/bge-base-en-v1.5-course-recommender-v5](https://huggingface.co/datasocietyco/bge-base-en-v1.5-course-recommender-v5/tree/2b069eed51ce7bc2beeaa7e3eea1a01c48ef3d29) | pending | pending | 109M | 5 |
| - | [thenlper/gte-small](https://huggingface.co/thenlper/gte-small/tree/17e1f347d17fe144873b1201da91788898c639cd) | pending | pending | 33M | 6 |
| - | [sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/tree/a50ef00143b4d5391434df20ae11632588ac25be) | pending | pending | 33M | 7 |
| - | [sentence-transformers/all-distilroberta-v1](https://huggingface.co/sentence-transformers/all-distilroberta-v1/tree/842eaed40bee4d61673a81c92d5689a8fed7a09f) | pending | pending | 82M | 8 |
<!-- leaderboard:end -->

### Robustness across datasets

`ragvet suite scifact nfcorpus` — hybrid wins on both:

```
Robustness -- NDCG@10 across datasets (dense = model2vec)
retriever                  scifact    nfcorpus
----------------------------------------------
bm25                         0.560       0.267
dense (model2vec)            0.506       0.244
hybrid                       0.609       0.293
```

### SyllabusAI domain (representative Q&A)

`ragvet.benchmarks.syllabus` evaluates on the actual problem behind [SyllabusAI](https://syllabusai.net) — answering student questions from course syllabi — using a hand-labeled representative set (the live corpus has no ground-truth labels; point a pgvector retriever at it to run on real data):

```
SyllabusAI domain eval -- 12 syllabus sections, 16 labeled questions
retriever             hit@3  mrr@5  ndcg@5  recall@5
----------------------------------------------------
bm25                  0.625  0.622   0.653     0.750
dense (model2vec)     0.875  0.830   0.872     1.000
hybrid                0.875  0.762   0.805     0.938

Answer faithfulness (context = CS 250 grading section):
  grounded answer     -> 1.00
  hallucinated answer -> 0.00
```

Note the flip from scifact: on paraphrased student questions, **dense retrieval beats BM25** because the questions rarely share vocabulary with the syllabus text. The faithfulness check cleanly separates a grounded answer from a hallucinated one.

## Drift dashboard

```bash
streamlit run ragvet/dashboard.py
```

Paste a baseline window of queries and a current window; the dashboard embeds both and runs drift detection (PSI + centroid distance) so you can see when production traffic has wandered away from your eval set and your metrics have gone stale.

## Wire your own stack (pgvector + Claude)

`ragvet/retrievers_pgvector_claude.py` is a clearly-marked integration stub showing the production seam: Supabase pgvector for retrieval, Claude for generation, behind the same `Retriever` interface every retriever here uses. Point `eval_runner.evaluate_retriever()` at it once you fill in your keys.

## What's real vs stand-in

- `SparseRetriever` — real BM25 (`rank_bm25`).
- `DenseRetriever` — TF-IDF+SVD stand-in for the offline demo; use `ragvet.embeddings.sentence_transformer_embedder` (`[embeddings]`) for real semantic vectors, or model2vec (`[benchmark]`) for a keyless middle ground.
- Reranker — a lexical/dense blend in the demo; `ragvet.embeddings.cross_encoder_reranker` is a real cross-encoder.
- `generation.faithfulness` — real embedding-support heuristic; `ClaudeJudge` for entailment-grade scoring.
- `drift.py`, `cost.py` — real math (PSI + centroid cosine; token/$ against a configurable rate table).

## Project layout

```
ragvet/
  metrics.py          hit@k, recall@k, precision@k, mrr, ndcg@k
  cost.py             token + $ estimator
  retrievers.py       Sparse (BM25), Dense (stand-in), Hybrid (+rerank)
  embeddings.py       real sentence-transformer embedder + cross-encoder reranker
  generation.py       faithfulness + answer relevance (+ Claude judge)
  eval_runner.py      run / compare retrievers
  drift.py            centroid cosine + PSI drift detection
  dashboard.py        Streamlit drift dashboard
  cli.py              ragvet CLI
  benchmarks/
    beir.py           generalized BEIR engine (any dataset, pluggable embedder)
    scifact.py        scifact entrypoint
    syllabus.py       SyllabusAI-domain retrieval + faithfulness eval
  retrievers_pgvector_claude.py   integration stub
tests/                metrics, eval_runner, cost, drift, retrievers, generation, cli
```

## Roadmap

- More BEIR datasets in the suite (fiqa, trec-covid).
- Live Claude-generated-answer eval end to end (retrieve -> generate -> score) in one command.
- Ship the drift dashboard on a rolling production-query window with alerting.

## Tests

```bash
pip install -e ".[dev]" && pytest -q      # 54 tests, no mocks, no network
```

## License

MIT

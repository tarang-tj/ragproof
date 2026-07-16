# ragproof

[![ci](https://github.com/tarang-tj/ragproof/actions/workflows/ci.yml/badge.svg)](https://github.com/tarang-tj/ragproof/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Evaluation and retrieval-benchmark harness for RAG systems. Prove retrieval **and** answer quality with numbers instead of vibes.

## Why evals

Most RAG failures are retrieval failures, not generation failures: the model answers badly because the wrong chunks were fetched, not because it reasoned poorly. Most teams shipping LLM apps have no evals at all, so a retrieval regression ships silently and shows up as "the bot got worse" three weeks later with no way to diagnose it. `ragproof` scores retrieval (hit@k, MRR, nDCG, recall), scores the generated answer (faithfulness, relevance), tracks cost, and detects query drift, so you catch regressions before your users do.

## Install

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
ragproof demo                                  # offline retrieval demo, no downloads
ragproof benchmark scifact                     # BEIR benchmark, model2vec dense (keyless)
ragproof benchmark scifact --embeddings st --rerank   # bge-small + cross-encoder
ragproof suite scifact nfcorpus                # robustness table across datasets
ragproof version
```

## Retrieval quality (offline demo)

`ragproof demo` loads a baked-in 40-document corpus and 24 labeled queries and evaluates three retrievers, no keys or downloads:

```
retriever          hit@3  mrr    ndcg@3  recall@5
-------------------------------------------------
sparse (bm25)      0.917  0.849  0.860  0.958
dense (tfidf+svd)  0.958  0.861  0.886  0.958
hybrid (+rerank)   1.000  0.882  0.912  1.000
```

Hybrid wins because sparse and dense retrieval have complementary failure modes: BM25 misses paraphrased queries, dense misses exact-keyword ones, and score fusion recovers both. `tests/test_eval_runner.py::test_hybrid_beats_single_methods_on_demo_dataset` locks the claim in.

## Answer quality: faithfulness and relevance

Retrieval metrics say the right context was fetched. They say nothing about the answer the LLM then wrote. `ragproof.generation` scores that:

- **Faithfulness** — the fraction of the answer's sentences supported by the retrieved context. Low = hallucination.
- **Answer relevance** — how on-topic the answer is to the question.

Two backends: a keyless embedding heuristic (default) and a Claude LLM-as-judge (`[judge]` extra) for fine-grained factual entailment.

```python
from ragproof.generation import evaluate_answer
from ragproof.benchmarks.beir import model2vec_embedder
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

`ragproof benchmark scifact --embeddings st --rerank` swaps in a real sentence-transformer (`BAAI/bge-small-en-v1.5`) and a cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`):

```
retriever               hit@3  mrr@10  ndcg@10  recall@10
----------------------------------------------------------
bm25                    0.590   0.524    0.560      0.686
dense (bge-small)       0.727   0.684    0.720      0.845
hybrid                  0.687   0.627    0.660      0.788
hybrid + rerank         0.717   0.654    0.681      0.795
```

A real embedder lifts dense NDCG@10 from 0.506 (model2vec) to **0.720** — the single biggest quality jump in this repo. The honest surprise: on scifact, dense alone beats hybrid and hybrid+rerank, because naive score fusion dilutes a strong dense retriever with a weaker BM25 signal. Fusion helps when the two retrievers are comparably strong (see scifact with model2vec, and the SyllabusAI set below); it hurts when one dominates. That is exactly the kind of thing an eval harness is for.

### Robustness across datasets

`ragproof suite scifact nfcorpus` — hybrid wins on both:

```
Robustness -- NDCG@10 across datasets (dense = model2vec)
retriever                  scifact    nfcorpus
----------------------------------------------
bm25                         0.560       0.267
dense (model2vec)            0.506       0.244
hybrid                       0.609       0.293
```

### SyllabusAI domain (representative Q&A)

`ragproof.benchmarks.syllabus` evaluates on the actual problem behind [SyllabusAI](https://syllabusai.net) — answering student questions from course syllabi — using a hand-labeled representative set (the live corpus has no ground-truth labels; point a pgvector retriever at it to run on real data):

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
streamlit run ragproof/dashboard.py
```

Paste a baseline window of queries and a current window; the dashboard embeds both and runs drift detection (PSI + centroid distance) so you can see when production traffic has wandered away from your eval set and your metrics have gone stale.

## Wire your own stack (pgvector + Claude)

`ragproof/retrievers_pgvector_claude.py` is a clearly-marked integration stub showing the production seam: Supabase pgvector for retrieval, Claude for generation, behind the same `Retriever` interface every retriever here uses. Point `eval_runner.evaluate_retriever()` at it once you fill in your keys.

## What's real vs stand-in

- `SparseRetriever` — real BM25 (`rank_bm25`).
- `DenseRetriever` — TF-IDF+SVD stand-in for the offline demo; use `ragproof.embeddings.sentence_transformer_embedder` (`[embeddings]`) for real semantic vectors, or model2vec (`[benchmark]`) for a keyless middle ground.
- Reranker — a lexical/dense blend in the demo; `ragproof.embeddings.cross_encoder_reranker` is a real cross-encoder.
- `generation.faithfulness` — real embedding-support heuristic; `ClaudeJudge` for entailment-grade scoring.
- `drift.py`, `cost.py` — real math (PSI + centroid cosine; token/$ against a configurable rate table).

## Project layout

```
ragproof/
  metrics.py          hit@k, recall@k, precision@k, mrr, ndcg@k
  cost.py             token + $ estimator
  retrievers.py       Sparse (BM25), Dense (stand-in), Hybrid (+rerank)
  embeddings.py       real sentence-transformer embedder + cross-encoder reranker
  generation.py       faithfulness + answer relevance (+ Claude judge)
  eval_runner.py      run / compare retrievers
  drift.py            centroid cosine + PSI drift detection
  dashboard.py        Streamlit drift dashboard
  cli.py              ragproof CLI
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

## Tests and publishing

```bash
pip install -e ".[dev]" && pytest -q      # 54 tests, no mocks, no network
python -m build && twine upload dist/*    # publish to PyPI (needs a PyPI token)
```

## License

MIT

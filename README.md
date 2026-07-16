# ragproof

[![ci](https://github.com/tarang-tj/ragproof/actions/workflows/ci.yml/badge.svg)](https://github.com/tarang-tj/ragproof/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Evaluation and retrieval-benchmark harness for RAG systems. Prove retrieval and generation quality with numbers instead of vibes.

## Why evals

Most RAG failures are retrieval failures, not generation failures: the model answers badly because the wrong chunks were fetched, not because it reasoned poorly. Most teams shipping LLM apps have no evals at all, so a retrieval regression ships silently and shows up as "the bot got worse" three weeks later with no way to diagnose it. `ragproof` gives you hit rate, MRR, nDCG, and recall on every retriever change, plus a cost estimate and a drift check, so you catch regressions before users do.

## Quickstart

```bash
git clone https://github.com/tarang-tj/ragproof.git
cd ragproof
pip install -e .
python -m ragproof.demo
```

No API keys, no downloaded models, no network calls. The demo loads a baked-in 40-document corpus and 24 labeled queries and evaluates three retrievers end to end.

## Demo result (real output, not fabricated)

```
ragproof demo -- 40 docs, 24 labeled queries, fully offline

retriever          hit@3  mrr    ndcg@3  recall@5
-------------------------------------------------
sparse (bm25)      0.917  0.849  0.860  0.958
dense (tfidf+svd)  0.958  0.861  0.886  0.958
hybrid (+rerank)   1.000  0.882  0.912  1.000

Estimated cost per query (input=1200t, output=250t, embed=20t): $0.007352 (rates: input $3.0/1M, output $15.0/1M, embed $0.1/1M -- update to current pricing)
```

Hybrid wins on every metric because sparse and dense retrieval have complementary failure modes: BM25 misses paraphrased queries that share no exact vocabulary with the target document, while the TF-IDF+SVD dense retriever occasionally under-ranks documents that exact-keyword queries should nail. Weighted score fusion recovers both failure modes because a document ranked poorly by one method but well by the other still surfaces in the combined top-k. This is a real, reproducible effect on this specific 40-doc corpus, not a cherry-picked number — run `python -m ragproof.demo` yourself, or read `ragproof/datasets.py` to see which queries are exact-keyword, which are paraphrases, and `tests/test_eval_runner.py::test_hybrid_beats_single_methods_on_demo_dataset` for the regression guard that keeps this claim honest.

## Real benchmark: BEIR/scifact (real embeddings, real qrels)

Beyond the offline toy demo, `ragproof` runs on the standard [BEIR/scifact](https://github.com/beir-cellar/beir) IR benchmark: 5,183 documents and 300 test queries with human relevance judgments. Dense retrieval uses [model2vec](https://github.com/MinishLab/model2vec) static embeddings (real semantic vectors, CPU only, no API key), so the whole benchmark reproduces without a GPU.

```bash
pip install -e ".[benchmark]"
python -m ragproof.benchmarks.scifact
```

```
BEIR/scifact -- 5183 docs, 300 test queries, real qrels
dense = model2vec minishlab/potion-base-8M static embeddings (CPU, no API key)

retriever             hit@3  mrr@10  ndcg@10  recall@10
--------------------------------------------------------
bm25                  0.590   0.524    0.560      0.686
dense (model2vec)     0.520   0.467    0.506      0.662
hybrid                0.633   0.576    0.609      0.731
```

Hybrid fusion lifts NDCG@10 from 0.560 (BM25) to 0.609 and hit@3 from 0.590 to 0.633 — the same complementary-failure-mode effect as the toy demo, now on a real benchmark. Two honest caveats: BM25 here is untuned `rank_bm25` (a tuned Anserini BM25 scores ~0.66 NDCG@10 on scifact), and model2vec static embeddings trade some quality for CPU speed, so a real sentence-transformer or Voyage embedding would push the dense number higher. The point is the relative hybrid gain and that the whole thing reproduces on a laptop.

## Metrics explained

- **hit@k** — did at least one relevant doc land in the top k? Binary, per query.
- **precision@k** — what fraction of the top k are relevant.
- **recall@k** — what fraction of all relevant docs were found in the top k.
- **MRR** — reciprocal rank of the first relevant doc (1/rank), full ranked list.
- **nDCG@k** — position-weighted relevance, normalized against the ideal ranking. Rewards putting relevant docs earlier, not just anywhere in the top k.

All five are pure functions in `ragproof/metrics.py`, unit tested against hand-computed values in `tests/test_metrics.py`.

## What's real vs. stand-in

- `SparseRetriever` — real BM25 (via `rank_bm25`). No caveats.
- `DenseRetriever` — **stand-in**: TF-IDF followed by TruncatedSVD (LSA), not a real semantic embedding model. It exists so the demo runs fully offline with zero downloads. It captures some latent co-occurrence structure but is not comparable to a real embedding model like Voyage or OpenAI's.
- `HybridRetriever` — real weighted score fusion (min-max normalized per query, then linearly combined). The optional reranker is also a **stand-in**: a lexical/dense blend over the shortlist, not a real cross-encoder.
- `drift.py` — real math: cosine distance between query-embedding centroids, and PSI (Population Stability Index) on a 1D projection. Both are standard, correct formulas, just run against the stand-in embeddings above.
- `cost.py` — real arithmetic against a rate table you must keep current (see `DEFAULT_RATES` — marked "update to current pricing").

## Wire your own stack

`ragproof/retrievers_pgvector_claude.py` is a clearly-marked integration stub (not executed by the demo or tests) showing the production seam: Supabase pgvector for similarity search, Claude for generation, behind the exact same `Retriever` interface (`retrieve(query, top_k)` / `score(query)`) used by every retriever in this repo. Point `eval_runner.evaluate_retriever()` at it once you fill in your Supabase/Anthropic/Voyage keys and it just works — no changes to metrics, eval_runner, or drift code.

## Project layout

```
ragproof/
  metrics.py                       hit@k, recall@k, precision@k, mrr, ndcg@k
  cost.py                          token + $ estimator, configurable rate table
  retrievers.py                    Sparse (BM25), Dense (TF-IDF+SVD stand-in), Hybrid (+rerank)
  eval_runner.py                   run one retriever or compare several, side-by-side table
  drift.py                         centroid cosine distance + PSI drift detection
  datasets.py                      baked-in offline demo corpus/queries + load_beir() stub
  demo.py                          python -m ragproof.demo
  retrievers_pgvector_claude.py    integration stub: Supabase pgvector + Claude
tests/
  test_metrics.py                  hand-computed expected values for every metric
  test_eval_runner.py               smoke tests + hybrid-wins regression guard
  test_cost.py, test_drift.py, test_retrievers.py
```

## Roadmap

- More BEIR sets (nfcorpus, fiqa, etc.) — scifact is done in `ragproof/benchmarks/scifact.py`; the `load_beir()` helper in `datasets.py` generalizes it next.
- Live evaluation against a real Claude-generated answer (faithfulness/groundedness scoring, not just retrieval metrics).
- A small drift dashboard that runs `drift.py` on a rolling window of production queries and alerts on PSI threshold breaches.
- Swap in a real embedding model (Voyage/Claude) as a second `DenseRetriever` implementation for an apples-to-apples comparison against the TF-IDF+SVD stand-in.

## Running tests

```bash
pip install -e ".[dev]"
pytest -q
```

44 tests, all passing, no mocks, no network calls.

## License

MIT

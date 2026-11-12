"""Monthly embedding leaderboard: today's most-downloaded small embedders, scored on BEIR/scifact.

    python -m ragvet.leaderboard [--max-new 2] [--top 8] [--offline]

Each run asks the Hugging Face Hub for the most-downloaded sentence-transformers models,
keeps the top `--top` that are small enough to run on a CPU runner (<= 150M parameters, no
custom remote code), and benchmarks any model whose revision (commit sha) is not already in
leaderboard/results.json. Scores come from the same engine as `ragvet benchmark`
(ragvet.benchmarks.beir.run): dense retrieval only, no instruction prefixes, nDCG@10 over
scifact's 300 test queries. leaderboard/LEADERBOARD.md and the README table are generated
from the JSON. Download ranks move every month; a score never changes for a pinned revision.

Exit 0: nothing changed. Exit 3: files changed (the workflow commits them). API errors raise.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()  # run from a checkout of the repo: the files below live there, not in site-packages
DIR = ROOT / "leaderboard"
RESULTS = DIR / "results.json"
BOARD = DIR / "LEADERBOARD.md"
README = ROOT / "README.md"
API = "https://huggingface.co/api/models"
QUERY = ("?pipeline_tag=sentence-similarity&library=sentence-transformers&sort=downloads&direction=-1"
         "&limit=40&expand[]=safetensors&expand[]=downloads&expand[]=sha&expand[]=tags")
MAX_PARAMS = 150_000_000
DATASET = "scifact"
START, END = "<!-- leaderboard:start -->", "<!-- leaderboard:end -->"


def fetch_models() -> list[dict]:
    url = (os.environ.get("RAGVET_HF_API") or API) + QUERY  # the workflow's failure drill sets a dead one
    req = urllib.request.Request(url, headers={"User-Agent": "ragvet-leaderboard"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    if not isinstance(data, list) or not data or "id" not in data[0]:
        raise ValueError(f"unexpected Hub API response: {str(data)[:200]}")
    return data


def candidates(models: list[dict], top: int) -> list[dict]:
    """The `top` most-downloaded models that fit a CPU runner, in download order."""
    out = []
    for m in models:
        params = (m.get("safetensors") or {}).get("total")
        if not params or params > MAX_PARAMS or "custom_code" in (m.get("tags") or []):
            continue
        out.append({"model": m["id"], "sha": m["sha"], "params": int(params), "rank": len(out) + 1})
        if len(out) == top:
            break
    return out


MAX_MODEL_S = 1500  # a model projected past this on a CPU runner is not scored (mpnet-base took 1,611 s and fit)
PROBE_DOCS = 64


def too_slow(seconds_for_probe: float, n_docs: int, limit: float = MAX_MODEL_S) -> float | None:
    """Projected seconds to embed the corpus, if over the limit; else None."""
    projected = seconds_for_probe * n_docs / PROBE_DOCS
    return projected if projected > limit else None


def score(model: str, sha: str) -> dict:
    """Dense-retrieval scores for one pinned model revision on scifact.

    Times the model on PROBE_DOCS documents first: one model that is too heavy for a CPU runner
    must not consume the whole job (a 109M fine-tune once took a run past its 45-minute timeout).
    """
    from ragvet.benchmarks.beir import load_beir, run
    from ragvet.embeddings import sentence_transformer_embedder
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    embed = sentence_transformer_embedder(model, revision=sha)
    corpus, _, _ = load_beir(DATASET)
    docs = list(corpus.values())
    embed(docs[:8])  # warm-up: the first call carries one-time setup and overstated gte-small 2x
    t0 = time.monotonic()
    embed(docs[PROBE_DOCS:2 * PROBE_DOCS])
    projected = too_slow(time.monotonic() - t0, len(corpus))
    if projected:
        return {"skipped": f"projected {projected / 60:.0f} min to embed {len(corpus):,} docs on this CPU",
                "limit_s": MAX_MODEL_S, "scored": today}
    t0 = time.monotonic()
    means = run(DATASET, embed=embed, dense_label=model, quiet=True)
    hit3, mrr10, ndcg10, rec10 = (round(float(v), 4) for v in means[f"dense ({model})"])
    return {"ndcg10": ndcg10, "mrr10": mrr10, "recall10": rec10, "hit3": hit3,
            "seconds": round(time.monotonic() - t0), "scored": today}


def load() -> dict:
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else {"scores": {}, "current": []}


def update(state: dict, cands: list[dict], max_new: int, scorer=score, budget_s: float = float("inf"),
           on_scored=None, clock=time.monotonic) -> list[str]:
    """Score up to `max_new` unscored revisions, starting no new one after `budget_s` seconds.

    The budget is what keeps a run inside the workflow's timeout: a job that times out commits
    nothing. `on_scored(state)` runs after each model (the CLI rewrites the files and logs it).
    Returns the models scored.
    """
    state["current"] = cands
    start, done = clock(), []
    for c in cands:
        key = f"{c['model']}@{c['sha']}"
        prev = state["scores"].get(key)
        if prev and not (prev.get("skipped") and prev.get("limit_s", 0) < MAX_MODEL_S):
            continue  # scored, or skipped under the current limit; a skip under a lower limit is retried
        if len(done) >= max_new or clock() - start > budget_s:
            break
        state["scores"][key] = scorer(c["model"], c["sha"])
        done.append(c["model"])
        if on_scored:
            on_scored(state)
    return done


def table(state: dict) -> str:
    """Current models ranked by nDCG@10; unscored ones listed last as pending."""
    rows = []
    for c in state["current"]:
        s = state["scores"].get(f"{c['model']}@{c['sha']}")
        rows.append((s["ndcg10"] if s and "ndcg10" in s else -1, c, s))
    rows.sort(key=lambda r: (-r[0], r[1]["rank"]))
    lines = ["| # | model | nDCG@10 | recall@10 | params | downloads rank |", "|---|---|---|---|---|---|"]
    for i, (_, c, s) in enumerate(rows, 1):
        link = f"[{c['model']}](https://huggingface.co/{c['model']}/tree/{c['sha']})"
        if s and "skipped" in s:
            vals = "too slow for a CPU runner | -"
        else:
            vals = f"{s['ndcg10']:.3f} | {s['recall10']:.3f}" if s else "pending | pending"
        ranked = bool(s and "ndcg10" in s)
        lines.append(f"| {i if ranked else '-'} | {link} | {vals} | {c['params'] / 1e6:.0f}M | {c['rank']} |")
    return "\n".join(lines)


def render(state: dict, checked: str) -> str:
    return (f"Dense retrieval on BEIR/{DATASET} (5,183 docs, 300 test queries), no instruction prefixes, so "
            "models trained with query prefixes (e5, bge) score below their model cards. Candidates: the "
            f"{len(state['current'])} most-downloaded sentence-transformers models on Hugging Face under "
            f"{MAX_PARAMS // 1_000_000}M parameters without custom code, re-checked monthly (last check "
            f"{checked}). Each score is pinned to the linked revision. BM25 on the same set: 0.560.\n\n"
            + table(state))


def write(state: dict, checked: str) -> None:
    DIR.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    body = render(state, checked)
    BOARD.write_text(f"# Embedding leaderboard\n\n{body}\n\nGenerated by `python -m ragvet.leaderboard`.\n")
    text = README.read_text()
    if START not in text or END not in text:
        raise ValueError("README has no leaderboard markers")
    README.write_text(re.sub(re.escape(START) + r".*?" + re.escape(END),
                             lambda _: f"{START}\n{body}\n{END}", text, flags=re.S))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ragvet leaderboard")
    ap.add_argument("--max-new", type=int, default=2, help="score at most this many new revisions")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--offline", action="store_true", help="re-render from results.json, no Hub call")
    ap.add_argument("--budget-min", type=float, default=12,
                    help="start no new model after this many minutes: ~5 setup + 12 + one model up to 25 fits the 45-minute timeout")
    a = ap.parse_args(argv)
    before = [p.read_text() if p.exists() else "" for p in (RESULTS, BOARD, README)]
    if START not in (README.read_text() if README.exists() else ""):  # fail before minutes of scoring
        raise SystemExit(f"run from the repository root: {README} has no leaderboard markers")
    state = load()
    checked = state.get("checked", "never")
    if not a.offline:
        cands = candidates(fetch_models(), a.top)
        checked = state["checked"] = datetime.now(timezone.utc).strftime("%Y-%m")
        done = update(state, cands, a.max_new, budget_s=a.budget_min * 60,
                      on_scored=lambda st: (write(st, checked), print("scored", list(st["scores"])[-1], flush=True)))
        print("scored:", ", ".join(done) or "nothing new")
    write(state, checked)
    print(render(state, checked))
    after = [p.read_text() for p in (RESULTS, BOARD, README)]
    return 3 if after != before else 0


if __name__ == "__main__":
    sys.exit(main())

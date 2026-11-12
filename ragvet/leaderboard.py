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

ROOT = Path(__file__).resolve().parents[1]
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


def score(model: str, sha: str) -> dict:
    """Dense-retrieval scores for one pinned model revision on scifact."""
    from ragvet.benchmarks.beir import run
    from ragvet.embeddings import sentence_transformer_embedder
    t0 = time.monotonic()
    means = run(DATASET, embed=sentence_transformer_embedder(model, revision=sha), dense_label=model, quiet=True)
    hit3, mrr10, ndcg10, rec10 = (round(float(v), 4) for v in means[f"dense ({model})"])
    return {"ndcg10": ndcg10, "mrr10": mrr10, "recall10": rec10, "hit3": hit3,
            "seconds": round(time.monotonic() - t0), "scored": datetime.now(timezone.utc).strftime("%Y-%m-%d")}


def load() -> dict:
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else {"scores": {}, "current": []}


def update(state: dict, cands: list[dict], max_new: int, scorer=score) -> list[str]:
    """Score up to `max_new` unscored revisions; set the current list. Returns models scored."""
    done = []
    for c in cands:
        key = f"{c['model']}@{c['sha']}"
        if key in state["scores"] or len(done) >= max_new:
            continue
        state["scores"][key] = scorer(c["model"], c["sha"])
        done.append(c["model"])
    state["current"] = cands
    return done


def table(state: dict) -> str:
    """Current models ranked by nDCG@10; unscored ones listed last as pending."""
    rows = []
    for c in state["current"]:
        s = state["scores"].get(f"{c['model']}@{c['sha']}")
        rows.append((s["ndcg10"] if s else -1, c, s))
    rows.sort(key=lambda r: (-r[0], r[1]["rank"]))
    lines = ["| # | model | nDCG@10 | recall@10 | params | downloads rank |", "|---|---|---|---|---|---|"]
    for i, (_, c, s) in enumerate(rows, 1):
        link = f"[{c['model']}](https://huggingface.co/{c['model']}/tree/{c['sha']})"
        vals = f"{s['ndcg10']:.3f} | {s['recall10']:.3f}" if s else "pending | pending"
        lines.append(f"| {i if s else '-'} | {link} | {vals} | {c['params'] / 1e6:.0f}M | {c['rank']} |")
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
    a = ap.parse_args(argv)
    before = [p.read_text() if p.exists() else "" for p in (RESULTS, BOARD, README)]
    state = load()
    checked = state.get("checked", "never")
    if not a.offline:
        done = update(state, candidates(fetch_models(), a.top), a.max_new)
        checked = state["checked"] = datetime.now(timezone.utc).strftime("%Y-%m")
        print("scored:", ", ".join(done) or "nothing new")
    write(state, checked)
    print(render(state, checked))
    after = [p.read_text() for p in (RESULTS, BOARD, README)]
    return 3 if after != before else 0


if __name__ == "__main__":
    sys.exit(main())

"""Leaderboard logic without the network: candidate filtering, the revision cache, ranking,
README rendering, and a dead API failing loud. Scoring itself is the BEIR engine, run for real
by the monthly workflow; here a fixed scorer stands in so no model is downloaded."""
import json

import pytest

from ragvet import leaderboard as lb

HUB = [  # the Hub API's shape, as returned on 2026-11-11 (fields trimmed)
    {"id": "sentence-transformers/all-MiniLM-L6-v2", "sha": "aaa", "safetensors": {"total": 22_713_728}, "tags": []},
    {"id": "BAAI/bge-m3", "sha": "bbb", "safetensors": None, "tags": []},                  # no size: skip
    {"id": "nomic-ai/nomic-embed-text-v1.5", "sha": "ccc", "safetensors": {"total": 136_731_648},
     "tags": ["custom_code"]},                                                            # remote code: skip
    {"id": "big/model", "sha": "ddd", "safetensors": {"total": 567_754_752}, "tags": []},  # too big: skip
    {"id": "thenlper/gte-small", "sha": "eee", "safetensors": {"total": 33_360_512}, "tags": []},
    {"id": "sentence-transformers/all-mpnet-base-v2", "sha": "fff", "safetensors": {"total": 109_486_978},
     "tags": []},
]


def fake_score(scores):
    calls = []

    def scorer(model, sha):
        calls.append(model)
        return {"ndcg10": scores[model], "mrr10": 0.5, "recall10": 0.8, "hit3": 0.6, "seconds": 1,
                "scored": "2026-11-11"}
    return scorer, calls


def test_candidates_keep_small_models_without_remote_code_in_download_order():
    c = lb.candidates(HUB, top=8)
    assert [x["model"] for x in c] == ["sentence-transformers/all-MiniLM-L6-v2", "thenlper/gte-small",
                                       "sentence-transformers/all-mpnet-base-v2"]
    assert [x["rank"] for x in c] == [1, 2, 3]
    assert lb.candidates(HUB, top=2)[-1]["model"] == "thenlper/gte-small"


def test_a_scored_revision_is_never_rescored_and_new_ones_are_capped():
    scorer, calls = fake_score({"sentence-transformers/all-MiniLM-L6-v2": 0.6, "thenlper/gte-small": 0.7,
                                "sentence-transformers/all-mpnet-base-v2": 0.65})
    state = {"scores": {}, "current": []}
    assert lb.update(state, lb.candidates(HUB, 8), max_new=2, scorer=scorer) == calls[:2]
    assert len(calls) == 2
    lb.update(state, lb.candidates(HUB, 8), max_new=2, scorer=scorer)
    assert len(calls) == 3                       # only the one left over
    hub2 = [dict(m, sha="new") if m["id"] == "thenlper/gte-small" else m for m in HUB]
    lb.update(state, lb.candidates(hub2, 8), max_new=2, scorer=scorer)
    assert calls[-1] == "thenlper/gte-small" and len(calls) == 4   # a new revision is a new score


def test_time_budget_stops_new_models_and_each_score_is_saved_as_it_lands():
    scorer, calls = fake_score({m["id"]: 0.6 for m in HUB})
    ticks = iter([0, 0, 100, 2000])          # start, before model 1, before model 2, before model 3
    saved = []
    state = {"scores": {}, "current": []}
    done = lb.update(state, lb.candidates(HUB, 8), max_new=9, scorer=scorer, budget_s=1500,
                     on_scored=lambda st: saved.append(len(st["scores"])), clock=lambda: next(ticks))
    assert len(done) == 2 and saved == [1, 2]   # third model not started: 2000 s > 1500 s budget
    assert len(state["current"]) == 3           # the ranking is still refreshed in full


def test_table_ranks_by_ndcg_and_lists_pending_last():
    scorer, _ = fake_score({"sentence-transformers/all-MiniLM-L6-v2": 0.6, "thenlper/gte-small": 0.7})
    state = {"scores": {}, "current": []}
    lb.update(state, lb.candidates(HUB, 8), max_new=2, scorer=scorer)
    rows = lb.table(state).splitlines()[2:]
    assert "gte-small" in rows[0] and rows[0].startswith("| 1 ") and "0.700" in rows[0]
    assert "all-MiniLM-L6-v2" in rows[1] and rows[1].startswith("| 2 ")
    assert "all-mpnet-base-v2" in rows[2] and "pending" in rows[2] and rows[2].startswith("| - ")
    assert "/tree/eee)" in rows[0]               # each score links its pinned revision


def test_write_replaces_only_the_readme_block(tmp_path, monkeypatch):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{lb.START}\nold table\n{lb.END}\noutro\n")
    for name, path in (("README", readme), ("DIR", tmp_path / "leaderboard"),
                       ("RESULTS", tmp_path / "leaderboard" / "results.json"),
                       ("BOARD", tmp_path / "leaderboard" / "LEADERBOARD.md")):
        monkeypatch.setattr(lb, name, path)
    scorer, _ = fake_score({"sentence-transformers/all-MiniLM-L6-v2": 0.6})
    state = {"scores": {}, "current": []}
    lb.update(state, lb.candidates(HUB, 1), max_new=2, scorer=scorer)
    lb.write(state, "2026-11")
    text = readme.read_text()
    assert text.startswith("intro\n") and text.endswith("outro\n") and "old table" not in text
    assert "0.600" in text and "last check 2026-11" in text
    assert json.loads((tmp_path / "leaderboard" / "results.json").read_text()) == state
    readme.write_text("no markers\n")
    with pytest.raises(ValueError):
        lb.write(state, "2026-11")


def test_a_dead_api_fails_loud(monkeypatch):
    monkeypatch.setenv("RAGVET_HF_API", "http://127.0.0.1:9/api/models")
    with pytest.raises(OSError):
        lb.fetch_models()

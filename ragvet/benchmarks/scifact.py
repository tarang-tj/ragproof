"""BEIR/scifact benchmark -- thin wrapper over the generalized engine.

Kept so the documented `python -m ragvet.benchmarks.scifact` command still works.
The retrieval/metric logic lives in `ragvet.benchmarks.beir`.

    python -m ragvet.benchmarks.scifact
"""
from ragvet.benchmarks.beir import load_beir, run


def load_scifact(*_args, **_kwargs):
    """Backwards-compatible loader; returns (corpus, queries, qrels) for scifact."""
    return load_beir("scifact")


if __name__ == "__main__":
    run("scifact")

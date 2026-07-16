"""ragproof command-line interface.

    ragproof demo                         # offline retrieval-metrics demo, no downloads
    ragproof benchmark scifact            # BEIR benchmark (model2vec dense, keyless)
    ragproof benchmark scifact --embeddings st --rerank   # bge-small + cross-encoder
    ragproof suite scifact nfcorpus       # robustness table across datasets
    ragproof version
"""
from __future__ import annotations

import argparse
import sys

from ragproof import __version__


def _build_embed_and_reranker(embeddings: str, rerank: bool):
    """Resolve the dense embedder + optional reranker from CLI flags (lazy heavy imports)."""
    if embeddings == "st":
        from ragproof.embeddings import sentence_transformer_embedder
        embed, label = sentence_transformer_embedder(), "bge-small"
    else:
        embed, label = None, "model2vec"  # engine default
    reranker = None
    if rerank:
        from ragproof.embeddings import cross_encoder_reranker
        reranker = cross_encoder_reranker()
    return embed, reranker, label


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ragproof", description="RAG evaluation harness.")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("demo", help="run the offline retrieval-metrics demo")

    bench = sub.add_parser("benchmark", help="run a BEIR benchmark on one dataset")
    bench.add_argument("dataset", nargs="?", default="scifact")
    bench.add_argument("--embeddings", choices=["model2vec", "st"], default="model2vec",
                       help="dense backend: model2vec (keyless) or st (sentence-transformer)")
    bench.add_argument("--rerank", action="store_true", help="add a cross-encoder reranker")

    suite = sub.add_parser("suite", help="robustness table (NDCG@10) across datasets")
    suite.add_argument("datasets", nargs="*", default=["scifact", "nfcorpus"])
    suite.add_argument("--embeddings", choices=["model2vec", "st"], default="model2vec")

    sub.add_parser("version", help="print the ragproof version")

    args = parser.parse_args(argv)

    if args.command == "demo":
        from ragproof.demo import main as demo_main
        demo_main()
    elif args.command == "benchmark":
        from ragproof.benchmarks.beir import run
        embed, reranker, label = _build_embed_and_reranker(args.embeddings, args.rerank)
        run(args.dataset, embed=embed, reranker=reranker, dense_label=label)
    elif args.command == "suite":
        from ragproof.benchmarks.beir import run_suite
        embed, _, label = _build_embed_and_reranker(args.embeddings, False)
        run_suite(args.datasets, embed=embed, dense_label=label)
    elif args.command == "version":
        print(f"ragproof {__version__}")
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

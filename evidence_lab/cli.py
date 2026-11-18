"""Small command line interface for downloading and searching the corpus."""
import argparse
import json
import sys
from pathlib import Path

from .data import load_corpus
from .download import download_scifact
from .retrieval import BM25Index, DenseIndex, HybridIndex, DEFAULT_MODEL, DEFAULT_REVISION


def main(argv=None):
    parser = argparse.ArgumentParser(description="Search the BEIR SciFact corpus locally")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download and verify the official SciFact archive")
    download.add_argument("--data", default="data/scifact", help="New destination directory")
    search = commands.add_parser("search", help="Retrieve corpus documents for a query")
    search.add_argument("query")
    search.add_argument("--data", default="data/scifact")
    search.add_argument("--method", choices=("bm25", "dense", "hybrid"), default="bm25")
    search.add_argument("--k", type=int, default=10)
    search.add_argument("--candidate-k", type=int, default=100)
    search.add_argument("--cache-dir", default=None, help="Optional directory for validated dense embeddings")
    search.add_argument("--model", default=DEFAULT_MODEL)
    search.add_argument("--revision", default=DEFAULT_REVISION, help="Hugging Face revision; override with --model")
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            print(f"Downloaded verified SciFact data to {download_scifact(args.data)}")
            return 0
        corpus = load_corpus(Path(args.data) / "corpus.jsonl")
        bm25 = BM25Index(corpus) if args.method in ("bm25", "hybrid") else None
        dense = DenseIndex(corpus, model_name=args.model, revision=args.revision, cache_dir=args.cache_dir) if args.method in ("dense", "hybrid") else None
        index = bm25 if args.method == "bm25" else dense if args.method == "dense" else HybridIndex(bm25, dense, candidate_k=args.candidate_k)
        for doc_id, score in index.search(args.query, args.k):
            print(json.dumps({"id": doc_id, "score": score, **corpus[doc_id]}, ensure_ascii=False))
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

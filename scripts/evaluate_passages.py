"""Exploratory SciFact document vs passage BM25 comparison.

The existing test set was inspected previously. Document relevance judgments
cannot validate passage relevance or the quality of a citation.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

from evidence_lab.data import load_corpus, load_queries, load_qrels
from evidence_lab.evaluation import evaluate, paired_bootstrap, run_queries
from evidence_lab.passages import CHUNKING_VERSION, aggregate_passages, build_passages
from evidence_lab.retrieval import BM25Index
from scripts.evaluate import cross_split_duplicates, sha256


def build_systems(corpus):
    """Return fixed full-document and service-equivalent passage BM25 systems."""
    start = time.perf_counter()
    document_index = BM25Index(corpus, k1=1.2, b=0.75)
    document_build_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    passages = build_passages(corpus, chunk_words=180, overlap_words=30)
    if not passages:
        raise ValueError("no passages produced")
    searchable = {pid: {"title": p["title"] if p["chunk_index"] == 0 else "", "text": p["text"]}
                  for pid, p in passages.items()}
    passage_index = BM25Index(searchable, k1=1.2, b=0.75)
    passage_build_ms = (time.perf_counter() - start) * 1000

    def search_passages(query, k=10):
        # Retrieve all matching chunks, so duplicates cannot crowd out parent IDs.
        ranked = passage_index.search(query, k=len(passages))
        parents = aggregate_passages(ranked, passages, k=k)
        return [(row["parent_id"], row["score"]) for row in parents]

    return document_index.search, search_passages, {
        "document_index_ms": document_build_ms, "passage_build_and_index_ms": passage_build_ms,
        "document_count": len(corpus), "passage_count": len(passages),
        "documents_without_passages": len(corpus) - len({p["parent_id"] for p in passages.values()}),
    }


def benchmark(corpus, queries, qrels, *, k=10, seed=42, bootstrap_samples=10000):
    missing_docs = {doc for grades in qrels.values() for doc in grades if doc not in corpus}
    if missing_docs:
        raise ValueError(f"qrels refer to {len(missing_docs)} missing documents")
    document_search, passage_search, indexing = build_systems(corpus)
    reports = {}
    for name, search in (("document_bm25", document_search), ("passage_bm25", passage_search)):
        ranks, timing = run_queries(search, queries, qrels, k)
        report = evaluate(ranks, qrels, (k,))
        report["latency"] = timing
        report["ranked_parent_ids"] = ranks
        reports[name] = report
    comparisons = {metric: paired_bootstrap(reports["document_bm25"]["per_query"], reports["passage_bm25"]["per_query"], metric, samples=bootstrap_samples, seed=seed)
                   for metric in reports["document_bm25"]["metrics"]}
    return {"indexing": indexing, "systems": reports, "comparisons_passage_minus_document": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/scifact"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-test", action="store_true", help="Acknowledge use of previously inspected official test queries")
    args = parser.parse_args()
    if not args.final_test:
        parser.error("--final-test is required to run on the official test set")
    corpus_file, queries_file = args.data / "corpus.jsonl", args.data / "queries.jsonl"
    test_file, train_file = args.data / "qrels/test.tsv", args.data / "qrels/train.tsv"
    corpus, queries, qrels = load_corpus(corpus_file), load_queries(queries_file), load_qrels(test_file)
    result = benchmark(corpus, queries, qrels)
    result["config"] = {"split": "official test, previously inspected", "k": 10, "bm25_k1": 1.2, "bm25_b": 0.75,
                        "chunk_words": 180, "overlap_words": 30, "chunking_version": CHUNKING_VERSION,
                        "passage_title": "first chunk only", "aggregation": "max passage score; all matching chunks",
                        "bootstrap_samples": 10000, "seed": 42, "python": sys.version, "platform": platform.platform(),
                        "hashes_sha256": {str(p): sha256(p) for p in (corpus_file, queries_file, test_file, train_file)},
                        "limitations": "Exploratory: test set previously inspected; document qrels do not judge passage text or citation correctness; unjudged documents count as non-relevant."}
    result["cross_split_overlap"] = cross_split_duplicates(queries, load_qrels(train_file), qrels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"indexing": result["indexing"], "systems": {k: {"metrics": v["metrics"], "latency": v["latency"]} for k, v in result["systems"].items()}, "comparisons_passage_minus_document": result["comparisons_passage_minus_document"]}, indent=2))
    print(f"Full report: {args.output}")


if __name__ == "__main__":
    main()

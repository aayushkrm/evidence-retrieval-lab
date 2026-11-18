"""Run an offline, query-level SciFact retrieval benchmark.

Example: python -m scripts.evaluate --data data/scifact --split dev --systems bm25
The development split is a deterministic 20% sample of official train queries.
Test is reserved for a final, preselected configuration (--final-test).
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
from pathlib import Path
from evidence_lab.data import load_corpus, load_queries, load_qrels
from evidence_lab.evaluation import evaluate, paired_bootstrap, run_queries


def split_train(qrels, seed=42, dev_fraction=0.2):
    """Split by query, with no overlap. Seed and sorted input fix the split."""
    ids = sorted(qrels)
    random.Random(seed).shuffle(ids)
    n_dev = max(1, round(len(ids) * dev_fraction))
    if n_dev >= len(ids):
        raise ValueError("at least two training queries are needed")
    dev = set(ids[:n_dev])
    return ({q: qrels[q] for q in sorted(ids) if q not in dev},
            {q: qrels[q] for q in sorted(ids) if q in dev})


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cross_split_duplicates(queries, train_qrels, test_qrels):
    """Find exact query text matches after whitespace normalization and casefolding."""
    def normalized(text):
        return " ".join(text.split()).casefold()
    by_text = {}
    for qid in sorted(train_qrels):
        if qid not in queries:
            raise ValueError(f"missing training query text: {qid}")
        by_text.setdefault(normalized(queries[qid]), []).append(qid)
    matches = []
    for test_id in sorted(test_qrels):
        if test_id not in queries:
            raise ValueError(f"missing test query text: {test_id}")
        for train_id in by_text.get(normalized(queries[test_id]), []):
            matches.append({"train_query_id": train_id, "test_query_id": test_id,
                            "same_relevance_judgments": train_qrels[train_id] == test_qrels[test_id]})
    return matches


def package_versions():
    versions = {}
    for package in ("sentence-transformers", "transformers", "tokenizers", "torch", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/scifact"))
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True)
    parser.add_argument("--final-test", action="store_true", help="Confirm configuration was selected using train/dev only")
    parser.add_argument("--overlap-sensitivity", action="store_true", help="For final test, also report scores excluding test queries whose normalized text occurs in official train")
    parser.add_argument("--systems", nargs="+", choices=["bm25", "dense", "hybrid"], default=["bm25"])
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional directory for validated dense embeddings")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--model-revision", default="1110a243fdf4706b3f48f1d95db1a4f5529b4d41", help="Pinned model commit; supply a matching commit when changing --model")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.split == "test" and not args.final_test:
        parser.error("test is reserved for final reporting; pass --final-test after selecting configuration on train/dev")
    if args.overlap_sensitivity and args.split != "test":
        parser.error("--overlap-sensitivity requires --split test")
    if args.k < 1:
        parser.error("--k must be positive")
    if len(set(args.systems)) != len(args.systems):
        parser.error("--systems must not contain duplicates")
    from evidence_lab.retrieval import BM25Index, DenseIndex, HybridIndex
    corpus_file, query_file = args.data / "corpus.jsonl", args.data / "queries.jsonl"
    qrel_file = args.data / "qrels" / ("test.tsv" if args.split == "test" else "train.tsv")
    corpus, queries, qrels = load_corpus(corpus_file), load_queries(query_file), load_qrels(qrel_file)
    if args.split != "test":
        train, dev = split_train(qrels, seed=args.seed)
        qrels = train if args.split == "train" else dev
    unknown_docs = sorted({doc for judgments in qrels.values() for doc in judgments if doc not in corpus})
    if unknown_docs:
        raise ValueError(f"qrels refer to {len(unknown_docs)} missing documents")
    bm25 = BM25Index(corpus) if any(s in args.systems for s in ("bm25", "hybrid")) else None
    if any(s in args.systems for s in ("dense", "hybrid")):
        print("Loading the neural model and corpus embeddings; first-time CPU encoding can take a long time.", file=sys.stderr, flush=True)
        dense = DenseIndex(corpus, model_name=args.model, revision=args.model_revision, cache_dir=args.cache_dir)
    else:
        dense = None
    indexes = {"bm25": bm25, "dense": dense}
    if "hybrid" in args.systems:
        indexes["hybrid"] = HybridIndex(bm25, dense)
    result = {"config": {"split": args.split, "split_method": "seeded shuffle of sorted official train query IDs, 20% dev" if args.split != "test" else "official BEIR test", "seed": args.seed, "k": args.k, "systems": args.systems, "dense_model": args.model if dense else None, "dense_revision": args.model_revision if dense else None, "dense_cache_hit": dense.cache_hit if dense else None, "dense_cache_key": dense.cache_key if dense else None, "dense_max_seq_length": dense.model.max_seq_length if dense else None, "package_versions": package_versions(), "hashes_sha256": {str(p): sha256(p) for p in (corpus_file, query_file, qrel_file)}, "python": sys.version, "platform": platform.platform()}, "systems": {}, "comparisons": {}}
    overlaps = []
    if args.split == "test":
        train_file = args.data / "qrels" / "train.tsv"
        overlaps = cross_split_duplicates(queries, load_qrels(train_file), qrels)
        result["config"]["hashes_sha256"][str(train_file)] = sha256(train_file)
        result["cross_split_overlap"] = {"normalization": "casefold and collapse whitespace", "matching_pairs": overlaps, "overlapping_test_query_count": len({pair["test_query_id"] for pair in overlaps})}
        print(f"Train/test normalized exact-text overlap: {len(overlaps)} pairs, {len({pair['test_query_id'] for pair in overlaps})} test queries")
    for system in args.systems:
        ranks, latency = run_queries(indexes[system].search, queries, qrels, args.k)
        report = evaluate(ranks, qrels, (args.k,))
        report["latency"] = latency
        report["queries"] = {qid: {"query": queries[qid], "retrieved": ranks[qid], "relevant": sorted(doc for doc, grade in qrels[qid].items() if grade > 0), "missed_relevant": sorted(doc for doc, grade in qrels[qid].items() if grade > 0 and doc not in ranks[qid])} for qid in ranks}
        if args.overlap_sensitivity:
            excluded = {pair["test_query_id"] for pair in overlaps}
            filtered_qrels = {qid: grades for qid, grades in qrels.items() if qid not in excluded}
            sensitivity = evaluate(ranks, filtered_qrels, (args.k,))
            sensitivity["excluded_overlap_query_ids"] = sorted(excluded)
            report["excluding_train_text_overlap"] = sensitivity
        result["systems"][system] = report
    for i, first in enumerate(args.systems):
        for second in args.systems[i + 1:]:
            result["comparisons"][f"{second}_minus_{first}"] = {metric: paired_bootstrap(result["systems"][first]["per_query"], result["systems"][second]["per_query"], metric, seed=args.seed, samples=args.bootstrap_samples) for metric in result["systems"][first]["metrics"]}
    if args.overlap_sensitivity:
        result["comparisons_excluding_train_text_overlap"] = {}
        for i, first in enumerate(args.systems):
            for second in args.systems[i + 1:]:
                a = result["systems"][first]["excluding_train_text_overlap"]["per_query"]
                b = result["systems"][second]["excluding_train_text_overlap"]["per_query"]
                result["comparisons_excluding_train_text_overlap"][f"{second}_minus_{first}"] = {metric: paired_bootstrap(a, b, metric, seed=args.seed, samples=args.bootstrap_samples) for metric in result["systems"][first]["metrics"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({name: {"metrics": v["metrics"], "latency": v["latency"], "query_count": v["query_count"]} for name, v in result["systems"].items()}, indent=2))
    print(f"Full report: {args.output}")

if __name__ == "__main__":
    main()

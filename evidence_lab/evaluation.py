"""Evaluate ranked document IDs against human relevance judgments.

Only queries with at least one positive judgment are included. Missing judgments
are treated as non-relevant, as in the standard BEIR evaluation protocol.
"""
from __future__ import annotations

import math
import random
import statistics
import time
from typing import Callable, Mapping, Sequence

Qrels = Mapping[str, Mapping[str, int]]
Ranking = Mapping[str, Sequence[str]]


def query_metrics(ranked: Sequence[str], judgments: Mapping[str, int], k: int) -> dict[str, float]:
    """Recall, reciprocal rank, and graded nDCG at k (gain = 2**grade - 1)."""
    if k < 1:
        raise ValueError("k must be positive")
    relevant = {doc: grade for doc, grade in judgments.items() if grade > 0}
    if not relevant:
        raise ValueError("query must have at least one positive judgment")
    if len(ranked) != len(set(ranked)):
        raise ValueError("ranking contains duplicate document IDs")
    hits = [max(0, judgments.get(doc, 0)) for doc in ranked[:k]]
    recall = sum(grade > 0 for grade in hits) / len(relevant)
    reciprocal_rank = next((1.0 / (i + 1) for i, grade in enumerate(hits) if grade > 0), 0.0)
    dcg = sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(hits))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(ideal))
    return {"recall": recall, "mrr": reciprocal_rank, "ndcg": dcg / idcg}


def evaluate(rankings: Ranking, qrels: Qrels, ks: Sequence[int] = (5, 10)) -> dict:
    """Return macro means and per-query scores, including unreturned queries."""
    if not ks or any(k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")
    eligible = sorted(q for q, judgments in qrels.items() if any(g > 0 for g in judgments.values()))
    if not eligible:
        raise ValueError("no queries with positive relevance judgments")
    per_query = {}
    for qid in eligible:
        scores = {}
        for k in ks:
            scores.update({f"{name}@{k}": value for name, value in query_metrics(rankings.get(qid, []), qrels[qid], k).items()})
        per_query[qid] = scores
    averages = {metric: statistics.fmean(scores[metric] for scores in per_query.values()) for metric in next(iter(per_query.values()))}
    return {"query_count": len(eligible), "excluded_queries_without_positive_judgments": len(qrels) - len(eligible), "metrics": averages, "per_query": per_query}


def paired_bootstrap(a: Mapping[str, Mapping[str, float]], b: Mapping[str, Mapping[str, float]], metric: str, *, samples: int = 10000, seed: int = 42) -> dict:
    """Paired query bootstrap for mean B - A; percentile 95% interval."""
    if samples < 1:
        raise ValueError("samples must be positive")
    if set(a) != set(b) or not a:
        raise ValueError("paired systems must have the same nonempty query set")
    diffs = [b[q][metric] - a[q][metric] for q in sorted(a)]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    return {"mean_difference_b_minus_a": statistics.fmean(diffs), "ci95_low": means[int(0.025 * (samples - 1))], "ci95_high": means[int(0.975 * (samples - 1))], "samples": samples, "seed": seed, "query_count": n}


def run_queries(search: Callable, queries: Mapping[str, str], qrels: Qrels, k: int, *, warmup: int = 3) -> tuple[dict[str, list[str]], dict]:
    """Time sequential single-query searches, excluding a short warm-up."""
    ids = sorted(q for q in qrels if q in queries and any(g > 0 for g in qrels[q].values()))
    missing = sorted(q for q in qrels if q not in queries and any(g > 0 for g in qrels[q].values()))
    if missing:
        raise ValueError(f"{len(missing)} judged queries are missing query text")
    if not ids:
        raise ValueError("no eligible queries")
    for qid in ids[:warmup]:
        search(queries[qid], k=k)
    rankings, durations = {}, []
    for qid in ids:
        start = time.perf_counter()
        result = search(queries[qid], k=k)
        durations.append((time.perf_counter() - start) * 1000)
        rankings[qid] = [str(docid) for docid, _score in result]
    ordered = sorted(durations)
    return rankings, {"unit": "milliseconds", "query_count": len(ids), "warmup_queries": min(warmup, len(ids)), "mean": statistics.fmean(durations), "median": statistics.median(durations), "p95": ordered[math.ceil(0.95 * len(ordered)) - 1], "scope": "sequential search calls; excludes index construction"}

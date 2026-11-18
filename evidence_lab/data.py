"""Load BEIR JSONL and qrels TSV without executing serialized code."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def _jsonl(path):
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
                if not isinstance(row, dict) or not isinstance(row.get("_id"), str) or not row["_id"]:
                    raise ValueError(f"{path}:{line_no}: expected an object with nonempty string _id")
                yield row
    except OSError as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc


def load_corpus(path):
    docs = {}
    for row in _jsonl(path):
        doc_id = row["_id"]
        if doc_id in docs:
            raise ValueError(f"Duplicate corpus ID: {doc_id}")
        if not isinstance(row.get("text"), str) or not isinstance(row.get("title", ""), str):
            raise ValueError(f"Document {doc_id}: title and text must be strings")
        docs[doc_id] = {"title": row.get("title", ""), "text": row["text"]}
    if not docs:
        raise ValueError(f"Empty corpus: {path}")
    return docs


def load_queries(path):
    queries = {}
    for row in _jsonl(path):
        query_id = row["_id"]
        if query_id in queries:
            raise ValueError(f"Duplicate query ID: {query_id}")
        if not isinstance(row.get("text"), str):
            raise ValueError(f"Query {query_id}: text must be a string")
        queries[query_id] = row["text"]
    if not queries:
        raise ValueError(f"Empty queries: {path}")
    return queries


def load_qrels(path):
    path = Path(path)
    result = {}
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            required = {"query-id", "corpus-id", "score"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"{path}: expected TSV columns {sorted(required)}")
            for line_no, row in enumerate(reader, 2):
                try:
                    qid, did, score = row["query-id"], row["corpus-id"], int(row["score"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{line_no}: invalid qrel") from exc
                if not qid or not did or score < 0 or did in result.get(qid, {}):
                    raise ValueError(f"{path}:{line_no}: invalid or duplicate qrel")
                result.setdefault(qid, {})[did] = score
    except OSError as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    if not result:
        raise ValueError(f"Empty qrels: {path}")
    return result

"""Deterministic word-window passages and parent-level result aggregation.

Offsets are Python character indices into the original document text, with an
inclusive start and exclusive end. Whitespace between words is retained.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import math
import re

CHUNKING_VERSION = "word-window-v1"
_WORD = re.compile(r"\S+")


def build_passages(corpus: dict, *, chunk_words: int = 180, overlap_words: int = 30) -> dict:
    """Build a searchable passage corpus with source offsets and stable IDs.

    Windows use runs of non-whitespace characters as words. Documents with no
    such runs yield no passages. Passages cover every word; offsets do not
    include leading/trailing whitespace outside each window.
    """
    if not isinstance(chunk_words, int) or isinstance(chunk_words, bool) or chunk_words < 1:
        raise ValueError("chunk_words must be a positive integer")
    if not isinstance(overlap_words, int) or isinstance(overlap_words, bool) or not 0 <= overlap_words < chunk_words:
        raise ValueError("overlap_words must be a nonnegative integer less than chunk_words")
    if not isinstance(corpus, dict):
        raise ValueError("corpus must be a dictionary")
    if any(not isinstance(key, str) or not key for key in corpus):
        raise ValueError("corpus must map nonempty string IDs to document objects")
    passages = {}
    for parent_id in sorted(corpus):
        doc = corpus[parent_id]
        if not isinstance(parent_id, str) or not parent_id or not isinstance(doc, dict):
            raise ValueError("corpus must map nonempty string IDs to document objects")
        title, text = doc.get("title", ""), doc.get("text")
        if not isinstance(title, str) or not isinstance(text, str):
            raise ValueError(f"Invalid title or text in document {parent_id}")
        words = iter(_WORD.finditer(text))
        window = deque()
        for _ in range(chunk_words):
            match = next(words, None)
            if match is None:
                break
            window.append(match)
        index = 0
        while window:
            start_char, end_char = window[0].start(), window[-1].end()
            passage_text = text[start_char:end_char]
            identity = [CHUNKING_VERSION, chunk_words, overlap_words, parent_id, title,
                        start_char, end_char, passage_text]
            payload = json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            passage_id = "p1-" + hashlib.sha256(payload).hexdigest()
            if passage_id in passages:
                raise ValueError("passage ID collision")
            passages[passage_id] = {"title": title, "text": passage_text, "parent_id": parent_id,
                                    "start_char": start_char, "end_char": end_char,
                                    "chunk_index": index, "chunking_version": CHUNKING_VERSION}
            if len(window) < chunk_words:
                break
            for _ in range(chunk_words - overlap_words):
                window.popleft()
            added = 0
            while len(window) < chunk_words:
                match = next(words, None)
                if match is None:
                    break
                window.append(match)
                added += 1
            if not added:
                break
            index += 1
    return passages


def aggregate_passages(ranked: list[tuple[str, float]], passages: dict, *, k: int = 10) -> list[dict]:
    """Rank parents by their best retrieved passage and return that passage's offsets.

    Ties between passages of a parent use passage ID; ties between parents use
    parent ID. This aggregation describes retrieval scores, not relevance.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    winners = {}
    seen = set()
    for passage_id, score in ranked:
        if passage_id in seen:
            raise ValueError("ranking contains duplicate passage IDs")
        seen.add(passage_id)
        if passage_id not in passages:
            raise ValueError(f"unknown passage ID: {passage_id}")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("passage scores must be finite numbers")
        passage = passages[passage_id]
        parent_id = passage["parent_id"]
        current = winners.get(parent_id)
        if current is None or (-score, passage_id) < (-current["score"], current["passage_id"]):
            winners[parent_id] = {"parent_id": parent_id, "score": score, "passage_id": passage_id,
                                  "start_char": passage["start_char"], "end_char": passage["end_char"],
                                  "text": passage["text"], "title": passage["title"]}
    return sorted(winners.values(), key=lambda item: (-item["score"], item["parent_id"]))[:k]

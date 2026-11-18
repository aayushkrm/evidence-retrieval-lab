"""CPU BM25, optional dense retrieval, and reciprocal-rank fusion."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

_TOKEN = re.compile(r"\w+", re.UNICODE)
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def tokenize(text):
    return _TOKEN.findall(text.lower())


def _check_k(k):
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")


def _check_corpus(corpus):
    if not corpus:
        raise ValueError("corpus must not be empty")
    for doc_id, doc in corpus.items():
        if not isinstance(doc_id, str) or not doc_id or not isinstance(doc, dict):
            raise ValueError("corpus must map nonempty string IDs to document objects")
        if not isinstance(doc.get("text"), str) or not isinstance(doc.get("title", ""), str):
            raise ValueError(f"Invalid title or text in document {doc_id}")


class BM25Index:
    """Okapi BM25; lowercase Unicode word tokens, title concatenated with text."""

    def __init__(self, corpus, k1=1.2, b=0.75):
        _check_corpus(corpus)
        if not math.isfinite(k1) or not math.isfinite(b) or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("k1 must be positive and b must be between 0 and 1")
        self.k1, self.b = k1, b
        self.ids = sorted(corpus)
        self.postings = defaultdict(list)
        self.lengths = {}
        for doc_id in self.ids:
            doc = corpus[doc_id]
            tokens = tokenize(doc.get("title", "") + " " + doc["text"])
            self.lengths[doc_id] = len(tokens)
            for term, count in Counter(tokens).items():
                self.postings[term].append((doc_id, count))
        self.avgdl = sum(self.lengths.values()) / len(self.ids)

    def search(self, query, k=10):
        _check_k(k)
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if not self.avgdl:
            return []
        scores = defaultdict(float)
        for term, query_count in sorted(Counter(tokenize(query)).items()):
            postings = self.postings.get(term, ())
            if not postings:
                continue
            idf = math.log1p((len(self.ids) - len(postings) + 0.5) / (len(postings) + 0.5))
            for doc_id, tf in postings:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[doc_id] / self.avgdl)
                scores[doc_id] += query_count * idf * tf * (self.k1 + 1) / (tf + norm)
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]


class DenseIndex:
    """Normalized CPU embeddings with an optional trusted-local-data cache.

    NPZ loading disables pickle but does not limit allocations from hostile ZIPs.
    Use a cache directory controlled by the current user.
    """

    def __init__(self, corpus, model_name=DEFAULT_MODEL, batch_size=32, revision=DEFAULT_REVISION, cache_dir=None):
        _check_corpus(corpus)
        _check_k(batch_size)
        if cache_dir is not None:
            if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
                raise ValueError("dense cache requires an immutable 40-character model commit revision")
            if Path(model_name).exists():
                raise ValueError("dense cache does not support local model paths")
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Dense retrieval needs the optional dependency: pip install 'evidence-retrieval-lab[dense]'") from exc
        self.ids = sorted(corpus)
        self.model_name = model_name
        self.revision = revision
        self.cache_hit = False
        self.cache_key = None
        try:
            self.model = SentenceTransformer(model_name, device="cpu", revision=revision)
            dimension = self.model.get_sentence_embedding_dimension()
            if not isinstance(dimension, int) or dimension < 1:
                raise ValueError("model returned an invalid embedding dimension")
            texts = [corpus[i].get("title", "") + "\n" + corpus[i]["text"] for i in self.ids]
            if cache_dir is not None:
                versions = {}
                for package in ("sentence-transformers", "transformers", "tokenizers", "torch", "numpy"):
                    try:
                        versions[package] = importlib.metadata.version(package)
                    except importlib.metadata.PackageNotFoundError:
                        versions[package] = "unavailable"
                config = {"format": 1, "model": model_name, "revision": revision,
                          "dimension": dimension, "max_seq_length": self.model.max_seq_length,
                          "batch_size": batch_size, "normalized": True, "dtype": "float32",
                          "document_format": "title+newline+text", "device": "cpu",
                          "package_versions": versions}
                digest = hashlib.sha256()
                for item in [config] + [[doc_id, corpus[doc_id].get("title", ""), corpus[doc_id]["text"]] for doc_id in self.ids]:
                    digest.update(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
                    digest.update(b"\n")
                self.cache_key = digest.hexdigest()
                directory = Path(cache_dir)
                directory.mkdir(parents=True, exist_ok=True)
                cache_path = directory / (self.cache_key + ".npz")
            else:
                cache_path = None
            if cache_path is not None and cache_path.exists():
                try:
                    with np.load(cache_path, allow_pickle=False) as cached:
                        if set(cached.files) != {"key", "embeddings"} or cached["key"].shape != () or str(cached["key"].item()) != self.cache_key:
                            raise ValueError("cache metadata mismatch")
                        embeddings = cached["embeddings"]
                        self._validate_embeddings(embeddings, len(self.ids), dimension, np)
                        self.embeddings = embeddings.copy()
                    self.cache_hit = True
                except Exception as exc:
                    raise ValueError(f"Invalid dense embedding cache {cache_path}: {exc}") from exc
            else:
                self.embeddings = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                                    convert_to_numpy=True, show_progress_bar=sys.stderr.isatty())
                self._validate_embeddings(self.embeddings, len(self.ids), dimension, np)
                if cache_path is not None:
                    temporary = None
                    try:
                        with tempfile.NamedTemporaryFile(mode="wb", prefix=".embedding-", suffix=".tmp", dir=directory, delete=False) as out:
                            temporary = Path(out.name)
                            np.savez(out, key=np.array(self.cache_key), embeddings=self.embeddings)
                            out.flush()
                            os.fsync(out.fileno())
                        os.replace(temporary, cache_path)
                    finally:
                        if temporary is not None:
                            temporary.unlink(missing_ok=True)
        except Exception as exc:
            raise RuntimeError(f"Unable to load or encode with model {model_name!r}: {exc}") from exc

    @staticmethod
    def _validate_embeddings(embeddings, count, dimension, np):
        if not isinstance(embeddings, np.ndarray) or embeddings.shape != (count, dimension):
            raise ValueError("embedding shape mismatch")
        if embeddings.dtype != np.float32:
            raise ValueError("embedding dtype must be float32")
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings contain non-finite values")

    def search(self, query, k=10):
        _check_k(k)
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        try:
            vector = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
            scores = self.embeddings @ vector
        except Exception as exc:
            raise RuntimeError(f"Dense query encoding failed: {exc}") from exc
        return sorted(((doc_id, float(score)) for doc_id, score in zip(self.ids, scores)),
                      key=lambda item: (-item[1], item[0]))[:k]


class HybridIndex:
    """Fuse candidate ranks, not raw scores. candidate_k applies to each retriever."""

    def __init__(self, bm25, dense, rrf_k=60, candidate_k=100):
        if not isinstance(rrf_k, (int, float)) or not math.isfinite(rrf_k) or rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        _check_k(candidate_k)
        self.bm25, self.dense = bm25, dense
        self.rrf_k, self.candidate_k = rrf_k, candidate_k

    def search(self, query, k=10):
        _check_k(k)
        if k > self.candidate_k:
            raise ValueError("k cannot exceed candidate_k")
        scores = defaultdict(float)
        for retriever in (self.bm25, self.dense):
            for rank, (doc_id, _) in enumerate(retriever.search(query, self.candidate_k), 1):
                scores[doc_id] += 1 / (self.rrf_k + rank)
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]

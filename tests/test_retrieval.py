"""Offline tests for ranking, fusion, and the public search command."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evidence_lab.retrieval import BM25Index, HybridIndex


class FixedIndex:
    def __init__(self, results):
        self.results = results
    def search(self, query, k=10):
        return self.results[:k]


class RetrievalTests(unittest.TestCase):
    def test_bm25_relevance_and_deterministic_ties(self):
        corpus = {
            "z": {"title": "", "text": "alpha beta"},
            "a": {"title": "", "text": "alpha beta"},
            "irrelevant": {"title": "", "text": "gamma delta"},
        }
        index = BM25Index(corpus)
        self.assertEqual([doc for doc, _ in index.search("alpha")], ["a", "z"])
        self.assertEqual(index.search("missing"), [])
        self.assertEqual(index.search("alpha", 1), index.search("alpha")[:1])
        self.assertEqual(index.search("ALPHA"), index.search("alpha"))
        with self.assertRaises(ValueError):
            index.search("alpha", 0)

    def test_bm25_indexes_titles(self):
        index = BM25Index({"title-hit": {"title": "vaccine", "text": "study"},
                           "other": {"text": "unrelated"}})
        self.assertEqual(index.search("vaccine")[0][0], "title-hit")

    def test_hybrid_rrf_rewards_agreement_and_resolves_ties(self):
        first = FixedIndex([("x", 100.0), ("shared", 1.0), ("z", 0.1)])
        second = FixedIndex([("z", -20.0), ("shared", -50.0), ("x", -100.0)])
        index = HybridIndex(first, second, rrf_k=1, candidate_k=3)
        scores = index.search("ignored", 3)
        self.assertEqual([doc for doc, _ in scores], ["x", "z", "shared"])
        self.assertAlmostEqual(scores[0][1], 1 / 2 + 1 / 4)
        self.assertAlmostEqual(scores[2][1], 2 / 3)
        with self.assertRaises(ValueError):
            index.search("ignored", 4)

    def test_cli_search_smoke_and_invalid_input(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "corpus.jsonl").write_text(json.dumps({"_id": "1", "title": "A", "text": "vaccine evidence"}) + "\n")
            command = [sys.executable, "-m", "evidence_lab.cli", "search", "vaccine", "--data", temp]
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertEqual((result["id"], result["title"]), ("1", "A"))
            self.assertGreater(result["score"], 0)
            invalid = subprocess.run(command + ["--k", "0"], text=True, capture_output=True, check=False)
            self.assertEqual(invalid.returncode, 1)
            self.assertIn("positive", invalid.stderr)

if __name__ == "__main__":
    unittest.main()

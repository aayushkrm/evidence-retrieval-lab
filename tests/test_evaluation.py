import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from evidence_lab.evaluation import evaluate, paired_bootstrap, query_metrics, run_queries
from scripts.evaluate import cross_split_duplicates, split_train

class EvaluationTests(unittest.TestCase):
    def test_graded_metrics_and_unjudged_document(self):
        scores = query_metrics(["unjudged", "a", "b"], {"a": 3, "b": 1, "c": 1}, 3)
        self.assertAlmostEqual(scores["recall"], 2 / 3)
        self.assertAlmostEqual(scores["mrr"], 0.5)
        self.assertAlmostEqual(scores["ndcg"], (7 / math.log2(3) + 1 / 2) / (7 + 1 / math.log2(3) + 1 / 2))

    def test_duplicate_does_not_inflate_recall(self):
        with self.assertRaises(ValueError):
            query_metrics(["a", "a", "b"], {"a": 1, "b": 1}, 2)

    def test_missing_result_counted_and_empty_judgments_excluded(self):
        report = evaluate({"q1": ["x"]}, {"q1": {"x": 1}, "q2": {"y": 1}, "q3": {"z": 0}}, (1,))
        self.assertEqual(report["query_count"], 2)
        self.assertEqual(report["excluded_queries_without_positive_judgments"], 1)
        self.assertEqual(report["metrics"]["recall@1"], 0.5)

    def test_bootstrap_is_paired_and_reproducible(self):
        a = {"1": {"m": 0.0}, "2": {"m": 0.5}}
        b = {"1": {"m": 0.25}, "2": {"m": 0.75}}
        report = paired_bootstrap(a, b, "m", samples=100, seed=9)
        self.assertEqual(report["mean_difference_b_minus_a"], 0.25)
        self.assertEqual(report["ci95_low"], 0.25)
        self.assertEqual(report, paired_bootstrap(a, b, "m", samples=100, seed=9))
        with self.assertRaises(ValueError):
            paired_bootstrap(a, {"1": {"m": 1}}, "m")

    def test_cross_split_normalized_overlap_and_labels(self):
        queries = {"train": "  Clinical   Trial\nWorks ", "test": "clinical trial works", "other": "unrelated"}
        pairs = cross_split_duplicates(queries, {"train": {"a": 1}}, {"test": {"a": 1}, "other": {"b": 1}})
        self.assertEqual(pairs, [{"train_query_id": "train", "test_query_id": "test", "same_relevance_judgments": True}])
        pairs = cross_split_duplicates(queries, {"train": {"a": 2}}, {"test": {"a": 1}})
        self.assertFalse(pairs[0]["same_relevance_judgments"])

    def test_sensitivity_report_preserves_full_test_result(self):
        from scripts.evaluate import main
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "qrels").mkdir()
            (base / "corpus.jsonl").write_text(''.join(json.dumps(x) + "\n" for x in [
                {"_id": "a", "text": "alpha trial"}, {"_id": "b", "text": "beta trial"}]))
            (base / "queries.jsonl").write_text(''.join(json.dumps(x) + "\n" for x in [
                {"_id": "tr", "text": "Alpha trial"}, {"_id": "t1", "text": "alpha trial"},
                {"_id": "t2", "text": "beta trial"}]))
            header = "query-id\tcorpus-id\tscore\n"
            (base / "qrels" / "train.tsv").write_text(header + "tr\ta\t1\n")
            (base / "qrels" / "test.tsv").write_text(header + "t1\ta\t1\nt2\tb\t1\n")
            out = base / "report.json"
            argv = ["evaluate", "--data", str(base), "--split", "test", "--final-test", "--overlap-sensitivity", "--systems", "bm25", "--output", str(out)]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            report = json.loads(out.read_text())
            self.assertEqual(report["systems"]["bm25"]["query_count"], 2)
            sensitivity = report["systems"]["bm25"]["excluding_train_text_overlap"]
            self.assertEqual(sensitivity["query_count"], 1)
            self.assertEqual(sensitivity["excluded_overlap_query_ids"], ["t1"])
            self.assertEqual(set(sensitivity["per_query"]), {"t2"})

    def test_query_split_no_leakage(self):
        qrels = {str(i): {"x": 1} for i in range(20)}
        train, dev = split_train(qrels)
        self.assertEqual(len(dev), 4)
        self.assertFalse(set(train) & set(dev))
        self.assertEqual(set(train) | set(dev), set(qrels))
        self.assertEqual((train, dev), split_train(qrels))

    def test_run_queries_rejects_missing_text(self):
        with self.assertRaises(ValueError):
            run_queries(lambda q, k: [], {}, {"q": {"d": 1}}, 1)

if __name__ == "__main__":
    unittest.main()

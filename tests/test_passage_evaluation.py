import unittest
from unittest.mock import patch
from evidence_lab.retrieval import BM25Index
from scripts.evaluate_passages import benchmark, build_systems


class PassageEvaluationTests(unittest.TestCase):
    def test_full_rank_aggregation_and_title_only_first_passage(self):
        corpus = {"long": {"title": "needle", "text": " ".join(["alpha"] * 330)},
                  "short": {"title": "", "text": "needle"}}
        with patch("scripts.evaluate_passages.BM25Index", wraps=BM25Index) as index_class:
            doc_search, passage_search, stats = build_systems(corpus)
            indexed = index_class.call_args_list[1].args[0]
        long_chunks = [p for p in indexed.values() if "alpha" in p["text"]]
        self.assertEqual(sum(p["title"] == "needle" for p in long_chunks), 1)
        self.assertGreater(stats["passage_count"], stats["document_count"])
        self.assertEqual({d for d, _ in doc_search("needle", k=2)}, {"long", "short"})
        ranked = passage_search("needle", k=2)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(len({doc for doc, _ in ranked}), 2)
        # A later chunk without the title contributes no title-only match.
        self.assertEqual(passage_search("alpha", k=2)[0][0], "long")

    def test_paired_report_uses_same_judged_queries(self):
        corpus = {"a": {"title": "", "text": "apple tree"}, "b": {"title": "", "text": "banana tree"}}
        result = benchmark(corpus, {"1": "apple", "2": "banana"}, {"1": {"a": 1}, "2": {"b": 1}}, bootstrap_samples=20)
        for name in ("document_bm25", "passage_bm25"):
            report = result["systems"][name]
            self.assertEqual(report["query_count"], 2)
            self.assertEqual(report["metrics"]["recall@10"], 1.0)
            self.assertEqual(set(report["per_query"]), {"1", "2"})
        self.assertEqual(result["comparisons_passage_minus_document"]["recall@10"]["query_count"], 2)
        self.assertEqual(result["comparisons_passage_minus_document"]["recall@10"]["mean_difference_b_minus_a"], 0)

if __name__ == "__main__": unittest.main()

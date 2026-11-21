import math
import unittest
from evidence_lab.passages import aggregate_passages, build_passages


class PassageTests(unittest.TestCase):
    def test_windows_overlap_and_source_offsets(self):
        text = "  one\ttwo\nthree four  five six seven   "
        chunks = list(build_passages({"doc": {"title": "Title", "text": text}}, chunk_words=3, overlap_words=1).values())
        self.assertEqual([c["text"] for c in chunks], ["one\ttwo\nthree", "three four  five", "five six seven"])
        self.assertEqual([c["chunk_index"] for c in chunks], [0, 1, 2])
        for c in chunks:
            self.assertEqual(text[c["start_char"]:c["end_char"]], c["text"])
            self.assertEqual(c["parent_id"], "doc")
            self.assertEqual(c["title"], "Title")
        self.assertEqual({word for c in chunks for word in c["text"].split()}, set(text.split()))

    def test_unicode_offsets_and_empty_documents(self):
        text = " 🧬  café\n你好  e\u0301  "
        chunks = build_passages({"u": {"text": text}, "empty": {"text": " \n\t"}}, chunk_words=2, overlap_words=1)
        self.assertEqual(len(chunks), 3)
        for c in chunks.values():
            self.assertEqual(text[c["start_char"]:c["end_char"]], c["text"])
            self.assertEqual(c["parent_id"], "u")
        self.assertEqual(build_passages({"x": {"text": ""}}), {})

    def test_stability_config_changes_and_collision_resistance(self):
        a = {"x:y": {"title": "z", "text": "one two three"}, "x": {"title": "y:z", "text": "one two three"}}
        first = build_passages(a, chunk_words=2, overlap_words=1)
        self.assertEqual(first, build_passages(dict(reversed(list(a.items()))), chunk_words=2, overlap_words=1))
        self.assertEqual(len(first), len(set(first)))
        self.assertEqual(len(first), 4)
        changed = build_passages(a, chunk_words=2, overlap_words=0)
        self.assertFalse(set(first) & set(changed))
        self.assertFalse(set(first) & set(build_passages({"x:y": {"title": "other", "text": "one two three"}}, chunk_words=2, overlap_words=1)))

    def test_long_text_has_no_missing_words_and_no_duplicate_tail(self):
        text = " ".join("word%d" % i for i in range(1003))
        chunks = list(build_passages({"d": {"text": text}}, chunk_words=100, overlap_words=20).values())
        self.assertTrue(all(len(c["text"].split()) <= 100 for c in chunks))
        self.assertEqual(chunks[-1]["text"].split()[-1], "word1002")
        self.assertEqual(set(text.split()), {w for chunk in chunks for w in chunk["text"].split()})
        self.assertTrue(all(len(c["text"].split()) > 20 for c in chunks))

    def test_streaming_window_matches_expected_for_many_boundaries(self):
        import re
        for total in range(1, 48):
            text = "  " + " \t".join(str(i) for i in range(total)) + "  "
            spans = list(re.finditer(r"\S+", text))
            for width, overlap in ((1, 0), (3, 0), (3, 2), (7, 3)):
                expected = []
                for start in range(0, total, width - overlap):
                    end = min(start + width, total)
                    expected.append((spans[start].start(), spans[end - 1].end()))
                    if end == total: break
                chunks = build_passages({"d": {"text": text}}, chunk_words=width, overlap_words=overlap)
                self.assertEqual([(p["start_char"], p["end_char"]) for p in chunks.values()], expected)

    def test_aggregate_best_passage_and_ties(self):
        passages = build_passages({"b": {"text": "one two three"}, "a": {"text": "four five"}}, chunk_words=2, overlap_words=1)
        by_parent = {}
        for pid, p in passages.items(): by_parent.setdefault(p["parent_id"], []).append(pid)
        a, b = by_parent["a"][0], by_parent["b"]
        results = aggregate_passages([(b[1], 2.0), (a, 2.0), (b[0], 2.0)], passages)
        self.assertEqual([r["parent_id"] for r in results], ["a", "b"])
        self.assertEqual(results[1]["passage_id"], min(b))
        selected = passages[results[1]["passage_id"]]
        self.assertEqual((results[1]["start_char"], results[1]["end_char"]), (selected["start_char"], selected["end_char"]))
        self.assertEqual(len(aggregate_passages([(a, 1), (b[0], 2)], passages, k=1)), 1)

    def test_invalid_inputs_and_unknown_rankings(self):
        corpus = {"x": {"text": "one"}}
        for args in ({"chunk_words": 0}, {"chunk_words": 2, "overlap_words": 2}, {"chunk_words": True}, {"overlap_words": -1}):
            with self.assertRaises(ValueError): build_passages(corpus, **args)
        with self.assertRaises(ValueError): build_passages({1: {"text": "a"}, "x": {"text": "b"}})
        passages = build_passages(corpus)
        pid = next(iter(passages))
        with self.assertRaises(ValueError): aggregate_passages([("missing", 1.0)], passages)
        with self.assertRaises(ValueError): aggregate_passages([(pid, 1.0), (pid, 2.0)], passages)
        for score in (math.nan, math.inf, "1"):
            with self.assertRaises(ValueError): aggregate_passages([(pid, score)], passages)

if __name__ == "__main__": unittest.main()

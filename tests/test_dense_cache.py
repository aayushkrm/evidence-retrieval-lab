"""Offline cache tests with deterministic, lightweight fake embeddings."""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None
from evidence_lab.retrieval import DenseIndex


class FakeModel:
    encodes = 0
    max_seq_length = 128
    def __init__(self, name, device, revision):
        pass
    def get_sentence_embedding_dimension(self):
        return 2
    def encode(self, texts, **kwargs):
        type(self).encodes += 1
        return np.array([[1.0, 0.0] if "alpha" in t else [0.0, 1.0] for t in texts], dtype=np.float32)


@unittest.skipIf(np is None, "optional numpy dependency is not installed")
class DenseCacheTests(unittest.TestCase):
    def setUp(self):
        FakeModel.encodes = 0
        self.fake = patch.dict(sys.modules, {"sentence_transformers": types.SimpleNamespace(SentenceTransformer=FakeModel)})
        self.fake.start()
        self.addCleanup(self.fake.stop)
        self.corpus = {"1": {"title": "", "text": "alpha"}, "2": {"text": "beta"}}

    def test_opt_in_cache_reuses_embeddings_and_results(self):
        with tempfile.TemporaryDirectory() as directory:
            first = DenseIndex(self.corpus, cache_dir=directory)
            self.assertFalse(first.cache_hit)
            self.assertEqual(FakeModel.encodes, 1)
            second = DenseIndex(self.corpus, cache_dir=directory)
            self.assertTrue(second.cache_hit)
            self.assertEqual(FakeModel.encodes, 1)
            self.assertEqual(first.search("alpha"), second.search("alpha"))
            self.assertEqual(first.search("alpha")[0][0], "1")
            self.assertEqual(len(list(Path(directory).glob("*.npz"))), 1)
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_identity_changes_with_corpus_id_content_model_revision_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            base = DenseIndex(self.corpus, cache_dir=directory)
            variants = [
                DenseIndex({"3": {"text": "alpha"}, "2": {"text": "beta"}}, cache_dir=directory),
                DenseIndex({"1": {"text": "changed"}, "2": {"text": "beta"}}, cache_dir=directory),
                DenseIndex(self.corpus, model_name="other", cache_dir=directory),
                DenseIndex(self.corpus, revision="a" * 40, cache_dir=directory),
                DenseIndex(self.corpus, batch_size=1, cache_dir=directory),
            ]
            self.assertTrue(all(not index.cache_hit and index.cache_key != base.cache_key for index in variants))
            self.assertEqual(len({base.cache_key, *(index.cache_key for index in variants)}), 6)

    def test_dependency_versions_are_part_of_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("evidence_lab.retrieval.importlib.metadata.version", return_value="v1"):
                first = DenseIndex(self.corpus, cache_dir=directory)
            for package in ("sentence-transformers", "transformers", "tokenizers", "torch", "numpy"):
                with self.subTest(package=package), patch("evidence_lab.retrieval.importlib.metadata.version", side_effect=lambda name: "v2" if name == package else "v1"):
                    changed = DenseIndex(self.corpus, cache_dir=directory)
                    self.assertNotEqual(first.cache_key, changed.cache_key)
                    self.assertFalse(changed.cache_hit)

    def test_cache_rejects_mutable_revision_and_local_model(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "immutable"):
                DenseIndex(self.corpus, revision="main", cache_dir=directory)
            with self.assertRaisesRegex(ValueError, "local model"):
                DenseIndex(self.corpus, model_name=directory, cache_dir=directory)

    def test_invalid_cache_is_rejected_without_reencoding(self):
        for kind in ("shape", "dtype", "nonfinite", "metadata", "pickle"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                FakeModel.encodes = 0
                first = DenseIndex(self.corpus, cache_dir=directory)
                path = Path(directory) / (first.cache_key + ".npz")
                key = first.cache_key if kind != "metadata" else "wrong"
                arr = np.ones((2, 2), dtype=np.float32)
                if kind == "shape": arr = arr[:1]
                if kind == "dtype": arr = arr.astype(np.float64)
                if kind == "nonfinite": arr[0, 0] = np.nan
                if kind == "pickle": arr = np.array([[object(), object()], [object(), object()]], dtype=object)
                np.savez(path, key=np.array(key), embeddings=arr)
                with self.assertRaisesRegex(RuntimeError, "Invalid dense embedding cache"):
                    DenseIndex(self.corpus, cache_dir=directory)
                self.assertEqual(FakeModel.encodes, 1)

    def test_cache_disabled_by_default(self):
        DenseIndex(self.corpus)
        DenseIndex(self.corpus)
        self.assertEqual(FakeModel.encodes, 2)

if __name__ == "__main__":
    unittest.main()

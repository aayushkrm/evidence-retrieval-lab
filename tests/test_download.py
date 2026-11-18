"""Exercise archive verification without downloading or executing external data."""
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from evidence_lab.download import REQUIRED, download_scifact


def archive_bytes(missing=None):
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        for name in REQUIRED:
            if name != missing:
                archive.writestr("scifact/" + name, "fixture " + name)
        archive.writestr("scifact/../../escape.txt", "must not extract")
        archive.writestr("scifact/unexpected.txt", "must not extract")
    return stream.getvalue()


class DownloadTests(unittest.TestCase):
    def test_verified_archive_extracts_only_allowlisted_paths(self):
        content = archive_bytes()
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "data" / "scifact"
            with patch("evidence_lab.download.urlopen", return_value=io.BytesIO(content)), patch("evidence_lab.download.MD5", hashlib.md5(content).hexdigest()):
                self.assertEqual(download_scifact(target), target)
            for name in REQUIRED:
                self.assertEqual((target / name).read_text(), "fixture " + name)
            self.assertFalse((Path(temp) / "escape.txt").exists())
            self.assertFalse((target / "unexpected.txt").exists())
            self.assertFalse(any(p.name == "escape.txt" for p in Path(temp).rglob("*")))

    def test_checksum_mismatch_leaves_no_destination(self):
        content = archive_bytes()
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "scifact"
            with patch("evidence_lab.download.urlopen", return_value=io.BytesIO(content)), patch("evidence_lab.download.MD5", "0" * 32):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    download_scifact(target)
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_missing_member_leaves_no_destination(self):
        content = archive_bytes(missing="queries.jsonl")
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "scifact"
            with patch("evidence_lab.download.urlopen", return_value=io.BytesIO(content)), patch("evidence_lab.download.MD5", hashlib.md5(content).hexdigest()):
                with self.assertRaisesRegex(RuntimeError, "extract"):
                    download_scifact(target)
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_existing_destination_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "scifact"
            target.mkdir()
            (target / "sentinel").write_text("keep")
            with patch("evidence_lab.download.urlopen") as request:
                with self.assertRaises(FileExistsError):
                    download_scifact(target)
            request.assert_not_called()
            self.assertEqual((target / "sentinel").read_text(), "keep")

if __name__ == "__main__":
    unittest.main()

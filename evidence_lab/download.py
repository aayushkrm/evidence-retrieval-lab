"""Download and verify the official BEIR SciFact release."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen
from zipfile import ZipFile, BadZipFile

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
# Published in the BEIR dataset listing. MD5 detects accidental changes; it is not authentication.
MD5 = "5f7d1de60b170fc8027bb7898e2efca1"
REQUIRED = ("corpus.jsonl", "queries.jsonl", "qrels/train.tsv", "qrels/test.tsv")


def download_scifact(destination="data/scifact"):
    """Download to a temporary file, verify BEIR's published checksum, then extract known files."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}; choose a new path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        root = Path(temp)
        archive = root / "scifact.zip"
        digest = hashlib.md5()  # nosec: comparison with published dataset checksum, not security
        try:
            request = Request(URL, headers={"User-Agent": "evidence-retrieval-lab/0.1"})
            with urlopen(request, timeout=60) as response, archive.open("wb") as out:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > 100 * 1024 * 1024:
                        raise ValueError("Download exceeds 100 MiB limit")
                    digest.update(chunk)
                    out.write(chunk)
            if digest.hexdigest() != MD5:
                raise ValueError(f"SciFact checksum mismatch: got {digest.hexdigest()}, expected {MD5}")
            extracted = root / "output"
            extracted.mkdir()
            with ZipFile(archive) as zf:
                for relative in REQUIRED:
                    member = "scifact/" + relative
                    info = zf.getinfo(member)
                    if info.is_dir() or info.file_size > 100 * 1024 * 1024:
                        raise ValueError(f"Invalid archive member: {member}")
                    target = extracted / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as source, target.open("wb") as out:
                        shutil.copyfileobj(source, out)
            extracted.rename(destination)
        except (OSError, BadZipFile, KeyError) as exc:
            raise RuntimeError(f"Could not download or extract SciFact: {exc}") from exc
    return destination

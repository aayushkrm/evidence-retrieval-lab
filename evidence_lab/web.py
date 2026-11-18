"""Run a local, read-only search demo over a BEIR-format corpus."""
from __future__ import annotations

import argparse
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .data import load_corpus
from .retrieval import BM25Index


def render(query: str, results: list[tuple[str, float]], corpus: dict) -> str:
    """Render only escaped corpus and request data; document text is untrusted."""
    h = html.escape
    cards = []
    for rank, (doc_id, score) in enumerate(results, 1):
        doc = corpus[doc_id]
        cards.append(
            '<article><div class="meta">'
            f'RANK {rank} · DOCUMENT {h(doc_id)} · BM25 {score:.3f}'
            f'</div><h2>{h(doc["title"] or "Untitled document")}</h2>'
            f'<p>{h(doc["text"])}</p></article>'
        )
    if query and not results:
        cards.append('<article><h2>No matching documents</h2><p>Try different terms.</p></article>')
    body = "".join(cards)
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Evidence Retrieval Lab</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f5f6f3;color:#142b2a;font:16px/1.55 system-ui,sans-serif}}
main{{max-width:860px;margin:auto;padding:52px 24px}}.eyebrow,.meta{{font-size:12px;font-weight:700;letter-spacing:.1em;color:#55766c}}
h1{{font-size:clamp(32px,6vw,50px);line-height:1.08;margin:12px 0}}.intro{{max-width:680px;color:#425a54}}
form{{display:flex;gap:10px;margin:28px 0}}input{{flex:1;min-width:0;padding:14px;border:1px solid #b6c8be;border-radius:8px;font:inherit}}
button{{border:0;border-radius:8px;padding:12px 22px;background:#174b3b;color:white;font:inherit;font-weight:700;cursor:pointer}}
article{{background:white;border:1px solid #dbe5df;border-radius:12px;padding:22px;margin:14px 0}}h2{{font-size:20px;line-height:1.3;margin:9px 0}}article p{{color:#344941;white-space:pre-wrap}}.notice{{font-size:14px;color:#52635e}}@media(max-width:520px){{form{{flex-direction:column}}}}
</style></head><body><main><div class="eyebrow">EVIDENCE RETRIEVAL LAB</div>
<h1>Search the research corpus.</h1><p class="intro">Explore document retrieval with BM25 keyword search. Results are ranked matches, not verified answers or medical advice.</p>
<form action="/" method="get"><label for="query" hidden>Search query</label><input id="query" name="q" maxlength="500" value="{h(query, quote=True)}" placeholder="Enter a scientific claim or question" required><button>Search</button></form>
<p class="notice">Local demo · Up to 10 documents · Corpus content may be incomplete or outdated</p>{body}</main></body></html>'''


def make_handler(corpus: dict, index: BM25Index):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # The default HTTP log includes the full query string.
            pass

        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path != "/":
                self.send_error(404)
                return
            query = parse_qs(parsed.query).get("q", [""])[0].strip()
            if len(query) > 500:
                self.send_error(400, "Query exceeds 500 characters")
                return
            results = index.search(query, k=10) if query else []
            page = render(query, results, corpus).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(page)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/scifact/corpus.jsonl"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    corpus = load_corpus(args.corpus)
    index = BM25Index(corpus)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(corpus, index))
    print(f"Search demo at http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

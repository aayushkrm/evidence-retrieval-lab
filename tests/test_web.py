import unittest
from urllib.parse import quote
from urllib.request import urlopen
from threading import Thread
from http.server import ThreadingHTTPServer

from evidence_lab.retrieval import BM25Index
from evidence_lab.web import make_handler, render


class WebTests(unittest.TestCase):
    def test_untrusted_content_is_escaped(self):
        corpus = {"1": {"title": "<script>alert(1)</script>", "text": "<b>not markup</b>"}}
        page = render('" onfocus="alert(1)', [("1", 1.0)], corpus)
        self.assertNotIn("<script>", page)
        self.assertNotIn("<b>not markup</b>", page)
        self.assertIn("&lt;script&gt;", page)

    def test_search_over_local_http(self):
        corpus = {"1": {"title": "Heart study", "text": "Heart cells and evidence"}}
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(corpus, BM25Index(corpus)))
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urlopen(f"http://127.0.0.1:{server.server_port}/?q={quote('heart')}") as response:
                self.assertEqual(response.status, 200)
                self.assertIn("Heart study", response.read().decode())
                self.assertIn("default-src 'none'", response.headers["Content-Security-Policy"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()

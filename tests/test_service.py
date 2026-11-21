import asyncio
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from evidence_lab.service import BoundedBodyMiddleware, create_app


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "workspace.sqlite3"
        self.client = TestClient(create_app(self.path), base_url="http://127.0.0.1")
        self.base = "/v1/tenants/alice/collections/research"

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def put(self, docs, expected=None, base=None):
        payload = {"documents": docs}
        if expected is not None:
            payload["expected_generation"] = expected
        return self.client.post((base or self.base) + "/documents", json=payload)

    def search(self, query, base=None):
        return self.client.post((base or self.base) + "/search", json={"query": query})

    def test_ingest_search_update_delete_and_restart(self):
        doc = {"id": "paper-1", "title": "Research", "text": "alpha beta gamma"}
        result = self.put([doc], 0)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["generation"], 1)
        first = self.search("alpha").json()
        self.assertEqual(first["results"][0]["parent_id"], "paper-1")
        passage = first["results"][0]
        self.assertEqual(doc["text"][passage["start_char"]:passage["end_char"]], passage["text"])
        self.assertFalse(first["cache_hit"])
        self.assertTrue(self.search("alpha").json()["cache_hit"])
        self.assertEqual(self.put([{"id": "paper-1", "text": "delta evidence"}], 1).json()["generation"], 2)
        self.assertEqual(self.search("alpha").json()["results"], [])
        self.assertEqual(self.search("delta").json()["generation"], 2)
        again = TestClient(create_app(self.path), base_url="http://127.0.0.1")
        try:
            self.assertEqual(again.post(self.base + "/search", json={"query": "delta"}).json()["results"][0]["parent_id"], "paper-1")
        finally:
            again.close()
        deleted = self.client.post(self.base + "/delete", json={"ids": ["paper-1"], "expected_generation": 2})
        self.assertEqual(deleted.json()["generation"], 3)
        self.assertEqual(self.search("delta").json()["results"], [])
        self.assertEqual([e["operation"] for e in self.client.get(self.base + "/audit").json()["events"]], ["delete", "upsert", "upsert"])

    def test_scoped_collection_and_conflicts(self):
        self.put([{"id": "x", "text": "alpha secret"}])
        other = "/v1/tenants/bob/collections/research"
        self.put([{"id": "x", "text": "beta document"}], base=other)
        self.assertEqual(self.search("alpha", other).json()["results"], [])
        self.assertEqual(self.search("beta").json()["results"], [])
        conflict = self.put([{"id": "y", "text": "stale"}], 0)
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(self.client.get(self.base).json()["document_count"], 1)
        self.assertEqual(self.client.get("/v1/tenants/bob/collections").json()["collections"][0]["name"], "research")

    def test_validation_and_atomic_batch(self):
        self.put([{"id": "x", "text": "original"}])
        bad = self.put([{"id": "ok", "text": "valid"}, {"id": "bad", "text": ""}])
        self.assertEqual(bad.status_code, 422)
        self.assertEqual(self.client.get(self.base).json()["document_count"], 1)
        self.assertEqual(self.search(" ").status_code, 422)
        self.assertEqual(self.client.post(self.base + "/search", json={"query": "x", "limit": 51}).status_code, 422)
        self.assertEqual(self.put([{"id": "x", "text": "x", "extra": "bad"}]).status_code, 422)
        self.assertEqual(self.client.get("/health", headers={"host": "attacker.example"}).status_code, 400)
        self.assertEqual(self.client.get("/health", headers={"origin": "https://attacker.example"}).status_code, 403)

    def test_search_limit_includes_title_bytes(self):
        self.put([{"id": "a", "title": "long title", "text": "x"}])
        with patch("evidence_lab.service.MAX_SEARCH_BYTES", 5):
            response = self.search("x")
        self.assertEqual(response.status_code, 413)

    def test_parallel_search_reads_consistent_results(self):
        self.put([{"id": "a", "text": "concurrent search example"}])
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(lambda _: self.search("concurrent").json(), range(8)))
        self.assertTrue(all(r["generation"] == 1 and r["results"][0]["parent_id"] == "a" for r in responses))

    def test_search_during_updates_reports_snapshot_generation(self):
        self.put([{"id": "a", "text": "shared odd content"}])

        def update():
            for generation in range(2, 12):
                word = "even" if generation % 2 == 0 else "odd"
                response = self.put([{"id": "a", "text": f"shared {word} content"}])
                self.assertEqual(response.status_code, 200)

        def searches():
            for _ in range(20):
                result = self.search("shared").json()
                word = "even" if result["generation"] % 2 == 0 else "odd"
                self.assertIn(word, result["results"][0]["text"])

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(update), pool.submit(searches), pool.submit(searches)]
            for future in futures:
                future.result()

    def test_streamed_body_limit_does_not_trust_header(self):
        async def exercise():
            chunks = iter([
                {"type": "http.request", "body": b"a" * 7_000_000, "more_body": True},
                {"type": "http.request", "body": b"b" * 5_000_000, "more_body": False},
            ])
            messages = []
            async def receive():
                return next(chunks)
            async def send(message):
                messages.append(message)
            async def inner(scope, receive, send):
                self.fail("oversized request reached application")
            await BoundedBodyMiddleware(inner)({"type": "http", "method": "POST", "headers": [(b"host", b"127.0.0.1")]}, receive, send)
            self.assertEqual(messages[0]["status"], 413)
        asyncio.run(exercise())

    def test_fragment_limit_duplicate_host_and_timeout(self):
        async def exercise():
            async def inner(scope, receive, send):
                self.fail("request reached application")
            scope = {"type": "http", "method": "POST", "headers": [(b"host", b"localhost")]}
            async def run(receive, request_scope=scope):
                messages = []
                async def send(message):
                    messages.append(message)
                await BoundedBodyMiddleware(inner)(request_scope, receive, send)
                return messages[0]["status"]
            async def empty_fragment():
                return {"type": "http.request", "body": b"", "more_body": True}
            with patch("evidence_lab.service.MAX_REQUEST_EVENTS", 3):
                self.assertEqual(await run(empty_fragment), 413)
            duplicate = {"type": "http", "headers": [(b"host", b"localhost"), (b"host", b"localhost")]}
            self.assertEqual(await run(empty_fragment, duplicate), 400)
            async def timeout(*args, **kwargs):
                args[0].close()
                raise asyncio.TimeoutError()
            with patch("evidence_lab.service.asyncio.wait_for", timeout):
                self.assertEqual(await run(empty_fragment), 408)
        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()

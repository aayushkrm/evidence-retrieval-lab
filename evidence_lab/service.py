"""Local collection API. Namespaces are not authentication or access control."""
from __future__ import annotations

from collections import OrderedDict
import asyncio
import os
from pathlib import Path
from threading import Lock, Semaphore
from time import perf_counter
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .passages import aggregate_passages, build_passages
from .retrieval import BM25Index
from .workspace import GenerationConflict, SnapshotLimitExceeded, Workspace

MAX_REQUEST_BYTES = 11_000_000
MAX_SEARCH_BYTES = 2_000_000
MAX_REQUEST_EVENTS = 4096


class BoundedBodyMiddleware:
    """Bound received bytes, including requests without a Content-Length header."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = [(k.lower(), v) for k, v in scope.get("headers", [])]
        hosts = [v.decode("latin-1") for k, v in headers if k == b"host"]
        origins = [v.decode("latin-1") for k, v in headers if k == b"origin"]
        try:
            host = urlsplit("//" + hosts[0]) if len(hosts) == 1 else None
            valid_host = (host is not None and host.hostname in ("localhost", "127.0.0.1")
                          and host.username is None and host.password is None
                          and not host.path and not host.query and not host.fragment
                          and (host.port is None or 1 <= host.port <= 65535))
        except ValueError:
            valid_host = False
        if not valid_host:
            return await JSONResponse(status_code=400, content={"detail": "Invalid Host"})(scope, receive, send)
        if len(origins) > 1 or (origins and origins[0].lower() != f"{scope.get('scheme', 'http')}://{hosts[0]}".lower()):
            return await JSONResponse(status_code=403, content={"detail": "Cross-origin request denied"})(scope, receive, send)
        chunks, size, events = [], 0, 0
        deadline = asyncio.get_running_loop().time() + 30
        while True:
            try:
                remaining = deadline - asyncio.get_running_loop().time()
                message = await asyncio.wait_for(receive(), timeout=max(0, remaining))
            except asyncio.TimeoutError:
                return await JSONResponse(status_code=408, content={"detail": "Request timeout"})(scope, receive, send)
            events += 1
            if events > MAX_REQUEST_EVENTS:
                return await JSONResponse(status_code=413, content={"detail": "Too many request fragments"})(scope, receive, send)
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > MAX_REQUEST_BYTES:
                response = JSONResponse(status_code=413, content={"detail": "Request body too large"})
                return await response(scope, receive, send)
            if chunk:
                chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        return await self.app(scope, replay, send)


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str
    title: str = ""
    text: str


class UpsertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    documents: list[Document] = Field(min_length=1, max_length=500)
    expected_generation: int | None = Field(default=None, ge=0)


class DeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    ids: list[str] = Field(min_length=1, max_length=500)
    expected_generation: int | None = Field(default=None, ge=0)


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=10, ge=1, le=50)


def create_app(db_path: str | Path) -> FastAPI:
    """Create an app with a trusted local SQLite path and generation-aware indexes."""
    workspace = Workspace(db_path)
    app = FastAPI(title="Evidence Retrieval Lab", version="0.2.0")
    app.add_middleware(BoundedBodyMiddleware)
    cache: OrderedDict[tuple[str, str, int], tuple] = OrderedDict()
    cache_lock = Lock()
    build_slots = Semaphore(2)
    search_slots = Semaphore(4)

    @app.exception_handler(GenerationConflict)
    async def generation_conflict(_request: Request, exc: GenerationConflict):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(SnapshotLimitExceeded)
    async def snapshot_limit(_request: Request, exc: SnapshotLimitExceeded):
        return JSONResponse(status_code=413, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def invalid_value(_request: Request, exc: ValueError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health")
    def health():
        return {"status": "ok", "scope": "local prototype"}

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard():
        page = Path(__file__).with_name("dashboard.html")
        if not page.exists():
            raise HTTPException(status_code=503, detail="Dashboard is not installed")
        return HTMLResponse(page.read_text(encoding="utf-8"), headers={
            "Content-Security-Policy": "default-src 'self'; style-src 'unsafe-inline'; script-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'none'",
            "X-Content-Type-Options": "nosniff",
        })

    @app.get("/v1/tenants/{tenant}/collections")
    def collections(tenant: str):
        return {"collections": workspace.list_collections(tenant)}

    @app.get("/v1/tenants/{tenant}/collections/{collection}")
    def collection_info(tenant: str, collection: str):
        return workspace.collection_info(tenant, collection)

    @app.post("/v1/tenants/{tenant}/collections/{collection}/documents")
    def upsert(tenant: str, collection: str, request: UpsertRequest):
        return workspace.upsert(tenant, collection, [d.model_dump() for d in request.documents],
                                expected_generation=request.expected_generation)

    @app.post("/v1/tenants/{tenant}/collections/{collection}/delete")
    def delete(tenant: str, collection: str, request: DeleteRequest):
        return workspace.delete(tenant, collection, request.ids, expected_generation=request.expected_generation)

    @app.get("/v1/tenants/{tenant}/collections/{collection}/audit")
    def audit(tenant: str, collection: str, limit: int = Query(default=100, ge=1, le=1000)):
        return {"events": workspace.audit(tenant, collection, limit=limit)}

    @app.post("/v1/tenants/{tenant}/collections/{collection}/search")
    def search(tenant: str, collection: str, request: SearchRequest):
        if not request.query.strip():
            raise ValueError("query must not be blank")
        with search_slots:
            return perform_search(tenant, collection, request)

    def perform_search(tenant: str, collection: str, request: SearchRequest):
        started = perf_counter()
        snap = workspace.snapshot(tenant, collection, max_bytes=MAX_SEARCH_BYTES)
        docs = snap["documents"]
        key = (tenant, collection, snap["generation"])
        with cache_lock:
            item = cache.get(key)
            cache_hit = item is not None
            if item is not None:
                cache.move_to_end(key)
        if item is None and docs:
            with build_slots:
                with cache_lock:
                    item = cache.get(key)
                    if item is not None:
                        cache.move_to_end(key)
                        cache_hit = True
                if item is None:
                    passages = build_passages(docs)
                    searchable = {pid: {"title": p["title"] if p["chunk_index"] == 0 else "", "text": p["text"]}
                                  for pid, p in passages.items()}
                    item = (passages, BM25Index(searchable)) if passages else None
                    if item is not None:
                        with cache_lock:
                            cache[key] = item
                            while len(cache) > 4:
                                cache.popitem(last=False)
        results = []
        if item:
            passages, index = item
            results = aggregate_passages(index.search(request.query, k=len(passages)), passages, k=request.limit)
        return {"tenant": tenant, "collection": collection, "generation": snap["generation"],
                "method": "bm25-passages", "document_count": len(docs), "cache_hit": cache_hit,
                "elapsed_ms": round((perf_counter() - started) * 1000, 3),
                "note": "Title matches may select the first passage; excerpts are not verified evidence.", "results": results}

    return app


def main():
    import uvicorn
    db_path = os.environ.get("EVIDENCE_LAB_DB", "data/workspace.sqlite3")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    uvicorn.run(create_app(db_path), host="127.0.0.1", port=8766, access_log=False)


if __name__ == "__main__":
    main()

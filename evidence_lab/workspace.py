"""Tenant-scoped local document collections with atomic, versioned snapshots.

The database path is a trusted server configuration, never document input. Documents
are supplied as text; this module does not open document paths or execute content.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path

_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$', re.ASCII)
MAX_BATCH = 500
MAX_TEXT_BYTES = 1_000_000
MAX_TITLE_BYTES = 4096
MAX_BATCH_BYTES = 10_000_000
MAX_COLLECTION_DOCUMENTS = 10_000
MAX_COLLECTION_BYTES = 100_000_000


class GenerationConflict(ValueError):
    """The caller's snapshot generation is no longer current."""


class SnapshotLimitExceeded(ValueError):
    """The requested snapshot exceeds its maximum document byte size."""


def _name(value, label):
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError(f'{label} must be 1–64 ASCII letters, digits, dot, underscore, or hyphen; start with a letter or digit')
    return value


def _id(value):
    if not isinstance(value, str) or not value or value != value.strip() or '\x00' in value:
        raise ValueError('document id must be nonempty, trimmed, NUL-free, and at most 256 UTF-8 bytes')
    try:
        valid = len(value.encode('utf-8')) <= 256
    except UnicodeEncodeError as exc:
        raise ValueError('document id must be valid UTF-8') from exc
    if not valid:
        raise ValueError('document id exceeds 256 UTF-8 bytes')
    return value


def _generation(value):
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
        raise ValueError('expected_generation must be a nonnegative integer')


def _document(doc):
    if not isinstance(doc, dict) or set(doc) - {'id', 'title', 'text'} or 'id' not in doc or 'text' not in doc:
        raise ValueError('document must contain id and text, optional title, and no other fields')
    doc_id = _id(doc['id'])
    title, text = doc.get('title', ''), doc['text']
    if not isinstance(title, str) or not isinstance(text, str) or not text.strip() or '\x00' in title or '\x00' in text:
        raise ValueError('title and nonempty text must be strings without NUL characters')
    try:
        title_bytes, text_bytes = title.encode('utf-8'), text.encode('utf-8')
    except UnicodeEncodeError as exc:
        raise ValueError('title and text must be valid UTF-8') from exc
    if len(title_bytes) > MAX_TITLE_BYTES or len(text_bytes) > MAX_TEXT_BYTES:
        raise ValueError('document exceeds title or text byte limit')
    canonical = json.dumps([title, text], ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return doc_id, title, text, hashlib.sha256(canonical).hexdigest(), len(title_bytes) + len(text_bytes)


class Workspace:
    """SQLite state; safe to share between threads because each call has its own connection."""

    def __init__(self, db_path):
        self.path = str(Path(db_path).expanduser().absolute())
        if str(db_path) == ':memory:':
            raise ValueError('use a persistent database path, not :memory:')
        with self._connect() as connection:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS collections (
                    tenant TEXT NOT NULL, name TEXT NOT NULL, generation INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (tenant, name));
                CREATE TABLE IF NOT EXISTS documents (
                    tenant TEXT NOT NULL, collection TEXT NOT NULL, id TEXT NOT NULL,
                    title TEXT NOT NULL, text TEXT NOT NULL, content_id TEXT NOT NULL,
                    PRIMARY KEY (tenant, collection, id),
                    FOREIGN KEY (tenant, collection) REFERENCES collections(tenant, name));
                CREATE TABLE IF NOT EXISTS audit_log (
                    tenant TEXT NOT NULL, collection TEXT NOT NULL, generation INTEGER NOT NULL,
                    operation TEXT NOT NULL, changed INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                    PRIMARY KEY (tenant, collection, generation),
                    FOREIGN KEY (tenant, collection) REFERENCES collections(tenant, name));
            ''')

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute('PRAGMA foreign_keys=ON')
            connection.execute('PRAGMA busy_timeout=30000')
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _current(connection, tenant, collection):
        row = connection.execute('SELECT generation FROM collections WHERE tenant=? AND name=?', (tenant, collection)).fetchone()
        return row['generation'] if row else 0

    @staticmethod
    def _check(current, expected):
        if expected is not None and expected != current:
            raise GenerationConflict(f'expected generation {expected}, current generation {current}')

    @staticmethod
    def _advance(connection, tenant, collection, current, operation, changed):
        if not changed:
            return current
        generation = current + 1
        connection.execute('INSERT INTO collections(tenant,name,generation) VALUES(?,?,?) '
                           'ON CONFLICT(tenant,name) DO UPDATE SET generation=excluded.generation',
                           (tenant, collection, generation))
        connection.execute('INSERT INTO audit_log(tenant,collection,generation,operation,changed) VALUES(?,?,?,?,?)',
                           (tenant, collection, generation, operation, changed))
        return generation

    def upsert(self, tenant, collection, documents, *, expected_generation=None):
        tenant, collection = _name(tenant, 'tenant'), _name(collection, 'collection')
        _generation(expected_generation)
        if not isinstance(documents, list) or not 1 <= len(documents) <= MAX_BATCH:
            raise ValueError(f'documents must be a list of 1–{MAX_BATCH} documents')
        parsed = [_document(doc) for doc in documents]
        if len({doc[0] for doc in parsed}) != len(parsed):
            raise ValueError('duplicate document id in batch')
        if sum(doc[4] for doc in parsed) > MAX_BATCH_BYTES:
            raise ValueError('batch exceeds byte limit')
        with self._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            try:
                current = self._current(conn, tenant, collection)
                self._check(current, expected_generation)
                # Insert collection before documents for foreign key enforcement.
                conn.execute('INSERT OR IGNORE INTO collections(tenant,name,generation) VALUES(?,?,0)', (tenant, collection))
                changed = 0
                for doc_id, title, text, content_id, _ in parsed:
                    cursor = conn.execute('''INSERT INTO documents(tenant,collection,id,title,text,content_id)
                        VALUES(?,?,?,?,?,?) ON CONFLICT(tenant,collection,id) DO UPDATE SET
                        title=excluded.title,text=excluded.text,content_id=excluded.content_id
                        WHERE documents.content_id != excluded.content_id''',
                        (tenant, collection, doc_id, title, text, content_id))
                    changed += cursor.rowcount
                limits = conn.execute('''SELECT COUNT(*) AS n,
                    COALESCE(SUM(LENGTH(CAST(title AS BLOB)) + LENGTH(CAST(text AS BLOB))), 0) AS bytes
                    FROM documents WHERE tenant=? AND collection=?''', (tenant, collection)).fetchone()
                if limits['n'] > MAX_COLLECTION_DOCUMENTS or limits['bytes'] > MAX_COLLECTION_BYTES:
                    raise ValueError('collection exceeds document count or byte limit')
                generation = self._advance(conn, tenant, collection, current, 'upsert', changed)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return {'generation': generation, 'changed': changed,
                'documents': [{'id': d[0], 'content_id': d[3]} for d in parsed]}

    def delete(self, tenant, collection, ids, *, expected_generation=None):
        tenant, collection = _name(tenant, 'tenant'), _name(collection, 'collection')
        _generation(expected_generation)
        if not isinstance(ids, list) or not 1 <= len(ids) <= MAX_BATCH:
            raise ValueError(f'ids must be a list of 1–{MAX_BATCH} document ids')
        ids = [_id(value) for value in ids]
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate document id in batch')
        with self._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            try:
                current = self._current(conn, tenant, collection)
                self._check(current, expected_generation)
                changed = sum(conn.execute('DELETE FROM documents WHERE tenant=? AND collection=? AND id=?',
                                           (tenant, collection, doc_id)).rowcount for doc_id in ids)
                generation = self._advance(conn, tenant, collection, current, 'delete', changed)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return {'generation': generation, 'changed': changed}

    def snapshot(self, tenant, collection, *, max_bytes=None):
        """Read one snapshot, optionally limiting the UTF-8 bytes in titles and text."""
        tenant, collection = _name(tenant, 'tenant'), _name(collection, 'collection')
        if max_bytes is not None and (not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 0):
            raise ValueError('max_bytes must be a nonnegative integer or None')
        with self._connect() as conn:
            conn.execute('BEGIN')
            try:
                generation = self._current(conn, tenant, collection)
                if max_bytes is not None:
                    size = conn.execute('''SELECT COALESCE(SUM(LENGTH(CAST(title AS BLOB)) +
                        LENGTH(CAST(text AS BLOB))), 0) FROM documents WHERE tenant=? AND collection=?''',
                                        (tenant, collection)).fetchone()[0]
                    if size > max_bytes:
                        raise SnapshotLimitExceeded('snapshot exceeds byte limit')
                rows = conn.execute('SELECT id,title,text,content_id FROM documents WHERE tenant=? AND collection=? ORDER BY id',
                                    (tenant, collection)).fetchall()
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return {'generation': generation, 'documents': {row['id']: {'title': row['title'], 'text': row['text'],
                                                                      'content_id': row['content_id']} for row in rows}}

    def collection_info(self, tenant, collection):
        """Return a consistent generation and document count without loading text."""
        tenant, collection = _name(tenant, 'tenant'), _name(collection, 'collection')
        with self._connect() as conn:
            conn.execute('BEGIN')
            try:
                generation = self._current(conn, tenant, collection)
                count = conn.execute('SELECT COUNT(*) FROM documents WHERE tenant=? AND collection=?',
                                     (tenant, collection)).fetchone()[0]
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return {'generation': generation, 'document_count': count}

    def list_collections(self, tenant):
        tenant = _name(tenant, 'tenant')
        with self._connect() as conn:
            rows = conn.execute('''SELECT c.name,c.generation,COUNT(d.id) AS document_count FROM collections c
                LEFT JOIN documents d ON d.tenant=c.tenant AND d.collection=c.name
                WHERE c.tenant=? GROUP BY c.name,c.generation ORDER BY c.name''', (tenant,)).fetchall()
        return [dict(row) for row in rows]

    def audit(self, tenant, collection, limit=100):
        tenant, collection = _name(tenant, 'tenant'), _name(collection, 'collection')
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 1000:
            raise ValueError('limit must be between 1 and 1000')
        with self._connect() as conn:
            rows = conn.execute('''SELECT generation,operation,changed,created_at FROM audit_log
                WHERE tenant=? AND collection=? ORDER BY generation DESC LIMIT ?''', (tenant, collection, limit)).fetchall()
        return [dict(row) for row in rows]

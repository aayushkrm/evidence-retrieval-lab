import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from evidence_lab.workspace import Workspace, GenerationConflict, SnapshotLimitExceeded


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'workspace.sqlite'
        self.workspace = Workspace(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_isolation_updates_and_audit(self):
        w = self.workspace
        a = w.upsert('alice', 'papers', [{'id': 'x', 'text': 'alpha'}])
        b = w.upsert('bob', 'papers', [{'id': 'x', 'text': 'beta'}])
        self.assertNotEqual(a['documents'][0]['content_id'], b['documents'][0]['content_id'])
        self.assertEqual(w.snapshot('alice', 'papers')['documents']['x']['text'], 'alpha')
        self.assertEqual(w.snapshot('bob', 'papers')['documents']['x']['text'], 'beta')
        self.assertEqual(w.snapshot('alice', 'other'), {'generation': 0, 'documents': {}})
        same = w.upsert('alice', 'papers', [{'id': 'x', 'title': '', 'text': 'alpha'}])
        self.assertEqual((same['generation'], same['changed']), (1, 0))
        update = w.upsert('alice', 'papers', [{'id': 'x', 'title': 'new', 'text': 'alpha'}])
        self.assertEqual((update['generation'], update['changed']), (2, 1))
        self.assertNotEqual(update['documents'][0]['content_id'], a['documents'][0]['content_id'])
        self.assertEqual(w.delete('alice', 'papers', ['x', 'absent']), {'generation': 3, 'changed': 1})
        self.assertEqual(w.delete('alice', 'papers', ['x']), {'generation': 3, 'changed': 0})
        self.assertEqual([x['operation'] for x in w.audit('alice', 'papers')], ['delete', 'upsert', 'upsert'])
        self.assertEqual(w.audit('bob', 'papers')[0]['generation'], 1)
        self.assertEqual(w.list_collections('alice')[0]['document_count'], 0)

    def test_batch_validation_and_database_rollback(self):
        w = self.workspace
        w.upsert('a', 'c', [{'id': 'old', 'text': 'old'}])
        with self.assertRaises(ValueError):
            w.upsert('a', 'c', [{'id': 'good', 'text': 'yes'}, {'id': 'bad', 'text': ''}])
        self.assertEqual(w.snapshot('a', 'c')['generation'], 1)
        with patch('evidence_lab.workspace.MAX_COLLECTION_DOCUMENTS', 1):
            with self.assertRaises(ValueError):
                w.upsert('a', 'c', [{'id': 'new', 'text': 'new'}])
        self.assertEqual(list(w.snapshot('a', 'c')['documents']), ['old'])
        self.assertEqual(len(w.audit('a', 'c')), 1)
        with sqlite3.connect(self.path) as conn:
            conn.execute("CREATE TRIGGER fail_insert BEFORE INSERT ON documents WHEN NEW.id='fail' BEGIN SELECT RAISE(ABORT, 'failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            w.upsert('a', 'c', [{'id': 'first', 'text': 'yes'}, {'id': 'fail', 'text': 'no'}])
        self.assertEqual(list(w.snapshot('a', 'c')['documents']), ['old'])

    def test_optimistic_concurrency(self):
        w = self.workspace
        w.upsert('a', 'c', [{'id': 'initial', 'text': 'x'}])
        def insert(i):
            try:
                return w.upsert('a', 'c', [{'id': str(i), 'text': 'x'}], expected_generation=1)['generation']
            except GenerationConflict:
                return 'conflict'
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(insert, range(8)))
        self.assertEqual(results.count(2), 1)
        self.assertEqual(results.count('conflict'), 7)
        self.assertEqual(len(w.snapshot('a', 'c')['documents']), 2)
        self.assertEqual(len(w.audit('a', 'c')), 2)

    def test_concurrent_unconditional_writes_are_not_lost(self):
        w = self.workspace
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda i: w.upsert('a', 'c', [{'id': str(i), 'text': 'x'}]), range(20)))
        self.assertEqual(sorted(r['generation'] for r in results), list(range(1, 21)))
        snapshot = w.snapshot('a', 'c')
        self.assertEqual(snapshot['generation'], 20)
        self.assertEqual(len(snapshot['documents']), 20)

    def test_validation_and_content_ids(self):
        w = self.workspace
        for bad in ('../bad', '', 'a/b', 'a b'):
            with self.assertRaises(ValueError):
                w.snapshot(bad, 'x')
        for docs in ([{'id': 'x', 'text': 'a'}, {'id': 'x', 'text': 'b'}],
                     [{'id': 'x', 'text': 'a', 'path': '/etc/passwd'}],
                     [{'id': '\ud800', 'text': 'a'}]):
            with self.assertRaises(ValueError):
                w.upsert('a', 'c', docs)
        one = w.upsert('a', 'c', [{'id': 'one', 'text': 'same'}, {'id': 'two', 'text': 'same'}])
        self.assertEqual(one['documents'][0]['content_id'], one['documents'][1]['content_id'])
        with self.assertRaises(GenerationConflict):
            w.delete('a', 'c', ['one'], expected_generation=0)
        self.assertEqual(w.snapshot('a', 'c')['generation'], 1)

    def test_collection_info_is_scoped_and_tracks_changes(self):
        w = self.workspace
        self.assertEqual(w.collection_info('alice', 'papers'), {'generation': 0, 'document_count': 0})
        w.upsert('alice', 'papers', [{'id': 'one', 'text': 'first'}, {'id': 'two', 'text': 'second'}])
        w.upsert('bob', 'papers', [{'id': 'other', 'text': 'other'}])
        self.assertEqual(w.collection_info('alice', 'papers'), {'generation': 1, 'document_count': 2})
        self.assertEqual(w.collection_info('bob', 'papers'), {'generation': 1, 'document_count': 1})
        w.delete('alice', 'papers', ['one', 'two'])
        self.assertEqual(w.collection_info('alice', 'papers'), {'generation': 2, 'document_count': 0})
        with self.assertRaises(ValueError):
            w.collection_info('../bad', 'papers')

    def test_collection_info_reads_one_snapshot_during_concurrent_write(self):
        w = self.workspace
        w.upsert('a', 'c', [{'id': 'one', 'text': 'one'}])
        original_current = w._current

        def write_once(conn, tenant, collection):
            generation = original_current(conn, tenant, collection)
            with patch.object(w, '_current', side_effect=original_current):
                w.upsert('a', 'c', [{'id': 'two', 'text': 'two'}])
            return generation

        with patch.object(w, '_current', side_effect=write_once):
            self.assertEqual(w.collection_info('a', 'c'), {'generation': 1, 'document_count': 1})
        self.assertEqual(w.collection_info('a', 'c'), {'generation': 2, 'document_count': 2})

    def test_snapshot_byte_limit_and_validation(self):
        w = self.workspace
        w.upsert('a', 'c', [{'id': 'private-id', 'title': 'é', 'text': 'private text'}])
        size = len('éprivate text'.encode('utf-8'))
        original = w.snapshot('a', 'c')
        self.assertEqual(w.snapshot('a', 'c', max_bytes=size), original)
        statements = []
        original_connect = w._connect

        @contextmanager
        def traced_connect():
            with original_connect() as conn:
                conn.set_trace_callback(statements.append)
                yield conn

        with patch.object(w, '_connect', side_effect=traced_connect):
            with self.assertRaises(SnapshotLimitExceeded) as error:
                w.snapshot('a', 'c', max_bytes=size - 1)
        self.assertNotIn('private', str(error.exception))
        self.assertFalse(any('SELECT id,title,text,content_id' in sql for sql in statements))
        self.assertEqual(w.snapshot('a', 'missing', max_bytes=0), {'generation': 0, 'documents': {}})
        for invalid in (-1, True, 1.5, '100'):
            with self.subTest(max_bytes=invalid), self.assertRaises(ValueError):
                w.snapshot('a', 'c', max_bytes=invalid)

    def test_snapshot_limit_and_documents_use_same_read_snapshot(self):
        w = self.workspace
        w.upsert('a', 'c', [{'id': 'one', 'text': 'small'}])
        original_current = w._current

        def write_once(conn, tenant, collection):
            generation = original_current(conn, tenant, collection)
            with patch.object(w, '_current', side_effect=original_current):
                w.upsert('a', 'c', [{'id': 'one', 'text': 'much larger'}])
            return generation

        with patch.object(w, '_current', side_effect=write_once):
            result = w.snapshot('a', 'c', max_bytes=5)
        self.assertEqual(result['generation'], 1)
        self.assertEqual(result['documents']['one']['text'], 'small')
        with self.assertRaises(SnapshotLimitExceeded):
            w.snapshot('a', 'c', max_bytes=5)


if __name__ == '__main__':
    unittest.main()

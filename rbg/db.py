"""Turso / libSQL persistence layer for the harness.

One libSQL database per repository lives at ``<repo>/.rbg/rbg.db`` and
holds the index (files, chunks, symbols, reference edges), the dense chunk
vectors, the append-only execution ledger and the learned model artifacts.
A second, user-scoped database at ``~/.local/share/rbg/global.db`` is
``ATTACH``-ed as ``global`` so a repo-local model can be shrunk toward a
cross-repo pooled prior during training (small n -> pooled, large n -> local).

libSQL is a drop-in superset of SQLite: FTS5 gives lexical retrieval, a
``F32_BLOB`` column plus a ``libsql_vector_idx`` gives an on-disk approximate
vector index, and ``vector_distance_cos`` scores a query against it.  Setting
``RBG_DB_URL`` (+ ``RBG_DB_AUTH_TOKEN``, and optionally
``RBG_DB_SYNC_URL`` for an embedded replica) points the same code at a
remote Turso server; the default is the local file.

Dependency note: ``libsql`` is a hard dependency — Turso/libSQL is the only
store the harness supports.  (The shared/global tier is best-effort: if its
``ATTACH`` cannot be created the repo store still works, just without
cross-repo pooling.)
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from contextlib import contextmanager

import libsql
import numpy as np

from . import persist

SCHEMA_VERSION = 2
GLOBAL_DB = os.path.join(os.path.expanduser("~"), ".local", "share", "rbg", "global.db")


def global_db_path() -> str:
    """The shared, cross-repo database (overridable for tests/CI)."""
    return os.environ.get("RBG_GLOBAL_DB") or GLOBAL_DB


def db_path(root: str) -> str:
    """Where a repo's libSQL database lives.

    Default is a central, per-repo file under ``~/.local/share/rbg/repos``
    (keyed by the repo path), so probing an arbitrary snapshot — including a
    read-only one — never writes into the repo itself.  Set
    ``RBG_DB_IN_REPO=1`` for the portable ``<repo>/.rbg/rbg.db``,
    or ``RBG_DB_DIR`` / ``RBG_DB_URL`` to relocate/remote it."""
    default_root = os.environ.get("RBG_DB_ROOT") or os.path.join(
        os.path.expanduser("~"), ".local", "share", "rbg", "repos")
    if os.environ.get("RBG_DB_IN_REPO") or os.environ.get("RBG_DB_DIR"):
        base = os.environ.get("RBG_DB_DIR") or os.path.join(os.path.abspath(root), ".rbg")
        return os.path.join(base, "rbg.db")
    key = hashlib.sha1(os.path.abspath(root).encode()).hexdigest()[:16]
    return os.path.join(default_root, f"{key}.db")


def _floats_to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype="<f4").tobytes()


def _blob_to_floats(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f4").astype(float)


_BUSY_MARKERS = ("locked", "busy", "snapshot", "conflict")


def _is_busy(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _BUSY_MARKERS)


def _batched(rows, size: int = 1000):
    """The binding's ``executemany`` wants a list; batch so a 47k-chunk index
    does not materialise every row at once."""
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


class Store:
    """A connection to one repo's libSQL database (with the global DB attached).

    Every method is idempotent and keyed by revision: re-indexing a changed tree
    replaces the stored rows, and a query for an old revision is simply a miss.
    """

    def __init__(self, root: str, *, url: str | None = None, auth_token: str = "",
                 sync_url: str | None = None, global_url: str | None = None) -> None:
        self.root = os.path.abspath(root)
        self.path = db_path(self.root)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        url = url or os.environ.get("RBG_DB_URL") or f"file:{self.path}"
        auth_token = auth_token or os.environ.get("RBG_DB_AUTH_TOKEN", "")
        sync_url = sync_url or os.environ.get("RBG_DB_SYNC_URL") or None
        self._busy_timeout_s = int(os.environ.get("RBG_DB_BUSY_TIMEOUT_MS", "30000")) / 1000
        self._txn_depth = 0
        # AUTOCOMMIT: every multi-statement write takes its own BEGIN IMMEDIATE.
        self.conn = libsql.connect(url, auth_token=auth_token, sync_url=sync_url,
                                   timeout=self._busy_timeout_s, isolation_level=None)
        self._local_file = (url.startswith("file:") or "://" not in url) and not sync_url
        self._configure_connection()
        self._attached_global = False
        self._schema()
        self._attach_global(global_url)

    def _configure_connection(self) -> None:
        """WAL + a long busy handler: concurrent agents read in parallel and
        writers queue instead of failing with ``database is locked``."""
        try:
            self.conn.execute(f"PRAGMA busy_timeout = {int(self._busy_timeout_s * 1000)}")
        except Exception:  # pragma: no cover
            pass
        if not self._local_file:
            return
        sync = (os.environ.get("RBG_DB_SYNCHRONOUS") or "FULL").upper()
        try:
            self.conn.execute(f"PRAGMA synchronous = "
                              f"{sync if sync in ('FULL', 'NORMAL') else 'FULL'}")
        except Exception:  # pragma: no cover
            pass
        try:
            mode = self.conn.execute("PRAGMA journal_mode").fetchall()
            if not mode or str(mode[0][0]).lower() != "wal":
                self.conn.execute("PRAGMA journal_mode = WAL").fetchall()
        except Exception:  # pragma: no cover - e.g. a read-only directory
            pass

    @contextmanager
    def _write_txn(self):
        """One ``BEGIN IMMEDIATE`` write transaction, retried while busy.

        Taking the write lock up front avoids SQLite's non-retryable
        read-to-write upgrade (``SQLITE_BUSY_SNAPSHOT``) and makes the busy
        timeout effective across processes.  Re-entrant so callers can compose
        (``save_encoder`` -> ``save_vectors``).
        """
        if self._txn_depth:
            self._txn_depth += 1
            try:
                yield
            finally:
                self._txn_depth -= 1
            return
        deadline = time.monotonic() + self._busy_timeout_s
        delay = 0.01
        while True:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                break
            except Exception as exc:
                if not _is_busy(exc) or time.monotonic() >= deadline:
                    raise
                time.sleep(delay * (0.5 + random.random() * 0.5))
                delay = min(delay * 2, 0.5)
        self._txn_depth = 1
        try:
            yield
            self.conn.execute("COMMIT")
        except BaseException:
            try:
                self.conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            self._txn_depth = 0

    # -- schema ----------------------------------------------------------- #
    def _schema(self) -> None:
        rows = self.conn.execute("PRAGMA user_version").fetchall()
        if rows and int(rows[0][0] or 0) >= SCHEMA_VERSION:
            return  # warm open: no DDL, so no write lock on every connection
        with self._write_txn():
            self._create_schema()
            self.conn.execute(f"PRAGMA user_version = {int(SCHEMA_VERSION)}")

    def _create_schema(self) -> None:
        cur = self.conn

        cur.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
        cur.execute("""CREATE TABLE IF NOT EXISTS repo(
            id INTEGER PRIMARY KEY CHECK (id = 1),
            root TEXT, revision TEXT, updated_at REAL)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS chunks(
            id INTEGER PRIMARY KEY, path TEXT, symbol TEXT, kind TEXT,
            start_line INTEGER, end_line INTEGER, content_hash TEXT, content TEXT)""")
        cur.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(content, content='chunks', content_rowid='id')""")
        cur.execute("""CREATE TABLE IF NOT EXISTS symbols(
            id INTEGER PRIMARY KEY, name TEXT, kind TEXT, path TEXT,
            start_line INTEGER, end_line INTEGER)""")
        cur.execute("CREATE TABLE IF NOT EXISTS refs(id INTEGER PRIMARY KEY, name TEXT, path TEXT)")
        cur.execute("""CREATE TABLE IF NOT EXISTS ledger(
            id INTEGER PRIMARY KEY, task_id TEXT, component TEXT, revision TEXT,
            timestamp TEXT, model_name TEXT, confidence REAL, reason_codes TEXT,
            result TEXT, inputs_hash TEXT, prev_hash TEXT, hash TEXT)""")
        # Databases created before the chain existed get the columns here.
        for column in ("prev_hash", "hash"):
            try:
                cur.execute(f"ALTER TABLE ledger ADD COLUMN {column} TEXT")
            except Exception:  # noqa: BLE001 - already present
                pass
        cur.execute("""CREATE TABLE IF NOT EXISTS plans(
            task_id TEXT PRIMARY KEY, root TEXT, doc TEXT, version INTEGER,
            updated_at REAL)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS models(
            name TEXT PRIMARY KEY, revision TEXT, version TEXT,
            payload BLOB, metrics TEXT, updated_at REAL)""")
        # Experience memory (SWE-Exp): one row per task run, retrieved by dense
        # similarity.  Vector width is fixed so experiences written by any repo
        # are directly comparable (see EXPERIENCE_DIM / experience_encoder).
        cur.execute("""CREATE TABLE IF NOT EXISTS experiences(
            id INTEGER PRIMARY KEY, task_id TEXT, repo_id TEXT, revision TEXT,
            query TEXT, outcome TEXT, summary TEXT, vector F32_BLOB(512))""")
        cur.execute("CREATE INDEX IF NOT EXISTS experiences_idx "
                    "ON experiences(libsql_vector_idx(vector))")

    def _attach_global(self, global_url: str | None) -> None:
        global_url = global_url or global_db_path()
        try:
            os.makedirs(os.path.dirname(global_url), exist_ok=True)
            lit = global_url.replace("'", "''")
            self.conn.execute(f"ATTACH DATABASE 'file:{lit}' AS global")
            if self._local_file:
                try:  # WAL on the shared tier too; readers never block there
                    mode = self.conn.execute("PRAGMA global.journal_mode").fetchall()
                    if not mode or str(mode[0][0]).lower() != "wal":
                        self.conn.execute("PRAGMA global.journal_mode = WAL").fetchall()
                except Exception:  # pragma: no cover
                    pass
            names = {r[0] for r in self.conn.execute(
                "SELECT name FROM global.sqlite_master WHERE type='table'").fetchall()}
            missing = {"pooled_model", "repo_examples"} - names
            if missing:
                with self._write_txn():
                    self.conn.execute("""CREATE TABLE IF NOT EXISTS global.pooled_model(
                        kind TEXT PRIMARY KEY, version TEXT, weights BLOB,
                        n INTEGER, metrics TEXT, updated_at REAL)""")
                    self.conn.execute("""CREATE TABLE IF NOT EXISTS global.repo_examples(
                        repo_id TEXT, component TEXT, x BLOB, y REAL, ts REAL)""")
            self._attached_global = True
        except Exception:  # pragma: no cover - pooling is best effort
            self._attached_global = False

    # -- meta ------------------------------------------------------------- #
    def set_meta(self, key: str, value) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)))

    def get_meta(self, key: str, default=None):
        rows = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchall()
        return json.loads(rows[0][0]) if rows else default

    # -- index ------------------------------------------------------------ #
    def save_index(self, index) -> None:
        with self._write_txn():
            c = self.conn
            for tbl in ("chunks", "symbols", "refs"):
                c.execute(f"DELETE FROM {tbl}")
            # Vectors are revision-specific; drop them with the old index so a
            # re-index can never leave a stale matrix beside fresh chunks.  A bare
            # DELETE is not enough: libsql's vector index keeps shadow rows behind,
            # so re-inserting the same rowids then fails.
            if self._has_table("chunk_vectors"):
                self._drop_vectors()
            for batch in _batched(
                    (ch.path, ch.symbol, ch.kind, ch.start_line, ch.end_line,
                     ch.content_hash, ch.content) for ch in index.chunks):
                c.executemany(
                    "INSERT INTO chunks(path, symbol, kind, start_line, end_line, "
                    "content_hash, content) VALUES(?,?,?,?,?,?,?)", batch)
            c.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
            for batch in _batched(
                    (s.name, s.kind, s.path, s.start_line, s.end_line)
                    for s in index.symbols):
                c.executemany(
                    "INSERT INTO symbols(name, kind, path, start_line, end_line) "
                    "VALUES(?,?,?,?,?)", batch)
            for batch in _batched(
                    (name, p) for name, paths in index.references.items()
                    for p in set(paths)):
                c.executemany("INSERT INTO refs(name, path) VALUES(?,?)", batch)
            c.execute("INSERT INTO repo(id, root, revision, updated_at) VALUES(1,?,?,?) "
                      "ON CONFLICT(id) DO UPDATE SET root=excluded.root, "
                      "revision=excluded.revision, updated_at=excluded.updated_at",
                      (index.root, index.revision, time.time()))
            self.set_meta("files", list(index.files))

    def load_index(self, revision: str | None = None):
        from .index import Chunk, RepoIndex, Symbol

        rows = self.conn.execute("SELECT root, revision FROM repo WHERE id = 1").fetchall()
        if not rows:
            return None
        root, rev = rows[0]
        if revision is not None and revision != rev:
            return None
        idx = RepoIndex(root=root, revision=rev)
        idx.files = list(self.get_meta("files", []))
        for cid, path, symbol, kind, s, e, h, content in self.conn.execute(
                "SELECT id, path, symbol, kind, start_line, end_line, content_hash, content "
                "FROM chunks ORDER BY id").fetchall():
            idx.chunks.append(Chunk(path, s, e, content, symbol=symbol, kind=kind, content_hash=h))
        for name, kind, path, s, e in self.conn.execute(
                "SELECT name, kind, path, start_line, end_line FROM symbols ORDER BY id").fetchall():
            idx.symbols.append(Symbol(name, kind, path, s, e))
        refs: dict[str, list[str]] = {}
        for name, path in self.conn.execute("SELECT name, path FROM refs").fetchall():
            refs.setdefault(name, []).append(path)
        idx.references = {k: sorted(set(v)) for k, v in refs.items()}
        return idx

    # -- lexical candidates (FTS5) --------------------------------------- #
    def fts_search(self, query: str, limit: int) -> list[tuple[int, float]]:
        """BM25 lexical candidates from the persisted FTS5 index.

        Returns ``(position_in_index.chunks, score)`` pairs, best first, where
        position is the chunk's offset in the order ``save_index`` inserted them
        (so it lines up with a freshly loaded ``RepoIndex.chunks``).  An empty
        list means "no usable lexical index", never a raised error, so callers
        can fall back to the in-memory scorer.
        """
        terms = [t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", query)]
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))
        try:
            ids = [r[0] for r in self.conn.execute(
                "SELECT id FROM chunks ORDER BY id").fetchall()]
            pos = {cid: i for i, cid in enumerate(ids)}
            rows = self.conn.execute(
                "SELECT rowid, bm25(chunks_fts) AS s FROM chunks_fts "
                "WHERE chunks_fts MATCH ? ORDER BY s LIMIT ?",
                (match, int(limit))).fetchall()
        except Exception:
            return []
        return [(pos[r[0]], -float(r[1])) for r in rows if r[0] in pos]

    # -- vectors ---------------------------------------------------------- #
    def _has_table(self, name: str) -> bool:
        return bool(self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchall())

    def _drop_vectors(self) -> None:
        """Drop the vector table and its index together (see ``save_vectors``)."""
        self.conn.execute("DROP INDEX IF EXISTS chunk_vectors_idx")
        self.conn.execute("DROP TABLE IF EXISTS chunk_vectors")
        self.conn.execute("DELETE FROM meta WHERE key='dim'")

    def save_vectors(self, matrix: np.ndarray) -> None:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.size == 0:
            return
        dim = matrix.shape[1]
        # Recreate rather than DELETE rows: libsql's vector index keeps shadow
        # rows across a DELETE, so re-inserting the same rowids fails with
        # "failed to insert shadow row".  Dropping the table drops the index.
        with self._write_txn():
            self._drop_vectors()
            self.conn.execute(f"CREATE TABLE chunk_vectors(rowid INTEGER PRIMARY KEY, v F32_BLOB({dim}))")
            self.conn.execute("CREATE INDEX chunk_vectors_idx ON chunk_vectors(libsql_vector_idx(v))")
            self.set_meta("dim", dim)
            for batch in _batched(
                    (row, _floats_to_blob(vec))
                    for row, vec in enumerate(matrix, start=1)):
                self.conn.executemany(
                    "INSERT INTO chunk_vectors(rowid, v) VALUES(?, ?)", batch)

    def load_vectors(self, expected_rows: int | None = None) -> np.ndarray | None:
        if not self._has_table("chunk_vectors"):
            return None
        rows = self.conn.execute("SELECT v FROM chunk_vectors ORDER BY rowid").fetchall()
        if not rows:
            return None
        mat = np.vstack([_blob_to_floats(r[0]) for r in rows])
        if expected_rows is not None and len(mat) != expected_rows:
            return None
        return mat

    def vector_search(self, query_vec: np.ndarray, limit: int = 10) -> list[tuple[int, float]]:
        """On-disk ANN lookup through the ``libsql_vector_idx`` index.  Returns
        ``(rowid, cosine_distance)`` best-first; rowid is the chunk id (1-based).

        ``vector_top_k`` uses the index to pick candidates; joining back to the
        table recomputes an exact distance (ordering a projection straight off
        the index yields NULLs once the scan is index-driven).  A zero query has
        an undefined cosine, so it returns nothing rather than NULL rows."""
        qv = np.asarray(query_vec, dtype=float)
        if qv.size == 0 or not np.any(qv):
            return []
        blob = _floats_to_blob(qv)
        rows = self.conn.execute(
            "SELECT e.rowid, vector_distance_cos(e.v, ?) AS d "
            "FROM vector_top_k('chunk_vectors_idx', ?, ?) t "
            "JOIN chunk_vectors e ON e.rowid = t.id", (blob, blob, int(limit))).fetchall()
        out = [(int(r[0]), float(r[1])) for r in rows if r[1] is not None]
        out.sort(key=lambda kv: kv[1])
        return out

    # -- experience memory (SWE-Exp) -------------------------------------- #
    def save_experience(self, *, task_id: str, repo_id: str, revision: str,
                        query: str, outcome: str, summary: str,
                        vector: np.ndarray) -> None:
        """Append one (issue -> fix) experience, keyed by dense task vector.

        Insert-only: the rowid is autoincremented, so a libsql vector-index
        shadow row from a re-saved task is never re-inserted.  task_ids are
        unique per run by default, so duplicates are a non-issue in practice.
        ponytail: append-only, no dedupe; a replayed task may leave one stale
        row, harmless at N=1 retrieval.
        """
        vec = np.asarray(vector, dtype=float).ravel()
        if vec.size != 512:
            raise ValueError(f"experience vector must be 512-dim, got {vec.size}")
        with self._write_txn():
            self.conn.execute(
                "INSERT INTO experiences(task_id, repo_id, revision, query, outcome, "
                "summary, vector) VALUES(?,?,?,?,?,?,?)",
                (task_id, repo_id, revision, query, outcome, summary, _floats_to_blob(vec)))

    def search_experiences(self, query_vec: np.ndarray, limit: int = 1,
                           exclude_task: str = "") -> list[dict]:
        """Nearest prior experiences by cosine similarity, best first.

        ``limit=1`` is deliberate: SWE-Exp found a single retrieved experience
        optimal, extra ones adding noise.  Join back to the table for the
        payload (index-only projection yields NULLs), same as ``vector_search``.
        """
        qv = np.asarray(query_vec, dtype=float).ravel()
        if qv.size != 512 or not np.any(qv):
            return []
        blob = _floats_to_blob(qv)
        rows = self.conn.execute(
            "SELECT e.task_id, e.repo_id, e.outcome, e.summary, "
            "vector_distance_cos(e.vector, ?) AS d "
            "FROM vector_top_k('experiences_idx', ?, ?) t "
            "JOIN experiences e ON e.id = t.id",
            (blob, blob, int(limit) + 1)).fetchall()
        out = [{"task_id": r[0], "repo_id": r[1], "outcome": r[2],
                "summary": r[3], "distance": float(r[4])}
               for r in rows if r[4] is not None and r[0] != exclude_task]
        out.sort(key=lambda e: e["distance"])
        return out[:limit]

    # -- encoder ---------------------------------------------------------- #
    def save_encoder(self, revision: str, state: dict, matrix: np.ndarray) -> None:
        dim = int(np.asarray(matrix).shape[1]) if np.asarray(matrix).size else 0
        payload, digest = persist.dump(state)
        metrics = {"dim": dim, "artifact_sha256": digest}
        with self._write_txn():
            self.conn.execute("INSERT INTO models(name, revision, version, payload, metrics, updated_at) "
                              "VALUES('encoder',?,?,?,?,?) "
                              "ON CONFLICT(name) DO UPDATE SET revision=excluded.revision, "
                              "version=excluded.version, payload=excluded.payload, "
                              "metrics=excluded.metrics, updated_at=excluded.updated_at",
                              (revision, state.get("kind", ""), payload,
                               json.dumps(metrics), time.time()))
            self.save_vectors(matrix)

    @staticmethod
    def _decode_artifact(payload, metrics_json: str):
        """Decode a model payload or return ``None`` for a legacy/bad artifact."""
        try:
            metrics = json.loads(metrics_json or "{}")
        except ValueError:
            return None
        digest = metrics.get("artifact_sha256", "")
        try:
            return persist.load(payload, digest), metrics
        except persist.UnsafeArtifact:
            return None

    def load_encoder(self, revision: str, expected_rows: int):
        from .retrieval import encoder_from_state

        rows = self.conn.execute(
            "SELECT revision, payload, metrics FROM models WHERE name='encoder'").fetchall()
        if not rows or rows[0][0] != revision:
            return None
        decoded = self._decode_artifact(rows[0][1], rows[0][2])
        if decoded is None:
            return None
        state, _ = decoded
        matrix = self.load_vectors(expected_rows)
        if matrix is None:
            return None
        return encoder_from_state(state, matrix)

    def load_prev_encoder(self):
        """The last persisted encoder ``(revision, state)`` ignoring the current
        revision and row count, so a changed tree can warm-start instead of
        paying a full SVD refit."""
        rows = self.conn.execute(
            "SELECT revision, payload, metrics FROM models WHERE name='encoder'").fetchall()
        if not rows:
            return None
        decoded = self._decode_artifact(rows[0][1], rows[0][2])
        if decoded is None:
            return None
        return rows[0][0], decoded[0]

    # -- plan documents --------------------------------------------------- #
    def save_plan(self, root: str, task_id: str, doc: dict) -> None:
        with self._write_txn():
            self.conn.execute(
                "INSERT INTO plans(task_id, root, doc, version, updated_at) "
                "VALUES(?,?,?,?,?) ON CONFLICT(task_id) DO UPDATE SET "
                "root=excluded.root, doc=excluded.doc, version=excluded.version, "
                "updated_at=excluded.updated_at",
                (task_id, root, json.dumps(doc), int(doc.get("version") or 1), time.time()))

    def load_plan(self, root: str, task_id: str) -> dict | None:
        rows = self.conn.execute(
            "SELECT doc, root FROM plans WHERE task_id = ?", (task_id,)).fetchall()
        if not rows:
            return None
        doc = json.loads(rows[0][0])
        if root and rows[0][1] != root:
            return None
        return doc

    # -- ledger ----------------------------------------------------------- #
    @staticmethod
    def _ledger_hash(entry: dict, prev_hash: str) -> str:
        canon = json.dumps([
            entry.get("task_id"), entry.get("component"), entry.get("revision"),
            entry.get("timestamp"), entry.get("model_name"), entry.get("confidence"),
            entry.get("reason_codes", []), entry.get("result", {}),
            entry.get("inputs_hash", ""), prev_hash,
        ], sort_keys=True, default=str)
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()

    def append_ledger(self, entry: dict) -> None:
        """Append one row and chain its hash to the previous row.

        The ledger trains models, so a silently edited row is a correctness
        problem, not only an audit one.  ``verify_ledger`` recomputes the chain.
        """
        with self._write_txn():
            row = self.conn.execute(
                "SELECT hash FROM ledger ORDER BY id DESC LIMIT 1").fetchall()
            prev = (row[0][0] if row and row[0][0] else "")
            digest = self._ledger_hash(entry, prev)
            self.conn.execute(
                "INSERT INTO ledger(task_id, component, revision, timestamp, model_name, "
                "confidence, reason_codes, result, inputs_hash, prev_hash, hash) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (entry.get("task_id"), entry.get("component"), entry.get("revision"),
                 entry.get("timestamp"), entry.get("model_name"), entry.get("confidence"),
                 json.dumps(entry.get("reason_codes", [])), json.dumps(entry.get("result", {})),
                 entry.get("inputs_hash", ""), prev, digest))

    def verify_ledger(self) -> dict:
        """Recompute the hash chain; return the first bad row, if any."""
        rows = self.conn.execute(
            "SELECT id, task_id, component, revision, timestamp, model_name, confidence, "
            "reason_codes, result, inputs_hash, prev_hash, hash FROM ledger ORDER BY id").fetchall()
        prev = ""
        for row in rows:
            entry = {
                "task_id": row[1], "component": row[2], "revision": row[3],
                "timestamp": row[4], "model_name": row[5], "confidence": row[6],
                "reason_codes": json.loads(row[7] or "[]"),
                "result": json.loads(row[8] or "{}"), "inputs_hash": row[9],
            }
            expected = self._ledger_hash(entry, prev)
            if (row[10] or "") != prev or (row[11] or "") != expected:
                return {"ok": False, "rows": len(rows), "first_bad_id": row[0],
                        "reason": "hash chain mismatch"}
            prev = expected
        return {"ok": True, "rows": len(rows), "first_bad_id": None,
                "reason": "chain intact"}

    def repo_revision(self) -> str:
        rows = self.conn.execute("SELECT revision FROM repo WHERE id = 1").fetchall()
        return rows[0][0] if rows else ""

    def ledger_count(self) -> int:
        """Cheap ledger-size probe: lets the read path notice new rows without
        loading and parsing the whole ledger."""
        row = self.conn.execute("SELECT COUNT(*) FROM ledger").fetchone()
        return int(row[0]) if row else 0

    def read_ledger(self) -> list[dict]:
        out = []
        for row in self.conn.execute(
                "SELECT task_id, component, revision, timestamp, model_name, confidence, "
                "reason_codes, result, inputs_hash, hash FROM ledger ORDER BY id").fetchall():
            out.append({
                "task_id": row[0], "component": row[1], "revision": row[2],
                "timestamp": row[3], "model_name": row[4], "confidence": row[5],
                "reason_codes": json.loads(row[6] or "[]"),
                "result": json.loads(row[7] or "{}"), "inputs_hash": row[8],
                "hash": row[9],
            })
        return out

    # -- model artifacts -------------------------------------------------- #
    def save_model(self, name: str, version: str, obj, metrics: dict | None = None,
                   revision: str = "") -> None:
        payload, digest = persist.dump(obj)
        stored = {**(metrics or {}), "artifact_sha256": digest}
        with self._write_txn():
            self.conn.execute(
                "INSERT INTO models(name, revision, version, payload, metrics, updated_at) "
                "VALUES(?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
                "revision=excluded.revision, version=excluded.version, payload=excluded.payload, "
                "metrics=excluded.metrics, updated_at=excluded.updated_at",
                (name, revision, version, payload, json.dumps(stored), time.time()))

    def load_model(self, name: str):
        rows = self.conn.execute("SELECT payload, version, metrics FROM models WHERE name=?",
                                 (name,)).fetchall()
        if not rows:
            return None
        decoded = self._decode_artifact(rows[0][0], rows[0][2])
        if decoded is None:
            return None
        return decoded[0], rows[0][1], decoded[1]

    def load_model_at(self, name: str, revision: str):
        """Load a persisted artifact only if it was saved for this exact revision."""
        rows = self.conn.execute("SELECT payload, version, metrics FROM models "
                                 "WHERE name=? AND revision=?", (name, revision)).fetchall()
        if not rows:
            return None
        decoded = self._decode_artifact(rows[0][0], rows[0][2])
        if decoded is None:
            return None
        return decoded[0], rows[0][1], decoded[1]

    # -- pooled prior (global tier) --------------------------------------- #
    def save_pooled(self, kind: str, weights, n: int, version: str, metrics: dict | None = None) -> None:
        if not self._attached_global:
            return
        payload, digest = persist.dump(weights)
        stored = {**(metrics or {}), "artifact_sha256": digest}
        try:
            with self._write_txn():
                self.conn.execute(
                    "INSERT INTO global.pooled_model(kind, version, weights, n, metrics, updated_at) "
                    "VALUES(?,?,?,?,?,?) ON CONFLICT(kind) DO UPDATE SET version=excluded.version, "
                    "weights=excluded.weights, n=excluded.n, metrics=excluded.metrics, "
                    "updated_at=excluded.updated_at",
                    (kind, version, payload, int(n), json.dumps(stored), time.time()))
        except Exception:  # pragma: no cover
            pass

    def save_examples(self, repo_id: str, component: str, X: np.ndarray, y: np.ndarray) -> None:
        """Persist this repo's curated training rows in the shared tier (replacing
        the previous rows for the same repo/component, so retrains do not stack)."""
        if not self._attached_global or len(y) == 0:
            return
        try:
            rows = ((repo_id, component, _floats_to_blob(xi), float(yi), time.time())
                    for xi, yi in zip(np.asarray(X, dtype=float), np.asarray(y, dtype=float)))
            with self._write_txn():
                self.conn.execute("DELETE FROM global.repo_examples WHERE repo_id=? AND component=?",
                                  (repo_id, component))
                for batch in _batched(rows):
                    self.conn.executemany(
                        "INSERT INTO global.repo_examples(repo_id, component, x, y, ts) "
                        "VALUES(?,?,?,?,?)", batch)
        except Exception:  # pragma: no cover
            pass

    def load_examples(self, component: str, exclude_repo: str | None = None):
        """Cross-repo training rows, optionally excluding one repo (the local one)."""
        if not self._attached_global:
            return np.zeros((0, 0)), np.zeros(0)
        try:
            sql = "SELECT x, y FROM global.repo_examples WHERE component=?"
            params: list = [component]
            if exclude_repo is not None:
                sql += " AND repo_id != ?"
                params.append(exclude_repo)
            rows = self.conn.execute(sql, tuple(params)).fetchall()
        except Exception:  # pragma: no cover
            return np.zeros((0, 0)), np.zeros(0)
        if not rows:
            return np.zeros((0, 0)), np.zeros(0)
        X = np.vstack([_blob_to_floats(r[0]) for r in rows])
        y = np.asarray([r[1] for r in rows], dtype=float)
        return X, y

    def pooled_repo_count(self, component: str) -> int:
        if not self._attached_global:
            return 0
        try:
            return self.conn.execute("SELECT count(DISTINCT repo_id) FROM global.repo_examples "
                                     "WHERE component=?", (component,)).fetchall()[0][0]
        except Exception:  # pragma: no cover
            return 0

    def load_pooled(self, kind: str):
        if not self._attached_global:
            return None
        try:
            rows = self.conn.execute("SELECT weights, n, metrics FROM global.pooled_model "
                                     "WHERE kind=?", (kind,)).fetchall()
        except Exception:  # pragma: no cover
            return None
        if not rows:
            return None
        decoded = self._decode_artifact(rows[0][0], rows[0][2])
        if decoded is None:
            return None
        return decoded[0], int(rows[0][1]), decoded[1]

    def stats(self) -> dict:
        def one(sql, default=0):
            try:
                return self.conn.execute(sql).fetchall()[0][0]
            except Exception:
                return default
        return {
            "path": self.path,
            "attached_global": self._attached_global,
            "revision": (self.conn.execute("SELECT revision FROM repo WHERE id=1").fetchall()
                         or [("",)])[0][0],
            "chunks": one("SELECT count(*) FROM chunks"),
            "symbols": one("SELECT count(*) FROM symbols"),
            "refs": one("SELECT count(*) FROM refs"),
            "vectors": one("SELECT count(*) FROM chunk_vectors"),
            "ledger_entries": one("SELECT count(*) FROM ledger"),
            "models": one("SELECT count(*) FROM models"),
        }

    def commit(self) -> None:
        if not self._txn_depth:
            self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:  # pragma: no cover
            pass

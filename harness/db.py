"""Turso / libSQL persistence layer for the harness.

One libSQL database per repository lives at ``<repo>/.harness/harness.db`` and
holds the index (files, chunks, symbols, reference edges), the dense chunk
vectors, the append-only execution ledger and the learned model artifacts.
A second, user-scoped database at ``~/.local/share/harness/global.db`` is
``ATTACH``-ed as ``global`` so a repo-local model can be shrunk toward a
cross-repo pooled prior during training (small n -> pooled, large n -> local).

libSQL is a drop-in superset of SQLite: FTS5 gives lexical retrieval, a
``F32_BLOB`` column plus a ``libsql_vector_idx`` gives an on-disk approximate
vector index, and ``vector_distance_cos`` scores a query against it.  Setting
``HARNESS_DB_URL`` (+ ``HARNESS_DB_AUTH_TOKEN``, and optionally
``HARNESS_DB_SYNC_URL`` for an embedded replica) points the same code at a
remote Turso server; the default is the local file.

Dependency note: ``libsql`` is a hard dependency — Turso/libSQL is the only
store the harness supports.  (The shared/global tier is best-effort: if its
``ATTACH`` cannot be created the repo store still works, just without
cross-repo pooling.)
"""

from __future__ import annotations

import json
import os
import pickle
import time
import hashlib

import numpy as np
import libsql

SCHEMA_VERSION = 1
GLOBAL_DB = os.path.join(os.path.expanduser("~"), ".local", "share", "harness", "global.db")


def global_db_path() -> str:
    """The shared, cross-repo database (overridable for tests/CI)."""
    return os.environ.get("HARNESS_GLOBAL_DB") or GLOBAL_DB


def db_path(root: str) -> str:
    """Where a repo's libSQL database lives.

    Default is a central, per-repo file under ``~/.local/share/harness/repos``
    (keyed by the repo path), so probing an arbitrary snapshot — including a
    read-only one — never writes into the repo itself.  Set
    ``HARNESS_DB_IN_REPO=1`` for the portable ``<repo>/.harness/harness.db``,
    or ``HARNESS_DB_DIR`` / ``HARNESS_DB_URL`` to relocate/remote it."""
    default_root = os.environ.get("HARNESS_DB_ROOT") or os.path.join(
        os.path.expanduser("~"), ".local", "share", "harness", "repos")
    if os.environ.get("HARNESS_DB_IN_REPO") or os.environ.get("HARNESS_DB_DIR"):
        base = os.environ.get("HARNESS_DB_DIR") or os.path.join(os.path.abspath(root), ".harness")
        return os.path.join(base, "harness.db")
    key = hashlib.sha1(os.path.abspath(root).encode()).hexdigest()[:16]
    return os.path.join(default_root, f"{key}.db")


def _floats_to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype="<f4").tobytes()


def _blob_to_floats(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f4").astype(float)


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
        url = url or os.environ.get("HARNESS_DB_URL") or f"file:{self.path}"
        auth_token = auth_token or os.environ.get("HARNESS_DB_AUTH_TOKEN", "")
        sync_url = sync_url or os.environ.get("HARNESS_DB_SYNC_URL") or None
        self.conn = libsql.connect(url, auth_token=auth_token, sync_url=sync_url)
        self._attached_global = False
        self._schema()
        self._attach_global(global_url)

    # -- schema ----------------------------------------------------------- #
    def _schema(self) -> None:
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
            result TEXT, inputs_hash TEXT)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS models(
            name TEXT PRIMARY KEY, revision TEXT, version TEXT,
            payload BLOB, metrics TEXT, updated_at REAL)""")
        cur.commit()

    def _attach_global(self, global_url: str | None) -> None:
        global_url = global_url or global_db_path()
        try:
            os.makedirs(os.path.dirname(global_url), exist_ok=True)
            lit = global_url.replace("'", "''")
            self.conn.execute(f"ATTACH DATABASE 'file:{lit}' AS global")
            self.conn.execute("""CREATE TABLE IF NOT EXISTS global.pooled_model(
                kind TEXT PRIMARY KEY, version TEXT, weights BLOB,
                n INTEGER, metrics TEXT, updated_at REAL)""")
            self.conn.execute("""CREATE TABLE IF NOT EXISTS global.repo_examples(
                repo_id TEXT, component TEXT, x BLOB, y REAL, ts REAL)""")
            self.conn.commit()
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
        c = self.conn
        for tbl in ("chunks", "symbols", "refs"):
            c.execute(f"DELETE FROM {tbl}")
        # Vectors are revision-specific; drop them with the old index so a
        # re-index can never leave a stale matrix beside fresh chunks.  A bare
        # DELETE is not enough: libsql's vector index keeps shadow rows behind,
        # so re-inserting the same rowids then fails.
        if self._has_table("chunk_vectors"):
            self._drop_vectors()
        for ch in index.chunks:
            c.execute("INSERT INTO chunks(path, symbol, kind, start_line, end_line, "
                      "content_hash, content) VALUES(?,?,?,?,?,?,?)",
                      (ch.path, ch.symbol, ch.kind, ch.start_line, ch.end_line,
                       ch.content_hash, ch.content))
        c.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        for s in index.symbols:
            c.execute("INSERT INTO symbols(name, kind, path, start_line, end_line) "
                      "VALUES(?,?,?,?,?)", (s.name, s.kind, s.path, s.start_line, s.end_line))
        for name, paths in index.references.items():
            for p in set(paths):
                c.execute("INSERT INTO refs(name, path) VALUES(?,?)", (name, p))
        c.execute("INSERT INTO repo(id, root, revision, updated_at) VALUES(1,?,?,?) "
                  "ON CONFLICT(id) DO UPDATE SET root=excluded.root, "
                  "revision=excluded.revision, updated_at=excluded.updated_at",
                  (index.root, index.revision, time.time()))
        self.set_meta("files", list(index.files))
        self.commit()

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
        self._drop_vectors()
        self.conn.execute(f"CREATE TABLE chunk_vectors(rowid INTEGER PRIMARY KEY, v F32_BLOB({dim}))")
        self.conn.execute("CREATE INDEX chunk_vectors_idx ON chunk_vectors(libsql_vector_idx(v))")
        self.set_meta("dim", dim)
        for row, vec in enumerate(matrix, start=1):
            self.conn.execute("INSERT INTO chunk_vectors(rowid, v) VALUES(?, ?)",
                              (row, _floats_to_blob(vec)))
        self.commit()

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

    # -- encoder ---------------------------------------------------------- #
    def save_encoder(self, revision: str, state: dict, matrix: np.ndarray) -> None:
        dim = int(np.asarray(matrix).shape[1]) if np.asarray(matrix).size else 0
        self.conn.execute("INSERT INTO models(name, revision, version, payload, metrics, updated_at) "
                          "VALUES('encoder',?,?,?,?,?) "
                          "ON CONFLICT(name) DO UPDATE SET revision=excluded.revision, "
                          "version=excluded.version, payload=excluded.payload, "
                          "metrics=excluded.metrics, updated_at=excluded.updated_at",
                          (revision, state.get("kind", ""), pickle.dumps(state),
                           json.dumps({"dim": dim}), time.time()))
        self.save_vectors(matrix)

    def load_encoder(self, revision: str, expected_rows: int):
        from .retrieval import encoder_from_state

        rows = self.conn.execute("SELECT revision, payload FROM models WHERE name='encoder'").fetchall()
        if not rows or rows[0][0] != revision:
            return None
        state = pickle.loads(rows[0][1])
        matrix = self.load_vectors(expected_rows)
        if matrix is None:
            return None
        return encoder_from_state(state, matrix)

    # -- ledger ----------------------------------------------------------- #
    def append_ledger(self, entry: dict) -> None:
        self.conn.execute(
            "INSERT INTO ledger(task_id, component, revision, timestamp, model_name, "
            "confidence, reason_codes, result, inputs_hash) VALUES(?,?,?,?,?,?,?,?,?)",
            (entry.get("task_id"), entry.get("component"), entry.get("revision"),
             entry.get("timestamp"), entry.get("model_name"), entry.get("confidence"),
             json.dumps(entry.get("reason_codes", [])), json.dumps(entry.get("result", {})),
             entry.get("inputs_hash", "")))
        self.commit()

    def ledger_count(self) -> int:
        """Cheap ledger-size probe: lets the read path notice new rows without
        loading and parsing the whole ledger."""
        row = self.conn.execute("SELECT COUNT(*) FROM ledger").fetchone()
        return int(row[0]) if row else 0

    def read_ledger(self) -> list[dict]:
        out = []
        for row in self.conn.execute(
                "SELECT task_id, component, revision, timestamp, model_name, confidence, "
                "reason_codes, result, inputs_hash FROM ledger ORDER BY id").fetchall():
            out.append({
                "task_id": row[0], "component": row[1], "revision": row[2],
                "timestamp": row[3], "model_name": row[4], "confidence": row[5],
                "reason_codes": json.loads(row[6] or "[]"),
                "result": json.loads(row[7] or "{}"), "inputs_hash": row[8],
            })
        return out

    # -- model artifacts -------------------------------------------------- #
    def save_model(self, name: str, version: str, obj, metrics: dict | None = None,
                   revision: str = "") -> None:
        self.conn.execute(
            "INSERT INTO models(name, revision, version, payload, metrics, updated_at) "
            "VALUES(?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
            "revision=excluded.revision, version=excluded.version, payload=excluded.payload, "
            "metrics=excluded.metrics, updated_at=excluded.updated_at",
            (name, revision, version, pickle.dumps(obj), json.dumps(metrics or {}), time.time()))
        self.commit()

    def load_model(self, name: str):
        rows = self.conn.execute("SELECT payload, version, metrics FROM models WHERE name=?",
                                 (name,)).fetchall()
        if not rows:
            return None
        return pickle.loads(rows[0][0]), rows[0][1], json.loads(rows[0][2] or "{}")

    def load_model_at(self, name: str, revision: str):
        """Load a persisted artifact only if it was saved for this exact revision."""
        rows = self.conn.execute("SELECT payload, version, metrics FROM models "
                                 "WHERE name=? AND revision=?", (name, revision)).fetchall()
        if not rows:
            return None
        return pickle.loads(rows[0][0]), rows[0][1], json.loads(rows[0][2] or "{}")

    # -- pooled prior (global tier) --------------------------------------- #
    def save_pooled(self, kind: str, weights, n: int, version: str, metrics: dict | None = None) -> None:
        if not self._attached_global:
            return
        try:
            self.conn.execute(
                "INSERT INTO global.pooled_model(kind, version, weights, n, metrics, updated_at) "
                "VALUES(?,?,?,?,?,?) ON CONFLICT(kind) DO UPDATE SET version=excluded.version, "
                "weights=excluded.weights, n=excluded.n, metrics=excluded.metrics, "
                "updated_at=excluded.updated_at",
                (kind, version, pickle.dumps(weights), int(n),
                 json.dumps(metrics or {}), time.time()))
            self.conn.commit()
        except Exception:  # pragma: no cover
            pass

    def save_examples(self, repo_id: str, component: str, X: np.ndarray, y: np.ndarray) -> None:
        """Persist this repo's curated training rows in the shared tier (replacing
        the previous rows for the same repo/component, so retrains do not stack)."""
        if not self._attached_global or len(y) == 0:
            return
        try:
            self.conn.execute("DELETE FROM global.repo_examples WHERE repo_id=? AND component=?",
                              (repo_id, component))
            for xi, yi in zip(np.asarray(X, dtype=float), np.asarray(y, dtype=float)):
                self.conn.execute("INSERT INTO global.repo_examples(repo_id, component, x, y, ts) "
                                  "VALUES(?,?,?,?,?)",
                                  (repo_id, component, _floats_to_blob(xi), float(yi), time.time()))
            self.conn.commit()
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
        return pickle.loads(rows[0][0]), int(rows[0][1]), json.loads(rows[0][2] or "{}")

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
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:  # pragma: no cover
            pass

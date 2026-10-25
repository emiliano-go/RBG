"""Repository indexing: files, chunks, symbols, and reference edges.

Python source is parsed with the stdlib ``ast`` module.  Every other text
file is split into line windows.  No third-party parser is required, which
keeps the harness runnable with only numpy installed.
"""

from __future__ import annotations

import ast
import hashlib
import os
import subprocess
from dataclasses import dataclass, field

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".ruff_cache",
}
# Generated / cache dirs that are not always named in .gitignore but never hold
# source worth retrieving.
GENERATED_SKIP = {
    ".cache", ".zensical-debug", ".seo-debug", "_test-env", ".tox", ".nox",
    "htmlcov", ".hypothesis", ".idea", ".vscode",
}

_EXTRA_SKIP_CACHE: tuple[str, frozenset[str]] = ("", frozenset())

# Files that commonly hold credentials are never indexed: retrieval output is
# fed to an LLM and persisted in the ledger, so a tracked secret must not be
# reachable through search_code/read_chunk.  Operators can opt back in for a
# repository that legitimately contains fixtures.
_SENSITIVE_NAMES = {
    ".netrc", ".npmrc", ".pypirc", ".git-credentials", ".env",
    "credentials", "credentials.json", "secrets", "secrets.json",
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
}
_SENSITIVE_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".keystore", ".jks")


def _sensitive(rel: str) -> bool:
    if os.environ.get("RBG_ALLOW_SENSITIVE", "").lower() in ("1", "true", "yes"):
        return False
    name = os.path.basename(rel).lower()
    return (name in _SENSITIVE_NAMES
            or name.startswith(".env.")
            or name.endswith(_SENSITIVE_SUFFIXES))


def _extra_skip_dirs() -> frozenset[str]:
    """Operator-supplied directory names to prune (``RBG_EXTRA_SKIP_DIRS``).

    A monorepo root often mixes source with generated campaign/site artefacts;
    this lets a deployment exclude those without hard-coding repo-specific
    names in the indexer.
    """
    global _EXTRA_SKIP_CACHE
    raw = os.environ.get("RBG_EXTRA_SKIP_DIRS", "")
    if _EXTRA_SKIP_CACHE[0] != raw:
        names = frozenset(p.strip().strip("/") for p in raw.split(",") if p.strip())
        _EXTRA_SKIP_CACHE = (raw, names)
    return _EXTRA_SKIP_CACHE[1]


def _git_ignored(root: str) -> set[str]:
    """Repo-relative paths (dirs and files) excluded by git's ignore rules.

    Uses ``git ls-files --others --ignored --directory`` so a vendored or
    generated tree (a packaged copy, a test venv) is pruned the way the
    project itself intends, rather than relying on a hard-coded list.
    """
    try:
        out = subprocess.run(
            ["git", "-C", root, "ls-files", "--others", "--ignored",
             "--exclude-standard", "--directory", "-z"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    if out.returncode != 0:
        return set()
    return {p.rstrip("/") for p in out.stdout.split("\0") if p}
TEXT_EXT = {
    ".py", ".pyi", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
    ".c", ".h", ".cpp", ".hpp", ".rb", ".sh", ".toml", ".yaml", ".yml",
    ".json", ".md", ".cfg", ".ini", ".txt", ".sql",
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class Chunk:
    path: str
    start_line: int
    end_line: int
    content: str
    symbol: str = ""
    kind: str = "block"
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = _sha(self.content)


def chunk_key(path: str, start_line: int, end_line: int) -> str:
    """Stable identity for a chunk, usable across requests (see read_chunk)."""
    return f"{path}:{start_line}-{end_line}"


@dataclass
class Symbol:
    name: str
    kind: str
    path: str
    start_line: int
    end_line: int


@dataclass
class RepoIndex:
    root: str
    files: list[str] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    symbols: list[Symbol] = field(default_factory=list)
    references: dict[str, list[str]] = field(default_factory=dict)
    revision: str = ""

    @classmethod
    def build(cls, root: str, revision: str | None = None) -> "RepoIndex":
        root = os.path.abspath(root)
        idx = cls(root=root, revision=revision or _tree_hash(root))
        ignored = _git_ignored(root)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if not cls._skip_dir(d)
                and os.path.relpath(os.path.join(dirpath, d), root) not in ignored
            ]
            for name in sorted(filenames):
                if os.path.splitext(name)[1] not in TEXT_EXT:
                    continue
                rel = os.path.relpath(os.path.join(dirpath, name), root)
                if rel in ignored or _sensitive(rel):
                    continue
                full = os.path.join(dirpath, name)
                idx.files.append(rel)
                try:
                    with open(full, "r", encoding="utf-8", errors="replace") as fh:
                        text = fh.read()
                except OSError:
                    continue
                if name.endswith(".py"):
                    idx._index_python(rel, text)
                elif not idx._index_tree_sitter(rel, text):
                    idx._index_text(rel, text)
        return idx

    @staticmethod
    def _skip_dir(name: str) -> bool:
        return (name in SKIP_DIRS or name in GENERATED_SKIP
                or name in _extra_skip_dirs()
                or (name.startswith(".") and name not in {".github", ".circleci"}))

    def _index_python(self, rel: str, text: str) -> None:
        lines = text.splitlines()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            self._index_text(rel, text)
            return
        covered: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                start = node.lineno
                end = getattr(node, "end_lineno", start) or start
                covered.update(range(start, end + 1))
                body = "\n".join(lines[start - 1:end])
                chunk = Chunk(rel, start, end, body, symbol=node.name, kind=kind)
                self.chunks.append(chunk)
                self.symbols.append(Symbol(node.name, kind, rel, start, end))
            elif isinstance(node, ast.Call):
                fn = node.func
                name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                if name:
                    self.references.setdefault(name, []).append(rel)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    self.references.setdefault(alias.name.split(".")[0], []).append(rel)
        # Module-level assignments are symbols too: rule constants, compiled
        # regexes and lookup tables are prime retrieval targets and are not
        # functions or classes.
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if not names:
                continue
            start = node.lineno
            end = getattr(node, "end_lineno", start) or start
            covered.update(range(start, end + 1))
            body = "\n".join(lines[start - 1:end])
            for name in names:
                self.symbols.append(Symbol(name, "constant", rel, start, end))
            self.chunks.append(Chunk(rel, start, end, body, symbol=names[0], kind="constant"))
        rest = [i for i in range(1, len(lines) + 1) if i not in covered and lines[i - 1].strip()]
        self._pack_lines(rel, lines, rest, "module")

    def _index_tree_sitter(self, rel: str, text: str) -> bool:
        from . import parsers
        parsed = parsers.parse(rel, text)
        if parsed is None:
            return False
        chunks, symbols, references = parsed
        for start, end, body, name, kind in chunks:
            self.chunks.append(Chunk(rel, start, end, body, symbol=name, kind=kind))
        for name, kind, start, end in symbols:
            self.symbols.append(Symbol(name, kind, rel, start, end))
        for name, paths in references.items():
            self.references.setdefault(name, []).extend(paths)
        covered = set()
        for start, end, _body, _name, _kind in chunks:
            covered.update(range(start, end + 1))
        lines = text.splitlines()
        rest = [i for i in range(1, len(lines) + 1)
                if i not in covered and lines[i - 1].strip()]
        if rest:
            self._pack_lines(rel, lines, rest, "module")
        return True

    def _index_text(self, rel: str, text: str) -> None:
        lines = text.splitlines()
        self._pack_lines(rel, lines, range(1, len(lines) + 1), "text")

    def _pack_lines(self, rel: str, lines: list[str], line_numbers, kind: str, window: int = 40) -> None:
        nums = [n for n in line_numbers if n - 1 < len(lines) and lines[n - 1].strip()]
        for i in range(0, len(nums), window):
            group = nums[i:i + window]
            if not group:
                continue
            start, end = group[0], group[-1]
            body = "\n".join(lines[start - 1:end])
            self.chunks.append(Chunk(rel, start, end, body, kind=kind))

    def chunk_by_id(self) -> dict[str, Chunk]:
        return {f"{c.path}:{c.start_line}-{c.end_line}": c for c in self.chunks}

    def callers_of(self, symbol: str) -> list[str]:
        return sorted(set(self.references.get(symbol, [])))


def is_test_chunk(c: "Chunk") -> bool:
    """Whether a chunk is a test function (not a helper, fixture, or class).

    The previous heuristic ``"test" in c.path`` also counted helpers like
    ``_snapshot_sql`` and fixtures, polluting test ranking.  Class-level
    chunks (``Test*``) are excluded because their content spans all methods,
    making them match too many symbols.
    """
    if "test" not in c.path:
        return False
    if c.kind != "function":
        return False
    sym = c.symbol or ""
    return sym.startswith("test_") or sym.startswith("Test")


def _walk_indexable(root: str):
    """Yield (rel, full) for files the index would read, in a stable order."""
    ignored = _git_ignored(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if not RepoIndex._skip_dir(d)
            and os.path.relpath(os.path.join(dirpath, d), root) not in ignored
        ]
        for name in sorted(filenames):
            if os.path.splitext(name)[1] not in TEXT_EXT:
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root)
            if rel in ignored:
                continue
            yield rel, full


def _tree_signature(root: str) -> str:
    """Cheap freshness signature: path, size and nanosecond mtime only.

    Used to decide whether the cached index is still current without reading
    file contents; a metadata-only change still trips it, a content-only change
    with the same size *and* mtime is the one case it cannot see, which is what
    the content fingerprint below exists to catch on the next real comparison.
    """
    h = hashlib.sha256()
    for rel, full in _walk_indexable(root):
        try:
            st = os.stat(full)
        except OSError:
            continue
        h.update(f"{rel}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def _tree_hash(root: str) -> str:
    """Content fingerprint over the indexable tree: the source of truth for a
    revision, so a same-size, mtime-preserving edit still changes it."""
    h = hashlib.sha256()
    for rel, full in _walk_indexable(root):
        try:
            with open(full, "rb") as fh:
                data = fh.read()
        except OSError:
            continue
        h.update(rel.encode())
        h.update(hashlib.sha256(data).digest())
    return h.hexdigest()[:16]

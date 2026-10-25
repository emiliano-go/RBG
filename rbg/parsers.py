"""Tree-sitter chunking and symbol extraction for non-Python languages.

Python keeps the AST path in :mod:`rbg.index` (better call edges); every
other supported language gets real function/class chunks instead of a blind
40-line window.  If a grammar is unavailable the caller falls back to text.
"""

from __future__ import annotations

import re

_SUFFIXES = {
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx",
    ".go": "go", ".rs": "rust", ".java": "java",
    ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".hpp": "cpp",
    ".cs": "c_sharp", ".rb": "ruby", ".php": "php",
    ".sh": "bash", ".bash": "bash",
}

_DEF_TYPES = {
    "function_definition", "function_declaration", "function_item",
    "method_definition", "method_declaration", "constructor_declaration",
    "class_definition", "class_declaration", "class_specifier",
    "struct_item", "struct_specifier", "interface_declaration",
    "trait_item", "impl_item", "module", "module_declaration",
    "object_declaration", "decorated_definition",
}
_CLASS_HINTS = ("class", "struct", "interface", "trait", "impl", "module", "object")
_CALL_TYPES = {"call_expression", "call", "method_invocation", "function_call"}
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def supported(suffix: str) -> bool:
    return suffix.lower() in _SUFFIXES


def _parser(language: str):
    from tree_sitter_language_pack import get_parser
    return get_parser(language)


def _name(node, source: bytes) -> str:
    named = node.child_by_field_name("name")
    if named is not None:
        return source[named.start_byte:named.end_byte].decode("utf-8", "replace")
    declarator = node.child_by_field_name("declarator")
    if declarator is not None:
        found = _IDENT.search(source[declarator.start_byte:declarator.end_byte]
                              .decode("utf-8", "replace"))
        if found:
            return found.group(0)
    for child in node.named_children:
        if child.type in ("identifier", "field_identifier", "type_identifier"):
            return source[child.start_byte:child.end_byte].decode("utf-8", "replace")
    return ""


def parse(rel: str, text: str):
    """Return ``(chunks, symbols, references)`` or ``None`` when unsupported.

    ``chunks`` are ``(start_line, end_line, body, symbol, kind)``;
    ``symbols`` are ``(name, kind, start_line, end_line)``;
    ``references`` maps a called name to ``rel``.
    """
    language = _SUFFIXES.get(rel.rsplit(".", 1)[-1].lower() and "." + rel.rsplit(".", 1)[-1].lower())
    if not language:
        return None
    try:
        parser = _parser(language)
        source = text.encode("utf-8")
        tree = parser.parse(source)
    except Exception:
        return None

    lines = text.splitlines()
    definitions = []
    references: dict[str, list[str]] = {}

    def walk(node):
        if node.type in _DEF_TYPES:
            name = _name(node, source)
            if name:
                definitions.append((node, name))
        if node.type in _CALL_TYPES:
            fn = node.child_by_field_name("function") or node.child_by_field_name("name")
            if fn is not None:
                ident = _IDENT.search(source[fn.start_byte:fn.end_byte]
                                      .decode("utf-8", "replace"))
                if ident:
                    references.setdefault(ident.group(0), []).append(rel)
        for child in node.children:
            walk(child)

    walk(tree.root_node)

    # Keep the outermost definition when a decorated/unwrapped node repeats.
    definitions.sort(key=lambda item: (item[0].start_byte, -item[0].end_byte))
    chunks, symbols = [], []
    last_end = -1
    for node, name in definitions:
        if node.start_byte <= last_end:
            continue
        last_end = node.end_byte
        start = node.start_point[0] + 1
        end = node.end_point[0] + 1
        kind = "class" if any(h in node.type for h in _CLASS_HINTS) else "function"
        body = "\n".join(lines[start - 1:end])
        if body.strip():
            chunks.append((start, end, body, name, kind))
            symbols.append((name, kind, start, end))
    return chunks, symbols, references

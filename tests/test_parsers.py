from rbg import parsers


def test_javascript_symbols_and_references():
    out = parsers.parse("a.js", "class A {\n  greet(x) { return x; }\n}\nfunction helper(y) { return A(y); }\n")
    assert out is not None
    chunks, symbols, refs = out
    assert [s[0] for s in symbols] == ["A", "helper"]
    assert "A" in refs


def test_go_and_rust_functions():
    go = parsers.parse("a.go", "package p\nfunc Add(a int) int { return a }\n")
    assert go and go[1][0][0] == "Add"
    rust = parsers.parse("a.rs", "fn alpha() -> i32 { 1 }\nstruct S;\n")
    assert rust and {s[0] for s in rust[1]} >= {"alpha", "S"}


def test_unsupported_suffix_falls_back():
    assert parsers.parse("a.xyz", "text") is None

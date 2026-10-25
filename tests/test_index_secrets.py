"""Credential-shaped files never enter the retrieval index."""

from rbg.index import RepoIndex


def test_secret_files_are_not_indexed(tmp_path):
    (tmp_path / "normal.py").write_text("def ok():\n    return 1\n")
    (tmp_path / ".env").write_text("API_KEY=supersecret\n")
    (tmp_path / ".env.production").write_text("TOKEN=abc\n")
    (tmp_path / "id_rsa").write_text("private\n")
    (tmp_path / "server.pem").write_text("cert\n")
    idx = RepoIndex.build(str(tmp_path))
    assert "normal.py" in idx.files
    for name in (".env", ".env.production", "id_rsa", "server.pem"):
        assert name not in idx.files


def test_override_allows_sensitive(tmp_path, monkeypatch):
    monkeypatch.setenv("RBG_ALLOW_SENSITIVE", "1")
    (tmp_path / "credentials.json").write_text("{\"k\": 1}\n")
    idx = RepoIndex.build(str(tmp_path))
    assert "credentials.json" in idx.files

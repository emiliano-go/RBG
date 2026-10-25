import numpy as np

from rbg.retrieval_static import StaticCodeEncoder


class _Model:
    def encode(self, texts):
        return np.asarray([[len(t), 1.0] for t in texts], dtype=float)


def test_static_encoder_scores_documents():
    enc = StaticCodeEncoder(path=None, dim=2)
    enc.model = _Model()
    chunks = [type("C", (), {"content": "a", "content_hash": "a"})(),
              type("C", (), {"content": "longer", "content_hash": "b"})()]
    ranked = enc.rank("longer", chunks, limit=2)
    assert ranked and ranked[0][0] == 1
    assert enc.document_vectors(chunks).shape == (2, 2)


def test_missing_model_returns_no_encoder(monkeypatch, tmp_path):
    monkeypatch.setenv("RBG_STATIC_MODEL", str(tmp_path / "missing"))
    from rbg.retrieval_static import encoder
    assert encoder() is None

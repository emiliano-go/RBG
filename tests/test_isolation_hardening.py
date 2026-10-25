from pathlib import Path

import pytest

from rbg import isolation
from rbg.verify import Sandbox


def test_prefix_unshares_and_drops_caps(tmp_path):
    if isolation.available_backend() is None:
        pytest.skip("no isolation backend on this host")
    iso = isolation.select(False, str(tmp_path))
    assert "--unshare-all" in iso.prefix
    assert "--cap-drop" in iso.prefix
    assert "--tmpfs" in iso.prefix


def test_sandbox_blocks_writes_outside_worktree_and_network(tmp_path):
    if isolation.available_backend() is None:
        pytest.skip("no isolation backend on this host")
    root = tmp_path / "repo"
    root.mkdir()
    code = (
        "import os,socket\n"
        "try:\n"
        " open('/usr/.rbg-escape','w').write('x'); print('WROTE')\n"
        "except OSError: print('READONLY')\n"
        "s=socket.socket(); s.settimeout(0.5)\n"
        "try:\n"
        " s.connect(('1.1.1.1',80)); print('NET')\n"
        "except OSError: print('OFFLINE')\n"
        f"print('HOME', os.path.exists({str(Path.home() / '.ssh')!r}))\n"
    )
    res = Sandbox(str(root)).run(["python3", "-c", code])
    assert res.isolated
    assert "READONLY" in res.stdout and "OFFLINE" in res.stdout
    assert "HOME False" in res.stdout

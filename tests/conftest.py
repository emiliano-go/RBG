"""Test bootstrap.

Isolation is required in production, but a host without bwrap/nsjail would
otherwise fail every test with IsolationUnavailable.  When no backend binary
exists, opt into the documented development escape hatch so the suite still
runs (isolation-specific behaviour is asserted explicitly in test_sandbox).
"""

import os
import shutil
import tempfile

if not (shutil.which("bwrap") or shutil.which("nsjail")):
    os.environ.setdefault("RBG_ALLOW_UNISOLATED", "1")

# Test repositories must not depend on the developer's global commit signing.
if "GIT_CONFIG_COUNT" not in os.environ:
    os.environ["GIT_CONFIG_COUNT"] = "1"
    os.environ["GIT_CONFIG_KEY_0"] = "commit.gpgsign"
    os.environ["GIT_CONFIG_VALUE_0"] = "false"

# The MCP server denies by default; tests opt in to the temp dir and this repo.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault(
    "RBG_ALLOWED_ROOTS", os.pathsep.join({tempfile.gettempdir(), _REPO}))

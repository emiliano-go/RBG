"""Test bootstrap.

Isolation is required in production, but a host without bwrap/nsjail would
otherwise fail every test with IsolationUnavailable.  When no backend binary
exists, opt into the documented development escape hatch so the suite still
runs (isolation-specific behaviour is asserted explicitly in test_sandbox).
"""

import os
import shutil

if not (shutil.which("bwrap") or shutil.which("nsjail")):
    os.environ.setdefault("HARNESS_ALLOW_UNISOLATED", "1")

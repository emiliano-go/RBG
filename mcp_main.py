#!/usr/bin/env python3
"""Launcher for the RBG MCP server (stdio transport).

Referenced by opencode's global config so the server runs regardless of the
client's working directory.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rbg.mcp_server import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

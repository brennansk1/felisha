"""Allow ``python -m causalrag.tui`` to launch the TUI directly.

Flags:
    --auto       Mount the live queue + chain-forest panels (for use
                 with /auto run).
    <project>    Optional project directory (defaults to CWD).
"""

from __future__ import annotations

import sys
from pathlib import Path

from causalrag.tui.app import run

if __name__ == "__main__":
    args = sys.argv[1:]
    auto_mode = "--auto" in args
    args = [a for a in args if a != "--auto"]
    project_dir = Path(args[0]).resolve() if args else None
    run(project_dir=project_dir, auto_mode=auto_mode)

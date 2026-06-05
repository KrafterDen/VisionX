"""Advanced autonomous kegelring package."""

from __future__ import annotations

import sys
from pathlib import Path


# Existing Visioners modules use top-level imports such as ``from config import ...``.
# Keep the parent Visioners directory ahead of this package so those imports keep
# resolving to Visioners/config.py instead of Visioners/advanced/config.py.
VISIONERS_DIR = Path(__file__).resolve().parents[1]
if str(VISIONERS_DIR) not in sys.path:
    sys.path.insert(0, str(VISIONERS_DIR))


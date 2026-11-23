"""Compatibility re-export for TankNodes manga/MOSS layout helpers."""

from pathlib import Path
import sys

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from manga import moss_nodes as _moss_mod

globals().update({k: v for k, v in vars(_moss_mod).items() if not k.startswith("__")})
__all__ = [k for k in vars(_moss_mod) if not k.startswith("__")]

"""Exec a local model server that exits when its ComfyUI parent dies (Linux)."""

from __future__ import annotations

import ctypes
import os
import signal
import sys


def main() -> None:
    parent = int(sys.argv[1])
    binary = sys.argv[2]
    if sys.platform.startswith("linux"):
        # This helper is a fresh Python process, so no ComfyUI threads exist
        # when prctl is called. The kernel then stops the server even if the
        # ComfyUI parent is killed without running Python cleanup hooks.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG failed")
    if os.getppid() != parent:
        raise RuntimeError("ComfyUI exited before the model server started")
    os.execv(binary, [binary, *sys.argv[3:]])


if __name__ == "__main__":
    main()

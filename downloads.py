"""Resumable HTTPS downloads used to set up models and runtimes on first use."""
from __future__ import annotations

import hashlib
import os
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

_LOCK = threading.Lock()
_CHUNK = 8 * 1024 * 1024
Progress = Callable[[int, int | None], None]


def auto_download_enabled() -> bool:
    """Downloads are on by default; TANKNODES_AUTO_DOWNLOAD=0 disables them."""
    value = os.environ.get("TANKNODES_AUTO_DOWNLOAD", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def hf_url(repo: str, filename: str) -> str:
    """Hugging Face file URL; HF_ENDPOINT selects a mirror such as hf-mirror.com."""
    base = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return f"{base}/{repo}/resolve/main/{urllib.parse.quote(filename)}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _console_progress(label: str) -> Progress:
    state = {"time": 0.0}

    def report(done: int, total: int | None) -> None:
        now = time.monotonic()
        if now - state["time"] < 5 and (total is None or done < total):
            return
        state["time"] = now
        size = f"{done / 2**30:.2f}/{total / 2**30:.2f} GiB ({done * 100 // total}%)" if total else f"{done / 2**30:.2f} GiB"
        print(f"[TankNodes] Downloading {label}: {size}", flush=True)

    return report


def fetch(url: str, dest: Path, *, sha256: str | None = None, label: str | None = None,
          progress: Progress | None = None) -> Path:
    """Download ``url`` to ``dest`` through ``dest.part``, resuming partial files.

    The final file appears only after the size (and SHA-256 when given) checks
    pass, so an interrupted download is never mistaken for a usable model.
    """
    dest = Path(dest)
    label = label or dest.name
    report = progress or _console_progress(label)
    with _LOCK:
        if dest.is_file():
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        done = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "ComfyUI-TankNodes"}
        if done:
            headers["Range"] = f"bytes={done}-"
        print(f"[TankNodes] Downloading {label} from {urllib.parse.urlsplit(url).netloc}", flush=True)
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
            if done and response.status != 206:
                done = 0  # The server ignored the range request; start again.
            length = response.headers.get("Content-Length")
            total = done + int(length) if length else None
            with part.open("ab" if done else "wb") as file:
                while True:
                    block = response.read(_CHUNK)
                    if not block:
                        break
                    file.write(block)
                    done += len(block)
                    report(done, total)
        if total is not None and done != total:
            raise RuntimeError(f"Download of {label} stopped at {done} of {total} bytes; run again to resume.")
        if sha256 and _sha256(part) != sha256:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch for {label}; the partial file was removed.")
        os.replace(part, dest)
        print(f"[TankNodes] Downloaded {label}", flush=True)
        return dest

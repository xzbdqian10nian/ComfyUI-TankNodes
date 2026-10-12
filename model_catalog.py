"""Local GGUF discovery and descriptive catalogue data for TankNodes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .downloads import auto_download_enabled, fetch, hf_url

try:
    import folder_paths
except Exception:  # pragma: no cover - only used by standalone tests
    folder_paths = None


PLUGIN_DIR = Path(__file__).resolve().parent
CATALOG_PATH = PLUGIN_DIR / "models.json"
DEFAULT_MODEL_SUBDIR = Path("LLM/Qwen3.8")
DEFAULT_MODEL = "Qwen3.8-27B-UD-Q4_K_M.gguf"
DEFAULT_MMPROJ = "mmproj-BF16.gguf"
BONSAI_MMPROJ = "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"
BONSAI_MODELS = {
    "Ternary-Bonsai-2-27B-PQ2_0.gguf": "stock",
    "Ternary-Bonsai-2-27B-Abliterated-v2-PQ2_0-MTP.gguf": "abliterated_v2",
}
MISSING_FILES = {
    "model": "(No model found / 未找到主模型)",
    "mmproj": "(No projector found / 未找到视觉投影)",
}


def _load_catalog() -> dict[str, Any]:
    try:
        return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


CATALOG = _load_catalog()


def _known_files() -> dict[str, dict[str, str]]:
    """Catalogue filenames that can be downloaded on first use, by kind."""
    known: dict[str, dict[str, str]] = {"model": {}, "mmproj": {}}
    for source in CATALOG.get("sources", []):
        repo = source.get("repo")
        if source.get("model") and repo:
            known["model"].setdefault(source["model"], repo)
        if source.get("mmproj") and repo:
            known["mmproj"].setdefault(source["mmproj"], source.get("mmproj_repo", repo))
    return known


KNOWN_FILES = _known_files()


def _model_root() -> Path:
    configured = os.environ.get("QWEN38_MODEL_DIR")
    if configured:
        return Path(configured).expanduser()
    if folder_paths is not None:
        return Path(folder_paths.models_dir) / CATALOG.get("model_subdir", DEFAULT_MODEL_SUBDIR)
    # Standalone fallback: custom_nodes/<plugin> is two levels below the
    # ComfyUI root, so keep the same portable models/ layout without knowing
    # anything about a platform-specific storage mount.
    return PLUGIN_DIR.parents[1] / "models" / CATALOG.get("model_subdir", DEFAULT_MODEL_SUBDIR)


def _is_complete_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 1024 * 1024 and not Path(f"{path}.aria2").exists()
    except OSError:
        return False


def _choices(kind: str) -> list[str]:
    """Return local GGUF files plus downloadable catalogue files, default first."""
    root = _model_root()
    found: list[str] = []
    if root.exists():
        for path in root.rglob("*"):
            if path.suffix.lower() != ".gguf":
                continue
            if ("mmproj" in path.name.lower()) != (kind == "mmproj"):
                continue
            if _is_complete_file(path):
                found.append(path.relative_to(root).as_posix())

    preferred = DEFAULT_MMPROJ if kind == "mmproj" else DEFAULT_MODEL
    configured = CATALOG.get("default_mmproj" if kind == "mmproj" else "default_model", preferred)
    local = set(found)
    names = local | set(KNOWN_FILES[kind])
    ordered = sorted(names, key=lambda name: (name != configured, name not in local, name.lower()))
    return ordered or [MISSING_FILES[kind]]


def _resolve_file(filename: str, kind: str) -> Path:
    """Resolve a selected model file and reject incomplete downloads."""
    if not filename or filename.startswith("("):
        raise FileNotFoundError(f"No {kind} file found. Add matching GGUF files to {_model_root()} and refresh the model list.")
    path = Path(filename).expanduser()
    if not path.is_absolute():
        path = _model_root() / path
    if Path(f"{path}.aria2").exists():
        raise RuntimeError(f"{kind} is still downloading: {path}")
    repo = KNOWN_FILES.get(kind, {}).get(filename)
    if not path.is_file() and repo:
        if not auto_download_enabled():
            raise FileNotFoundError(f"{kind} not found: {path}. Download {filename} from https://huggingface.co/{repo} or unset TANKNODES_AUTO_DOWNLOAD.")
        fetch(hf_url(repo, filename), path, label=filename)
    if not path.is_file():
        raise FileNotFoundError(f"{kind} not found: {path}")
    if path.stat().st_size <= 1024 * 1024:
        raise RuntimeError(f"{kind} is incomplete or too small to be a model: {path}")
    # Keep ComfyUI's configured path instead of exposing the target of a
    # platform-specific storage symlink in errors and status messages.
    return path.absolute()


def selection_info(model_file: str, mmproj_file: str) -> str:
    """Describe known releases without rewriting the selected file paths."""
    name = Path(model_file).name
    for source in CATALOG.get("sources", []):
        if name in {source.get("model"), source.get("alternative_model")}:
            expected = source.get("mmproj", "")
            return (
                f"source={source.get('repo', '')}\n"
                f"variant={source.get('name', '')}\n"
                f"recommended_mmproj={expected}\n"
                f"projector_filename_match={Path(mmproj_file).name == expected}"
            )
    return "source=unlisted local file; verify model/projector compatibility"


def bonsai_variant(model_file: str) -> str | None:
    return BONSAI_MODELS.get(Path(model_file).name)

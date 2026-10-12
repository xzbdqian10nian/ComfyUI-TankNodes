"""Locate or install the PrismML llama-server used for local GGUF chat.

Lookup order: ``BONSAI_LLAMA_SERVER`` -> this plugin's ``runtime/`` folder ->
download the pinned official PrismML release for this platform.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

from .downloads import auto_download_enabled, fetch

RELEASE = "prism-b10743-adfffbe"
RELEASE_URL = f"https://github.com/PrismML-Eng/llama.cpp/releases/download/{RELEASE}/"
PLUGIN_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = PLUGIN_DIR / "runtime"
_P = f"llama-{RELEASE}-bin-"
_CUDART = "cudart-llama-bin-win-cuda-"
# Official release assets and their GitHub SHA-256 digests.
ASSETS: dict[str, list[tuple[str, str]]] = {
    "linux-cuda-12.4": [(_P + "linux-cuda-12.4-x64.tar.gz", "fef4c7e8d83ff261d89809c1b302fe5f826adc1a74e13654b67ec056fbc1c639")],
    "linux-cuda-12.8": [(_P + "linux-cuda-12.8-x64.tar.gz", "43b73a24d5cd83c4482750ee52e59afac497c669c008a319e44e43a0033757e2")],
    "linux-cuda-13.3": [(_P + "linux-cuda-13.3-x64.tar.gz", "bfa6e1e0af2f21295e863cf26444c49b5d0094221ea785836cd810325fca2159")],
    "linux-cpu": [(_P + "ubuntu-x64.tar.gz", "1bb340929fddae8667c97ec6d4064a5768fe1dd08eeee0074b8d5e08f07dfc31")],
    "windows-cuda-12.4": [
        (_P + "win-cuda-12.4-x64.zip", "1b849f713bee42fda258de83770cd422e8f48dd631ce370eb0641f6458c69d87"),
        (_CUDART + "12.4-x64.zip", "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"),
    ],
    "windows-cuda-13.3": [
        (_P + "win-cuda-13.3-x64.zip", "c4f937e2a873180db5cceafca08fae73b5fa0b3fbbf020aeb555c2efe424c7bb"),
        (_CUDART + "13.3-x64.zip", "1462a050eb4c684921ba51dcc4cc488a036674c3e73e9945ee705b854808d03e"),
    ],
    "macos-arm64": [(_P + "macos-arm64.tar.gz", "596d257973080ca5011a4be50477c5f93ed1d231fcccfc2afb44bb573bb9629a")],
}
MANUAL_HELP = (
    f"Download a matching archive from https://github.com/PrismML-Eng/llama.cpp/releases/tag/{RELEASE} "
    f"and extract it into {RUNTIME_DIR / RELEASE}, or set BONSAI_LLAMA_SERVER to the llama-server executable."
)


class RuntimeUnavailable(RuntimeError):
    pass


def _executable_name(system: str | None = None) -> str:
    return "llama-server.exe" if (system or platform.system()) == "Windows" else "llama-server"


def _torch_cuda() -> str | None:
    try:
        import torch
        return torch.version.cuda if torch.cuda.is_available() else None
    except Exception:
        return None


def select_build(system: str | None = None, machine: str | None = None, cuda: str | None = "detect") -> str:
    """Pick the release build matching the OS, CPU and PyTorch CUDA version."""
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    cuda = _torch_cuda() if cuda == "detect" else cuda
    version = tuple(int(part) for part in cuda.split(".")[:2]) if cuda else None
    if system == "Linux" and machine in {"x86_64", "amd64"}:
        if version is None:
            return "linux-cpu"
        if version >= (13, 0):
            return "linux-cuda-13.3"
        return "linux-cuda-12.8" if version >= (12, 8) else "linux-cuda-12.4"
    if system == "Windows" and machine in {"amd64", "x86_64"} and version is not None:
        return "windows-cuda-13.3" if version >= (13, 0) else "windows-cuda-12.4"
    if system == "Darwin" and machine == "arm64":
        return "macos-arm64"
    raise RuntimeUnavailable(f"No prebuilt PrismML llama-server for {system}/{machine} (CUDA {cuda}). {MANUAL_HELP}")


def find_server() -> Path | None:
    configured = os.environ.get("BONSAI_LLAMA_SERVER")
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return path
        raise RuntimeUnavailable(f"BONSAI_LLAMA_SERVER points to a missing file: {path}")
    name = _executable_name()
    preferred = sorted((RUNTIME_DIR / RELEASE).rglob(name)) if (RUNTIME_DIR / RELEASE).is_dir() else []
    others = sorted(RUNTIME_DIR.rglob(name)) if RUNTIME_DIR.is_dir() else []
    for path in [*preferred, *others]:
        if path.is_file():
            return path
    return None


def _cuda_library_dirs() -> list[Path]:
    """CUDA libraries shipped inside PyTorch's pip wheels (nvidia-* packages)."""
    dirs: list[Path] = []
    try:
        import nvidia  # namespace package installed with CUDA builds of torch
        for base in nvidia.__path__:
            dirs.extend(sorted(Path(base).glob("*/lib")))
    except Exception:
        pass
    try:
        import torch
        dirs.append(Path(torch.__file__).parent / "lib")
    except Exception:
        pass
    return [path for path in dirs if path.is_dir()]


def _missing_libraries(files: list[Path]) -> set[str]:
    missing: set[str] = set()
    for file in files:
        try:
            output = subprocess.run(["ldd", str(file)], capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            return set()
        missing.update(line.split()[0] for line in output.splitlines() if "not found" in line)
    return missing


def link_cuda_libraries(server: Path) -> list[str]:
    """Make CUDA runtime libraries from PyTorch visible to llama-server.

    The official Linux builds load libcudart/libcublas from the system, which
    most ComfyUI installs only have inside PyTorch's wheels. The binaries use
    RUNPATH=$ORIGIN, so links placed beside them are enough, also for other
    plugins that start the same executable directly.
    """
    if platform.system() != "Linux" or shutil.which("ldd") is None:
        return []
    home = server.parent
    linked: list[str] = []
    for _ in range(3):  # libcublas itself needs libcublasLt, so resolve in rounds
        missing = _missing_libraries([server, *home.glob("libggml-cuda*.so")])
        added = False
        for name in sorted(missing):
            for directory in _cuda_library_dirs():
                candidate = directory / name
                if candidate.exists():
                    link = home / name
                    if link.is_symlink() or link.exists():
                        link.unlink()
                    link.symlink_to(candidate)
                    linked.append(name)
                    added = True
                    break
        if not added:
            break
    return linked


def _safe_extract(archive: Path, target: Path) -> None:
    target = target.resolve()
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                if not (target / member).resolve().is_relative_to(target):
                    raise RuntimeUnavailable(f"Unsafe path in {archive.name}: {member}")
            bundle.extractall(target)
        return
    with tarfile.open(archive) as bundle:
        if hasattr(tarfile, "data_filter"):
            bundle.extractall(target, filter="data")
            return
        for member in bundle.getmembers():
            path = (target / member.name).resolve()
            link_ok = not (member.issym() or member.islnk()) or (path.parent / member.linkname).resolve().is_relative_to(target)
            if not path.is_relative_to(target) or not link_ok:
                raise RuntimeUnavailable(f"Unsafe path in {archive.name}: {member.name}")
        bundle.extractall(target)


def install(build: str | None = None) -> Path:
    """Download, verify and extract the pinned release into ``runtime/``."""
    build = build or select_build()
    target = RUNTIME_DIR / RELEASE
    staging = RUNTIME_DIR / ".downloads"
    extracting = RUNTIME_DIR / f".{RELEASE}.partial"
    shutil.rmtree(extracting, ignore_errors=True)
    extracting.mkdir(parents=True)
    for name, digest in ASSETS[build]:
        archive = fetch(RELEASE_URL + name, staging / name, sha256=digest, label=name)
        _safe_extract(archive, extracting)
    servers = list(extracting.rglob(_executable_name()))
    if not servers:
        raise RuntimeUnavailable(f"{build} archive did not contain llama-server. {MANUAL_HELP}")
    # Windows CUDA runtime DLLs must sit beside llama-server.exe.
    home = servers[0].parent
    for dll in extracting.rglob("*.dll"):
        if dll.parent != home and not (home / dll.name).exists():
            shutil.move(str(dll), home / dll.name)
    servers[0].chmod(servers[0].stat().st_mode | 0o111)
    shutil.rmtree(target, ignore_errors=True)
    extracting.rename(target)
    link_cuda_libraries(next(target.rglob(_executable_name())))
    shutil.rmtree(staging, ignore_errors=True)
    found = find_server()
    if found is None:
        raise RuntimeUnavailable(f"llama-server was installed but not found under {target}.")
    return found


def ensure_server() -> Path:
    found = find_server()
    if found is not None:
        if found.is_relative_to(RUNTIME_DIR):
            link_cuda_libraries(found)  # repair links after a PyTorch upgrade
        return found
    if not auto_download_enabled():
        raise RuntimeUnavailable(f"PrismML llama-server is not installed and TANKNODES_AUTO_DOWNLOAD=0. {MANUAL_HELP}")
    print(f"[TankNodes] Installing PrismML llama-server {RELEASE} into {RUNTIME_DIR} (one time)", flush=True)
    return install()


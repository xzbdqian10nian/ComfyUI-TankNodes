import importlib.util
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import pytest


if "folder_paths" not in sys.modules:
    try:
        import folder_paths as _folder_paths  # noqa: F401
    except ImportError:
        _stub = ModuleType("folder_paths")
        _stub.models_dir = tempfile.gettempdir()
        _stub.get_temp_directory = lambda: tempfile.gettempdir()
        _stub.get_user_directory = lambda: tempfile.gettempdir()
        _stub.get_output_directory = lambda: tempfile.gettempdir()
        sys.modules["folder_paths"] = _stub


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("tanknodes_test_package", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
PLUGIN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLUGIN
SPEC.loader.exec_module(PLUGIN)
# Pytest otherwise imports the dotted/hyphenated checkout's __init__.py as a
# top-level module. Reuse the package already loaded exactly as ComfyUI does.
sys.modules.setdefault("__init__", PLUGIN)


@pytest.fixture
def plugin():
    return PLUGIN


@pytest.fixture
def module():
    return lambda name: importlib.import_module(f"tanknodes_test_package.{name}")

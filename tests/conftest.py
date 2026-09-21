import importlib.util
import sys
from pathlib import Path

import pytest


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

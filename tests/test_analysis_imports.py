"""Import check of every module in paper/analysis. Run: python -m pytest tests/test_analysis_imports.py -q
Each module is imported in a fresh Python process, the way it is loaded when the script is started from the command line: an import cycle (for example
between a module and paper/analysis/common.py) only shows up when that module is the first one imported in its process, so a shared process would hide it.
The check imports and nothing else: no data are read and no model is fitted."""
import glob
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS = os.path.join(ROOT, "paper", "analysis")
MODULES = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(ANALYSIS, "*.py")))


def test_every_analysis_module_is_found():
    assert "common" in MODULES and len(MODULES) >= 30


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_in_a_fresh_process(module):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([ROOT, ANALYSIS, os.environ.get("PYTHONPATH", "")]), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", f"import {module}"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"{module} failed to import:\n{r.stderr[-1500:]}"

"""pytest configuration shared by all tests.

(1) The results and cache folders of the whole test session are fixed to temporary folders. They are read from environment variables when
defectshift.config is first imported, so the assignment must come before the first import of defectshift in this file, which pytest loads before any
test file: the fake runs of the tests then stay out of the real results and cache folders and the working folder, even if DEFECTSHIFT_RESULTS_DIR and
DEFECTSHIFT_CACHE_DIR of a main run are still set in the shell. The NODE and TabNet temporary-folder variables are cleared too, so that the temporary
folders under the cache are used. Both temporary folders are removed when the tests end.
(2) A test (or a fixture) that fails only because the released data file is absent is reported as skipped, with the path of the missing file."""
import atexit
import os
import shutil
import tempfile

os.environ["DEFECTSHIFT_CACHE_DIR"] = tempfile.mkdtemp(prefix="defectshift_test_cache_")
os.environ["DEFECTSHIFT_RESULTS_DIR"] = tempfile.mkdtemp(prefix="defectshift_test_results_")
for _d in (os.environ["DEFECTSHIFT_CACHE_DIR"], os.environ["DEFECTSHIFT_RESULTS_DIR"]):   # removed at the end
    atexit.register(shutil.rmtree, _d, ignore_errors=True)
for _k in ("DEFECTSHIFT_NODE_TMP", "DEFECTSHIFT_TABNET_TMP"):
    os.environ.pop(_k, None)

import pytest  # noqa: E402

from defectshift import config  # noqa: E402


def _skip_if_data_missing(outcome):
    exc = outcome.excinfo[1] if outcome.excinfo else None
    if isinstance(exc, FileNotFoundError) and str(config.DATA_PATH) in str(exc):
        outcome.force_exception(pytest.skip.Exception(f"data file not found: {config.DATA_PATH} (see docs/usage.md)"))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    _skip_if_data_missing((yield))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    _skip_if_data_missing((yield))

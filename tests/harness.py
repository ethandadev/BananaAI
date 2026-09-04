"""A very small test harness.

pytest would work fine, but this keeps the test suite runnable with nothing
but the training dependencies -- useful on a fresh box, or inside WSL before
anything optional is installed.

Each test returns a short string describing what it verified, so a passing run
is a readable report rather than a wall of dots.
"""

from __future__ import annotations

import contextlib
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


class Suite:
    def __init__(self, name: str):
        self.name = name
        self.tests = []

    def test(self, fn):
        self.tests.append(fn)
        return fn

    def run(self, verbose: bool = True) -> tuple[int, int]:
        if verbose:
            print(f"\n  {self.name}")
        width = max((len(f.__name__) for f in self.tests), default=10)
        passed = 0
        for fn in self.tests:
            try:
                detail = fn() or ""
                passed += 1
                if verbose:
                    print(f"    [PASS]  {fn.__name__.ljust(width)}  {detail}")
            except AssertionError as e:
                if verbose:
                    print(f"    [FAIL]  {fn.__name__.ljust(width)}  {e}")
            except Exception:
                if verbose:
                    print(f"    [ERR ]  {fn.__name__.ljust(width)}  "
                          f"{traceback.format_exc().strip().splitlines()[-1]}")
        return passed, len(self.tests)


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def ensure_fixtures() -> None:
    """Generate the test corpus if it is not already on disk.

    The fixtures are derived from the repo itself, so they are built on demand
    rather than committed -- a stale checked-in corpus would drift out of step
    with the filters it is meant to exercise.
    """
    import importlib.util

    fixtures = REPO / "tests" / "fixtures"
    needed = [
        (fixtures / "corpus" / "expected.json", "build.py"),
        (fixtures / "prefs.jsonl", "build_post.py"),
    ]
    for marker, script in needed:
        if marker.exists():
            continue
        path = fixtures / script
        spec = importlib.util.spec_from_file_location(f"_fixture_{script[:-3]}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.build()


@contextlib.contextmanager
def temp_dir():
    """A temporary directory whose cleanup cannot fail the test.

    tempfile.TemporaryDirectory raises on Windows if anything still holds a
    handle -- and GGUFReader memory-maps the file it opens without exposing a
    way to close it. That is a cleanup detail, not a test result, so it must
    not turn a passing assertion into an error.
    """
    path = Path(tempfile.mkdtemp())
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)

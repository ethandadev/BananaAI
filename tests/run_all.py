"""Run every test suite.

    python tests/run_all.py
    python tests/run_all.py --quick     # skip the slow integration suite

Exit code is non-zero if anything fails, so this works as a CI gate.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FAST = [
    "tests.test_model",
    "tests.test_tokenizer",
    "tests.test_data",
    "tests.test_sample",
    "tests.test_training",
    "tests.test_post",
    "tests.test_export",
    "tests.test_sidecar",
]
SLOW = ["tests.test_integration"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--quick", action="store_true", help="skip the integration suite")
    p.add_argument("--only", help="run a single suite, e.g. export")
    args = p.parse_args(argv)

    modules = FAST + ([] if args.quick else SLOW)
    if args.only:
        modules = [m for m in modules if m.endswith(args.only)]
        if not modules:
            raise SystemExit(f"no suite matching {args.only!r}")

    t0 = time.perf_counter()
    total_passed = total_run = 0
    failed_suites = []

    for name in modules:
        module = importlib.import_module(name)
        passed, count = module.suite.run()
        total_passed += passed
        total_run += count
        if passed != count:
            failed_suites.append((module.suite.name, count - passed))

    elapsed = time.perf_counter() - t0
    print(f"\n  {'=' * 62}")
    if failed_suites:
        detail = ", ".join(f"{n} ({c} failing)" for n, c in failed_suites)
        print(f"  {total_passed}/{total_run} passed in {elapsed:.1f}s -- FAILURES in {detail}\n")
        return 1
    print(f"  {total_passed}/{total_run} passed in {elapsed:.1f}s\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

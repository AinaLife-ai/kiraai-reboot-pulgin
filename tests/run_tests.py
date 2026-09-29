#!/usr/bin/env python3
"""Run the reboot plugin test suite against a real KiraAI checkout.

Usage:
    KIRA_CORE_PATH=/path/to/KiraAI python3 tests/run_tests.py
    KIRA_CORE_PATH=/path/to/KiraAI python3 tests/run_tests.py --filter summary

Each test runs in a fresh event loop with its own sandbox, SessionManager and
plugin instance.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import harness  # noqa: E402

TEST_MODULES = [
    "test_reset",
    "test_summary",
    "test_commands",
    "test_tool_gate",
    "test_prompt_layout",
    "test_ads_compat",
    "test_version_bump",
    "test_config_migration",
]


def collect(module):
    tests = getattr(module, "TESTS", None)
    if tests:
        return list(tests)
    out = []
    for name, obj in vars(module).items():
        if name.startswith("test_") and (inspect.isfunction(obj) or inspect.iscoroutinefunction(obj)):
            out.append((name, obj))
    return sorted(out)


def run_one(fn):
    if inspect.iscoroutinefunction(fn):
        return asyncio.run(fn())
    fn()
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", default="", help="substring filter on test names")
    parser.add_argument("--module", default="", help="only run this test module")
    args = parser.parse_args()

    # Prepare paths/cwd before the framework modules are imported.
    state = harness.bootstrap()
    print(f"[harness] framework : {state['core_path']}")
    print(f"[harness] sandbox   : {state['sandbox']}")
    print(f"[harness] plugin pkg: {state['plugin_dir']}")
    print()

    passed, failed, skipped = [], [], []
    started = time.time()

    for module_name in TEST_MODULES:
        if args.module and args.module != module_name:
            continue
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            failed.append((module_name, "<import>", e, traceback.format_exc()))
            print(f"FAIL  {module_name}.<import>: {e}")
            continue
        for name, fn in collect(module):
            label = f"{module_name}.{name}"
            if args.filter and args.filter not in label:
                continue
            try:
                run_one(fn)
            except harness.FrameworkNotFound as e:
                skipped.append((label, str(e)))
                continue
            except Exception as e:
                failed.append((label, name, e, traceback.format_exc()))
                print(f"FAIL  {label}: {type(e).__name__}: {e}")
                continue
            passed.append(label)
            print(f"ok    {label}")

    duration = time.time() - started
    print()
    print("=" * 68)
    print(f"passed {len(passed)} | failed {len(failed)} | skipped {len(skipped)} | {duration:.2f}s")
    if failed:
        print()
        for label, _, e, tb in failed:
            print("-" * 68)
            print(f"FAILED: {label}\n{tb}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

"""Run the original regression and capture allocations after measurement."""
import json
import platform
import os
import pytest
import sys
import sysconfig
import tracemalloc
from pathlib import Path

from numpy._core.tests.test_stringdtype import (
    test_ufunc_fixed_width_out_releases_buffer,
)

assert platform.machine() == "aarch64"
assert sys.version_info[:3] == (3, 15, 2)
assert sysconfig.get_config_var("Py_GIL_DISABLED") == 1
print(sys.version, flush=True)
original_stop = tracemalloc.stop
snapshots = []


def capture_then_stop():
    # The helper has already saved its final byte count before stopping tracing.
    snapshot = tracemalloc.take_snapshot()
    original_stop()
    snapshots.append([
        {"traceback": str(item.traceback), "bytes": item.size, "count": item.count}
        for item in snapshot.statistics("traceback")[:25]
    ])


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    if item.name != "test_ufunc_fixed_width_out_releases_buffer":
        yield
        return
    tracemalloc.stop = capture_then_stop
    try:
        yield
    finally:
        tracemalloc.stop = original_stop
        output = Path(f"/evidence/suite-allocations-{os.getpid()}.json")
        output.write_text(json.dumps(snapshots, indent=2))
        snapshots.clear()


if __name__ == "__main__":
    tracemalloc.stop = capture_then_stop
    results = []
    try:
        for iteration in range(20):
            try:
                test_ufunc_fixed_width_out_releases_buffer()
            except AssertionError as exc:
                result = {"iteration": iteration, "outcome": "failed", "error": str(exc)}
            else:
                result = {"iteration": iteration, "outcome": "passed"}
            result["allocations_after_measurement"] = (snapshots.pop() if snapshots else [])
            results.append(result)
            print(json.dumps(result), flush=True)
    finally:
        tracemalloc.stop = original_stop
        Path("/evidence/probe.json").write_text(json.dumps(results, indent=2))
    sys.exit(any(item["outcome"] == "failed" for item in results))

"""Collect evidence for NumPy #33369 using an installed development build."""

import argparse
import json
import mmap
import os
from pathlib import Path
import platform
import subprocess
import sys
import tracemalloc

import numpy as np
from numpy._core.tests.test_stringdtype import _large_string_args


def _allocation_control():
    with mmap.mmap(-1, 32 * 1024**2):
        pass


def _control_args():
    return {"32-MiB-mapping": ()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform not in ("linux", "darwin"):
        parser.error("the memory-limit helper requires Linux or macOS")

    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "numpy_path": np.__file__,
        "allocator_environment": {
            key: os.environ.get(key)
            for key in ("PYTHONMALLOC", "MallocMediumZone", "MallocNanoZone")
        },
        "traced_allocations": [],
        "runs": [],
    }
    for label in tuple(_large_string_args()):
        function, *call_args = _large_string_args()[label]
        tracemalloc.start()
        output = function(*call_args)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result["traced_allocations"].append(
            {"label": label, "retained": current, "peak": peak}
        )
        del output, function, call_args

    locations = [
        ("strings", ("_operator", None, "call"),
         ("numpy._core.tests.test_stringdtype", None, "_large_string_args")),
        ("control", (Path(__file__).stem, __file__, "_allocation_control"),
         (Path(__file__).stem, __file__, "_control_args")),
    ]
    modes = ["default"]
    if sys.platform == "darwin":
        modes.append("medium-zone-disabled")
    for mode in modes:
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
        if mode == "medium-zone-disabled":
            env["MallocMediumZone"] = "0"
        for allowance in (6, 8, 12, 16):
            for kind, function_location, args_location in locations:
                code = (
                    "from numpy.testing._private.utils import "
                    "_run_with_memory_limit; "
                    f"_run_with_memory_limit({function_location!r}, "
                    f"{args_location!r}, {allowance * 1024**2})"
                )
                completed = subprocess.run(
                    [sys.executable, "-c", code], env=env,
                    capture_output=True, text=True, timeout=120,
                )
                record = {
                    "mode": mode, "allowance_MiB": allowance,
                    "kind": kind, "returncode": completed.returncode,
                    "stdout": completed.stdout, "stderr": completed.stderr,
                }
                result["runs"].append(record)
                args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(mode, allowance, kind, completed.returncode, flush=True)


if __name__ == "__main__":
    main()

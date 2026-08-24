#!/usr/bin/env python3
"""Run the reviewer-safe lifetime-reference development benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lifetime_reference_benchmark import (  # noqa: E402
    execute_benchmark,
    verify_benchmark_artifacts,
    write_benchmark_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "docs" / "lifetime-reference-benchmark" / "v1",
    )
    args = parser.parse_args()
    execution = execute_benchmark()
    result = write_benchmark_artifacts(execution, args.output_dir)
    verification = verify_benchmark_artifacts(args.output_dir)
    summary = result["summary"]
    print(
        "Completed lifetime-reference benchmark: "
        f"{summary['counts']['labelled_executed']} labelled, "
        f"{summary['counts']['shadow_executed']} safe shadow, "
        f"{summary['counts']['shadow_withheld_blind_overlap']} blind-overlap withheld."
    )
    print(f"Artifacts: {args.output_dir}")
    print(
        "Verified bundle: "
        f"{verification['setup_json_files']} setup JSON files; sealed overlap 0."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Convert finalized schema-v5 workbench exports into blind benchmark labels."""
from __future__ import annotations

import argparse
import json
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_v24 import (  # noqa: E402
    labels_from_review_exports,
    load_manifest,
    verify_freeze,
)
from coil_validation_v24 import ValidationConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exports", type=Path, nargs="+")
    parser.add_argument("--partition", choices=("development", "validation", "holdout"), required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "review_snapshots" / "benchmark_2026-08-13_v24_72",
    )
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "manifest.json"
    protocol_path = root / "protocol.json"
    manifest = load_manifest(manifest_path)
    if args.partition == "holdout":
        for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
            if subprocess.run(command, cwd=PROJECT_ROOT).returncode != 0:
                raise SystemExit("holdout import requires a clean tracked worktree")
        if args.freeze is None:
            raise SystemExit("holdout reveal requires --freeze")
        freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
        verify_freeze(
            freeze,
            manifest_path=manifest_path,
            protocol_path=protocol_path,
            selection_report_path=root / "validation-selection.json",
            current_code_commit=subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            config=ValidationConfig(
                **freeze["detectors"]["v2_4_validation"]["config"]
            ),
        )
    exports = [json.loads(path.read_text(encoding="utf-8")) for path in args.exports]
    internal = json.loads(
        (root / "review-task-manifest.internal.json").read_text(encoding="utf-8")
    )
    labels = labels_from_review_exports(
        exports,
        manifest,
        partition=args.partition,
        manifest_sha256=manifest["_manifest_sha256"],
        task_manifest=internal,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(labels, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create or extend the v2.4 benchmark configuration-freeze artifact."""
from __future__ import annotations

import argparse
import json
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_v24 import (  # noqa: E402
    file_sha256,
    make_freeze,
    register_holdout_labels,
    validate_selection_report,
)
from coil_validation_v24 import (  # noqa: E402
    DEFAULT_CONFIG,
    ValidationConfig,
    config_fingerprint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "review_snapshots" / "benchmark_2026-08-13_v24_72",
    )
    parser.add_argument("--register-holdout-labels", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "configuration-freeze.json"
    selection_report = root / "validation-selection.json"
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        if subprocess.run(command, cwd=PROJECT_ROOT).returncode != 0:
            raise SystemExit("configuration freeze requires a clean tracked worktree")
    if output.exists():
        freeze = json.loads(output.read_text(encoding="utf-8"))
        if args.register_holdout_labels is None:
            raise SystemExit("freeze already exists; labels were not registered")
    else:
        if not selection_report.is_file():
            raise SystemExit("configuration freeze requires validation-selection.json")
        selection = json.loads(selection_report.read_text(encoding="utf-8"))
        config = validate_selection_report(
            selection,
            manifest_path=root / "manifest.json",
            protocol_path=root / "protocol.json",
            current_code_commit=current_commit,
        )
        freeze = make_freeze(
            manifest_path=root / "manifest.json",
            protocol_path=root / "protocol.json",
            code_commit=current_commit,
            evaluation_command=str(protocol.get("evaluation_command") or ""),
            selection_report_sha256=file_sha256(selection_report),
            config=config,
        )
    if args.register_holdout_labels is not None:
        from benchmark_v24 import verify_freeze

        verify_freeze(
            freeze,
            manifest_path=root / "manifest.json",
            protocol_path=root / "protocol.json",
            selection_report_path=selection_report,
            current_code_commit=current_commit,
            config=ValidationConfig(
                **freeze["detectors"]["v2_4_validation"]["config"]
            ),
        )
        freeze = register_holdout_labels(
            freeze,
            args.register_holdout_labels,
            revealed_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
    output.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

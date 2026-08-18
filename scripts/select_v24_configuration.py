#!/usr/bin/env python3
"""Run the registered development sweep and one validation audit."""
from __future__ import annotations

import argparse
import json
import sys
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_v24 import (  # noqa: E402
    BENCHMARK_ID,
    detector_metrics,
    history_coverage_audit,
    file_sha256,
    label_stability,
    load_labels,
    load_manifest,
    registered_validation_configs,
    run_detectors,
    select_development_configuration,
)
from coil_validation_v24 import config_fingerprint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-labels", type=Path, required=True)
    parser.add_argument("--validation-labels", type=Path, required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "review_snapshots" / "benchmark_2026-08-13_v24_72",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "validation-selection.json"
    canonical_development_labels = root / "development-labels.json"
    canonical_validation_labels = root / "validation-labels.json"
    if args.development_labels.resolve() != canonical_development_labels.resolve():
        raise SystemExit(f"development labels must be {canonical_development_labels}")
    if args.validation_labels.resolve() != canonical_validation_labels.resolve():
        raise SystemExit(f"validation labels must be {canonical_validation_labels}")
    if output.exists():
        raise SystemExit("validation-selection.json already exists; validation may be used only once")
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        if subprocess.run(command, cwd=PROJECT_ROOT).returncode != 0:
            raise SystemExit("configuration selection requires a clean tracked worktree")
    manifest_path = root / "manifest.json"
    protocol_path = root / "protocol.json"
    manifest = load_manifest(manifest_path)
    coverage = history_coverage_audit(manifest)
    if not coverage["valid"]:
        raise SystemExit(
            "configuration selection requires verified listing-quarter history"
        )
    development = load_labels(
        args.development_labels,
        partition="development",
        manifest_path=manifest_path,
        protocol_path=protocol_path,
    )
    validation = load_labels(
        args.validation_labels,
        partition="validation",
        manifest_path=manifest_path,
        protocol_path=protocol_path,
    )
    stability = label_stability(development["labels"])
    if not stability["passed"]:
        raise SystemExit("label-quality gate failed; configuration sweep is prohibited")
    development_primary_count = sum(
        row.get("attempt_role", "primary") == "primary"
        for row in development["labels"]
    )
    validation_primary_count = sum(
        row.get("attempt_role", "primary") == "primary"
        for row in validation["labels"]
    )
    if development_primary_count != 36 or validation_primary_count != 18:
        raise SystemExit(
            "configuration selection requires 36 development and 18 validation primary labels"
        )

    development_manifest = {
        "items": [item for item in manifest["items"] if item["partition"] == "development"]
    }
    scored = []
    rows = []
    for config in registered_validation_configs():
        outputs = run_detectors(development_manifest, root, config=config)
        metrics = detector_metrics(
            development["labels"],
            outputs,
            variant="v2_4_validation",
            manifest=development_manifest,
        )
        scored.append((config, metrics))
        rows.append(
            {
                "config": asdict(config),
                "config_fingerprint": config_fingerprint(config),
                "development_metrics": metrics,
            }
        )
    selected_config, selected_development_metrics = select_development_configuration(scored)

    # The frozen protocol permits exactly one validation use, on the selected
    # development winner. Exclusive output creation makes accidental reruns
    # explicit rather than silently replacing that audit.
    validation_manifest = {
        "items": [item for item in manifest["items"] if item["partition"] == "validation"]
    }
    validation_outputs = run_detectors(
        validation_manifest, root, config=selected_config
    )
    validation_metrics = detector_metrics(
        validation["labels"],
        validation_outputs,
        variant="v2_4_validation",
        manifest=validation_manifest,
    )
    report = {
        "schema_version": 1,
        "kind": "coilingview.v24-development-selection",
        "benchmark_id": BENCHMARK_ID,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "manifest_sha256": manifest["_manifest_sha256"],
        "protocol_sha256": file_sha256(protocol_path),
        "code_commit": current_commit,
        "development_labels_sha256": file_sha256(canonical_development_labels),
        "validation_labels_sha256": file_sha256(canonical_validation_labels),
        "label_stability": stability,
        "ordering": [
            "major_top_precision",
            "pattern_precision",
            "band_agreement",
            "major_top_recall",
        ],
        "registered_configuration_count": len(rows),
        "development_sweep": rows,
        "selected": {
            "config": asdict(selected_config),
            "config_fingerprint": config_fingerprint(selected_config),
            "development_metrics": selected_development_metrics,
            "single_use_validation_metrics": validation_metrics,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit or score the frozen v2.4 benchmark without leaking holdout labels."""
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
    BENCHMARK_ID,
    DECISION_SCHEMA_VERSION,
    HoldoutSealedError,
    detector_metrics,
    gate_decision,
    label_stability,
    load_labels,
    load_manifest,
    point_in_time_audit,
    prediction_summary,
    run_detectors,
    validate_manifest,
)
from coil_validation_v24 import DEFAULT_CONFIG, ValidationConfig  # noqa: E402


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "review_snapshots" / "benchmark_2026-08-13_v24_72",
    )
    value.add_argument("--freeze", type=Path)
    value.add_argument("--holdout-labels", type=Path)
    value.add_argument("--repeat-labels", type=Path)
    value.add_argument("--output", type=Path)
    return value


def main() -> int:
    args = parser().parse_args()
    root = args.root.resolve()
    manifest_path = root / "manifest.json"
    protocol_path = root / "protocol.json"
    selection_report_path = root / "validation-selection.json"
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = load_manifest(manifest_path)
    config = DEFAULT_CONFIG
    if args.freeze:
        for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
            if subprocess.run(command, cwd=PROJECT_ROOT).returncode != 0:
                raise SystemExit("frozen evaluation requires a clean tracked worktree")
        freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
        raw_config = (
            (freeze.get("detectors") or {}).get("v2_4_validation") or {}
        ).get("config")
        if not isinstance(raw_config, dict):
            raise SystemExit("configuration freeze lacks the registered v2.4 config")
        config = ValidationConfig(**raw_config)
        from benchmark_v24 import verify_freeze

        verify_freeze(
            freeze,
            manifest_path=manifest_path,
            protocol_path=protocol_path,
            selection_report_path=selection_report_path,
            current_code_commit=current_commit,
            config=config,
        )
    outputs = run_detectors(manifest, root, config=config)
    operating_behavior = prediction_summary(manifest, outputs)
    pit = point_in_time_audit(manifest, root, outputs, config=config)

    stability = None
    if args.repeat_labels:
        repeat_labels = load_labels(
            args.repeat_labels,
            partition="development",
            manifest_path=manifest_path,
            protocol_path=protocol_path,
        )
        stability = label_stability(repeat_labels["labels"])

    v23_metrics = v24_metrics = None
    holdout_state = "not_requested"
    if args.holdout_labels:
        try:
            holdout = load_labels(
                args.holdout_labels,
                partition="holdout",
                manifest_path=manifest_path,
                protocol_path=protocol_path,
                freeze_path=args.freeze,
                selection_report_path=selection_report_path,
                current_code_commit=current_commit,
                config=config,
            )
        except HoldoutSealedError:
            holdout_state = "sealed"
        else:
            holdout_state = "revealed_after_verified_freeze"
            v23_metrics = detector_metrics(
                holdout["labels"], outputs, variant="v2_3_1", manifest=manifest
            )
            v24_metrics = detector_metrics(
                holdout["labels"], outputs, variant="v2_4_validation", manifest=manifest
            )

    invalid_accepted = 0
    for item in manifest["items"]:
        if not (item.get("data_quality") or {}).get("expected_hard_invalid"):
            continue
        result = outputs[item["sample_id"]]["v2_4_validation"]
        if result.get("abstained") is not True:
            invalid_accepted += 1
    validation = validate_manifest(manifest)
    decision = gate_decision(
        manifest_validation=validation,
        stability=stability,
        v24_metrics=v24_metrics,
        point_in_time=pit,
        accepted_hard_invalid=invalid_accepted,
    )
    report = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "kind": "coilingview.v24-benchmark-decision",
        "benchmark_id": BENCHMARK_ID,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "manifest_sha256": manifest["_manifest_sha256"],
        "holdout_state": holdout_state,
        "manifest_validation": validation,
        "point_in_time": pit,
        "paired_operating_behavior": operating_behavior,
        "label_stability": stability,
        "metrics": {"v2_3_1": v23_metrics, "v2_4_validation": v24_metrics},
        "decision": decision,
        "guardrails": {
            "production_default_unchanged": True,
            "advanced_ml_added": False,
            "production_trading_logic_added": False,
        },
    }
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0 if decision["outcome"] == "go" else 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_SCHEMA_VERSION = 1
PREDICTION_SCHEMA_VERSION = 1
REVIEW_SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_segment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    if not cleaned:
        raise ValueError("Path segment cannot be empty.")
    return cleaned


def make_run_id(now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"vision_{stamp}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class VisionRunStore:
    """Filesystem-backed store under ``coil-screener/vision_runs``."""

    def __init__(self, root: Path):
        self.root = root

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        return self.root / safe_segment(run_id)

    def prediction_path(self, run_id: str, ticker: str) -> Path:
        return self.run_dir(run_id) / "predictions" / f"{safe_segment(ticker.upper())}.json"

    def raw_path(self, run_id: str, ticker: str) -> Path:
        return self.run_dir(run_id) / "raw" / f"{safe_segment(ticker.upper())}.json"

    def mapped_path(self, run_id: str, ticker: str) -> Path:
        return self.run_dir(run_id) / "mapped" / f"{safe_segment(ticker.upper())}.json"

    def image_path(self, run_id: str, ticker: str) -> Path:
        return self.run_dir(run_id) / "images" / f"{safe_segment(ticker.upper())}.png"

    def debug_image_path(self, run_id: str, ticker: str) -> Path:
        return self.run_dir(run_id) / "debug" / f"{safe_segment(ticker.upper())}_annotated.png"

    def run_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "run.json"

    def manifest_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "manifest.json"

    def write_run(self, run_id: str, payload: dict[str, Any]) -> None:
        data = {"schema_version": RUN_SCHEMA_VERSION, **payload, "run_id": run_id}
        write_json(self.run_path(run_id), data)

    def write_manifest(self, run_id: str, payload: dict[str, Any]) -> Path:
        path = self.manifest_path(run_id)
        data = {"schema_version": RUN_SCHEMA_VERSION, **payload, "run_id": run_id}
        write_json(path, data)
        return path

    def read_run(self, run_id: str) -> dict[str, Any]:
        path = self.run_path(run_id)
        if not path.exists():
            raise FileNotFoundError(run_id)
        return read_json(path)

    def list_runs(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        runs: list[dict[str, Any]] = []
        for path in self.root.iterdir():
            if not path.is_dir():
                continue
            run_file = path / "run.json"
            if not run_file.exists():
                continue
            try:
                run = read_json(run_file)
            except json.JSONDecodeError:
                continue
            run.setdefault("run_id", path.name)
            runs.append(run)
        return sorted(
            runs,
            key=lambda run: str(run.get("created_at") or run.get("run_id") or ""),
            reverse=True,
        )

    def latest_run_id(self) -> str | None:
        runs = self.list_runs()
        return str(runs[0]["run_id"]) if runs else None

    def write_raw(self, run_id: str, ticker: str, payload: dict[str, Any]) -> Path:
        path = self.raw_path(run_id, ticker)
        write_json(path, payload)
        return path

    def write_mapped(self, run_id: str, ticker: str, payload: dict[str, Any]) -> Path:
        path = self.mapped_path(run_id, ticker)
        write_json(path, payload)
        return path

    def write_prediction(self, run_id: str, ticker: str, payload: dict[str, Any]) -> Path:
        path = self.prediction_path(run_id, ticker)
        data = {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            **payload,
            "run_id": run_id,
            "ticker": ticker.upper(),
        }
        write_json(path, data)
        return path

    def read_prediction(
        self,
        ticker: str,
        interval: str | None = None,
        run_id: str = "latest",
    ) -> dict[str, Any]:
        symbol = ticker.upper()
        run_ids = [run_id]
        if run_id == "latest":
            run_ids = [str(run["run_id"]) for run in self.list_runs()]

        for rid in run_ids:
            path = self.prediction_path(rid, symbol)
            if not path.exists():
                continue
            prediction = read_json(path)
            if interval and prediction.get("interval") != interval:
                continue
            return prediction
        raise FileNotFoundError(f"No vision prediction for {symbol}.")

    def append_review(self, review: dict[str, Any]) -> dict[str, Any]:
        run_id = safe_segment(str(review["run_id"]))
        ticker = safe_segment(str(review["ticker"]).upper())
        path = self.run_dir(run_id) / "reviews" / f"{ticker}.jsonl"
        record = {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "created_at": utc_now_iso(),
            **review,
            "ticker": ticker,
            "run_id": run_id,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")

        prediction_path = self.prediction_path(run_id, ticker)
        if prediction_path.exists():
            prediction = read_json(prediction_path)
            prediction["review"] = record
            write_json(prediction_path, prediction)
        return record

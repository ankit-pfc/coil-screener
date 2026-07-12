from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .capture import CaptureConfig, capture_chart
from .inference import RoboflowHostedClient, normalize_detections, write_debug_annotation
from .mapping import map_detections_to_chart_points, map_detections_to_highs
from .storage import VisionRunStore, make_run_id, utc_now_iso
from .trendlines import suggest_resistance_trendlines


DEMO_DEFAULT_RUN = "demo_curated_coils_results.csv"


@dataclass(frozen=True)
class VisionRunConfig:
    project_root: Path
    tickers: list[str] | None = None
    saved_run: str | None = None
    interval: str = "3M"
    timeframe: str = "10Y"
    chart_type: str = "candles"
    base_url: str = "http://127.0.0.1:5173"
    limit: int | None = None
    run_id: str | None = None
    headless: bool = True
    max_highs: int = 3
    confidence: float = 0.35
    max_trendlines: int = 5
    touch_tolerance_pct: float = 1.5


def saved_run_files(project_root: Path) -> list[Path]:
    return sorted(
        project_root.glob("*.csv"),
        key=lambda path: (path.name == DEMO_DEFAULT_RUN, path.stat().st_mtime),
        reverse=True,
    )


def resolve_saved_run(project_root: Path, filename: str | None = None) -> Path:
    if filename:
        if "/" in filename or "\\" in filename:
            raise ValueError("Invalid saved run filename.")
        path = project_root / filename
        if path.exists() and path.suffix.lower() == ".csv":
            return path
        raise FileNotFoundError(filename)
    files = saved_run_files(project_root)
    if not files:
        raise FileNotFoundError("No saved screener runs found.")
    return files[0]


def tickers_from_saved_run(project_root: Path, filename: str | None, limit: int | None) -> list[str]:
    path = resolve_saved_run(project_root, filename)
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError(f"{path.name} does not contain a ticker column.")
    tickers = [
        str(value).strip().upper()
        for value in df["ticker"].tolist()
        if str(value).strip()
    ]
    if limit is not None:
        tickers = tickers[: max(0, limit)]
    return tickers


def resolve_tickers(config: VisionRunConfig) -> tuple[list[str], str | None]:
    if config.tickers:
        tickers = [ticker.strip().upper() for ticker in config.tickers if ticker.strip()]
        if config.limit is not None:
            tickers = tickers[: max(0, config.limit)]
        return tickers, None
    path = resolve_saved_run(config.project_root, config.saved_run)
    return tickers_from_saved_run(config.project_root, path.name, config.limit), path.name


def relative_to_run(path: Path, run_dir: Path) -> str:
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


CaptureFn = Callable[[CaptureConfig, Path], Any]


def _suggestion_bucket(mapped_points: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [point for point in mapped_points if point.get("kind") == kind]


def _legacy_trendline(
    trendlines: list[dict[str, Any]],
    mapped_highs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if trendlines:
        anchors = trendlines[0].get("anchors") or []
        if len(anchors) >= 2:
            return {
                "from": anchors[0],
                "to": anchors[1],
                "source": "vision",
                "confidence": trendlines[0].get("confidence"),
                "suggestion_id": trendlines[0].get("id"),
            }
    if len(mapped_highs) < 2:
        return None
    return {"from": mapped_highs[0], "to": mapped_highs[-1], "source": "vision"}


def run_vision_pipeline(
    config: VisionRunConfig,
    *,
    store: VisionRunStore | None = None,
    inference_client: Any | None = None,
    capture_fn: CaptureFn = capture_chart,
) -> dict[str, Any]:
    run_id = config.run_id or make_run_id()
    store = store or VisionRunStore(config.project_root / "vision_runs")
    store.ensure()
    run_dir = store.run_dir(run_id)
    tickers, saved_run_name = resolve_tickers(config)
    client = inference_client or RoboflowHostedClient(confidence=config.confidence)

    request_payload = {
        "tickers": tickers,
        "saved_run": saved_run_name or config.saved_run,
        "interval": config.interval,
        "timeframe": config.timeframe,
        "chart_type": config.chart_type,
        "base_url": config.base_url,
        "limit": config.limit,
        "max_highs": config.max_highs,
        "confidence": config.confidence,
        "max_trendlines": config.max_trendlines,
        "touch_tolerance_pct": config.touch_tolerance_pct,
    }
    run_payload: dict[str, Any] = {
        "created_at": utc_now_iso(),
        "completed_at": None,
        "status": "running",
        "request": request_payload,
        "summary": {"requested": len(tickers), "completed": 0, "failed": 0},
        "predictions": [],
        "errors": [],
    }
    store.write_run(run_id, run_payload)

    for ticker in tickers:
        image_path = store.image_path(run_id, ticker)
        debug_path = store.debug_image_path(run_id, ticker)
        try:
            capture = capture_fn(
                CaptureConfig(
                    base_url=config.base_url,
                    ticker=ticker,
                    interval=config.interval,
                    timeframe=config.timeframe,
                    chart_type=config.chart_type,
                    headless=config.headless,
                ),
                image_path,
            )
            raw_result = client.infer(image_path)
            raw_path = store.write_raw(run_id, ticker, raw_result)
            detections = [
                detection
                for detection in normalize_detections(raw_result)
                if float(detection.get("confidence", 0.0) or 0.0) >= config.confidence
            ]
            mapped_points = map_detections_to_chart_points(detections, capture.state)
            mapped_highs = map_detections_to_highs(
                detections,
                capture.state,
                max_highs=config.max_highs,
            )
            trendlines = suggest_resistance_trendlines(
                mapped_points,
                max_lines=config.max_trendlines,
                touch_tolerance_pct=config.touch_tolerance_pct,
            )
            trendline = _legacy_trendline(trendlines, mapped_highs)
            write_debug_annotation(image_path, detections, debug_path)

            mapped_payload = {
                "schema_version": 1,
                "run_id": run_id,
                "ticker": ticker,
                "interval": config.interval,
                "timeframe": config.timeframe,
                "chart_type": config.chart_type,
                "status": "suggested",
                "source_of_truth": "human_review",
                "suggestions": {
                    "major_highs": _suggestion_bucket(mapped_points, "major_high"),
                    "resistance_touches": _suggestion_bucket(mapped_points, "resistance_touch"),
                    "resistance_trendlines": trendlines,
                },
            }
            mapped_path = store.write_mapped(run_id, ticker, mapped_payload)

            prediction = {
                "ticker": ticker,
                "interval": config.interval,
                "timeframe": config.timeframe,
                "chart_type": config.chart_type,
                "captured_at": utc_now_iso(),
                "capture_url": capture.url,
                "image_path": relative_to_run(image_path, run_dir),
                "debug_image_path": relative_to_run(debug_path, run_dir),
                "raw_path": relative_to_run(raw_path, run_dir),
                "mapped_path": relative_to_run(mapped_path, run_dir),
                "raw_detections": detections,
                "mapped_highs": mapped_highs,
                "trendline": trendline,
                "suggestions": mapped_payload["suggestions"],
                "capture_state": capture.state,
                "review": {"status": "pending"},
            }
            prediction_path = store.write_prediction(run_id, ticker, prediction)
            run_payload["predictions"].append(
                {
                    "ticker": ticker,
                    "path": relative_to_run(prediction_path, run_dir),
                    "mapped_path": relative_to_run(mapped_path, run_dir),
                    "high_count": len(mapped_highs),
                    "trendline_count": len(trendlines),
                }
            )
            run_payload["summary"]["completed"] += 1
        except Exception as exc:  # noqa: BLE001 - per-ticker failures are recorded.
            run_payload["summary"]["failed"] += 1
            run_payload["errors"].append({"ticker": ticker, "error": str(exc)})
        finally:
            store.write_run(run_id, run_payload)

    run_payload["completed_at"] = utc_now_iso()
    run_payload["status"] = "completed" if run_payload["summary"]["failed"] == 0 else "completed_with_errors"
    store.write_run(run_id, run_payload)
    store.write_manifest(
        run_id,
        {
            "created_at": run_payload["created_at"],
            "completed_at": run_payload["completed_at"],
            "status": run_payload["status"],
            "request": request_payload,
            "defaults": {
                "run": "latest_saved_screener_run",
                "interval": "3M",
                "range": "10Y",
                "chart_type": "candles",
            },
            "files": {
                "run": "run.json",
                "predictions_dir": "predictions",
                "raw_predictions_dir": "raw",
                "mapped_predictions_dir": "mapped",
                "captures_dir": "images",
                "annotated_dir": "debug",
            },
            "predictions": run_payload["predictions"],
            "errors": run_payload["errors"],
        },
    )
    return store.read_run(run_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CoilingView chart vision tagging.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--base-url", default="http://127.0.0.1:5173")
    parser.add_argument("--saved-run")
    parser.add_argument("--ticker", action="append", dest="tickers")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--interval", default="3M")
    parser.add_argument("--timeframe", default="10Y")
    parser.add_argument("--chart-type", default="candles")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--max-highs", type=int, default=3)
    parser.add_argument("--max-trendlines", type=int, default=5)
    parser.add_argument("--touch-tolerance-pct", type=float, default=1.5)
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()
    run = run_vision_pipeline(
        VisionRunConfig(
            project_root=args.project_root,
            base_url=args.base_url,
            saved_run=args.saved_run,
            tickers=args.tickers,
            limit=args.limit,
            interval=args.interval,
            timeframe=args.timeframe,
            chart_type=args.chart_type,
            confidence=args.confidence,
            max_highs=args.max_highs,
            max_trendlines=args.max_trendlines,
            touch_tolerance_pct=args.touch_tolerance_pct,
            headless=not args.headed,
        )
    )
    print(run["run_id"])


if __name__ == "__main__":
    main()

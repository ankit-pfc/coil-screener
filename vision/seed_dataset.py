from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import requests

from .capture import CaptureConfig, ChartCapture, capture_chart
from .run import VisionRunConfig, relative_to_run, resolve_tickers
from .storage import make_run_id, safe_segment, utc_now_iso, write_json


VALID_SPLITS = {"train", "valid", "test"}
DEFAULT_TAGS = ("coilingview", "chart-capture")


@dataclass(frozen=True)
class RoboflowUploadConfig:
    project_id: str
    api_key: str | None = None
    split: str = "train"
    batch: str | None = None
    tags: tuple[str, ...] = DEFAULT_TAGS
    timeout_s: int = 120
    endpoint: str = "https://api.roboflow.com"


@dataclass(frozen=True)
class DatasetSeedConfig:
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
    upload: bool = False
    roboflow: RoboflowUploadConfig | None = None


CaptureFn = Callable[[CaptureConfig, Path], ChartCapture]
UploadFn = Callable[[Path, str, RoboflowUploadConfig], dict[str, Any]]


def dataset_slug(project_id: str) -> str:
    cleaned = project_id.strip().strip("/")
    if not cleaned:
        raise ValueError("Roboflow project id is required.")
    return cleaned.split("/")[-1]


def image_filename(
    *,
    ticker: str,
    interval: str,
    timeframe: str,
    chart_type: str,
    run_id: str,
) -> str:
    parts = [
        safe_segment(ticker.upper()),
        safe_segment(interval),
        safe_segment(timeframe),
        safe_segment(chart_type),
        safe_segment(run_id),
    ]
    return "_".join(parts) + ".png"


def upload_image_to_roboflow(
    image_path: Path,
    filename: str,
    config: RoboflowUploadConfig,
) -> dict[str, Any]:
    if config.split not in VALID_SPLITS:
        raise ValueError(f"Roboflow split must be one of: {', '.join(sorted(VALID_SPLITS))}.")
    api_key = config.api_key or os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is required when --upload is set.")

    upload_url = (
        f"{config.endpoint.rstrip('/')}/dataset/"
        f"{quote(dataset_slug(config.project_id), safe='')}/upload"
    )
    form_fields: list[tuple[str, str]] = [
        ("name", filename),
        ("split", config.split),
    ]
    if config.batch:
        form_fields.append(("batch", config.batch))
    for tag in config.tags:
        if tag.strip():
            form_fields.append(("tag", tag.strip()))

    with image_path.open("rb") as handle:
        response = requests.post(
            upload_url,
            params={"api_key": api_key},
            data=form_fields,
            files={"file": (filename, handle, "image/png")},
            timeout=config.timeout_s,
        )

    try:
        payload: dict[str, Any] = response.json()
    except ValueError:
        payload = {"text": response.text}
    if response.status_code >= 400:
        message = payload.get("error") or payload.get("message") or response.text
        raise RuntimeError(f"Roboflow upload failed ({response.status_code}): {message}")
    return payload


def run_dataset_seed(
    config: DatasetSeedConfig,
    *,
    capture_fn: CaptureFn = capture_chart,
    upload_fn: UploadFn = upload_image_to_roboflow,
) -> dict[str, Any]:
    run_id = config.run_id or make_run_id().replace("vision_", "dataset_")
    run_dir = config.project_root / "vision_dataset_uploads" / safe_segment(run_id)
    images_dir = run_dir / "images"
    states_dir = run_dir / "states"
    tickers, saved_run_name = resolve_tickers(
        VisionRunConfig(
            project_root=config.project_root,
            tickers=config.tickers,
            saved_run=config.saved_run,
            interval=config.interval,
            timeframe=config.timeframe,
            chart_type=config.chart_type,
            base_url=config.base_url,
            limit=config.limit,
            run_id=run_id,
            headless=config.headless,
        )
    )
    upload_config = config.roboflow
    if config.upload and upload_config is None:
        raise RuntimeError("Roboflow upload config is required when --upload is set.")

    payload: dict[str, Any] = {
        "created_at": utc_now_iso(),
        "completed_at": None,
        "status": "running",
        "run_id": run_id,
        "request": {
            "tickers": tickers,
            "saved_run": saved_run_name or config.saved_run,
            "interval": config.interval,
            "timeframe": config.timeframe,
            "chart_type": config.chart_type,
            "base_url": config.base_url,
            "limit": config.limit,
            "upload": config.upload,
            "roboflow": {
                "project_id": upload_config.project_id if upload_config else None,
                "dataset_slug": dataset_slug(upload_config.project_id) if upload_config else None,
                "split": upload_config.split if upload_config else None,
                "batch": upload_config.batch if upload_config else None,
                "tags": list(upload_config.tags) if upload_config else [],
            },
        },
        "summary": {"requested": len(tickers), "captured": 0, "uploaded": 0, "failed": 0},
        "captures": [],
        "errors": [],
    }
    write_json(run_dir / "manifest.json", payload)

    for ticker in tickers:
        filename = image_filename(
            ticker=ticker,
            interval=config.interval,
            timeframe=config.timeframe,
            chart_type=config.chart_type,
            run_id=run_id,
        )
        image_path = images_dir / filename
        state_path = states_dir / f"{safe_segment(ticker.upper())}.json"
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
            write_json(
                state_path,
                {
                    "ticker": ticker.upper(),
                    "captured_at": utc_now_iso(),
                    "capture_url": capture.url,
                    "state": capture.state,
                },
            )
            payload["summary"]["captured"] += 1

            upload_response: dict[str, Any] | None = None
            if config.upload:
                upload_response = upload_fn(image_path, filename, upload_config)  # type: ignore[arg-type]
                payload["summary"]["uploaded"] += 1

            payload["captures"].append(
                {
                    "ticker": ticker.upper(),
                    "status": "uploaded" if config.upload else "captured",
                    "filename": filename,
                    "image_path": relative_to_run(image_path, run_dir),
                    "state_path": relative_to_run(state_path, run_dir),
                    "capture_url": capture.url,
                    "roboflow": upload_response,
                }
            )
        except Exception as exc:  # noqa: BLE001 - per-ticker failures are recorded.
            payload["summary"]["failed"] += 1
            payload["errors"].append({"ticker": ticker.upper(), "error": str(exc)})
        finally:
            write_json(run_dir / "manifest.json", payload)

    payload["completed_at"] = utc_now_iso()
    payload["status"] = "completed" if payload["summary"]["failed"] == 0 else "completed_with_errors"
    write_json(run_dir / "manifest.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture CoilingView charts and optionally upload them to Roboflow for labeling."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--base-url", default="http://127.0.0.1:5173")
    parser.add_argument("--saved-run")
    parser.add_argument("--ticker", action="append", dest="tickers")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--interval", default="3M")
    parser.add_argument("--timeframe", default="10Y")
    parser.add_argument("--chart-type", default="candles")
    parser.add_argument("--run-id")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--project-id", default=os.getenv("ROBOFLOW_PROJECT_ID") or os.getenv("ROBOFLOW_PROJECT"))
    parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="train")
    parser.add_argument("--batch")
    parser.add_argument("--tag", action="append", dest="tags")
    args = parser.parse_args()

    run_id = args.run_id or make_run_id().replace("vision_", "dataset_")
    upload_config = None
    if args.upload:
        if not args.project_id:
            raise SystemExit("--project-id or ROBOFLOW_PROJECT_ID is required when --upload is set.")
        upload_config = RoboflowUploadConfig(
            project_id=args.project_id,
            split=args.split,
            batch=args.batch or f"coilingview-captures-{run_id}",
            tags=tuple(args.tags or DEFAULT_TAGS),
        )

    manifest = run_dataset_seed(
        DatasetSeedConfig(
            project_root=args.project_root,
            base_url=args.base_url,
            saved_run=args.saved_run,
            tickers=args.tickers,
            limit=args.limit,
            interval=args.interval,
            timeframe=args.timeframe,
            chart_type=args.chart_type,
            run_id=run_id,
            headless=not args.headed,
            upload=args.upload,
            roboflow=upload_config,
        )
    )
    print(manifest["run_id"])


if __name__ == "__main__":
    main()

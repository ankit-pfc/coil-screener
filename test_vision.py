from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

import app as app_module
from vision.capture import CaptureConfig, ChartCapture
from vision.inference import _model_parts
from vision.mapping import build_trendline, map_detections_to_chart_points, map_detections_to_highs
from vision.run import VisionRunConfig, run_vision_pipeline
from vision.storage import VisionRunStore
from vision.trendlines import suggest_resistance_trendlines


def client() -> TestClient:
    return TestClient(app_module.app)


def test_model_parts_supports_workspace_qualified_model_ids():
    assert _model_parts("coiling-view/1") == ("coiling-view", "1")
    assert _model_parts("ankits-workspace-kyy0z/coiling-view/1") == (
        "ankits-workspace-kyy0z",
        "coiling-view/1",
    )


def test_map_detections_to_highs_snaps_x_and_interpolates_price():
    capture_state = {
        "visible_bars": [
            {"idx": 0, "date": "2024-01-01", "x": 10, "high": 110, "close": 100},
            {"idx": 1, "date": "2024-04-01", "x": 40, "high": 130, "close": 120},
            {"idx": 2, "date": "2024-07-01", "x": 70, "high": 125, "close": 118},
        ],
        "price_scale": {
            "samples": [
                {"y": 0, "price": 200},
                {"y": 100, "price": 100},
            ]
        },
    }
    detections = [
        {
            "class_name": "major_high",
            "confidence": 0.91,
            "bbox": [34, 18, 46, 26],
        },
        {
            "class_name": "noise",
            "confidence": 0.99,
            "bbox": [70, 10, 90, 20],
        },
    ]

    highs = map_detections_to_highs(detections, capture_state)

    assert highs == [
        {
            "idx": 1,
            "date": "2024-04-01",
            "price": 130.0,
            "confidence": 0.91,
            "class_name": "major_high",
            "source": "vision",
            "image_anchor": {"x": 40.0, "y": 22.0},
            "bbox": [34, 18, 46, 26],
        }
    ]


def test_map_detections_to_chart_points_supports_plot_meta_shape():
    chart_meta = {
        "plot_area": {"left": 0, "top": 0, "width": 100, "height": 100},
        "y_min": 100,
        "y_max": 200,
        "candles": [
            {"idx": 0, "date": "2024-01-01", "high": 110},
            {"idx": 1, "date": "2024-04-01", "high": 130},
            {"idx": 2, "date": "2024-07-01", "high": 125},
        ],
    }
    suggestions = map_detections_to_chart_points(
        [{"class_name": "resistance_touch", "confidence": 0.8, "bbox": [43, 65, 57, 75]}],
        chart_meta,
    )

    assert suggestions[0]["kind"] == "resistance_touch"
    assert suggestions[0]["status"] == "suggested"
    assert suggestions[0]["chart_point"]["idx"] == 1
    assert suggestions[0]["chart_point"]["price"] == 130.0


def test_suggest_resistance_trendlines_returns_ranked_candidates():
    mapped_points = [
        {
            "id": "ai-0",
            "kind": "major_high",
            "confidence": 0.9,
            "chart_point": {"idx": 0, "date": "2024-01-01", "price": 120},
        },
        {
            "id": "ai-1",
            "kind": "major_high",
            "confidence": 0.8,
            "chart_point": {"idx": 3, "date": "2024-10-01", "price": 114},
        },
        {
            "id": "ai-2",
            "kind": "resistance_touch",
            "confidence": 0.85,
            "chart_point": {"idx": 6, "date": "2025-07-01", "price": 108},
        },
    ]

    trendlines = suggest_resistance_trendlines(mapped_points, touch_tolerance_pct=0.1)

    assert trendlines[0]["kind"] == "resistance_trendline"
    assert len(trendlines[0]["touches"]) == 3


def test_build_trendline_uses_first_and_last_high():
    highs = [
        {"idx": 1, "date": "2024-01-01", "price": 100},
        {"idx": 3, "date": "2024-07-01", "price": 105},
    ]
    assert build_trendline(highs) == {
        "from": highs[0],
        "to": highs[1],
        "source": "vision",
    }


def test_vision_run_endpoint_invokes_runner(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(app_module, "VISION_RUNS_DIR", tmp_path / "vision_runs")

    def fake_runner(config: app_module.VisionRunConfig, **kwargs: Any) -> dict[str, Any]:
        assert config.project_root == tmp_path
        assert config.interval == "3M"
        assert config.timeframe == "10Y"
        assert config.chart_type == "candles"
        assert config.tickers == ["AAPL"]
        assert kwargs["store"].root == tmp_path / "vision_runs"
        return {"run_id": "vision_test", "status": "completed"}

    monkeypatch.setattr(app_module, "run_vision_pipeline", fake_runner)

    resp = client().post("/api/vision/run", json={"tickers": ["AAPL"]})

    assert resp.status_code == 200
    assert resp.json() == {"run_id": "vision_test", "status": "completed"}


def test_run_vision_pipeline_writes_mapped_artifacts(tmp_path):
    capture_state = {
        "visible_bars": [
            {"idx": 0, "date": "2024-01-01", "x": 10, "high": 120, "close": 116},
            {"idx": 3, "date": "2024-10-01", "x": 40, "high": 114, "close": 110},
            {"idx": 6, "date": "2025-07-01", "x": 70, "high": 108, "close": 105},
        ],
        "price_scale": {
            "samples": [
                {"y": 0, "price": 130},
                {"y": 100, "price": 100},
            ]
        },
        "element": {
            "width": 100,
            "height": 100,
            "padding": {"top": 0, "right": 0, "bottom": 0, "left": 0},
            "chart_width": 100,
            "chart_height": 100,
        },
    }

    def fake_capture(config: CaptureConfig, image_path: Path) -> ChartCapture:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"fake image")
        return ChartCapture(config.ticker, "http://capture", image_path, capture_state)

    class FakeInferenceClient:
        def infer(self, _image_path: Path) -> dict[str, Any]:
            return {
                "predictions": [
                    {
                        "class": "major_high",
                        "confidence": 0.9,
                        "x": 10,
                        "y": 33,
                        "width": 8,
                        "height": 8,
                    },
                    {
                        "class": "major_high",
                        "confidence": 0.86,
                        "x": 40,
                        "y": 53,
                        "width": 8,
                        "height": 8,
                    },
                    {
                        "class": "resistance_touch",
                        "confidence": 0.88,
                        "x": 70,
                        "y": 73,
                        "width": 8,
                        "height": 8,
                    },
                ]
            }

    store = VisionRunStore(tmp_path / "vision_runs")
    run = run_vision_pipeline(
        VisionRunConfig(
            project_root=tmp_path,
            tickers=["AAPL"],
            run_id="vision_test",
            confidence=0.35,
            max_highs=5,
        ),
        store=store,
        inference_client=FakeInferenceClient(),
        capture_fn=fake_capture,
    )

    assert run["status"] == "completed"
    prediction = store.read_prediction("AAPL", "3M", "vision_test")
    assert prediction["suggestions"]["major_highs"][0]["status"] == "suggested"
    assert prediction["suggestions"]["resistance_trendlines"]
    assert (store.run_dir("vision_test") / prediction["mapped_path"]).exists()
    assert store.manifest_path("vision_test").exists()


def _write_prediction(
    root: Path,
    run_id: str,
    ticker: str = "AAPL",
    interval: str = "3M",
    created_at: str = "2026-06-01T00:00:00Z",
) -> None:
    store = VisionRunStore(root / "vision_runs")
    store.write_run(
        run_id,
        {
            "created_at": created_at,
            "completed_at": created_at,
            "status": "completed",
            "request": {"interval": interval},
            "summary": {"requested": 1, "completed": 1, "failed": 0},
            "predictions": [{"ticker": ticker, "path": f"predictions/{ticker}.json"}],
            "errors": [],
        },
    )
    store.write_prediction(
        run_id,
        ticker,
        {
            "interval": interval,
            "timeframe": "10Y",
            "chart_type": "candles",
            "mapped_highs": [{"idx": 1, "date": "2024-04-01", "price": 178}],
            "trendline": None,
            "raw_detections": [],
        },
    )


def test_vision_predictions_latest_reads_filesystem_json(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "VISION_RUNS_DIR", tmp_path / "vision_runs")
    _write_prediction(tmp_path, "vision_old", created_at="2026-01-01T00:00:00Z")
    _write_prediction(tmp_path, "vision_new", created_at="2026-06-01T00:00:00Z")

    resp = client().get("/api/vision/predictions/AAPL", params={"interval": "3M"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == "vision_new"
    assert body["ticker"] == "AAPL"
    assert body["mapped_highs"][0]["date"] == "2024-04-01"


def test_vision_review_appends_review_and_updates_prediction(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "VISION_RUNS_DIR", tmp_path / "vision_runs")
    _write_prediction(tmp_path, "vision_review")

    resp = client().post(
        "/api/vision/reviews",
        json={
            "run_id": "vision_review",
            "ticker": "AAPL",
            "interval": "3M",
            "decision": "accepted",
            "accepted_highs": [{"idx": 1, "date": "2024-04-01", "price": 178}],
        },
    )

    assert resp.status_code == 200
    review = resp.json()["review"]
    assert review["decision"] == "accepted"
    assert review["ticker"] == "AAPL"

    store = VisionRunStore(tmp_path / "vision_runs")
    prediction = store.read_prediction("AAPL", "3M", "vision_review")
    assert prediction["review"]["decision"] == "accepted"
    review_log = tmp_path / "vision_runs" / "vision_review" / "reviews" / "AAPL.jsonl"
    assert review_log.exists()

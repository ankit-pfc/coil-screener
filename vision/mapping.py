from __future__ import annotations

from dataclasses import dataclass
from typing import Any


HIGH_CLASS_HINTS = ("high", "resistance", "touch", "peak")
HIGH_SUGGESTION_KINDS = {"major_high", "resistance_touch"}


@dataclass(frozen=True)
class PlotMeta:
    left: float
    top: float
    width: float
    height: float
    y_min: float
    y_max: float
    candles: list[dict[str, Any]]

    @classmethod
    def from_capture_state(cls, capture_state: dict[str, Any]) -> "PlotMeta":
        """Normalize frontend capture metadata into a chart plot model.

        The current frontend publishes ``window.__CV_CAPTURE_STATE__`` with
        visible bar coordinates. The V1 Roboflow scaffold also described a
        ``window.__COILINGVIEW_CHART_META__`` shape with ``plot_area`` and
        ``candles``. Supporting both keeps the Python mapper decoupled from the
        exact browser contract while still using structured chart metadata.
        """

        nested_meta = capture_state.get("chart_meta") or capture_state.get("chartMeta")
        if isinstance(nested_meta, dict) and not (
            capture_state.get("plot_area") or capture_state.get("plotArea")
        ):
            return cls.from_capture_state(nested_meta)

        plot = capture_state.get("plot_area") or capture_state.get("plotArea")
        if isinstance(plot, dict):
            candles = capture_state.get("candles") or capture_state.get("visible_bars") or []
            return cls(
                left=_float(plot.get("left"), 0.0),
                top=_float(plot.get("top"), 0.0),
                width=max(_float(plot.get("width"), 0.0), 1.0),
                height=max(_float(plot.get("height"), 0.0), 1.0),
                y_min=_float(capture_state.get("y_min") or capture_state.get("yMin"), 0.0),
                y_max=_float(capture_state.get("y_max") or capture_state.get("yMax"), 0.0),
                candles=_dict_rows(candles),
            )

        element = capture_state.get("element") if isinstance(capture_state.get("element"), dict) else {}
        padding = element.get("padding") if isinstance(element.get("padding"), dict) else {}
        left = _float(padding.get("left"), 0.0)
        top = _float(padding.get("top"), 0.0)
        width = _float(element.get("chart_width") or element.get("chartWidth"), 0.0)
        height = _float(element.get("chart_height") or element.get("chartHeight"), 0.0)
        if width <= 0:
            width = max(_float(element.get("width"), 0.0) - left - _float(padding.get("right"), 0.0), 1.0)
        if height <= 0:
            height = max(_float(element.get("height"), 0.0) - top - _float(padding.get("bottom"), 0.0), 1.0)

        y_min = 0.0
        y_max = 0.0
        scale = capture_state.get("price_scale") or capture_state.get("priceScale") or {}
        samples = scale.get("samples") if isinstance(scale, dict) else []
        price_samples = [
            (_float(sample.get("y"), 0.0), _float(sample.get("price"), 0.0))
            for sample in samples
            if isinstance(sample, dict) and sample.get("price") is not None
        ]
        if len(price_samples) >= 2:
            y_top, price_top = min(price_samples, key=lambda item: item[0])
            y_bottom, price_bottom = max(price_samples, key=lambda item: item[0])
            top = y_top
            height = max(y_bottom - y_top, 1.0)
            y_max = price_top
            y_min = price_bottom

        candles = capture_state.get("visible_bars") or capture_state.get("visibleBars") or []
        if not candles:
            candles = capture_state.get("candles") or capture_state.get("bars") or []
        return cls(
            left=left,
            top=top,
            width=max(width, 1.0),
            height=max(height, 1.0),
            y_min=y_min,
            y_max=y_max,
            candles=_dict_rows(candles),
        )


def _float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _dict_rows(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _class_name(detection: dict[str, Any]) -> str:
    value = (
        detection.get("class_name")
        or detection.get("class")
        or detection.get("label")
        or ""
    )
    return str(value)


def canonical_detection_kind(detection: dict[str, Any]) -> str:
    name = _class_name(detection).strip()
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    if normalized in HIGH_SUGGESTION_KINDS:
        return normalized
    if not normalized or "high" in normalized or "peak" in normalized:
        return "major_high"
    if "resistance" in normalized or "touch" in normalized:
        return "resistance_touch"
    return normalized or "unknown"


def is_high_detection(detection: dict[str, Any]) -> bool:
    name = canonical_detection_kind(detection).lower()
    if not name:
        return True
    return name in HIGH_SUGGESTION_KINDS or any(hint in name for hint in HIGH_CLASS_HINTS)


def detection_bbox(detection: dict[str, Any]) -> list[float] | None:
    if all(key in detection for key in ("x_min", "y_min", "x_max", "y_max")):
        return [
            float(detection["x_min"]),
            float(detection["y_min"]),
            float(detection["x_max"]),
            float(detection["y_max"]),
        ]
    bbox = detection.get("bbox") or detection.get("xyxy")
    if isinstance(bbox, list | tuple) and len(bbox) >= 4:
        return [float(v) for v in bbox[:4]]
    if all(key in detection for key in ("x", "y", "width", "height")):
        x = float(detection["x"])
        y = float(detection["y"])
        width = float(detection["width"])
        height = float(detection["height"])
        return [x - width / 2, y - height / 2, x + width / 2, y + height / 2]
    return None


def detection_anchor(detection: dict[str, Any]) -> tuple[float, float]:
    if "x" in detection and "y" in detection:
        return float(detection["x"]), float(detection["y"])
    bbox = detection_bbox(detection)
    if bbox is not None:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        return (x1 + x2) / 2, (y1 + y2) / 2
    raise ValueError("Detection is missing an anchor point.")


def _visible_bars(capture_state: dict[str, Any]) -> list[dict[str, Any]]:
    bars = capture_state.get("visible_bars") or capture_state.get("visibleBars") or []
    return [bar for bar in bars if isinstance(bar, dict) and isinstance(bar.get("x"), int | float)]


def pixel_to_idx(x: float, meta: PlotMeta) -> int:
    if not meta.candles:
        return 0

    candles_with_x = [
        (pos, candle)
        for pos, candle in enumerate(meta.candles)
        if isinstance(candle.get("x"), int | float)
    ]
    if candles_with_x:
        _, candle = min(candles_with_x, key=lambda item: abs(float(item[1]["x"]) - x))
        return int(candle.get("idx", meta.candles.index(candle)))

    if len(meta.candles) <= 1:
        return int(meta.candles[0].get("idx", 0))
    rel = (x - meta.left) / max(meta.width, 1.0)
    pos = round(max(0.0, min(1.0, rel)) * (len(meta.candles) - 1))
    return int(meta.candles[pos].get("idx", pos))


def pixel_to_price(y: float, meta: PlotMeta) -> float:
    rel = (y - meta.top) / max(meta.height, 1.0)
    rel = max(0.0, min(1.0, rel))
    return meta.y_max - rel * (meta.y_max - meta.y_min)


def _nearest_candle_position(x: float, meta: PlotMeta) -> int:
    if not meta.candles:
        return 0
    candles_with_x = [
        (pos, candle)
        for pos, candle in enumerate(meta.candles)
        if isinstance(candle.get("x"), int | float)
    ]
    if candles_with_x:
        pos, _ = min(candles_with_x, key=lambda item: abs(float(item[1]["x"]) - x))
        return pos
    rel = (x - meta.left) / max(meta.width, 1.0)
    return round(max(0.0, min(1.0, rel)) * (len(meta.candles) - 1))


def _candle_date(candle: dict[str, Any]) -> str:
    return str(candle.get("date") or candle.get("time") or "")


def _candle_price(candle: dict[str, Any], key: str, fallback: float) -> float:
    return _float(candle.get(key) or candle.get("price") or candle.get("close"), fallback)


def snap_to_nearest_high(
    x: float,
    y: float,
    meta: PlotMeta,
    *,
    window: int = 3,
) -> dict[str, Any]:
    if not meta.candles:
        return {
            "idx": 0,
            "date": "",
            "price": pixel_to_price(y, meta),
            "pixel": {"x": x, "y": y},
            "snap": None,
        }

    source_pos = _nearest_candle_position(x, meta)
    predicted_price = pixel_to_price(y, meta)
    lo = max(0, source_pos - window)
    hi = min(len(meta.candles) - 1, source_pos + window)
    best_pos = source_pos
    best_score = float("inf")

    for pos in range(lo, hi + 1):
        candle = meta.candles[pos]
        high = _candle_price(candle, "high", predicted_price)
        x_score = abs(pos - source_pos)
        price_score = abs(high - predicted_price) / max(abs(meta.y_max - meta.y_min), 1e-9)
        y_high = candle.get("y_high") or candle.get("yHigh")
        y_score = (
            abs(float(y_high) - y) / max(meta.height, 1.0)
            if isinstance(y_high, int | float)
            else 0.0
        )
        score = x_score + price_score * 10.0 + y_score * 4.0
        if score < best_score:
            best_pos = pos
            best_score = score

    candle = meta.candles[best_pos]
    return {
        "idx": int(candle.get("idx", best_pos)),
        "date": _candle_date(candle),
        "price": _candle_price(candle, "high", predicted_price),
        "pixel": {"x": x, "y": y},
        "snap": {
            "source_idx": int(meta.candles[source_pos].get("idx", source_pos)),
            "snapped_idx": int(candle.get("idx", best_pos)),
            "predicted_price": predicted_price,
        },
    }


def _bbox_dict(bbox: list[float] | None) -> dict[str, float] | None:
    if bbox is None:
        return None
    return {
        "x_min": bbox[0],
        "y_min": bbox[1],
        "x_max": bbox[2],
        "y_max": bbox[3],
    }


def map_detections_to_chart_points(
    detections: list[dict[str, Any]],
    capture_state: dict[str, Any],
) -> list[dict[str, Any]]:
    meta = PlotMeta.from_capture_state(capture_state)
    if not meta.candles:
        return []
    mapped: list[dict[str, Any]] = []

    for detection in detections:
        try:
            anchor_x, anchor_y = detection_anchor(detection)
        except (TypeError, ValueError):
            continue

        kind = canonical_detection_kind(detection)
        confidence = float(detection.get("confidence", 0.0) or 0.0)
        if kind in HIGH_SUGGESTION_KINDS:
            chart_point = snap_to_nearest_high(anchor_x, anchor_y, meta)
        else:
            idx = pixel_to_idx(anchor_x, meta)
            candle = next(
                (
                    item
                    for item in meta.candles
                    if int(item.get("idx", -1)) == idx
                ),
                meta.candles[_nearest_candle_position(anchor_x, meta)] if meta.candles else {},
            )
            chart_point = {
                "idx": idx,
                "date": _candle_date(candle),
                "price": pixel_to_price(anchor_y, meta),
                "pixel": {"x": anchor_x, "y": anchor_y},
                "snap": None,
            }

        mapped.append(
            {
                "id": f"ai-{len(mapped)}",
                "kind": kind,
                "class_name": _class_name(detection),
                "confidence": confidence,
                "bbox": _bbox_dict(detection_bbox(detection)),
                "chart_point": chart_point,
                "status": "suggested",
            }
        )

    return mapped


def _price_from_y(capture_state: dict[str, Any], y: float, fallback: float) -> float:
    scale = capture_state.get("price_scale") or capture_state.get("priceScale") or {}
    samples = scale.get("samples") or []
    usable = [
        (float(s["y"]), float(s["price"]))
        for s in samples
        if isinstance(s, dict)
        and isinstance(s.get("y"), int | float)
        and isinstance(s.get("price"), int | float)
    ]
    if len(usable) >= 2:
        usable.sort(key=lambda pair: pair[0])
        (y1, p1), (y2, p2) = usable[0], usable[-1]
        if y2 != y1:
            ratio = (y - y1) / (y2 - y1)
            return p1 + ratio * (p2 - p1)
    return fallback


def map_detections_to_highs(
    detections: list[dict[str, Any]],
    capture_state: dict[str, Any],
    *,
    max_highs: int = 3,
) -> list[dict[str, Any]]:
    """Map image-space detections to ``{idx, date, price}`` chart anchors.

    The frontend capture state publishes visible bar x-coordinates and price
    scale metadata. High-like detections snap to the nearest candle high inside
    a small local window. Duplicate dates keep the highest-confidence
    detection.
    """

    by_date: dict[str, dict[str, Any]] = {}
    for suggestion in map_detections_to_chart_points(detections, capture_state):
        if suggestion["kind"] not in HIGH_SUGGESTION_KINDS:
            continue
        chart_point = suggestion["chart_point"]
        anchor = chart_point.get("pixel") or {}
        bbox = suggestion.get("bbox")
        mapped = {
            "idx": int(chart_point["idx"]),
            "date": str(chart_point["date"]),
            "price": float(chart_point["price"]),
            "confidence": float(suggestion.get("confidence", 0.0) or 0.0),
            "class_name": str(suggestion.get("class_name") or suggestion["kind"]),
            "source": "vision",
            "image_anchor": {
                "x": float(anchor.get("x", 0.0) or 0.0),
                "y": float(anchor.get("y", 0.0) or 0.0),
            },
            "bbox": [bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]]
            if isinstance(bbox, dict)
            else None,
        }
        existing = by_date.get(mapped["date"])
        if not existing or mapped["confidence"] >= float(existing.get("confidence", 0.0)):
            by_date[mapped["date"]] = mapped

    highs = sorted(by_date.values(), key=lambda high: high["idx"])
    if max_highs > 0:
        highs = highs[:max_highs]
    return highs


def build_trendline(highs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(highs) < 2:
        return None
    return {"from": highs[0], "to": highs[-1], "source": "vision"}

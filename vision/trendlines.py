from __future__ import annotations

import itertools
from typing import Any


HIGH_SUGGESTION_KINDS = {"major_high", "resistance_touch"}


def _chart_point(suggestion: dict[str, Any]) -> dict[str, Any]:
    point = suggestion.get("chart_point")
    return point if isinstance(point, dict) else {}


def _line_price_at_idx(p1: dict[str, Any], p2: dict[str, Any], idx: int) -> float:
    x1 = int(p1["idx"])
    y1 = float(p1["price"])
    x2 = int(p2["idx"])
    y2 = float(p2["price"])
    if x1 == x2:
        return y1
    slope = (y2 - y1) / (x2 - x1)
    return y1 + slope * (idx - x1)


def suggest_resistance_trendlines(
    mapped_points: list[dict[str, Any]],
    *,
    max_lines: int = 5,
    touch_tolerance_pct: float = 1.5,
) -> list[dict[str, Any]]:
    highs = [
        point
        for point in mapped_points
        if point.get("kind") in HIGH_SUGGESTION_KINDS
        and isinstance(point.get("chart_point"), dict)
    ]
    candidates: list[dict[str, Any]] = []

    for p1, p2 in itertools.combinations(highs, 2):
        cp1 = _chart_point(p1)
        cp2 = _chart_point(p2)
        idx1 = int(cp1.get("idx", 0))
        idx2 = int(cp2.get("idx", 0))
        if abs(idx2 - idx1) < 2:
            continue

        touches: list[dict[str, Any]] = []
        for point in highs:
            chart_point = _chart_point(point)
            idx = int(chart_point.get("idx", 0))
            actual = float(chart_point.get("price", 0.0) or 0.0)
            if actual == 0:
                continue
            expected = _line_price_at_idx(cp1, cp2, idx)
            err_pct = abs(actual - expected) / abs(actual) * 100.0
            if err_pct <= touch_tolerance_pct:
                touches.append(
                    {
                        "suggestion_id": point.get("id"),
                        "idx": idx,
                        "date": chart_point.get("date"),
                        "price": actual,
                        "line_price": expected,
                        "error_pct": err_pct,
                    }
                )

        if len(touches) < 2:
            continue

        touch_ids = {touch["suggestion_id"] for touch in touches}
        touch_confidences = [
            float(point.get("confidence", 0.0) or 0.0)
            for point in highs
            if point.get("id") in touch_ids
        ]
        avg_conf = (
            sum(touch_confidences) / len(touch_confidences)
            if touch_confidences
            else 0.0
        )
        candidates.append(
            {
                "id": f"trendline-{len(candidates)}",
                "kind": "resistance_trendline",
                "status": "suggested",
                "confidence": round(min(1.0, avg_conf * (1 + 0.1 * (len(touches) - 2))), 4),
                "anchors": [cp1, cp2],
                "touches": touches,
            }
        )

    candidates.sort(
        key=lambda candidate: (len(candidate["touches"]), candidate["confidence"]),
        reverse=True,
    )
    return candidates[:max_lines]

"""Deterministic shadow-comparison charts for current and lifetime detectors.

This module is deliberately presentation-only.  It accepts detector outputs as
already-computed inputs and never feeds reference geometry back into either
detector.  The renderer uses calendar-quarter x coordinates while preserving
the benchmark's exact scored normalization for every displayed line.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from coil_analysis import (  # noqa: E402
    _aggregate_quarterly_display_bars,
    _clean_bars,
    _completed_quarters,
)


__all__ = ["render_benchmark_chart"]


BACKGROUND = "#09111f"
PANEL = "#0d1728"
GRID = "#314159"
TEXT = "#e8eef8"
MUTED = "#9aa9bd"
UP = "#46c78c"
DOWN = "#e36b73"
CURRENT = "#f28e2b"
LIFETIME_PRIMARY = "#52a8ff"
LIFETIME_CHILD = "#f2b84b"
REFERENCE = "#dc78ff"


def _quarter_ordinal(value: Any) -> int:
    text = str(value)
    if len(text) < 7:
        raise ValueError(f"invalid calendar date: {text!r}")
    year = int(text[:4])
    month = int(text[5:7])
    if not 1 <= month <= 12:
        raise ValueError(f"invalid calendar month: {text!r}")
    return year * 4 + (month - 1) // 3


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _style_axis(ax: Any) -> None:
    ax.set_facecolor(PANEL)
    ax.grid(True, color=GRID, linewidth=0.55, alpha=0.45)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def _set_year_ticks(ax: Any, start: int, end: int) -> None:
    start_year = start // 4
    end_year = end // 4
    span_years = max(1, end_year - start_year)
    if span_years >= 30:
        every = 5
    elif span_years >= 16:
        every = 3
    elif span_years >= 8:
        every = 2
    else:
        every = 1
    first_year = start_year + (-start_year % every)
    years = list(range(first_year, end_year + 1, every))
    ticks = [year * 4 for year in years]
    ax.set_xticks(ticks, [str(year) for year in years])


def _draw_candles(
    ax: Any,
    quarters: list[dict[str, Any]],
    *,
    partial_ordinal: Optional[int] = None,
) -> None:
    for bar in quarters:
        x = _quarter_ordinal(bar["date"])
        open_price = float(bar.get("open", bar["close"]))
        close = float(bar["close"])
        low = float(bar["low"])
        high = float(bar["high"])
        color = UP if close >= open_price else DOWN
        is_partial = x == partial_ordinal
        ax.vlines(
            x,
            low,
            high,
            color=color,
            linewidth=0.8,
            alpha=0.48 if is_partial else 0.9,
            linestyles=(0, (2, 2)) if is_partial else "solid",
            zorder=3,
        )
        bottom = min(open_price, close)
        body_height = max(abs(close - open_price), max(high, 1.0) * 0.002)
        ax.add_patch(
            Rectangle(
                (x - 0.31, bottom),
                0.62,
                body_height,
                facecolor=color,
                edgecolor=MUTED if is_partial else color,
                linewidth=0.9 if is_partial else 0.5,
                alpha=0.38 if is_partial else 0.88,
                hatch="///" if is_partial else None,
                zorder=4,
            )
        )


def _active_lid(current_analysis: dict[str, Any]) -> Optional[dict[str, Any]]:
    value = current_analysis.get("active_lid")
    return value if isinstance(value, dict) else None


def _current_line_points(
    active_lid: dict[str, Any],
    monthly: list[dict[str, Any]],
    quarters: list[dict[str, Any]],
) -> list[tuple[int, float]]:
    """Sample the reported monthly-index line at calendar-quarter positions."""

    start = active_lid.get("from")
    if not isinstance(start, dict):
        anchors = active_lid.get("anchors")
        start = anchors[0] if isinstance(anchors, list) and anchors else None
    if not isinstance(start, dict):
        return []

    try:
        start_idx = int(start["idx"])
        start_x = _quarter_ordinal(start.get("date") or monthly[start_idx]["date"])
    except (KeyError, TypeError, ValueError, IndexError):
        return []
    start_price = _finite_float(start.get("price"))
    slope = _finite_float(active_lid.get("slope_per_bar"))
    if start_price is None or slope is None or not 0 <= start_idx < len(monthly):
        return []

    # One sample per displayed calendar quarter.  The y value remains the
    # current detector's monthly-index value at that quarter's closing source
    # month; it is never reconstructed from a quarterly endpoint slope.
    by_ordinal: dict[int, tuple[int, float]] = {}
    for quarter in quarters:
        try:
            source_idx = int(quarter["_close_source_idx"])
            x = _quarter_ordinal(quarter["date"])
        except (KeyError, TypeError, ValueError):
            continue
        if source_idx < start_idx or source_idx >= len(monthly):
            continue
        y = start_price + slope * (source_idx - start_idx)
        if math.isfinite(y) and y > 0:
            by_ordinal[x] = (source_idx, y)

    # Pin the first and projected values to the analyzer's own reported values;
    # intermediate samples use its reported monthly slope.
    by_ordinal[start_x] = (start_idx, start_price)
    projected = active_lid.get("projected")
    if isinstance(projected, dict):
        projected_price = _finite_float(projected.get("price"))
        try:
            projected_idx = int(projected.get("idx", len(monthly) - 1))
            projected_x = _quarter_ordinal(
                projected.get("date") or monthly[projected_idx]["date"]
            )
        except (TypeError, ValueError, IndexError):
            projected_idx = -1
            projected_x = -1
        if (
            projected_price is not None
            and projected_price > 0
            and start_idx <= projected_idx < len(monthly)
        ):
            by_ordinal[projected_x] = (projected_idx, projected_price)

    return [
        (ordinal, item[1])
        for ordinal, item in sorted(by_ordinal.items(), key=lambda pair: pair[0])
    ]


def _lifetime_structures(
    lifetime_analysis: dict[str, Any],
) -> list[tuple[dict[str, Any], bool]]:
    raw = lifetime_analysis.get("structures")
    structures = [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []
    if not structures:
        return []

    diagnostics = lifetime_analysis.get("diagnostics")
    primary_id = (
        diagnostics.get("primary_structure_id")
        if isinstance(diagnostics, dict)
        else None
    )
    primary_index = next(
        (index for index, item in enumerate(structures) if item.get("id") == primary_id),
        None,
    )
    if primary_index is None:
        primary_index = next(
            (
                index
                for index, item in enumerate(structures)
                if item.get("selection") == "active"
            ),
            0,
        )
    primary = structures[primary_index]
    return [(primary, True)] + [
        (item, False) for index, item in enumerate(structures) if index != primary_index
    ]


def _ordered_scored_structures(
    structures: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], bool]]:
    if not structures:
        return []
    primary_index = next(
        (
            index
            for index, structure in enumerate(structures)
            if structure.get("selection") == "primary"
            or structure.get("role") == "primary_lid"
        ),
        0,
    )
    return [(structures[primary_index], True)] + [
        (structure, False)
        for index, structure in enumerate(structures)
        if index != primary_index
    ]


def _structure_geometry(
    structure: dict[str, Any], chart_start: int, chart_end: int
) -> Optional[tuple[list[int], list[float], list[float], list[float]]]:
    # Benchmark-normalized structures are the canonical scored geometry.  Use
    # their exact intercept/slope instead of independently rebuilding a line
    # from raw detector endpoints in the presentation layer.
    normalized_slope = _finite_float(structure.get("slope_per_quarter"))
    normalized_intercept = _finite_float(structure.get("intercept"))
    if normalized_slope is not None and normalized_intercept is not None:
        anchor_ordinals = []
        for point in structure.get("anchor_points") or []:
            if not isinstance(point, dict) or not point.get("date"):
                continue
            try:
                anchor_ordinals.append(_quarter_ordinal(point["date"]))
            except (TypeError, ValueError):
                continue
        try:
            geometry_start = int(structure.get("geometry_start_ordinal"))
        except (TypeError, ValueError):
            geometry_start = min(anchor_ordinals) if anchor_ordinals else chart_start
        try:
            geometry_end = int(structure.get("geometry_end_ordinal"))
        except (TypeError, ValueError):
            geometry_end = chart_end
        visible_start = max(chart_start, geometry_start)
        visible_end = min(chart_end, geometry_end)
        if visible_start > visible_end:
            return None
        xs = list(range(visible_start, visible_end + 1))
        ys = [normalized_intercept + normalized_slope * x for x in xs]
        valid = [(x, y) for x, y in zip(xs, ys) if math.isfinite(y) and y > 0]
        if len(valid) < 2:
            return None
        xs = [item[0] for item in valid]
        ys = [item[1] for item in valid]

        band = structure.get("band")
        lower_pct = 0.0
        upper_pct = 0.0
        if isinstance(band, dict):
            lower_pct = max(0.0, _finite_float(band.get("lower_pct")) or 0.0) / 100.0
            upper_pct = max(0.0, _finite_float(band.get("upper_pct")) or 0.0) / 100.0
        lower = [value * (1.0 - lower_pct) for value in ys]
        upper = [value * (1.0 + upper_pct) for value in ys]
        return xs, ys, lower, upper

    line = structure.get("line")
    if not isinstance(line, dict):
        return None
    start = line.get("from")
    if not isinstance(start, dict):
        return None
    try:
        origin_x = (
            int(start["time_ordinal"])
            if start.get("time_ordinal") is not None
            else _quarter_ordinal(start["date"])
        )
    except (KeyError, TypeError, ValueError):
        return None
    origin_y = _finite_float(start.get("price"))
    slope = _finite_float(line.get("slope_per_quarter"))
    if slope is None:
        end = line.get("to")
        if not isinstance(end, dict):
            return None
        try:
            end_x = (
                int(end["time_ordinal"])
                if end.get("time_ordinal") is not None
                else _quarter_ordinal(end["date"])
            )
        except (KeyError, TypeError, ValueError):
            return None
        end_y = _finite_float(end.get("price"))
        if origin_y is None or end_y is None or end_x == origin_x:
            return None
        slope = (end_y - origin_y) / (end_x - origin_x)
    if origin_y is None:
        return None

    visible_start = max(chart_start, origin_x)
    if visible_start > chart_end:
        return None
    xs = list(range(visible_start, chart_end + 1))
    ys = [origin_y + slope * (x - origin_x) for x in xs]
    valid = [(x, y) for x, y in zip(xs, ys) if math.isfinite(y) and y > 0]
    if len(valid) < 2:
        return None
    xs = [item[0] for item in valid]
    ys = [item[1] for item in valid]

    band = structure.get("band")
    lower_pct = 0.0
    upper_pct = 0.0
    if isinstance(band, dict):
        lower_pct = max(0.0, _finite_float(band.get("lower_pct")) or 0.0) / 100.0
        upper_pct = max(0.0, _finite_float(band.get("upper_pct")) or 0.0) / 100.0
    lower = [value * (1.0 - lower_pct) for value in ys]
    upper = [value * (1.0 + upper_pct) for value in ys]
    return xs, ys, lower, upper


def _reference_lines(reference_setup: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(reference_setup, dict):
        return []
    raw = reference_setup.get("lines")
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(reference_setup.get("from"), dict) and isinstance(
        reference_setup.get("to"), dict
    ):
        return [reference_setup]
    return []


def _draw_reference_lines(
    ax: Any,
    reference_structures: list[dict[str, Any]],
    chart_start: int,
    chart_end: int,
) -> tuple[bool, list[float]]:
    drawn = False
    values: list[float] = []
    for line in reference_structures:
        geometry = _structure_geometry(line, chart_start, chart_end)
        if geometry is None:
            continue
        xs, ys, _, _ = geometry
        ax.plot(
            xs,
            ys,
            color=REFERENCE,
            linewidth=2.0,
            linestyle=(0, (1.2, 2.4)),
            alpha=0.95,
            zorder=9,
        )
        values.extend(ys)
        drawn = True

        anchors = line.get("anchor_points")
        if isinstance(anchors, list):
            anchor_points = []
            for anchor in anchors:
                if not isinstance(anchor, dict):
                    continue
                try:
                    x = _quarter_ordinal(anchor["date"])
                except (KeyError, TypeError, ValueError):
                    continue
                y = _finite_float(anchor.get("price"))
                if y is not None and chart_start <= x <= chart_end:
                    anchor_points.append((x, y))
            if anchor_points:
                ax.scatter(
                    [point[0] for point in anchor_points],
                    [point[1] for point in anchor_points],
                    marker="d",
                    s=42,
                    facecolor=PANEL,
                    edgecolor=REFERENCE,
                    linewidth=1.25,
                    zorder=10,
                )
                values.extend(point[1] for point in anchor_points)
    return drawn, values


def _partial_quarter_ordinal(
    quarters: list[dict[str, Any]], as_of: Optional[str]
) -> Optional[int]:
    if not quarters:
        return None
    completed = _completed_quarters(quarters, as_of=as_of)
    if len(completed) == len(quarters):
        return None
    return _quarter_ordinal(quarters[-1]["date"])


def render_benchmark_chart(
    ticker: str,
    monthly_bars: Iterable[dict[str, Any]],
    current_analysis: dict[str, Any],
    lifetime_analysis: dict[str, Any],
    output_path: str | Path,
    *,
    reference_setup: Optional[dict[str, Any]] = None,
    subtitle: Optional[str] = None,
) -> None:
    """Render a current-vs-lifetime detector shadow-comparison chart.

    The lines shown are the exact calendar-normalized structures used by the
    benchmark scorer. Human reference lines, when supplied, remain
    comparison-only overlays.
    """

    if not isinstance(current_analysis, dict) or not isinstance(lifetime_analysis, dict):
        raise TypeError("current_analysis and lifetime_analysis must be dictionaries")

    metadata = current_analysis.get("analysis_metadata")
    cutoff = metadata.get("history_end") if isinstance(metadata, dict) else None
    if not cutoff:
        cutoff = current_analysis.get("as_of")
    monthly = _clean_bars(monthly_bars, str(cutoff) if cutoff else None)
    if not monthly:
        raise ValueError("at least one valid monthly price bar is required")
    quarters = _aggregate_quarterly_display_bars(monthly)
    if not quarters:
        raise ValueError("monthly bars could not be aggregated into quarterly candles")

    expected_bar_count = current_analysis.get("bar_count")
    if isinstance(expected_bar_count, int) and expected_bar_count != len(monthly):
        raise ValueError(
            "monthly bars do not match current_analysis: "
            f"expected {expected_bar_count}, rendered {len(monthly)}"
        )

    # Import lazily to keep the renderer usable as a presentation-only module
    # while making the benchmark's canonical normalization the single source
    # of truth for both scoring and pixels.
    from lifetime_reference_benchmark import (  # noqa: PLC0415
        normalize_current_structures,
        normalize_lifetime_structures,
        normalize_reference_structures,
    )

    cutoff_date = str(monthly[-1]["date"])
    scored_current = normalize_current_structures(current_analysis, cutoff_date)
    scored_lifetime = normalize_lifetime_structures(lifetime_analysis, cutoff_date)
    scored_reference = normalize_reference_structures(
        reference_setup or {}, cutoff_date
    )

    decision_as_of = current_analysis.get("as_of") or lifetime_analysis.get("as_of")
    if not decision_as_of:
        decision_as_of = cutoff_date
    partial_ordinal = _partial_quarter_ordinal(quarters, str(decision_as_of))

    chart_start = _quarter_ordinal(quarters[0]["date"])
    chart_end = _quarter_ordinal(quarters[-1]["date"])
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BACKGROUND)
    _style_axis(ax)
    overlay_values: list[float] = []
    legend_handles: list[Any] = [
        Patch(facecolor=UP, edgecolor=UP, label="Up quarter"),
        Patch(facecolor=DOWN, edgecolor=DOWN, label="Down quarter"),
    ]
    if partial_ordinal is not None:
        legend_handles.append(
            Patch(
                facecolor=MUTED,
                edgecolor=MUTED,
                alpha=0.38,
                hatch="///",
                label="Partial quarter (observation only)",
            )
        )

    try:
        _draw_candles(ax, quarters, partial_ordinal=partial_ordinal)

        current_structure = next(
            (
                structure
                for structure in scored_current
                if structure.get("selection") == "primary"
            ),
            scored_current[0] if scored_current else None,
        )
        current_geometry = (
            _structure_geometry(current_structure, chart_start, chart_end)
            if current_structure
            else None
        )
        if current_geometry is not None:
            current_xs, current_ys, _, _ = current_geometry
            ax.plot(
                current_xs,
                current_ys,
                color=CURRENT,
                linewidth=2.25,
                linestyle=(0, (7, 4)),
                zorder=8,
            )
            overlay_values.extend(current_ys)
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=CURRENT,
                    linewidth=2.25,
                    linestyle=(0, (7, 4)),
                    label="Current boundary (scored normalization)",
                )
            )

        primary_drawn = False
        child_drawn = False
        primary_band_drawn = False
        child_band_drawn = False
        for structure, is_primary in _ordered_scored_structures(scored_lifetime):
            geometry = _structure_geometry(structure, chart_start, chart_end)
            if geometry is None:
                continue
            xs, ys, lower, upper = geometry
            color = LIFETIME_PRIMARY if is_primary else LIFETIME_CHILD
            band_has_width = any(
                abs(top - bottom) > 1e-12 for bottom, top in zip(lower, upper)
            )
            if band_has_width:
                ax.fill_between(
                    xs,
                    lower,
                    upper,
                    color=color,
                    alpha=0.14 if is_primary else 0.09,
                    linewidth=0,
                    zorder=1,
                )
                if is_primary:
                    primary_band_drawn = True
                else:
                    child_band_drawn = True
                overlay_values.extend(lower)
                overlay_values.extend(upper)
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=2.35 if is_primary else 1.85,
                zorder=7 if is_primary else 6,
            )
            overlay_values.extend(ys)
            if is_primary:
                primary_drawn = True
            else:
                child_drawn = True

        if primary_drawn:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=LIFETIME_PRIMARY,
                    linewidth=2.35,
                    label="Lifetime primary boundary",
                )
            )
        if primary_band_drawn:
            legend_handles.append(
                Patch(
                    facecolor=LIFETIME_PRIMARY,
                    alpha=0.22,
                    label="Lifetime primary band",
                )
            )
        if child_drawn:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=LIFETIME_CHILD,
                    linewidth=1.85,
                    label="Lifetime retained child",
                )
            )
        if child_band_drawn:
            legend_handles.append(
                Patch(
                    facecolor=LIFETIME_CHILD,
                    alpha=0.18,
                    label="Lifetime child band",
                )
            )

        reference_drawn, reference_values = _draw_reference_lines(
            ax, scored_reference, chart_start, chart_end
        )
        overlay_values.extend(reference_values)
        if reference_drawn:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=REFERENCE,
                    linewidth=2.0,
                    linestyle=(0, (1.2, 2.4)),
                    label="Human reference (comparison only)",
                )
            )

        ax.set_xlim(chart_start - 1, chart_end + 1)
        _set_year_ticks(ax, chart_start, chart_end)
        currency = (
            reference_setup.get("currency")
            if isinstance(reference_setup, dict)
            else None
        )
        ax.set_ylabel(f"Price ({currency})" if currency else "Price")
        ax.set_xlabel("Calendar time (quarterly candles)")

        visible_values = [
            value
            for bar in quarters
            for value in (float(bar["low"]), float(bar["high"]))
            if math.isfinite(value) and value > 0
        ]
        visible_values.extend(
            value for value in overlay_values if math.isfinite(value) and value > 0
        )
        if visible_values:
            low = min(visible_values)
            high = max(visible_values)
            padding = max((high - low) * 0.07, high * 0.02, 1e-9)
            ax.set_ylim(max(0.0, low - padding), high + padding)

        current_version = current_analysis.get("algorithm_version") or "current"
        lifetime_version = lifetime_analysis.get("algorithm_version") or "lifetime shadow"
        comparison_subtitle = subtitle or (
            f"Current detector {current_version} vs observed-lifetime detector "
            f"{lifetime_version}"
        )
        fig.suptitle(
            f"{str(ticker).strip().upper()} — boundary benchmark\n{comparison_subtitle}",
            x=0.055,
            y=0.965,
            ha="left",
            va="top",
            color=TEXT,
            fontsize=16,
            fontweight="bold",
            linespacing=1.35,
        )
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.055),
            ncol=min(4, max(1, len(legend_handles))),
            frameon=False,
            labelcolor=MUTED,
            fontsize=8.5,
        )
        footer = (
            "Development/shadow comparison only — not investment output. "
            "Human reference geometry is rendered after detection and is never detector input."
        )
        if partial_ordinal is not None:
            footer += (
                f" Hatched final candle is partial through {cutoff_date}; "
                "it is observation-only, not structural-top evidence."
            )
        fig.text(
            0.055,
            0.018,
            footer,
            color=MUTED,
            fontsize=7.8 if partial_ordinal is not None else 8.2,
            ha="left",
        )
        fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.205)

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            destination,
            dpi=170,
            facecolor=BACKGROUND,
        )
    finally:
        plt.close(fig)

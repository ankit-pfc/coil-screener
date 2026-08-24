#!/usr/bin/env python3
"""Render observed-lifetime detector output against frozen Amrut examples."""

from __future__ import annotations

import argparse
from datetime import date as calendar_date
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from coil_analysis import (  # noqa: E402
    _aggregate_quarterly_display_bars,
    _clean_bars,
)
from lifetime_structure import analyze_lifetime_references  # noqa: E402
from review_snapshots import load_review_snapshot  # noqa: E402


DEFAULT_SOURCE = "amrut_portfolio_exemplars_2026-08-21.csv"
DEFAULT_TICKERS = ("1070.HK", "0836.HK", "GMDCLTD.NS", "0981.HK")

BG = "#09111f"
PANEL = "#0d1728"
GRID = "#314159"
TEXT = "#e8eef8"
MUTED = "#9aa9bd"
UP = "#46c78c"
DOWN = "#e36b73"
PRIMARY = "#52a8ff"
PRIMARY_BAND = "#2f7fc9"
SECONDARY = "#f2b84b"
REFERENCE = "#dc78ff"
DEMOTED = "#ff6b6b"
TRACKING = "#b59cff"
TOUCH = "#51d6d0"


def _quarter_ordinal(value: str) -> int:
    year = int(str(value)[:4])
    month = int(str(value)[5:7])
    return year * 4 + (month - 1) // 3


def _ordinal_year_label(ordinal: int) -> str:
    return str(ordinal // 4)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _as_of_date(as_of: Optional[str]) -> Optional[calendar_date]:
    if not as_of:
        return None
    return calendar_date.fromisoformat(as_of[:10])


def _is_completed_quarter(bar: dict[str, Any], as_of: Optional[str]) -> bool:
    month = int(str(bar["date"])[5:7])
    if month % 3:
        return False
    year = int(str(bar["date"])[:4])
    cutoff = _as_of_date(as_of) or calendar_date.today()
    month_end = calendar_date(
        year,
        month,
        __import__("calendar").monthrange(year, month)[1],
    )
    return cutoff >= month_end


def _draw_candles(
    ax: plt.Axes,
    quarters: Iterable[dict[str, Any]],
    *,
    alpha: float = 0.9,
) -> None:
    for bar in quarters:
        x = _quarter_ordinal(str(bar["date"]))
        open_price = float(bar["open"])
        close = float(bar["close"])
        low = float(bar["low"])
        high = float(bar["high"])
        color = UP if close >= open_price else DOWN
        ax.vlines(x, low, high, color=color, linewidth=0.75, alpha=alpha * 0.9)
        bottom = min(open_price, close)
        height = max(abs(close - open_price), max(high, 1.0) * 0.002)
        ax.add_patch(
            Rectangle(
                (x - 0.31, bottom),
                0.62,
                height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.5,
                alpha=alpha,
            )
        )


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    ax.grid(True, color=GRID, linewidth=0.55, alpha=0.45)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)


def _set_year_ticks(ax: plt.Axes, start: int, end: int) -> None:
    start_year = start // 4
    end_year = end // 4
    span_years = max(1, end_year - start_year)
    every = 5 if span_years >= 20 else 3 if span_years >= 12 else 2
    first = start_year + (-start_year % every)
    ticks = [year * 4 for year in range(first, end_year + 1, every)]
    ax.set_xticks(ticks, [_ordinal_year_label(value) for value in ticks])


def _episode_roles(analysis: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in analysis["top_episodes"]}


def _draw_episode_markers(
    ax: plt.Axes,
    analysis: dict[str, Any],
    *,
    label_prices: bool,
    x_min: Optional[int] = None,
) -> None:
    primary = analysis["structures"][0] if analysis["structures"] else None
    primary_anchors = set(primary["construction_anchor_ids"]) if primary else set()
    primary_touches = set(primary["supporting_touch_ids"]) if primary else set()
    nested_points = {
        episode_id
        for structure in analysis["structures"][1:]
        for episode_id in (
            structure["construction_anchor_ids"] + structure["supporting_touch_ids"]
        )
    }
    for episode in analysis["top_episodes"]:
        x = int(episode["time_ordinal"])
        if x_min is not None and x < x_min:
            continue
        y = float(episode["price"])
        roles = set(episode["roles"])
        if "demoted_singleton" in roles:
            ax.scatter(x, y, marker="X", s=70, color=DEMOTED, zorder=9)
            label = "singleton"
        elif "tracking_new_high" in roles:
            ax.scatter(x, y, marker="D", s=54, color=TRACKING, zorder=9)
            label = "tracking"
        elif episode["id"] in primary_anchors:
            ax.scatter(
                x,
                y,
                marker="*",
                s=135,
                color=PRIMARY,
                edgecolor="#ffffff",
                linewidth=0.55,
                zorder=10,
            )
            label = "anchor"
        elif episode["id"] in primary_touches:
            ax.scatter(
                x,
                y,
                marker="o",
                s=47,
                facecolor=PANEL,
                edgecolor=TOUCH,
                linewidth=1.6,
                zorder=9,
            )
            label = "touch"
        elif episode["id"] in nested_points:
            ax.scatter(
                x,
                y,
                marker="s",
                s=36,
                facecolor=PANEL,
                edgecolor=SECONDARY,
                linewidth=1.35,
                zorder=8,
            )
            label = "nested"
        else:
            ax.scatter(x, y, marker="^", s=27, color=MUTED, alpha=0.62, zorder=7)
            label = "candidate"
        if label_prices and label in {"singleton", "tracking", "anchor"}:
            ax.annotate(
                f"{y:g}  {label}",
                (x, y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=TEXT if label == "anchor" else DEMOTED if label == "singleton" else TRACKING,
                fontsize=8,
                fontweight="bold",
                zorder=11,
            )


def _structure_value(structure: dict[str, Any], ordinal: int) -> float:
    line = structure["line"]
    start = line["from"]
    return float(start["price"]) + float(line["slope_per_quarter"]) * (
        ordinal - int(start["time_ordinal"])
    )


def _draw_structures(
    ax: plt.Axes,
    analysis: dict[str, Any],
    *,
    chart_end: int,
) -> None:
    for index, structure in enumerate(analysis["structures"]):
        line = structure["line"]
        start = int(line["from"]["time_ordinal"])
        x_values = list(range(start, chart_end + 1))
        y_values = [_structure_value(structure, value) for value in x_values]
        lower_pct = float(structure["band"]["lower_pct"]) / 100.0
        upper_pct = float(structure["band"]["upper_pct"]) / 100.0
        color = PRIMARY if index == 0 else SECONDARY
        fill = PRIMARY_BAND if index == 0 else SECONDARY
        ax.fill_between(
            x_values,
            [value * (1.0 - lower_pct) for value in y_values],
            [value * (1.0 + upper_pct) for value in y_values],
            color=fill,
            alpha=0.12 if index == 0 else 0.08,
            zorder=2,
        )
        ax.plot(
            x_values,
            y_values,
            color=color,
            linewidth=2.15 if index == 0 else 1.6,
            zorder=6,
        )


def _draw_reference(
    ax: plt.Axes,
    reference_setup: dict[str, Any],
    *,
    chart_start: int,
    chart_end: int,
) -> None:
    for line in reference_setup.get("lines", []):
        start = line.get("from")
        end = line.get("to")
        if not start or not end:
            continue
        x1 = _quarter_ordinal(str(start["date"]))
        x2 = _quarter_ordinal(str(end["date"]))
        y1 = float(start["price"])
        y2 = float(end["price"])
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        visible_start = max(chart_start, x1)
        visible_end = min(chart_end, x2)
        if visible_start <= visible_end:
            xs = [visible_start, visible_end]
            ys = [
                y1 + slope * (visible_start - x1),
                y1 + slope * (visible_end - x1),
            ]
            ax.plot(
                xs,
                ys,
                color=REFERENCE,
                linewidth=1.8,
                linestyle=(0, (5, 3)),
                alpha=0.95,
                zorder=7,
            )
        anchors = line.get("constructionAnchors") or []
        if anchors:
            ax.scatter(
                [_quarter_ordinal(str(item["date"])) for item in anchors],
                [float(item["price"]) for item in anchors],
                marker="d",
                s=48,
                facecolor="none",
                edgecolor=REFERENCE,
                linewidth=1.4,
                zorder=10,
            )


def _validation_metrics(
    analysis: dict[str, Any], reference_setup: dict[str, Any]
) -> dict[str, Any]:
    primary = next(
        (item for item in analysis["structures"] if item["kind"] == "line"),
        None,
    )
    reference = next(
        (
            item
            for item in reference_setup.get("lines", [])
            if item.get("role") == "primary_lid"
        ),
        None,
    )
    result = {
        "detected_structure_count": len(analysis["structures"]),
        "detected_primary_anchor_dates": [],
        "reference_primary_anchor_dates": [],
        "reference_anchor_rms_error_pct": None,
        "direction_match": None,
    }
    if primary is None or reference is None:
        return result
    episodes = _episode_roles(analysis)
    result["detected_primary_anchor_dates"] = [
        episodes[item]["date"] for item in primary["construction_anchor_ids"]
    ]
    anchors = reference.get("constructionAnchors") or []
    result["reference_primary_anchor_dates"] = [item["date"] for item in anchors]
    errors = []
    for anchor in anchors:
        ordinal = _quarter_ordinal(str(anchor["date"]))
        expected = float(anchor["price"])
        projected = _structure_value(primary, ordinal)
        if expected > 0:
            errors.append((projected - expected) / expected * 100.0)
    if errors:
        result["reference_anchor_rms_error_pct"] = round(
            math.sqrt(sum(value * value for value in errors) / len(errors)), 2
        )
    from_point = reference.get("from")
    to_point = reference.get("to")
    if from_point and to_point:
        reference_direction = math.copysign(
            1.0, float(to_point["price"]) - float(from_point["price"])
        )
        detector_direction = math.copysign(
            1.0, float(primary["line"]["slope_per_quarter"])
        )
        result["direction_match"] = reference_direction == detector_direction
    return result


def _summary_lines(analysis: dict[str, Any]) -> list[str]:
    ladder = analysis["reference_ladder"]
    key = [
        item
        for item in ladder
        if item["status"]
        in {"demoted_singleton", "tracking_new_high", "active_supported"}
    ][:5]
    ladder_text = "  →  ".join(
        f"{item['reference_price']:g} {item['status'].replace('_', ' ')}" for item in key
    )
    lines = [f"Reference ladder: {ladder_text}"]
    if analysis["structures"]:
        primary = analysis["structures"][0]
        lines.append(
            "Primary line: "
            f"{primary['line']['from']['date'][:7]} → {primary['line']['to']['date'][:7]}  |  "
            f"{primary['fit']['touch_count']} contacts  |  "
            f"{primary['line']['slope_pct_per_year']:+g}%/yr"
        )
        if len(analysis["structures"]) > 1:
            child = analysis["structures"][1]
            lines.append(
                "Nested family retained: "
                f"{child['fit']['touch_count']} contacts around {child['line']['from']['price']:g}"
            )
    return lines


def _replay_cutoffs(
    snapshot: dict[str, Any], reference_setup: dict[str, Any]
) -> dict[str, Any]:
    replays: dict[str, Any] = {}
    for field in (
        "firstRecognizableDate",
        "firstWatchDate",
        "firstActionableDate",
    ):
        cutoff = reference_setup.get(field)
        if not cutoff:
            continue
        analysis = analyze_lifetime_references(
            snapshot["monthly_bars"], as_of=str(cutoff)
        )
        metrics = _validation_metrics(analysis, reference_setup)
        rms = metrics["reference_anchor_rms_error_pct"]
        if not metrics["detected_primary_anchor_dates"]:
            assessment = "no_line_yet"
        elif metrics["direction_match"] is True and rms is not None and rms <= 10.0:
            assessment = "geometry_aligned"
        else:
            assessment = "different_geometry"
        replays[field] = {
            "as_of": cutoff,
            "assessment": assessment,
            "primary_anchor_dates": metrics["detected_primary_anchor_dates"],
            "direction_match": metrics["direction_match"],
            "reference_anchor_rms_error_pct": rms,
        }
    return replays


def _render_chart(
    ticker: str,
    snapshot: dict[str, Any],
    analysis: dict[str, Any],
    output_path: Path,
    *,
    as_of: Optional[str],
) -> None:
    monthly = _clean_bars(snapshot["monthly_bars"], as_of)
    quarters = _aggregate_quarterly_display_bars(monthly)
    reference_setup = (snapshot.get("corpus_labels") or {}).get(
        "reference_setup", {}
    )
    start = _quarter_ordinal(str(quarters[0]["date"]))
    end = _quarter_ordinal(str(quarters[-1]["date"]))
    primary = analysis["structures"][0] if analysis["structures"] else None
    focus_start = (
        max(start, int(primary["line"]["from"]["time_ordinal"]) - 4)
        if primary
        else start
    )
    focus_quarters = [
        item
        for item in quarters
        if focus_start <= _quarter_ordinal(str(item["date"])) <= end
    ]

    fig = plt.figure(figsize=(17, 10), facecolor=BG)
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.36, 0.64],
        hspace=0.12,
        left=0.055,
        right=0.975,
        top=0.88,
        bottom=0.11,
    )
    top_grid = grid[0].subgridspec(
        1, 2, width_ratios=[0.79, 0.21], wspace=0.025
    )
    ax_context = fig.add_subplot(top_grid[0])
    ax_info = fig.add_subplot(top_grid[1])
    ax_focus = fig.add_subplot(grid[1])
    _style_axis(ax_context)
    _style_axis(ax_focus)
    ax_info.set_facecolor(PANEL)
    ax_info.set_xticks([])
    ax_info.set_yticks([])
    for spine in ax_info.spines.values():
        spine.set_color(GRID)

    _draw_candles(ax_context, quarters, alpha=0.58)
    _draw_episode_markers(ax_context, analysis, label_prices=True)
    ax_context.set_xlim(start - 1, end + 1)
    _set_year_ticks(ax_context, start, end)
    ax_context.set_ylabel("Price")
    ax_context.set_title(
        "A. Observed lifetime — rank every confirmed top; keep unsupported extremes visible",
        loc="left",
        color=TEXT,
        fontsize=11,
        pad=8,
    )
    metrics = _validation_metrics(analysis, reference_setup)
    info_lines = _summary_lines(analysis)
    comparison_rms = metrics["reference_anchor_rms_error_pct"]
    info_lines.extend(
        [
            "",
            "Comparison only",
            f"Direction: {'match' if metrics['direction_match'] else 'mismatch' if metrics['direction_match'] is False else 'n/a'}",
            (
                f"Reference-anchor RMS: {comparison_rms:.2f}%"
                if comparison_rms is not None
                else "Reference-anchor RMS: n/a"
            ),
        ]
    )
    ax_info.text(
        0.06,
        0.94,
        "Detector reading",
        transform=ax_info.transAxes,
        ha="left",
        va="top",
        color=TEXT,
        fontsize=11,
        fontweight="bold",
    )
    ax_info.text(
        0.06,
        0.84,
        "\n\n".join(info_lines),
        transform=ax_info.transAxes,
        ha="left",
        va="top",
        color=MUTED,
        fontsize=8.2,
        linespacing=1.25,
        wrap=True,
    )

    _draw_candles(ax_focus, focus_quarters)
    _draw_structures(ax_focus, analysis, chart_end=end)
    _draw_reference(
        ax_focus,
        reference_setup,
        chart_start=focus_start,
        chart_end=end,
    )
    _draw_episode_markers(
        ax_focus,
        analysis,
        label_prices=False,
        x_min=focus_start,
    )
    recognize = reference_setup.get("firstRecognizableDate")
    if recognize:
        recognize_x = _quarter_ordinal(str(recognize))
        if focus_start <= recognize_x <= end:
            ax_focus.axvline(
                recognize_x,
                color=MUTED,
                linestyle=(0, (2, 4)),
                linewidth=1.0,
                alpha=0.8,
            )
            ax_focus.annotate(
                "Amrut: first recognisable",
                (recognize_x, 0.985),
                xycoords=("data", "axes fraction"),
                xytext=(5, -2),
                textcoords="offset points",
                color=MUTED,
                fontsize=8,
                va="top",
            )
    ax_focus.set_xlim(focus_start - 1, end + 1)
    _set_year_ticks(ax_focus, focus_start, end)
    ax_focus.set_ylabel("Price")
    ax_focus.set_xlabel("Calendar time (quarterly candles)")
    ax_focus.set_title(
        "B. Active geometry — fixed anchors, asymmetric price band, and retained lower family",
        loc="left",
        color=TEXT,
        fontsize=11,
        pad=8,
    )

    visible_values = [
        value
        for bar in focus_quarters
        for value in (float(bar["low"]), float(bar["high"]))
    ]
    if visible_values:
        low, high = min(visible_values), max(visible_values)
        padding = max((high - low) * 0.08, high * 0.03)
        ax_focus.set_ylim(max(0.0, low - padding), high + padding)

    company = reference_setup.get("company") or ""
    suffix = f" — {company}" if company else ""
    cutoff = f" | as of {as_of}" if as_of else " | current frozen history"
    fig.suptitle(
        f"{ticker}{suffix}\nObserved-lifetime reference ladder{cutoff}",
        x=0.055,
        y=0.965,
        ha="left",
        va="top",
        color=TEXT,
        fontsize=17,
        fontweight="bold",
        linespacing=1.35,
    )
    legend = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor=PRIMARY, markeredgecolor="white", markersize=11, label="Detector construction anchor"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PANEL, markeredgecolor=TOUCH, markersize=7, label="Detector supporting touch"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=PANEL, markeredgecolor=SECONDARY, markersize=7, label="Nested-family contact"),
        Line2D([0], [0], color=PRIMARY, linewidth=2.2, label="Detector outer line"),
        Patch(facecolor=PRIMARY_BAND, alpha=0.22, label="Detector asymmetric band"),
        Line2D([0], [0], color=SECONDARY, linewidth=1.8, label="Detector nested line"),
        Line2D([0], [0], color=REFERENCE, linestyle=(0, (5, 3)), linewidth=1.8, label="Amrut reference (comparison only)"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=DEMOTED, markersize=8, label="Demoted singleton"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=TRACKING, markersize=7, label="New high: tracking only"),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.028),
        ncol=5,
        frameon=False,
        labelcolor=MUTED,
        fontsize=8,
    )
    fig.text(
        0.055,
        0.012,
        "Development comparison on an outcome-revealed exemplar. The purple reference is never passed into detection; this is not a blind accuracy or investment claim.",
        color=MUTED,
        fontsize=8,
        ha="left",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, facecolor=BG)
    plt.close(fig)


def _report_markdown(
    output_dir: Path,
    results: list[dict[str, Any]],
    *,
    source: str,
    as_of: Optional[str],
) -> None:
    lines = [
        "# Observed-lifetime reference-line experiment",
        "",
        "This is a development comparison, not a production detector release. The experimental detector receives only frozen monthly OHLCV bars. Amrut's outcome-revealed geometry is added afterward as a dashed comparison overlay.",
        "",
        f"- Frozen source: `{source}`",
        f"- Cutoff: `{as_of or 'full frozen history'}`",
        "- Price definition: quarterly candle high",
        "- Time definition: calendar-quarter ordinal (missing quarters do not compress slope)",
        "- Production `analyze_coil()` remains unchanged",
        "",
        "| Ticker | Detector anchors | Amrut anchors | Direction | Reference-anchor RMS error | Structures retained |",
        "|---|---|---|---:|---:|---:|",
    ]
    for result in results:
        metrics = result["metrics"]
        detector_dates = ", ".join(metrics["detected_primary_anchor_dates"]) or "—"
        reference_dates = ", ".join(metrics["reference_primary_anchor_dates"]) or "—"
        direction = (
            "match"
            if metrics["direction_match"] is True
            else "mismatch"
            if metrics["direction_match"] is False
            else "n/a"
        )
        rms = metrics["reference_anchor_rms_error_pct"]
        rms_text = f"{rms:.2f}%" if rms is not None else "n/a"
        lines.append(
            f"| [{result['ticker']}]({result['image']}) | {detector_dates} | {reference_dates} | {direction} | {rms_text} | {metrics['detected_structure_count']} |"
        )
    lines.extend(
        [
            "",
            "## How to read the charts",
            "",
            "The top pane shows why the highest observed price is not automatically the active resistance. Red X markers remain in the record but are demoted when they cannot form a credible price-and-time family. The lower pane fixes the earliest credible pair of anchors, projects it through calendar time, accepts later tops inside an asymmetric band, and retains a lower repeated-price family when one exists.",
            "",
            "The numeric RMS comparison is descriptive only. These are teaching examples with estimated, outcome-revealed reference geometry, so it cannot establish population accuracy.",
            "",
            "## Historical cutoff replay",
            "",
            "This is the stricter check: what the detector could draw using only bars available at Amrut's dated milestones. `Aligned` means the direction matched and the detector line was within 10% RMS of the reference anchors; it is not a claim that the thresholds are final.",
            "",
            "| Ticker | First recognisable | First watch | First actionable |",
            "|---|---|---|---|",
        ]
    )
    replay_labels = {
        "geometry_aligned": "aligned",
        "different_geometry": "different line",
        "no_line_yet": "no line yet",
    }
    for result in results:
        replay = result.get("replays", {})

        def replay_cell(field: str) -> str:
            item = replay.get(field)
            if not item:
                return "—"
            dates = ", ".join(item["primary_anchor_dates"])
            suffix = f" ({dates})" if dates else ""
            return f"{replay_labels[item['assessment']]}{suffix}"

        lines.append(
            f"| {result['ticker']} | {replay_cell('firstRecognizableDate')} | {replay_cell('firstWatchDate')} | {replay_cell('firstActionableDate')} |"
        )
    lines.extend(
        [
            "",
            "The gap between first recognition and first watch is material: this version waits for a later price rejection before confirming a quarterly high. That makes it intentionally slower and prevents a live, still-rising quarter from becoming an anchor, but it also means some structures appear later than Amrut says they become visually recognisable.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--ticker", action="append", dest="tickers")
    parser.add_argument("--as-of")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "docs" / "lifetime-reference-validation" / "v1",
    )
    args = parser.parse_args()
    tickers = tuple(args.tickers or DEFAULT_TICKERS)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for ticker in tickers:
        snapshot = load_review_snapshot(args.source, ticker)
        analysis = analyze_lifetime_references(
            snapshot["monthly_bars"], as_of=args.as_of
        )
        reference_setup = (snapshot.get("corpus_labels") or {}).get(
            "reference_setup", {}
        )
        stem = ticker.replace("/", "-")
        detection_name = f"{stem}.detection.json"
        comparison_name = f"{stem}.comparison.json"
        image_name = f"{stem}.png"
        _write_json(args.output_dir / detection_name, analysis)
        _write_json(
            args.output_dir / comparison_name,
            {
                "ticker": ticker,
                "source": args.source,
                "comparison_only": True,
                "reference_setup": reference_setup,
            },
        )
        _render_chart(
            ticker,
            snapshot,
            analysis,
            args.output_dir / image_name,
            as_of=args.as_of,
        )
        results.append(
            {
                "ticker": ticker,
                "image": image_name,
                "detection": detection_name,
                "comparison": comparison_name,
                "metrics": _validation_metrics(analysis, reference_setup),
                "replays": _replay_cutoffs(snapshot, reference_setup),
            }
        )

    summary = {
        "experimental": True,
        "source": args.source,
        "as_of": args.as_of,
        "outcome_revealed_comparison": True,
        "results": results,
    }
    _write_json(args.output_dir / "validation_summary.json", summary)
    _report_markdown(
        args.output_dir,
        results,
        source=args.source,
        as_of=args.as_of,
    )
    print(f"Rendered {len(results)} chart(s) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

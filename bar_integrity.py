"""Strict, point-in-time OHLC inspection shared by analysis and benchmarks.

The detector must never obtain a trustworthy-looking structure by silently
dropping an invalid candle.  This module normalizes monthly bars, applies the
historical evidence cutoff using the end of each candle's calendar month, and
returns an explicit report.  Blocking issues leave the normalized bars visible
for diagnosis but callers must not classify them.
"""
from __future__ import annotations

import calendar
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Optional

DATA_QUALITY_VALID = "valid"
DATA_QUALITY_VALID_WITH_WARNINGS = "valid_with_warnings"
DATA_QUALITY_BLOCKED = "blocked"

ADJUSTMENT_SPLIT_ADJUSTED = "split_adjusted"
ADJUSTMENT_UNKNOWN = "unknown"


@dataclass(frozen=True)
class BarIntegrityResult:
    bars: list[dict[str, Any]]
    report: dict[str, Any]


def month_end(value: date) -> date:
    """Calendar month-end for a monthly candle label."""
    return date(value.year, value.month, calendar.monthrange(value.year, value.month)[1])


def _issue(
    severity: str,
    code: str,
    message: str,
    *,
    index: int | None = None,
    bar_date: str | None = None,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "severity": severity,
        "code": code,
        "message": message,
    }
    if index is not None:
        value["index"] = index
    if bar_date is not None:
        value["date"] = bar_date
    if fields:
        value["fields"] = fields
    return value


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _field(raw: dict[str, Any], name: str) -> Any:
    if name in raw:
        return raw[name]
    title = name.title()
    if title in raw:
        return raw[title]
    return None


def _months_between(left: date, right: date) -> int:
    return (right.year - left.year) * 12 + right.month - left.month


def _fingerprint(bars: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        bars,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _source_fingerprint(raw_bars: list[Any]) -> str:
    encoded = json.dumps(
        raw_bars,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def inspect_monthly_bars(
    raw_bars: Iterable[dict[str, Any]],
    *,
    as_of: Optional[str] = None,
    adjustment_mode: Optional[str] = None,
    today: Optional[date] = None,
    stale_after_days: int = 62,
) -> BarIntegrityResult:
    """Normalize, cutoff, and validate monthly OHLC without silent row loss.

    Historical ``as_of`` is an evidence timestamp, not a bar label.  A monthly
    row becomes available only at calendar month-end.  Live analysis (no
    ``as_of``) retains the current partial month as provisional price context;
    completed-quarter logic decides whether it may become structural evidence.
    """
    source = list(raw_bars)
    issues: list[dict[str, Any]] = []
    parsed_cutoff: date | None = None
    if as_of is not None:
        try:
            cutoff_text = str(as_of)
            parsed_cutoff = date.fromisoformat(cutoff_text)
            if parsed_cutoff.isoformat() != cutoff_text:
                raise ValueError
        except (TypeError, ValueError):
            issues.append(
                _issue("error", "invalid_as_of", "as_of must be ISO YYYY-MM-DD")
            )

    prepared: list[tuple[date, int, dict[str, Any]]] = []
    input_count = 0
    for index, raw in enumerate(source):
        input_count += 1
        if not isinstance(raw, dict):
            issues.append(
                _issue(
                    "error",
                    "invalid_bar",
                    "bar must be a JSON object",
                    index=index,
                )
            )
            continue

        raw_date = _field(raw, "date")
        date_text = str(raw_date or "")[:10]
        try:
            parsed = date.fromisoformat(date_text)
            if parsed.isoformat() != date_text:
                raise ValueError
        except (TypeError, ValueError):
            issues.append(
                _issue(
                    "error",
                    "invalid_date",
                    "bar date is not ISO YYYY-MM-DD",
                    index=index,
                    bar_date=date_text or None,
                )
            )
            continue

        # A stored monthly OHLC row represents its complete calendar month.  A
        # historical cutoff inside that month cannot use the row even when the
        # provider labels it on day one.
        if parsed_cutoff is not None and month_end(parsed) > parsed_cutoff:
            continue

        values: dict[str, float] = {}
        invalid_fields: list[str] = []
        for name in ("open", "high", "low", "close"):
            number = _finite_number(_field(raw, name))
            if number is None:
                invalid_fields.append(name)
            else:
                values[name] = number
        if invalid_fields:
            issues.append(
                _issue(
                    "error",
                    "missing_or_nonfinite_ohlc",
                    "OHLC values must be present and finite",
                    index=index,
                    bar_date=date_text,
                    fields=invalid_fields,
                )
            )
            continue

        nonpositive = [name for name, value in values.items() if value <= 0]
        if nonpositive:
            issues.append(
                _issue(
                    "error",
                    "nonpositive_ohlc",
                    "OHLC prices must be greater than zero",
                    index=index,
                    bar_date=date_text,
                    fields=nonpositive,
                )
            )

        if values["low"] > values["high"]:
            issues.append(
                _issue(
                    "error",
                    "low_above_high",
                    "low cannot exceed high",
                    index=index,
                    bar_date=date_text,
                )
            )
        if values["low"] > min(values["open"], values["close"]):
            issues.append(
                _issue(
                    "error",
                    "low_above_body",
                    "low must contain the candle body",
                    index=index,
                    bar_date=date_text,
                )
            )
        if values["high"] < max(values["open"], values["close"]):
            issues.append(
                _issue(
                    "error",
                    "high_below_body",
                    "high must contain the candle body",
                    index=index,
                    bar_date=date_text,
                )
            )

        normalized: dict[str, Any] = {"date": date_text, **values}
        raw_volume = _field(raw, "volume")
        if raw_volume is None:
            normalized["volume"] = None
            issues.append(
                _issue(
                    "warning",
                    "missing_volume",
                    "volume is unavailable",
                    index=index,
                    bar_date=date_text,
                )
            )
        else:
            volume = _finite_number(raw_volume)
            if volume is None or volume < 0:
                normalized["volume"] = None
                issues.append(
                    _issue(
                        "warning",
                        "invalid_volume",
                        "volume is non-finite or negative and cannot be evidence",
                        index=index,
                        bar_date=date_text,
                    )
                )
            else:
                normalized["volume"] = volume
        prepared.append((parsed, index, normalized))

    if any(
        prepared[pos][0] < prepared[pos - 1][0]
        for pos in range(1, len(prepared))
    ):
        issues.append(
            _issue(
                "warning",
                "out_of_order",
                "input bars were deterministically sorted by date",
            )
        )
    prepared.sort(key=lambda item: (item[0], item[1]))

    seen: set[date] = set()
    normalized_bars: list[dict[str, Any]] = []
    normalized_dates: list[date] = []
    for parsed, index, bar in prepared:
        if parsed in seen:
            issues.append(
                _issue(
                    "error",
                    "duplicate_date",
                    "bar date is duplicated",
                    index=index,
                    bar_date=bar["date"],
                )
            )
            continue
        seen.add(parsed)
        normalized_dates.append(parsed)
        normalized_bars.append(bar)

    gaps = [
        {
            "after": left.isoformat(),
            "before": right.isoformat(),
            "missing_months": _months_between(left, right) - 1,
        }
        for left, right in zip(normalized_dates, normalized_dates[1:])
        if _months_between(left, right) > 1
    ]
    if gaps:
        issue = _issue(
            "warning",
            "missing_periods",
            "one or more calendar months are missing",
        )
        issue["gaps"] = gaps
        issues.append(issue)

    discontinuities: list[dict[str, Any]] = []
    for previous, current in zip(normalized_bars, normalized_bars[1:]):
        prior_close = float(previous["close"])
        current_open = float(current["open"])
        ratio = current_open / prior_close if prior_close > 0 else 0.0
        if ratio >= 4.0 or ratio <= 0.25:
            discontinuities.append(
                {
                    "after": previous["date"],
                    "date": current["date"],
                    "open_to_prior_close_ratio": round(ratio, 6),
                }
            )
    if discontinuities:
        issue = _issue(
            "warning",
            "corporate_action_like_discontinuity",
            "large price discontinuity requires adjustment/provenance review",
        )
        issue["events"] = discontinuities
        issues.append(issue)

    reference = parsed_cutoff or today or date.today()
    if normalized_dates:
        final_period_end = month_end(normalized_dates[-1])
        if (reference - final_period_end).days > max(0, stale_after_days):
            issues.append(
                _issue(
                    "warning",
                    "stale_final_data",
                    "final monthly candle is stale for the analysis cutoff",
                    bar_date=normalized_dates[-1].isoformat(),
                )
            )
    else:
        issues.append(
            _issue("error", "no_usable_bars", "no monthly bars remain at the cutoff")
        )

    mode = str(adjustment_mode or ADJUSTMENT_UNKNOWN).strip() or ADJUSTMENT_UNKNOWN
    if mode != ADJUSTMENT_SPLIT_ADJUSTED:
        issues.append(
            _issue(
                "warning",
                "unverified_adjustment_mode",
                "canonical validation requires split-adjusted OHLC provenance",
            )
        )

    blocking = [issue for issue in issues if issue["severity"] == "error"]
    warnings = [issue for issue in issues if issue["severity"] == "warning"]
    status = (
        DATA_QUALITY_BLOCKED
        if blocking
        else DATA_QUALITY_VALID_WITH_WARNINGS
        if warnings
        else DATA_QUALITY_VALID
    )
    report = {
        "status": status,
        "blocked": bool(blocking),
        "input_bar_count": input_count,
        "accepted_bar_count": len(normalized_bars),
        "requested_cutoff": str(as_of) if as_of is not None else None,
        "effective_bar_cutoff": (
            month_end(normalized_dates[-1]).isoformat()
            if normalized_dates
            else None
        ),
        "evidence_cutoff": parsed_cutoff.isoformat() if parsed_cutoff else None,
        "adjustment_mode": mode,
        "source_fingerprint": _source_fingerprint(source),
        "bar_fingerprint": _fingerprint(normalized_bars),
        "blocking_issue_count": len(blocking),
        "warning_count": len(warnings),
        "issues": issues,
    }
    return BarIntegrityResult(bars=normalized_bars, report=report)

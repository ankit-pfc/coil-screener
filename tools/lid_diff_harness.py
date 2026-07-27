#!/usr/bin/env python3
"""Measurement harness for the lid-detection v2.2.0 change.

Three modes:

``run``
    Execute ``analyze_coil`` over every cached ticker of one repo tree and write
    a JSON snapshot.  ``--repo`` selects the tree, so the *same* harness can
    measure a frozen pre-change baseline and the live post-change working tree.

``diff``
    Compare two snapshots and emit a per-ticker change table plus aggregate
    counts (markdown to stdout/``--report``, machine-readable JSON to ``--json``).

``replay``
    Run the ``(ticker, as_of)`` regression assertions in ``REPLAY_ASSERTIONS``
    against one tree and report pass/fail.

The harness never lets two repo trees collide in one module namespace: each tree
is loaded with importlib under a module name derived from its own path, and the
tree root is only transiently on ``sys.path``.

The harness is read-only with respect to the trees it measures.  It writes
nothing but the snapshot / report files it is told to write.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from types import ModuleType
from typing import Any, Iterable, Optional

SNAPSHOT_VERSION = 1

# Role emitted by the pre-2.2.0 analyzer for a right-edge peak that has no
# future confirmation.  v2.2.0 must never emit it again; the harness counts it.
ROLE_PROVISIONAL_TOP = "provisional_top"

# Slope buckets aligned to the CoilConfig grade bands (grade_min -3, grade_a_min
# -1, grade_a_max 5, grade_b_max 6.5, grade_c_max 12).
SLOPE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("<-3 (too negative)", float("-inf"), -3.0),
    ("-3..-1", -3.0, -1.0),
    ("-1..5 (A band)", -1.0, 5.0),
    ("5..6.5 (B band)", 5.0, 6.5),
    ("6.5..12 (C band)", 6.5, 12.0),
    (">=12 (too steep)", 12.0, float("inf")),
)

# Data-driven regression assertions for ``replay`` mode.  Add entries here; each
# ``expect`` key is compared against the matching field of the ticker record
# produced by :func:`build_record`.  A value of ``None`` asserts the field is
# null.  ``proximity_pct`` and other floats accept a ``[lo, hi]`` two-element
# list to assert a range instead of equality.
REPLAY_ASSERTIONS: list[dict[str, Any]] = [
    {
        "ticker": "KN",
        "as_of": "2025-09-30",
        "expect": {"grade": "A"},
        "why": (
            "Reference set: KN is a genuine A coil as of 2025-09-30. It is "
            "correctly rejected as of today (price has left the lid band), so "
            "the historical replay must keep grading A or the fix has "
            "over-reached and destroyed a known-good name."
        ),
    },
]


# --------------------------------------------------------------------------
# repo tree loading
# --------------------------------------------------------------------------


def _module_suffix(repo: Path) -> str:
    """Stable, import-safe suffix identifying a repo tree."""
    return re.sub(r"[^0-9A-Za-z]+", "_", str(repo)).strip("_")


def _load_module_from(repo: Path, module_name: str) -> ModuleType:
    """Import ``<repo>/<module_name>.py`` under a repo-scoped module name.

    The repo root is prepended to ``sys.path`` only for the duration of the
    import so any sibling import inside the module resolves against the same
    tree, and the module is registered under a unique name so two trees can
    never overwrite one another in ``sys.modules``.
    """
    path = repo / f"{module_name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"{module_name}.py not found in repo tree {repo}")
    unique = f"{module_name}__{_module_suffix(repo)}"
    cached = sys.modules.get(unique)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(unique, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique] = module
    inserted = str(repo)
    sys.path.insert(0, inserted)
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(unique, None)
        raise
    finally:
        try:
            sys.path.remove(inserted)
        except ValueError:  # pragma: no cover - only if something else popped it
            pass
    return module


def load_tree(repo: Path, cache_dir: Optional[Path] = None) -> tuple[ModuleType, ModuleType, Path]:
    """Load ``coil_analysis`` and ``history_cache`` from ``repo``.

    Returns ``(coil_analysis, history_cache, resolved_cache_dir)``.  When
    ``cache_dir`` is given it is patched onto the loaded ``history_cache``
    module so its reader looks in the override directory.
    """
    repo = repo.resolve()
    coil_analysis = _load_module_from(repo, "coil_analysis")
    history_cache = _load_module_from(repo, "history_cache")
    resolved = (cache_dir.resolve() if cache_dir else Path(history_cache.CACHE_DIR))
    history_cache.CACHE_DIR = resolved
    return coil_analysis, history_cache, resolved


def list_tickers(cache_dir: Path) -> list[str]:
    """Every ticker with a cache payload, sorted, `.tmp` scratch files excluded."""
    return sorted(
        p.name[: -len(".json")]
        for p in cache_dir.glob("*.json")
        if p.is_file() and not p.name.endswith(".tmp.json")
    )


# --------------------------------------------------------------------------
# record extraction (pure given a result dict)
# --------------------------------------------------------------------------


def last_close_at(bars: Iterable[dict[str, Any]], as_of: Optional[str]) -> Optional[float]:
    """Close of the last bar on/before ``as_of`` (or the last bar outright)."""
    last: Optional[float] = None
    for bar in bars:
        date = str(bar.get("date") or "")[:10]
        if as_of and date > as_of:
            continue
        close = bar.get("close")
        if close is None:
            continue
        try:
            last = float(close)
        except (TypeError, ValueError):
            continue
    return last


def _anchor_list(active_lid: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(active_lid, dict):
        return []
    anchors = active_lid.get("anchors")
    if not isinstance(anchors, list):
        return []
    out: list[dict[str, Any]] = []
    for anchor in anchors:
        if not isinstance(anchor, dict):
            continue
        out.append(
            {
                "idx": anchor.get("idx"),
                "date": anchor.get("date"),
                "price": anchor.get("price"),
            }
        )
    return out


def build_record(
    ticker: str,
    result: dict[str, Any],
    *,
    last_close: Optional[float] = None,
) -> dict[str, Any]:
    """Flatten one ``analyze_coil`` result into the snapshot record shape.

    Pure: no I/O, no module state.  Missing fields (notably
    ``metrics.current_price_position``, which does not exist pre-2.2.0) become
    ``None`` rather than raising, so a baseline snapshot and a post-change
    snapshot share one schema.
    """
    metrics = result.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    active_lid = result.get("active_lid")
    active_lid = active_lid if isinstance(active_lid, dict) else None
    anchors = _anchor_list(active_lid)
    anchor_idxs = {a["idx"] for a in anchors if a.get("idx") is not None}

    points = result.get("points")
    points = points if isinstance(points, list) else []
    role_counts: dict[str, int] = {}
    provisional_idxs: set[Any] = set()
    provisional_lid_members = 0
    for point in points:
        if not isinstance(point, dict):
            continue
        role = str(point.get("role") or "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
        if role == ROLE_PROVISIONAL_TOP:
            provisional_idxs.add(point.get("idx"))
            if point.get("lid_member"):
                provisional_lid_members += 1

    major_highs = result.get("major_highs")
    major_highs = major_highs if isinstance(major_highs, list) else []

    provisional_anchor_idxs = sorted(
        idx for idx in provisional_idxs & anchor_idxs if idx is not None
    )

    return {
        "ticker": ticker,
        "ok": True,
        "error": None,
        "algorithm_version": result.get("algorithm_version"),
        "schema_version": result.get("schema_version"),
        "as_of": result.get("as_of"),
        "bar_count": result.get("bar_count"),
        "grade": result.get("grade"),
        "lifecycle": result.get("lifecycle"),
        "status": result.get("status"),
        "coil_score": result.get("coil_score"),
        "last_close": last_close,
        "proximity_pct": metrics.get("proximity_pct"),
        "current_price_position": metrics.get("current_price_position"),
        "has_structure": active_lid is not None,
        "anchors": anchors,
        "anchor_count": len(anchors),
        "slope_pct_per_year": (active_lid or {}).get("slope_pct_per_year"),
        "lid_grade": (active_lid or {}).get("grade"),
        "lid_value_at_last_bar": (active_lid or {}).get("value_at_last_bar"),
        "touch_count": (active_lid or {}).get("touch_count"),
        "span_years": (active_lid or {}).get("span_years"),
        "role_counts": role_counts,
        "role_point_count": len(points),
        "major_highs_count": len(major_highs),
        "provisional_top_count": len(provisional_idxs),
        "provisional_top_lid_member_count": provisional_lid_members,
        "provisional_anchor_idxs": provisional_anchor_idxs,
        "uses_provisional_anchor": bool(provisional_anchor_idxs),
        "notes": list(result.get("notes") or []),
    }


def error_record(ticker: str, exc: BaseException) -> dict[str, Any]:
    """Record for a ticker whose analysis blew up. Keeps the run going."""
    return {
        "ticker": ticker,
        "ok": False,
        "error": f"{type(exc).__name__}: {exc}",
        "algorithm_version": None,
        "schema_version": None,
        "as_of": None,
        "bar_count": None,
        "grade": None,
        "lifecycle": None,
        "status": None,
        "coil_score": None,
        "last_close": None,
        "proximity_pct": None,
        "current_price_position": None,
        "has_structure": False,
        "anchors": [],
        "anchor_count": 0,
        "slope_pct_per_year": None,
        "lid_grade": None,
        "lid_value_at_last_bar": None,
        "touch_count": None,
        "span_years": None,
        "role_counts": {},
        "role_point_count": 0,
        "major_highs_count": 0,
        "provisional_top_count": 0,
        "provisional_top_lid_member_count": 0,
        "provisional_anchor_idxs": [],
        "uses_provisional_anchor": False,
        "notes": [],
    }


# --------------------------------------------------------------------------
# run mode
# --------------------------------------------------------------------------


def run_snapshot(
    repo: Path,
    *,
    cache_dir: Optional[Path] = None,
    as_of: Optional[str] = None,
    tickers: Optional[list[str]] = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Analyze every cached ticker in ``repo`` and return the snapshot dict."""
    coil_analysis, history_cache, resolved_cache = load_tree(repo, cache_dir)
    symbols = tickers if tickers is not None else list_tickers(resolved_cache)

    records: dict[str, Any] = {}
    for symbol in symbols:
        try:
            payload = history_cache.read_cache(symbol)
            if not payload or not payload.get("bars"):
                raise ValueError(f"no cached bars for {symbol}")
            bars = payload["bars"]
            result = coil_analysis.analyze_coil(
                bars,
                config=coil_analysis.DEFAULT_CONFIG,
                as_of=as_of,
            )
            records[symbol] = build_record(
                symbol, result, last_close=last_close_at(bars, as_of)
            )
        except BaseException as exc:  # one bad ticker must not kill the run
            if isinstance(exc, KeyboardInterrupt):
                raise
            records[symbol] = error_record(symbol, exc)
        if progress:
            print(f"  {symbol}", file=sys.stderr)

    versions = sorted({r["algorithm_version"] for r in records.values() if r["algorithm_version"]})
    return {
        "snapshot_version": SNAPSHOT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo": str(repo.resolve()),
        "cache": str(resolved_cache),
        "as_of": as_of,
        "algorithm_versions": versions,
        "ticker_count": len(records),
        "records": records,
    }


# --------------------------------------------------------------------------
# diff mode (pure)
# --------------------------------------------------------------------------


def _anchor_key(record: dict[str, Any]) -> Optional[list[list[Any]]]:
    """Comparable anchor identity: ordered (date, price) pairs, or None."""
    if not record.get("has_structure"):
        return None
    return [[a.get("date"), a.get("price")] for a in record.get("anchors") or []]


def format_anchor_key(key: Optional[list[list[Any]]]) -> str:
    """Render an anchor key (from :func:`_anchor_key`) as one table cell.

    Members are comma-separated and bracketed so the list separator can never
    be confused with the ``before -> after`` arrow in the report table.
    """
    if key is None:
        return "-"
    if not key:
        return "[]"
    return "[" + ", ".join(f"{d}@{p}" for d, p in key) + "]"


def slope_distribution(slopes: Iterable[Optional[float]]) -> dict[str, Any]:
    """Bucket counts plus min/median/max over the non-null slopes."""
    values = [float(s) for s in slopes if s is not None]
    buckets = {label: 0 for label, _, _ in SLOPE_BUCKETS}
    for value in values:
        for label, lo, hi in SLOPE_BUCKETS:
            if lo <= value < hi:
                buckets[label] += 1
                break
    return {
        "count": len(values),
        "buckets": buckets,
        "min": round(min(values), 2) if values else None,
        "median": round(median(values), 2) if values else None,
        "max": round(max(values), 2) if values else None,
    }


def diff_records(before: Optional[dict[str, Any]], after: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Per-ticker change row. Either side may be ``None`` (ticker added/removed)."""
    empty: dict[str, Any] = {}
    b = before or empty
    a = after or empty
    fields = (
        "grade",
        "lifecycle",
        "status",
        "proximity_pct",
        "slope_pct_per_year",
        "coil_score",
        "current_price_position",
        "has_structure",
        "uses_provisional_anchor",
        "ok",
    )
    row: dict[str, Any] = {
        "ticker": (a.get("ticker") or b.get("ticker")),
        "present_before": before is not None,
        "present_after": after is not None,
    }
    for field in fields:
        row[f"{field}_before"] = b.get(field)
        row[f"{field}_after"] = a.get(field)
    row["anchors_before"] = _anchor_key(b) if before is not None else None
    row["anchors_after"] = _anchor_key(a) if after is not None else None
    row["anchors_changed"] = row["anchors_before"] != row["anchors_after"]
    row["error_before"] = b.get("error")
    row["error_after"] = a.get("error")
    row["notes_after"] = list(a.get("notes") or [])

    changed_fields = [
        field
        for field in ("grade", "lifecycle", "status", "proximity_pct", "slope_pct_per_year", "has_structure")
        if b.get(field) != a.get(field)
    ]
    if row["anchors_changed"]:
        changed_fields.append("anchors")
    row["changed_fields"] = changed_fields
    row["changed"] = bool(changed_fields) or before is None or after is None
    return row


def _graded(record: dict[str, Any]) -> bool:
    return record.get("grade") is not None


def build_diff(before_snapshot: dict[str, Any], after_snapshot: dict[str, Any]) -> dict[str, Any]:
    """Full machine-readable diff: per-ticker rows plus aggregates.

    Pure over two snapshot dicts.  Tickers present on only one side are still
    reported (``present_before`` / ``present_after``) and are excluded from the
    transition counts, which only make sense for a ticker on both sides.
    """
    before_recs: dict[str, Any] = before_snapshot.get("records") or {}
    after_recs: dict[str, Any] = after_snapshot.get("records") or {}
    tickers = sorted(set(before_recs) | set(after_recs))

    rows = [diff_records(before_recs.get(t), after_recs.get(t)) for t in tickers]
    both = [t for t in tickers if t in before_recs and t in after_recs]

    newly_rejected: list[dict[str, Any]] = []
    newly_graded: list[dict[str, Any]] = []
    lost_structure: list[str] = []
    gained_structure: list[str] = []
    grade_changed: list[str] = []
    lifecycle_changed: list[str] = []
    anchors_changed: list[str] = []
    for ticker in both:
        b, a = before_recs[ticker], after_recs[ticker]
        if _graded(b) and not _graded(a):
            newly_rejected.append(
                {
                    "ticker": ticker,
                    "grade_before": b.get("grade"),
                    "current_price_position": a.get("current_price_position"),
                    "proximity_pct_before": b.get("proximity_pct"),
                    "proximity_pct_after": a.get("proximity_pct"),
                    "lifecycle_after": a.get("lifecycle"),
                    "has_structure_after": bool(a.get("has_structure")),
                    "notes_after": list(a.get("notes") or []),
                }
            )
        if not _graded(b) and _graded(a):
            newly_graded.append(
                {
                    "ticker": ticker,
                    "grade_after": a.get("grade"),
                    "lifecycle_before": b.get("lifecycle"),
                    "proximity_pct_after": a.get("proximity_pct"),
                    "current_price_position": a.get("current_price_position"),
                }
            )
        if b.get("has_structure") and not a.get("has_structure"):
            lost_structure.append(ticker)
        if not b.get("has_structure") and a.get("has_structure"):
            gained_structure.append(ticker)
        if b.get("grade") != a.get("grade"):
            grade_changed.append(ticker)
        if b.get("lifecycle") != a.get("lifecycle"):
            lifecycle_changed.append(ticker)
        if _anchor_key(b) != _anchor_key(a):
            anchors_changed.append(ticker)

    position_counts: dict[str, int] = {}
    for entry in newly_rejected:
        key = str(entry["current_price_position"])
        position_counts[key] = position_counts.get(key, 0) + 1

    def _side(recs: dict[str, Any]) -> dict[str, Any]:
        values = list(recs.values())
        return {
            "tickers": len(values),
            "ok": sum(1 for r in values if r.get("ok")),
            "errors": sorted(r["ticker"] for r in values if not r.get("ok")),
            "graded": sum(1 for r in values if _graded(r)),
            "with_structure": sum(1 for r in values if r.get("has_structure")),
            "provisional_emitted": sum(1 for r in values if (r.get("provisional_top_count") or 0) > 0),
            "provisional_as_anchor": sum(1 for r in values if r.get("uses_provisional_anchor")),
            "grade_counts": _count_by(values, "grade"),
            "lifecycle_counts": _count_by(values, "lifecycle"),
            "current_price_position_counts": _count_by(values, "current_price_position"),
            "slope_distribution": slope_distribution(r.get("slope_pct_per_year") for r in values),
        }

    return {
        "before": {
            "path": before_snapshot.get("_path"),
            "repo": before_snapshot.get("repo"),
            "as_of": before_snapshot.get("as_of"),
            "algorithm_versions": before_snapshot.get("algorithm_versions"),
            **_side(before_recs),
        },
        "after": {
            "path": after_snapshot.get("_path"),
            "repo": after_snapshot.get("repo"),
            "as_of": after_snapshot.get("as_of"),
            "algorithm_versions": after_snapshot.get("algorithm_versions"),
            **_side(after_recs),
        },
        "summary": {
            "tickers_compared": len(both),
            "only_in_before": sorted(set(before_recs) - set(after_recs)),
            "only_in_after": sorted(set(after_recs) - set(before_recs)),
            "changed": sum(1 for row in rows if row["changed"]),
            "newly_rejected_count": len(newly_rejected),
            "newly_rejected": newly_rejected,
            "newly_rejected_by_position": position_counts,
            "newly_graded_count": len(newly_graded),
            "newly_graded": newly_graded,
            "lost_structure_count": len(lost_structure),
            "lost_structure": lost_structure,
            "gained_structure_count": len(gained_structure),
            "gained_structure": gained_structure,
            "grade_changed": grade_changed,
            "lifecycle_changed": lifecycle_changed,
            "anchors_changed_count": len(anchors_changed),
            "anchors_changed": anchors_changed,
        },
        "rows": rows,
    }


def _count_by(records: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = str(record.get(field))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_report(diff: dict[str, Any]) -> str:
    """Human-readable markdown for a diff produced by :func:`build_diff`."""
    before, after, summary = diff["before"], diff["after"], diff["summary"]
    out: list[str] = []
    out.append("# Lid diff report")
    out.append("")
    out.append(f"- before: `{before.get('repo')}` versions={before.get('algorithm_versions')}")
    out.append(f"- after:  `{after.get('repo')}` versions={after.get('algorithm_versions')}")
    out.append(f"- as_of:  before={before.get('as_of')} after={after.get('as_of')}")
    out.append("")

    out.append("## Aggregate")
    out.append("")
    out.append("| metric | before | after |")
    out.append("|---|---:|---:|")
    for label, key in (
        ("tickers", "tickers"),
        ("analyzed ok", "ok"),
        ("graded (grade not null)", "graded"),
        ("with structure (active lid)", "with_structure"),
        ("emitted provisional_top", "provisional_emitted"),
        ("provisional_top used as lid anchor", "provisional_as_anchor"),
    ):
        out.append(f"| {label} | {before[key]} | {after[key]} |")
    out.append("")
    out.append(f"- tickers compared on both sides: **{summary['tickers_compared']}**")
    out.append(f"- tickers with any change: **{summary['changed']}**")
    out.append(f"- newly rejected (graded -> ungraded): **{summary['newly_rejected_count']}**")
    if summary["newly_rejected_by_position"]:
        for pos, count in sorted(summary["newly_rejected_by_position"].items()):
            out.append(f"    - `current_price_position={pos}`: {count}")
    out.append(f"- newly graded (ungraded -> graded): **{summary['newly_graded_count']}**")
    out.append(f"- lost structure entirely: **{summary['lost_structure_count']}**"
               + (f" ({', '.join(summary['lost_structure'])})" if summary["lost_structure"] else ""))
    out.append(f"- gained structure: **{summary['gained_structure_count']}**"
               + (f" ({', '.join(summary['gained_structure'])})" if summary["gained_structure"] else ""))
    out.append(f"- anchors changed: **{summary['anchors_changed_count']}**")
    if before["errors"] or after["errors"]:
        out.append(f"- errors: before={before['errors']} after={after['errors']}")
    out.append("")

    out.append("### Grade counts")
    out.append("")
    out.append("| grade | before | after |")
    out.append("|---|---:|---:|")
    for key in sorted(set(before["grade_counts"]) | set(after["grade_counts"])):
        out.append(f"| {key} | {before['grade_counts'].get(key, 0)} | {after['grade_counts'].get(key, 0)} |")
    out.append("")

    out.append("### Lifecycle counts")
    out.append("")
    out.append("| lifecycle | before | after |")
    out.append("|---|---:|---:|")
    for key in sorted(set(before["lifecycle_counts"]) | set(after["lifecycle_counts"])):
        out.append(
            f"| {key} | {before['lifecycle_counts'].get(key, 0)} | {after['lifecycle_counts'].get(key, 0)} |"
        )
    out.append("")

    out.append("### current_price_position (after)")
    out.append("")
    out.append("| position | before | after |")
    out.append("|---|---:|---:|")
    positions = set(before["current_price_position_counts"]) | set(after["current_price_position_counts"])
    for key in sorted(positions):
        out.append(
            f"| {key} | {before['current_price_position_counts'].get(key, 0)} "
            f"| {after['current_price_position_counts'].get(key, 0)} |"
        )
    out.append("")

    out.append("### Slope distribution (%/yr)")
    out.append("")
    bd, ad = before["slope_distribution"], after["slope_distribution"]
    out.append("| bucket | before | after |")
    out.append("|---|---:|---:|")
    for label, _, _ in SLOPE_BUCKETS:
        out.append(f"| {label} | {bd['buckets'].get(label, 0)} | {ad['buckets'].get(label, 0)} |")
    out.append(f"| **n / min / median / max** | {bd['count']} / {_fmt(bd['min'])} / {_fmt(bd['median'])} "
               f"/ {_fmt(bd['max'])} | {ad['count']} / {_fmt(ad['min'])} / {_fmt(ad['median'])} "
               f"/ {_fmt(ad['max'])} |")
    out.append("")

    out.append("## Per-ticker changes")
    out.append("")
    changed_rows = [row for row in diff["rows"] if row["changed"]]
    if not changed_rows:
        out.append("_No ticker changed._")
        out.append("")
        return "\n".join(out)
    out.append("| ticker | grade | lifecycle | proximity % | slope %/yr | position | anchors |")
    out.append("|---|---|---|---|---|---|---|")
    for row in changed_rows:
        out.append(
            "| {t} | {g0} -> {g1} | {l0} -> {l1} | {p0} -> {p1} | {s0} -> {s1} | {pos} | {a0} -> {a1} |".format(
                t=row["ticker"],
                g0=_fmt(row["grade_before"]),
                g1=_fmt(row["grade_after"]),
                l0=_fmt(row["lifecycle_before"]),
                l1=_fmt(row["lifecycle_after"]),
                p0=_fmt(row["proximity_pct_before"]),
                p1=_fmt(row["proximity_pct_after"]),
                s0=_fmt(row["slope_pct_per_year_before"]),
                s1=_fmt(row["slope_pct_per_year_after"]),
                pos=_fmt(row["current_price_position_after"]),
                a0=format_anchor_key(row["anchors_before"]),
                a1=format_anchor_key(row["anchors_after"]),
            )
        )
    out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------
# replay mode (pure evaluation)
# --------------------------------------------------------------------------


def evaluate_expectation(record: dict[str, Any], expect: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare a record against an ``expect`` mapping. Returns per-field results.

    A two-element list value asserts an inclusive numeric range; anything else
    asserts equality (``None`` asserts the field is null).
    """
    checks: list[dict[str, Any]] = []
    for field, expected in expect.items():
        actual = record.get(field)
        if isinstance(expected, list) and len(expected) == 2 and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in expected
        ):
            lo, hi = expected
            ok = isinstance(actual, (int, float)) and not isinstance(actual, bool) and lo <= actual <= hi
            rendered = f"[{lo}, {hi}]"
        else:
            ok = actual == expected
            rendered = repr(expected)
        checks.append({"field": field, "expected": rendered, "actual": actual, "pass": bool(ok)})
    return checks


def run_replay(
    repo: Path,
    *,
    cache_dir: Optional[Path] = None,
    assertions: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Execute every replay assertion against ``repo`` and collect pass/fail."""
    cases = assertions if assertions is not None else REPLAY_ASSERTIONS
    coil_analysis, history_cache, resolved_cache = load_tree(repo, cache_dir)

    results: list[dict[str, Any]] = []
    for case in cases:
        ticker = case["ticker"]
        as_of = case.get("as_of")
        entry: dict[str, Any] = {
            "ticker": ticker,
            "as_of": as_of,
            "why": case.get("why"),
            "expect": case.get("expect", {}),
        }
        try:
            payload = history_cache.read_cache(ticker)
            if not payload or not payload.get("bars"):
                raise ValueError(f"no cached bars for {ticker}")
            bars = payload["bars"]
            result = coil_analysis.analyze_coil(
                bars, config=coil_analysis.DEFAULT_CONFIG, as_of=as_of
            )
            record = build_record(ticker, result, last_close=last_close_at(bars, as_of))
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            record = error_record(ticker, exc)
        entry["record"] = record
        entry["checks"] = evaluate_expectation(record, case.get("expect", {}))
        entry["pass"] = record["ok"] and all(c["pass"] for c in entry["checks"])
        results.append(entry)

    return {
        "repo": str(Path(repo).resolve()),
        "cache": str(resolved_cache),
        "total": len(results),
        "passed": sum(1 for r in results if r["pass"]),
        "failed": sum(1 for r in results if not r["pass"]),
        "cases": results,
    }


def render_replay(report: dict[str, Any]) -> str:
    """Human-readable replay result."""
    out = [f"Replay against {report['repo']}", ""]
    for case in report["cases"]:
        flag = "PASS" if case["pass"] else "FAIL"
        out.append(f"[{flag}] {case['ticker']} as_of={case['as_of']}")
        for check in case["checks"]:
            mark = "ok " if check["pass"] else "BAD"
            out.append(f"    {mark} {check['field']}: expected {check['expected']}, got {check['actual']!r}")
        if case["record"].get("error"):
            out.append(f"    error: {case['record']['error']}")
        if not case["pass"] and case.get("why"):
            out.append(f"    why it matters: {case['why']}")
    out.append("")
    out.append(f"{report['passed']}/{report['total']} passed, {report['failed']} failed")
    return "\n".join(out)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _read_snapshot(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        snapshot = json.load(fh)
    snapshot["_path"] = str(path)
    return snapshot


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lid_diff_harness",
        description="Measure lid-detection behaviour across two repo trees.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    run_p = sub.add_parser("run", help="analyze every cached ticker; write a snapshot")
    run_p.add_argument("--repo", required=True, type=Path, help="repo tree to import coil_analysis from")
    run_p.add_argument("--cache", type=Path, default=None, help="cache dir override (default <repo>/cache)")
    run_p.add_argument("--as-of", dest="as_of", default=None, help="YYYY-MM-DD replay cutoff")
    run_p.add_argument("--out", required=True, type=Path, help="snapshot JSON output path")
    run_p.add_argument("--ticker", action="append", default=None, help="limit to these tickers (repeatable)")
    run_p.add_argument("--progress", action="store_true", help="print tickers to stderr as they finish")

    diff_p = sub.add_parser("diff", help="compare two snapshots")
    diff_p.add_argument("--before", required=True, type=Path)
    diff_p.add_argument("--after", required=True, type=Path)
    diff_p.add_argument("--report", type=Path, default=None, help="markdown output path (default stdout)")
    diff_p.add_argument("--json", dest="json_out", type=Path, default=None, help="machine-readable diff JSON")

    replay_p = sub.add_parser("replay", help="run the (ticker, as_of) regression assertions")
    replay_p.add_argument("--repo", required=True, type=Path)
    replay_p.add_argument("--cache", type=Path, default=None)
    replay_p.add_argument("--json", dest="json_out", type=Path, default=None)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.mode == "run":
        snapshot = run_snapshot(
            args.repo,
            cache_dir=args.cache,
            as_of=args.as_of,
            tickers=args.ticker,
            progress=args.progress,
        )
        _write_json(args.out, snapshot)
        records = snapshot["records"].values()
        print(
            f"{snapshot['ticker_count']} tickers | "
            f"ok={sum(1 for r in records if r['ok'])} "
            f"errors={sum(1 for r in records if not r['ok'])} | "
            f"graded={sum(1 for r in records if r['grade'] is not None)} | "
            f"structure={sum(1 for r in records if r['has_structure'])} | "
            f"provisional_emitted={sum(1 for r in records if r['provisional_top_count'])} "
            f"provisional_as_anchor={sum(1 for r in records if r['uses_provisional_anchor'])} | "
            f"versions={snapshot['algorithm_versions']} -> {args.out}"
        )
        return 0

    if args.mode == "diff":
        diff = build_diff(_read_snapshot(args.before), _read_snapshot(args.after))
        report = render_report(diff)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report, encoding="utf-8")
            print(f"report -> {args.report}")
        else:
            print(report)
        if args.json_out:
            _write_json(args.json_out, diff)
            print(f"diff json -> {args.json_out}")
        return 0

    report = run_replay(args.repo, cache_dir=args.cache)
    print(render_replay(report))
    if args.json_out:
        _write_json(args.json_out, report)
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

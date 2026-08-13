#!/usr/bin/env python3
"""Create one protected v2.4 benchmark session and print its capability link."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_v24 import BENCHMARK_ID, BenchmarkError  # noqa: E402
from review_snapshots import REVIEW_CORPUS_MANIFEST_KIND  # noqa: E402

BATCHES = {
    "a": {
        "source": "benchmark_2026-08-13_v24_72_batch_a.csv",
        "partition": "development",
    },
    "b": {
        "source": "benchmark_2026-08-13_v24_72_batch_b.csv",
        "partition": "development_repeat",
    },
    "c": {
        "source": "benchmark_2026-08-13_v24_72_batch_c.csv",
        "partition": "validation",
    },
    "d": {
        "source": "benchmark_2026-08-13_v24_72_batch_d.csv",
        "partition": "holdout",
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"benchmark batch manifest is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise BenchmarkError("benchmark batch manifest must be a JSON object")
    return value


def load_batch(root: Path, batch: str) -> tuple[str, list[str]]:
    """Load an exact opaque batch queue from its frozen workbench manifest."""
    definition = BATCHES[batch]
    source = definition["source"]
    path = root / Path(source).stem / "manifest.json"
    manifest = _read_json(path)
    if manifest.get("schema_version") != 1:
        raise BenchmarkError("unsupported benchmark workbench manifest schema")
    if manifest.get("kind") != REVIEW_CORPUS_MANIFEST_KIND:
        raise BenchmarkError("benchmark batch is not a loadable workbench manifest")
    if manifest.get("benchmark_id") != BENCHMARK_ID:
        raise BenchmarkError("benchmark batch id mismatch")
    if manifest.get("benchmark_partition") != definition["partition"]:
        raise BenchmarkError("benchmark batch partition mismatch")
    if (manifest.get("source_run") or {}).get("filename") != source:
        raise BenchmarkError("benchmark batch source mismatch")

    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise BenchmarkError("benchmark batch has no workbench items")
    if not all(isinstance(item, dict) for item in items):
        raise BenchmarkError("benchmark batch contains an invalid workbench item")
    item_order = [
        str(item.get("ticker") or "").strip().upper()
        for item in items
    ]
    explicit_order = manifest.get("ordered_universe")
    ordered = (
        [str(value).strip().upper() for value in explicit_order]
        if isinstance(explicit_order, list) and explicit_order
        else item_order
    )
    if (
        not ordered
        or any(not ticker for ticker in ordered)
        or ordered != item_order
        or len(set(ordered)) != len(ordered)
    ):
        raise BenchmarkError("benchmark batch ordered universe is invalid")
    return source, ordered


def _validated_base_url(value: str, *, label: str) -> str:
    parsed = urlsplit(value.strip())
    is_local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if not parsed.netloc or parsed.scheme not in {"http", "https"}:
        raise BenchmarkError(f"{label} must be an absolute HTTP(S) URL")
    if parsed.scheme != "https" and not is_local:
        raise BenchmarkError(f"{label} must use HTTPS outside localhost")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise BenchmarkError(f"{label} must not contain credentials, query, or fragment")
    return value.strip().rstrip("/")


def review_link(app_base_url: str, session_id: int, token: str) -> str:
    """Put the capability in the fragment so it is never sent in HTTP requests."""
    base = _validated_base_url(app_base_url, label="app base URL")
    parsed = urlsplit(base)
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key not in {"view", "session", "run", "ticker"}
    ]
    query.extend((("view", "review"), ("session", str(session_id))))
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path or "/",
            urlencode(query),
            urlencode({"review_token": token}),
        )
    )


def create_session(
    *,
    api_base_url: str,
    source: str,
    tickers: list[str],
    reviewer: str,
    capability_token: str,
    admin_key: str | None,
    timeout: float = 30.0,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    """POST one exact protected queue to the existing review-session API."""
    base = _validated_base_url(api_base_url, label="API base URL")
    if len(capability_token) < 32 or capability_token.strip() != capability_token:
        raise BenchmarkError("review capability token must contain at least 32 characters")
    if len(reviewer.strip()) < 2:
        raise BenchmarkError("reviewer name is required")
    payload = {
        "source": source,
        "reviewerName": reviewer.strip(),
        "accessToken": capability_token,
        "requireFreshReview": True,
        "items": [{"ticker": ticker} for ticker in tickers],
        "snapshot": {
            "purpose": "lean v2.4 blind benchmark",
            "benchmark_id": BENCHMARK_ID,
        },
    }
    headers = {"Content-Type": "application/json"}
    if admin_key:
        headers["X-Review-Admin-Key"] = admin_key
    request = Request(
        f"{base}/api/review-sessions",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with opener(request, timeout=timeout) as response:
            body = json.loads(response.read())
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read()).get("detail")
        except (AttributeError, json.JSONDecodeError):
            detail = None
        raise BenchmarkError(
            f"review-session API rejected the batch ({exc.code}): {detail or exc.reason}"
        ) from exc
    except (URLError, OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"review-session API request failed: {exc}") from exc
    session = body.get("session") if isinstance(body, dict) else None
    if not isinstance(session, dict) or not isinstance(session.get("id"), int):
        raise BenchmarkError("review-session API returned an invalid session")
    if session.get("source") != source:
        raise BenchmarkError("review-session API returned the wrong source")
    returned_tickers = [
        str(item.get("ticker") or "").strip().upper()
        for item in session.get("items") or []
        if isinstance(item, dict)
    ]
    if returned_tickers != tickers:
        raise BenchmarkError("review-session API returned the wrong queue")
    if session.get("require_fresh_review") is not True:
        raise BenchmarkError("review-session API did not protect the benchmark queue")
    if session.get("reviewer_name") != reviewer.strip():
        raise BenchmarkError("review-session API returned the wrong reviewer")
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", choices=tuple(BATCHES), required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "review_snapshots",
        help="Directory containing the opaque batch workbench folders.",
    )
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("COILINGVIEW_API_BASE_URL", "http://localhost:8000"),
    )
    parser.add_argument(
        "--app-base-url",
        default=os.environ.get("COILINGVIEW_APP_BASE_URL", "http://localhost:5173"),
    )
    parser.add_argument("--reviewer", default="Amrut")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    token_env = f"COILINGVIEW_V24_BATCH_{args.batch.upper()}_TOKEN"
    token = os.environ.get(token_env, "")
    if not token:
        raise SystemExit(
            f"set {token_env} to a stable random capability token before creating this batch"
        )
    source, tickers = load_batch(args.root.resolve(), args.batch)
    result = create_session(
        api_base_url=args.api_base_url,
        source=source,
        tickers=tickers,
        reviewer=args.reviewer,
        capability_token=token,
        admin_key=os.environ.get("REVIEW_SESSION_CREATE_KEY") or None,
        timeout=args.timeout,
    )
    session = result["session"]
    output = {
        "batch": args.batch,
        "created": bool(result.get("created")),
        "session_id": session["id"],
        "source": source,
        "review_url": review_link(args.app_base_url, session["id"], token),
        "token_environment_variable": token_env,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as exc:
        raise SystemExit(str(exc)) from exc

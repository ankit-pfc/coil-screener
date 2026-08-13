from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from benchmark_v24 import BENCHMARK_ID, BenchmarkError
from review_snapshots import REVIEW_CORPUS_MANIFEST_KIND
from scripts.create_v24_review_session import (
    _RejectRedirects,
    create_session,
    load_batch,
    review_link,
)


class _Response:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.value).encode("utf-8")


def _write_batch(
    root: Path, *, ordered: list[str] | None = None, include_order: bool = True
) -> Path:
    source = "benchmark_2026-08-13_v24_72_batch_a.csv"
    folder = root / Path(source).stem
    folder.mkdir(exist_ok=True)
    tickers = ["AAA", "BBB"]
    manifest = {
        "schema_version": 1,
        "kind": REVIEW_CORPUS_MANIFEST_KIND,
        "benchmark_id": BENCHMARK_ID,
        "benchmark_partition": "development",
        "source_run": {"filename": source},
        "items": [{"ticker": ticker} for ticker in tickers],
    }
    if include_order:
        manifest["ordered_universe"] = ordered if ordered is not None else tickers
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


def test_load_batch_requires_exact_manifest_order(tmp_path):
    _write_batch(tmp_path)
    assert load_batch(tmp_path, "a") == (
        "benchmark_2026-08-13_v24_72_batch_a.csv",
        ["AAA", "BBB"],
    )

    _write_batch(tmp_path, include_order=False)
    assert load_batch(tmp_path, "a")[1] == ["AAA", "BBB"]

    _write_batch(tmp_path, ordered=["BBB", "AAA"])
    with pytest.raises(BenchmarkError, match="ordered universe"):
        load_batch(tmp_path, "a")


def test_create_session_posts_a_protected_exact_queue_without_logging_secrets():
    captured = {}

    def opener(request, *, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return _Response(
            {
                "created": True,
                "session": {
                    "id": 41,
                    "source": "benchmark_2026-08-13_v24_72_batch_a.csv",
                    "reviewer_name": "Amrut",
                    "require_fresh_review": True,
                    "items": [{"ticker": "AAA"}, {"ticker": "BBB"}],
                },
            }
        )

    result = create_session(
        api_base_url="https://review-api.example.com",
        source="benchmark_2026-08-13_v24_72_batch_a.csv",
        tickers=["AAA", "BBB"],
        reviewer="Amrut",
        capability_token="stable-secret-capability-token-0001",
        admin_key="server-admin-key",
        opener=opener,
    )

    assert result["session"]["id"] == 41
    assert captured["url"] == "https://review-api.example.com/api/review-sessions"
    assert captured["headers"]["X-review-admin-key"] == "server-admin-key"
    assert captured["payload"]["requireFreshReview"] is True
    assert captured["payload"]["accessToken"] == "stable-secret-capability-token-0001"
    assert captured["payload"]["items"] == [{"ticker": "AAA"}, {"ticker": "BBB"}]


def test_create_session_rejects_a_server_response_with_the_wrong_queue():
    def opener(_request, *, timeout):
        assert timeout == 30.0
        return _Response(
            {
                "created": True,
                "session": {
                    "id": 41,
                    "source": "benchmark_2026-08-13_v24_72_batch_a.csv",
                    "reviewer_name": "Amrut",
                    "require_fresh_review": True,
                    "items": [{"ticker": "BBB"}, {"ticker": "AAA"}],
                },
            }
        )

    with pytest.raises(BenchmarkError, match="wrong queue"):
        create_session(
            api_base_url="https://review-api.example.com",
            source="benchmark_2026-08-13_v24_72_batch_a.csv",
            tickers=["AAA", "BBB"],
            reviewer="Amrut",
            capability_token="stable-secret-capability-token-0001",
            admin_key=None,
            opener=opener,
        )


def test_review_link_keeps_capability_out_of_the_http_request():
    link = review_link(
        "https://review.example.com/workbench",
        41,
        "stable-secret-capability-token-0001",
    )
    parsed = urlsplit(link)
    assert parse_qs(parsed.query) == {"view": ["review"], "session": ["41"]}
    assert parse_qs(parsed.fragment) == {
        "review_token": ["stable-secret-capability-token-0001"]
    }
    assert "stable-secret" not in parsed.query


def test_remote_session_creation_requires_https():
    with pytest.raises(BenchmarkError, match="HTTPS"):
        create_session(
            api_base_url="http://review.example.com",
            source="benchmark_2026-08-13_v24_72_batch_a.csv",
            tickers=["AAA"],
            reviewer="Amrut",
            capability_token="stable-secret-capability-token-0001",
            admin_key=None,
            opener=lambda *_args, **_kwargs: None,
        )


def test_session_creation_refuses_all_http_redirects():
    handler = _RejectRedirects()
    assert (
        handler.redirect_request(
            None,
            None,
            302,
            "Found",
            {},
            "https://other.example.com/capture-admin-secret",
        )
        is None
    )

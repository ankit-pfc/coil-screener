from screen_monthly import INTERNATIONAL_REVIEW_TICKERS, build_ticker_list


def test_international_review_universe_is_large_and_diversified():
    tickers = build_ticker_list(universe="international")

    assert tickers == INTERNATIONAL_REVIEW_TICKERS
    assert len(tickers) >= 50
    assert len({ticker.rsplit(".", 1)[-1] for ticker in tickers}) >= 12


def test_international_limit_applies_before_explicit_tickers():
    tickers = build_ticker_list(
        explicit_tickers=["AAPL", "SHOP.TO"],
        universe="international",
        limit=3,
    )

    assert tickers == INTERNATIONAL_REVIEW_TICKERS[:3] + ["AAPL"]

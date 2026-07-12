from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode


@dataclass(frozen=True)
class CaptureConfig:
    base_url: str
    ticker: str
    interval: str = "3M"
    timeframe: str = "10Y"
    chart_type: str = "candles"
    timeout_ms: int = 60_000
    viewport_width: int = 1440
    viewport_height: int = 960
    headless: bool = True


@dataclass(frozen=True)
class ChartCapture:
    ticker: str
    url: str
    image_path: Path
    state: dict[str, Any]


def build_capture_url(config: CaptureConfig) -> str:
    params = urlencode(
        {
            "capture": "1",
            "ticker": config.ticker.upper(),
            "interval": config.interval,
            "timeframe": config.timeframe,
            "chartType": config.chart_type,
        }
    )
    return f"{config.base_url.rstrip('/')}/?{params}"


async def capture_chart_async(config: CaptureConfig, image_path: Path) -> ChartCapture:
    try:
        from playwright.async_api import async_playwright  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is required for chart capture. Install backend requirements "
            "and run `python -m playwright install chromium`."
        ) from exc

    image_path.parent.mkdir(parents=True, exist_ok=True)
    url = build_capture_url(config)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=config.headless)
        page = await browser.new_page(
            viewport={
                "width": config.viewport_width,
                "height": config.viewport_height,
            },
            device_scale_factor=1,
        )
        try:
            # Vite dev keeps a websocket open forever, so `networkidle` can
            # hang. The capture-state wait below is the authoritative chart
            # readiness gate.
            await page.goto(url, wait_until="domcontentloaded", timeout=config.timeout_ms)
            await page.wait_for_function(
                """() => (
                    window.__CV_CAPTURE_STATE__ &&
                    window.__CV_CAPTURE_STATE__.ready === true
                ) || (
                    window.__COILINGVIEW_CHART_META__ &&
                    window.__COILINGVIEW_CHART_META__.ready === true
                )""",
                timeout=config.timeout_ms,
            )
            state = await page.evaluate(
                """() => window.__CV_CAPTURE_STATE__ || window.__COILINGVIEW_CHART_META__"""
            )
            chart_meta = await page.evaluate(
                """() => window.__COILINGVIEW_CHART_META__ || null"""
            )
            chart_selector = (
                (chart_meta or {}).get("chart_selector")
                or (state or {}).get("chart_selector")
                or '[data-cv-capture-target="chart-plot"]'
            )
            target = page.locator(chart_selector).first
            await target.screenshot(path=str(image_path))
            if chart_meta:
                state = {**state, "chart_meta": chart_meta}
        finally:
            await browser.close()
    return ChartCapture(
        ticker=config.ticker.upper(),
        url=url,
        image_path=image_path,
        state=state,
    )


def capture_chart(config: CaptureConfig, image_path: Path) -> ChartCapture:
    return asyncio.run(capture_chart_async(config, image_path))

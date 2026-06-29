"""Vision-assisted chart tagging pipeline for CoilingView.

The package is intentionally import-light: Playwright, supervision, and image
annotation libraries are imported only inside the functions that need them so
the core FastAPI app and tests can start without the optional vision stack.
"""

from .mapping import map_detections_to_chart_points, map_detections_to_highs
from .storage import VisionRunStore
from .trendlines import suggest_resistance_trendlines

__all__ = [
    "VisionRunStore",
    "map_detections_to_chart_points",
    "map_detections_to_highs",
    "suggest_resistance_trendlines",
]

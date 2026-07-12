from __future__ import annotations

import base64
import os
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


SERVERLESS_API_URL = "https://serverless.roboflow.com"


class VisionInferenceError(RuntimeError):
    pass


def _model_parts(model_id: str) -> tuple[str, str]:
    parts = [part for part in model_id.strip("/").split("/") if part]
    if len(parts) < 2:
        raise VisionInferenceError(
            "ROBOFLOW_MODEL_ID must look like 'project-slug/version' for REST inference."
        )
    if len(parts) == 2:
        return parts[0], parts[1]
    return parts[0], "/".join(parts[1:])


class RoboflowHostedClient:
    """Roboflow hosted inference client.

    Uses the official ``inference_sdk`` when available, falling back to the
    documented serverless REST endpoint with a base64 image payload.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str | None = None,
        *,
        api_url: str = SERVERLESS_API_URL,
        confidence: float = 0.35,
    ):
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        self.model_id = model_id or os.getenv("ROBOFLOW_MODEL_ID")
        self.api_url = api_url.rstrip("/")
        self.confidence = confidence
        if not self.api_key:
            raise VisionInferenceError("ROBOFLOW_API_KEY is required for vision inference.")
        if not self.model_id:
            raise VisionInferenceError("ROBOFLOW_MODEL_ID is required for vision inference.")

    def infer(self, image_path: Path) -> dict[str, Any]:
        try:
            from inference_sdk import InferenceHTTPClient  # type: ignore

            client = InferenceHTTPClient(api_url=self.api_url, api_key=self.api_key)
            return client.infer(str(image_path), model_id=self.model_id)
        except ImportError:
            return self._infer_rest(image_path)

    def _infer_rest(self, image_path: Path) -> dict[str, Any]:
        dataset_id, version_id = _model_parts(self.model_id or "")
        with image_path.open("rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("ascii")
        response = requests.post(
            f"{self.api_url}/{quote(dataset_id, safe='')}/{quote(version_id, safe='')}",
            params={
                "api_key": self.api_key,
                "confidence": self.confidence,
                "format": "json",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=encoded,
            timeout=120,
        )
        if not response.ok:
            raise VisionInferenceError(
                f"Roboflow inference failed: {response.status_code} {response.text[:200]}"
            )
        return response.json()


def normalize_detections(raw_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert Roboflow output to a JSON-friendly detection list.

    ``supervision`` is used when installed; the fallback handles Roboflow's
    standard object-detection JSON shape directly.
    """

    try:
        import supervision as sv  # type: ignore

        detections = sv.Detections.from_inference(raw_result)
        class_names = detections.data.get("class_name") if detections.data else None
        rows: list[dict[str, Any]] = []
        for i, xyxy in enumerate(detections.xyxy):
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            class_name = ""
            if class_names is not None and i < len(class_names):
                class_name = str(class_names[i])
            rows.append(
                {
                    "x": (x1 + x2) / 2,
                    "y": (y1 + y2) / 2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "x_min": x1,
                    "y_min": y1,
                    "x_max": x2,
                    "y_max": y2,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(detections.confidence[i])
                    if detections.confidence is not None
                    else 0.0,
                    "class_id": int(detections.class_id[i])
                    if detections.class_id is not None
                    else None,
                    "class_name": class_name,
                }
            )
        return rows
    except Exception:
        predictions = raw_result.get("predictions", [])
        rows = []
        for prediction in predictions if isinstance(predictions, list) else []:
            if not isinstance(prediction, dict):
                continue
            x = float(prediction.get("x", 0.0) or 0.0)
            y = float(prediction.get("y", 0.0) or 0.0)
            width = float(prediction.get("width", 0.0) or 0.0)
            height = float(prediction.get("height", 0.0) or 0.0)
            x_min = x - width / 2
            y_min = y - height / 2
            x_max = x + width / 2
            y_max = y + height / 2
            rows.append(
                {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "bbox": [x_min, y_min, x_max, y_max],
                    "confidence": float(prediction.get("confidence", 0.0) or 0.0),
                    "class_id": prediction.get("class_id"),
                    "class_name": str(
                        prediction.get("class")
                        or prediction.get("class_name")
                        or prediction.get("label")
                        or ""
                    ),
                    "raw": prediction,
                }
            )
        return rows


def write_debug_annotation(
    image_path: Path,
    detections: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not detections:
        shutil.copyfile(image_path, output_path)
        return output_path

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        import supervision as sv  # type: ignore

        image = cv2.imread(str(image_path))
        if image is None:
            raise VisionInferenceError("OpenCV could not read captured image.")
        xyxy = np.array([d["bbox"] for d in detections], dtype=float)
        confidence = np.array([d.get("confidence", 0.0) for d in detections], dtype=float)
        class_names = [str(d.get("class_name") or "vision") for d in detections]
        sv_detections = sv.Detections(xyxy=xyxy, confidence=confidence)
        annotated = sv.BoxAnnotator().annotate(scene=image.copy(), detections=sv_detections)
        annotated = sv.LabelAnnotator().annotate(
            scene=annotated,
            detections=sv_detections,
            labels=[
                f"{name} {conf:.2f}"
                for name, conf in zip(class_names, confidence, strict=False)
            ],
        )
        cv2.imwrite(str(output_path), annotated)
    except Exception:
        shutil.copyfile(image_path, output_path)
    return output_path

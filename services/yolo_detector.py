"""
YOLODetector – Subject detection using YOLOv8 (ultralytics).

Returns the largest detected bounding box as the primary subject along
with positional and size metadata.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("framesense.yolo")


class YOLODetector:
    """Loads YOLOv8n once and reuses for frame inference."""

    # Confidence threshold; lower = more detections but noisier
    CONF_THRESHOLD = 0.35
    # Size buckets relative to image area
    SIZE_SMALL = 0.05
    SIZE_LARGE = 0.25

    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        try:
            from ultralytics import YOLO  # import here so startup is fast

            self._model = YOLO(model_name)
            self._model.fuse()  # fuse conv+bn layers for speed
            logger.info("YOLOv8 model loaded: %s", model_name)
        except Exception as exc:
            logger.error("Failed to load YOLO: %s", exc)
            self._model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run YOLOv8 on *frame* (BGR, H×W×3) and return subject info.

        Returns
        -------
        {
            "detected": bool,
            "subject_position": "left" | "center" | "right",
            "subject_size":     "small" | "medium" | "large",
            "bbox":             [x1, y1, x2, y2] | None,
            "confidence":       float | None,
            "label":            str | None,
        }
        """
        if self._model is None:
            return self._no_detection()

        h, w = frame.shape[:2]

        try:
            results = self._model.predict(
                source=frame,
                conf=self.CONF_THRESHOLD,
                verbose=False,
                device="cpu",   # stay on CPU for portability; swap to 0 for GPU
            )
        except Exception as exc:
            logger.warning("YOLO inference error: %s", exc)
            return self._no_detection()

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return self._no_detection()

        # Pick the largest box by area
        best_box = self._largest_box(boxes)
        if best_box is None:
            return self._no_detection()

        xyxy = best_box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        conf = float(best_box.conf[0].cpu())
        cls_id = int(best_box.cls[0].cpu())
        label = self._model.names.get(cls_id, "object")

        cx = (x1 + x2) / 2
        box_area = (x2 - x1) * (y2 - y1)
        img_area = h * w

        position = self._classify_position(cx, w)
        size = self._classify_size(box_area, img_area)

        return {
            "detected": True,
            "subject_position": position,
            "subject_size": size,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": round(conf, 3),
            "label": label,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _no_detection() -> dict:
        return {
            "detected": False,
            "subject_position": "center",
            "subject_size": "medium",
            "bbox": None,
            "confidence": None,
            "label": None,
        }

    @staticmethod
    def _largest_box(boxes):
        best, best_area = None, 0.0
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best = box
        return best

    @staticmethod
    def _classify_position(cx: float, width: int) -> str:
        third = width / 3
        if cx < third:
            return "left"
        if cx > 2 * third:
            return "right"
        return "center"

    @staticmethod
    def _classify_size(box_area: float, img_area: int) -> str:
        ratio = box_area / max(img_area, 1)
        if ratio < YOLODetector.SIZE_SMALL:
            return "small"
        if ratio > YOLODetector.SIZE_LARGE:
            return "large"
        return "medium"

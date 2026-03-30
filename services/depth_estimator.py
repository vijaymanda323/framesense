"""
DepthEstimator – per-frame relative depth using MiDaS (MiDaS_small).

MiDaS produces an *inverse* depth map (higher value = closer).  We sample
the region around the detected subject's bounding box (or the centre of the
frame if no subject was found) and bucket the result into close / medium / far.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("framesense.depth")

# Relative thresholds applied to the 0-1 normalised depth value
_CLOSE_THRESHOLD = 0.65  # norm_depth > this → "close"
_FAR_THRESHOLD = 0.35    # norm_depth < this → "far"


class DepthEstimator:
    """Singleton wrapper around MiDaS_small (cpu-friendly, ~60 ms / frame)."""

    _INPUT_SIZE = 256  # MiDaS_small internal resolution

    def __init__(self) -> None:
        self._model = None
        self._transform = None
        self._device = torch.device("cpu")
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(
        self,
        frame: np.ndarray,
        bbox: list[int] | None = None,
    ) -> dict:
        """
        Parameters
        ----------
        frame : np.ndarray  BGR image, H×W×3
        bbox  : [x1,y1,x2,y2] subject bounding box or None

        Returns
        -------
        {"distance": "close" | "medium" | "far", "depth_score": float}
        """
        if self._model is None:
            return {"distance": "medium", "depth_score": 0.5}

        try:
            depth_map = self._run_midas(frame)           # H×W float32
            score = self._sample_depth(depth_map, bbox, frame.shape)
            distance = self._classify_distance(score)
            return {"distance": distance, "depth_score": round(float(score), 3)}
        except Exception as exc:
            logger.warning("Depth estimation error: %s", exc)
            return {"distance": "medium", "depth_score": 0.5}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            self._model = torch.hub.load(
                "intel-isl/MiDaS",
                "MiDaS_small",
                trust_repo=True,
            )
            self._model.to(self._device).eval()

            transforms_hub = torch.hub.load(
                "intel-isl/MiDaS",
                "transforms",
                trust_repo=True,
            )
            self._transform = transforms_hub.small_transform
            logger.info("MiDaS_small loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load MiDaS: %s", exc)
            self._model = None

    def _run_midas(self, frame: np.ndarray) -> np.ndarray:
        """Return a normalised (0-1) depth map, same spatial size as *frame*."""
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self._transform(rgb).to(self._device)

        with torch.no_grad():
            prediction = self._model(input_batch)
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()

        # Normalise to 0-1
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.ones_like(depth) * 0.5

        return depth.astype(np.float32)

    @staticmethod
    def _sample_depth(depth_map: np.ndarray, bbox, shape) -> float:
        """Average depth in the subject bbox, or a centre crop if no bbox."""
        h, w = shape[:2]
        if bbox:
            x1, y1, x2, y2 = bbox
            # Clamp to frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                return float(depth_map[y1:y2, x1:x2].mean())
        # Fallback: centre 30% of frame
        ch, cw = int(h * 0.35), int(w * 0.35)
        cy, cx = h // 2, w // 2
        return float(depth_map[cy - ch // 2: cy + ch // 2,
                                cx - cw // 2: cx + cw // 2].mean())

    @staticmethod
    def _classify_distance(score: float) -> str:
        if score >= _CLOSE_THRESHOLD:
            return "close"
        if score <= _FAR_THRESHOLD:
            return "far"
        return "medium"

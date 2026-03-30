"""
SaliencyDetector – Rule-of-Thirds composition scoring via OpenCV saliency.

Uses SpectralResidual saliency (part of opencv-contrib) to find the most
visually salient region, then checks proximity to rule-of-thirds intersection
points and returns a 0-100 composition score contribution.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("framesense.saliency")

# Rule-of-thirds intersection offsets (relative)
_ROT_OFFSETS = [1 / 3, 2 / 3]


class SaliencyDetector:
    """Stateless saliency + RoT composition scorer (no ML model to load)."""

    # Maximum normalised distance at which we consider a subject "on" a RoT point
    _ROT_HIT_RADIUS = 0.18

    def __init__(self) -> None:
        # SpectralResidual is always available in opencv-contrib
        self._saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

    def score(
        self,
        frame: np.ndarray,
        subject_bbox: Optional[list[int]] = None,
    ) -> dict:
        """
        Parameters
        ----------
        frame         : BGR image H×W×3
        subject_bbox  : [x1,y1,x2,y2] from YOLO, or None

        Returns
        -------
        {
            "composition_score": int (0-100),
            "salient_point":     [cx, cy] normalised 0-1,
            "rot_aligned":       bool,
        }
        """
        h, w = frame.shape[:2]

        salient_cx, salient_cy = self._get_salient_point(frame, h, w)

        # If YOLO gave us a bbox, prefer its centre as the "subject point"
        if subject_bbox:
            x1, y1, x2, y2 = subject_bbox
            sub_cx = ((x1 + x2) / 2) / w
            sub_cy = ((y1 + y2) / 2) / h
        else:
            sub_cx, sub_cy = salient_cx, salient_cy

        rot_aligned, min_dist = self._check_rot(sub_cx, sub_cy)
        comp_score = self._compute_score(min_dist)

        return {
            "composition_score": comp_score,
            "salient_point": [round(salient_cx, 3), round(salient_cy, 3)],
            "rot_aligned": rot_aligned,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_salient_point(self, frame: np.ndarray, h: int, w: int):
        """Return (cx_norm, cy_norm) of the most salient region."""
        try:
            success, saliency_map = self._saliency.computeSaliency(frame)
            if not success:
                raise RuntimeError("saliency computation failed")
            # saliency_map is float32 in [0, 1]
            sm = (saliency_map * 255).astype(np.uint8)
            # find the centroid of the top-10% salient pixels
            threshold = np.percentile(sm, 90)
            hot_mask = (sm >= threshold).astype(np.uint8)
            moments = cv2.moments(hot_mask)
            if moments["m00"] > 0:
                cx = moments["m10"] / moments["m00"] / w
                cy = moments["m01"] / moments["m00"] / h
            else:
                cx, cy = 0.5, 0.5
        except Exception as exc:
            logger.debug("Saliency fallback: %s", exc)
            cx, cy = 0.5, 0.5
        return cx, cy

    @staticmethod
    def _rot_points():
        """Generate all 4 rule-of-thirds intersection points."""
        pts = []
        for rx in _ROT_OFFSETS:
            for ry in _ROT_OFFSETS:
                pts.append((rx, ry))
        return pts

    def _check_rot(self, cx: float, cy: float):
        """Return (is_aligned, min_euclidean_distance_to_rot_point)."""
        pts = self._rot_points()
        dists = [((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 for px, py in pts]
        min_dist = min(dists)
        return (min_dist <= self._ROT_HIT_RADIUS, min_dist)

    @staticmethod
    def _compute_score(min_dist: float) -> int:
        """
        Map min_dist (0 = perfect RoT, 0.5+ = far away) to 0-100 score.
        Score is 100 at distance 0, dropping to ~20 at max distance (≈0.47).
        """
        # Normalise over the diagonal of the image (0.0 – ~0.7)
        max_dist = 0.7
        proximity = 1.0 - min(min_dist / max_dist, 1.0)
        score = int(30 + 70 * proximity)  # floor 30, ceiling 100
        return max(0, min(100, score))

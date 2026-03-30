"""
CompositionAnalyzer – Brightness, Focus, Background Clutter, and Alignment analysis.

All methods are stateless OpenCV operations – no ML models required.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger("framesense.composition")


class CompositionAnalyzer:
    """
    Extracts four classical photography composition features using CV techniques:

    1. Brightness / Exposure  – grayscale mean intensity
    2. Focus quality          – Laplacian variance
    3. Background clutter     – Canny edge density
    4. Camera alignment       – Hough Line Transform horizon detection
    """

    # ── Brightness thresholds (0-255 scale) ──────────────────────────────
    _BRIGHTNESS_LOW  = 60
    _BRIGHTNESS_HIGH = 195

    # ── Focus thresholds (Laplacian variance) ────────────────────────────
    _BLUR_THRESHOLD  = 80.0   # below this → "blurry"

    # ── Edge density thresholds (fraction of edge pixels 0-1) ───────────
    _CLUTTER_LOW     = 0.05
    _CLUTTER_HIGH    = 0.15

    # ── Hough alignment ──────────────────────────────────────────────────
    _TILT_ANGLE_DEG  = 8.0   # deviation beyond this → "tilted"

    # ------------------------------------------------------------------
    # Brightness
    # ------------------------------------------------------------------

    def analyze_brightness(self, frame: np.ndarray) -> dict:
        """
        Returns
        -------
        {"brightness": "low" | "good" | "high", "mean_intensity": float}
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        if mean < self._BRIGHTNESS_LOW:
            label = "low"
        elif mean > self._BRIGHTNESS_HIGH:
            label = "high"
        else:
            label = "good"
        return {"brightness": label, "mean_intensity": round(mean, 2)}

    # ------------------------------------------------------------------
    # Focus
    # ------------------------------------------------------------------

    def analyze_focus(self, frame: np.ndarray) -> dict:
        """
        Returns
        -------
        {"focus": "blurry" | "sharp", "laplacian_variance": float}
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        focus = "blurry" if variance < self._BLUR_THRESHOLD else "sharp"
        return {"focus": focus, "laplacian_variance": round(variance, 2)}

    # ------------------------------------------------------------------
    # Background clutter
    # ------------------------------------------------------------------

    def analyze_clutter(self, frame: np.ndarray) -> dict:
        """
        Uses Canny edge detection.  Edge pixel density indicates visual complexity.

        Returns
        -------
        {"background_clutter": "low" | "medium" | "high", "edge_density": float}
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Adaptive Canny: median-based thresholds
        median = float(np.median(gray))
        sigma = 0.33
        lower = max(0, int((1.0 - sigma) * median))
        upper = min(255, int((1.0 + sigma) * median))
        edges = cv2.Canny(gray, lower, upper)
        density = float(np.sum(edges > 0)) / edges.size

        if density < self._CLUTTER_LOW:
            label = "low"
        elif density > self._CLUTTER_HIGH:
            label = "high"
        else:
            label = "medium"

        return {"background_clutter": label, "edge_density": round(density, 4)}

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------

    def analyze_alignment(self, frame: np.ndarray) -> dict:
        """
        Detects near-horizontal lines via Hough transform and computes their
        mean angular deviation from true horizontal.

        Returns
        -------
        {"alignment": "straight" | "tilted", "tilt_angle_deg": float}
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Blur to suppress texture noise before edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)

        if lines is None or len(lines) == 0:
            return {"alignment": "straight", "tilt_angle_deg": 0.0}

        # Collect angles of lines that are roughly horizontal
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle_deg = np.degrees(theta) - 90  # convert to deviation from horizontal
            if abs(angle_deg) < 45:             # only consider near-horizontal lines
                angles.append(angle_deg)

        if not angles:
            return {"alignment": "straight", "tilt_angle_deg": 0.0}

        mean_tilt = float(np.mean(np.abs(angles)))
        label = "tilted" if mean_tilt > self._TILT_ANGLE_DEG else "straight"

        return {"alignment": label, "tilt_angle_deg": round(mean_tilt, 2)}

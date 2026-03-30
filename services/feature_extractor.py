"""
FeatureExtractor – Orchestrates all sub-services into a single pipeline.

Design goals
------------
* Each sub-service is independent; failures degrade gracefully.
* The pipeline runs synchronously in a thread-pool executor (called from
  the async FastAPI handler) to avoid blocking the event loop.
* Total target latency: < 300 ms on CPU for 640×480 frames.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np

from services.composition_analyzer import CompositionAnalyzer
from services.depth_estimator import DepthEstimator
from services.rag import RAGPipeline
from services.saliency_detector import SaliencyDetector
from services.suggestion_engine import SuggestionEngine
from services.yolo_detector import YOLODetector

logger = logging.getLogger("framesense.extractor")

# Target resolution before any processing
_TARGET_W, _TARGET_H = 640, 480


class FeatureExtractor:
    """
    Main orchestrator.  Injected with singleton model services via the
    FastAPI application state so they are only loaded once.
    """

    def __init__(
        self,
        yolo: YOLODetector,
        depth: DepthEstimator,
        rag: RAGPipeline | None = None,
    ) -> None:
        self._yolo = yolo
        self._depth = depth
        self._rag = rag
        self._comp = CompositionAnalyzer()
        self._saliency = SaliencyDetector()
        self._suggestions = SuggestionEngine()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def extract(self, raw_frame: np.ndarray) -> dict:
        """
        Run the complete analysis pipeline on *raw_frame* (BGR) and
        return a structured results dict.

        Parameters
        ----------
        raw_frame : np.ndarray  Any-size BGR image coming from the decoder.

        Returns
        -------
        Full analysis dict ready to be serialised as JSON.
        """
        t0 = time.perf_counter()

        # ── 1. Resize to canonical resolution ──────────────────────────
        frame = self._resize(raw_frame)

        # ── 2. Parallel-ish feature extraction ─────────────────────────
        # (Python GIL means true parallelism isn't possible without
        # multiprocessing, but each call is short enough for < 300 ms)

        # Subject detection (YOLO)
        yolo_result = self._yolo.detect(frame)

        # Classical CV features (all fast, no GPU)
        brightness_result = self._comp.analyze_brightness(frame)
        focus_result = self._comp.analyze_focus(frame)
        clutter_result = self._comp.analyze_clutter(frame)
        alignment_result = self._comp.analyze_alignment(frame)

        # Depth estimation (MiDaS_small, ~60-80 ms)
        depth_result = self._depth.estimate(frame, bbox=yolo_result.get("bbox"))

        # Saliency + rule-of-thirds scoring
        saliency_result = self._saliency.score(frame, subject_bbox=yolo_result.get("bbox"))

        # ── 3. Merge all results into a flat dict ──────────────────────
        merged = {
            # Subject
            "subject_position": yolo_result["subject_position"],
            "subject_size":     yolo_result["subject_size"],
            "subject_label":    yolo_result.get("label"),
            "subject_detected": yolo_result["detected"],

            # Brightness
            "brightness":       brightness_result["brightness"],
            "mean_intensity":   brightness_result["mean_intensity"],

            # Focus
            "focus":            focus_result["focus"],
            "laplacian_variance": focus_result["laplacian_variance"],

            # Clutter
            "background_clutter": clutter_result["background_clutter"],
            "edge_density":     clutter_result["edge_density"],

            # Alignment
            "alignment":        alignment_result["alignment"],
            "tilt_angle_deg":   alignment_result["tilt_angle_deg"],

            # Depth
            "distance":         depth_result["distance"],
            "depth_score":      depth_result["depth_score"],

            # Composition
            "composition_score": saliency_result["composition_score"],
            "salient_point":     saliency_result["salient_point"],
            "rot_aligned":       saliency_result["rot_aligned"],
        }

        # ── 4. Adjust composition score based on penalties ─────────────
        merged["composition_score"] = self._apply_score_penalties(merged)

        # ── 5. Generate suggestions ────────────────────────────────────
        base_suggestions = self._suggestions.generate(merged)
        
        if getattr(self, "_rag", None):
            rag_suggestions = self._rag.generate_suggestion(merged)
            merged["suggestions"] = rag_suggestions + base_suggestions
        else:
            merged["suggestions"] = base_suggestions

        elapsed_ms = (time.perf_counter() - t0) * 1000
        merged["processing_time_ms"] = round(elapsed_ms, 1)

        logger.debug("Feature extraction completed in %.1f ms", elapsed_ms)
        return merged

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _resize(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w == _TARGET_W and h == _TARGET_H:
            return frame
        return cv2.resize(frame, (_TARGET_W, _TARGET_H), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _apply_score_penalties(f: dict) -> int:
        """
        Reduce composition score based on detected quality issues.

        Penalties are additive and capped so the final score stays in [0, 100].
        """
        score = f.get("composition_score", 70)

        # Penalty table: (condition, deduction)
        penalties = [
            (f.get("brightness") == "low",              10),
            (f.get("brightness") == "high",              6),
            (f.get("background_clutter") == "high",     12),
            (f.get("background_clutter") == "medium",    4),
            (f.get("alignment") == "tilted",            10),
            (f.get("distance") == "far",                 8),
            (f.get("focus") == "blurry",                14),
        ]

        for condition, penalty in penalties:
            if condition:
                score -= penalty

        return max(0, min(100, score))

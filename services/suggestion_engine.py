"""
SuggestionEngine – Generates human-readable photography suggestions
based on the feature extraction results.

Suggestions are prioritised and de-duplicated so the response stays concise.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("framesense.suggestions")

# Maximum number of suggestions to return per frame
_MAX_SUGGESTIONS = 5


class SuggestionEngine:
    """
    Stateless rule engine.  Each rule checks one extracted feature and
    appends an actionable suggestion when the feature is suboptimal.
    """

    def generate(self, features: dict) -> list[str]:
        """
        Parameters
        ----------
        features : dict – the merged output of all feature extractors

        Returns
        -------
        List of suggestion strings (max _MAX_SUGGESTIONS).
        """
        suggestions: list[str] = []

        self._check_brightness(features, suggestions)
        self._check_clutter(features, suggestions)
        self._check_alignment(features, suggestions)
        self._check_distance(features, suggestions)
        self._check_focus(features, suggestions)
        self._check_position(features, suggestions)
        self._check_composition(features, suggestions)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique.append(s)

        return unique[:_MAX_SUGGESTIONS]

    # ------------------------------------------------------------------
    # Individual rules
    # ------------------------------------------------------------------

    @staticmethod
    def _check_brightness(features: dict, out: list) -> None:
        b = features.get("brightness")
        if b == "low":
            out.append("Increase exposure or move to better lighting")
        elif b == "high":
            out.append("Reduce exposure or avoid strong backlighting")

    @staticmethod
    def _check_clutter(features: dict, out: list) -> None:
        if features.get("background_clutter") == "high":
            out.append("Simplify the background for a cleaner composition")

    @staticmethod
    def _check_alignment(features: dict, out: list) -> None:
        if features.get("alignment") == "tilted":
            tilt = features.get("tilt_angle_deg", 0)
            out.append(f"Straighten the camera (tilt detected: ~{tilt:.0f}°)")

    @staticmethod
    def _check_distance(features: dict, out: list) -> None:
        d = features.get("distance")
        if d == "far":
            out.append("Move closer to your subject")
        elif d == "close":
            out.append("Step back slightly for better framing")

    @staticmethod
    def _check_focus(features: dict, out: list) -> None:
        if features.get("focus") == "blurry":
            out.append("Hold the camera steady or tap to focus")

    @staticmethod
    def _check_position(features: dict, out: list) -> None:
        pos = features.get("subject_position")
        if pos == "left":
            out.append("Move camera slightly right to centre the subject")
        elif pos == "right":
            out.append("Move camera slightly left to centre the subject")

    @staticmethod
    def _check_composition(features: dict, out: list) -> None:
        score = features.get("composition_score", 100)
        rot_aligned = features.get("rot_aligned", True)
        if not rot_aligned and score < 60:
            out.append(
                "Try placing your subject on a rule-of-thirds grid point for a more dynamic composition"
            )

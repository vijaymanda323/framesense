"""
/api/v1/analyze-frame  – Main analysis endpoint.

Accepts a base64-encoded JPEG/PNG frame, decodes it, runs the full
feature extraction pipeline, and returns structured JSON.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from functools import partial

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, field_validator

from services.feature_extractor import FeatureExtractor

logger = logging.getLogger("framesense.router")

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    """Payload sent by the frontend every 2–3 seconds."""

    image: str  # base64-encoded JPEG (with or without data-URI prefix)

    @field_validator("image")
    @classmethod
    def strip_data_uri(cls, v: str) -> str:
        """Accept both raw base64 and data:image/jpeg;base64,<data>."""
        if "," in v:
            v = v.split(",", 1)[1]
        return v.strip()


class AnalyzeResponse(BaseModel):
    # Subject
    subject_position: str
    subject_size: str
    subject_label: str | None = None
    subject_detected: bool

    # Exposure
    brightness: str
    mean_intensity: float

    # Focus
    focus: str
    laplacian_variance: float

    # Clutter
    background_clutter: str
    edge_density: float

    # Alignment
    alignment: str
    tilt_angle_deg: float

    # Depth
    distance: str
    depth_score: float

    # Composition
    composition_score: int
    salient_point: list[float]
    rot_aligned: bool

    # Suggestions
    suggestions: list[str]

    # Meta
    processing_time_ms: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_image(b64_data: str) -> np.ndarray:
    """Decode base64 string → BGR numpy array."""
    try:
        img_bytes = base64.b64decode(b64_data)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("cv2.imdecode returned None – invalid image bytes")
        return frame
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Image decoding failed: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/analyze-frame",
    response_model=AnalyzeResponse,
    summary="Analyze a camera frame for photography composition",
    description=(
        "Accepts a base64-encoded JPEG/PNG frame captured from a live camera. "
        "Returns advanced photography feature extraction: brightness, focus, "
        "background clutter, alignment, depth, subject detection, rule-of-thirds "
        "composition score, and actionable suggestions."
    ),
)
async def analyze_frame(request: Request, payload: AnalyzeRequest):
    """
    Pipeline
    --------
    1. Decode base64 → numpy BGR frame
    2. Run FeatureExtractor in a thread-pool (keeps event loop free)
    3. Return AnalyzeResponse JSON
    """
    # Retrieve singleton models from app state
    yolo = request.app.state.yolo
    depth = request.app.state.depth
    rag = getattr(request.app.state, "rag", None)

    extractor = FeatureExtractor(yolo=yolo, depth=depth, rag=rag)

    # Decode image (fast, stays in this coroutine)
    frame = _decode_image(payload.image)

    # Run CPU-heavy extraction in thread pool
    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(
            None,
            partial(extractor.extract, frame),
        )
    except Exception as exc:
        logger.error("Feature extraction error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    return AnalyzeResponse(**results)

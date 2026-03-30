"""
FrameSense Backend – Main Application Entry Point
FastAPI server that exposes AI-powered photography composition analysis.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from routers.analyze import router as analyze_router
from routers.utils import router as utils_router
from services.yolo_detector import YOLODetector
from services.depth_estimator import DepthEstimator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("framesense")


from services.rag import RAGPipeline

# ---------------------------------------------------------------------------
# Lifespan – load heavyweight models once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FrameSense: loading models …")
    app.state.yolo = YOLODetector()
    app.state.depth = DepthEstimator()
    logger.info("FrameSense: initializing RAG pipeline …")
    app.state.rag = RAGPipeline()
    logger.info("FrameSense: all models ready ✓")
    yield
    logger.info("FrameSense: shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="FrameSense API",
    description=(
        "AI-powered photography composition analysis backend. "
        "Analyzes live camera frames and returns structured composition feedback."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins for local dev / cross-origin frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

app.include_router(analyze_router, prefix="/api/v1", tags=["Analysis"])
app.include_router(utils_router, prefix="/api/v1", tags=["Utilities"])


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "service": "FrameSense API", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    import os

    # To allow access from other devices (like Expo on your phone), 
    # the host MUST be '0.0.0.0'
    print("\n" + "="*50)
    print("FrameSense Backend starting...")
    print("Access locally: http://127.0.0.1:8000")
    print(f"Access from network: http://192.168.137.87:8000")
    print("="*50 + "\n")
    
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )

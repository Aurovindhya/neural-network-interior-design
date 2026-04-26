"""
main.py

FastAPI inference endpoint for the interior style classifier.

Endpoints:
  POST /predict   — upload a room image, get style prediction + palette
  GET  /styles    — list supported styles
  GET  /health    — health check

Usage:
    uvicorn api.main:app --reload
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from api.predictor import Predictor
from api.schemas import PredictionResponse, StyleListResponse, HealthResponse
from langfuse.tracing import get_tracer
from model.model import STYLE_CLASSES


# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Interior Style Classifier",
    description="Upload a room image to predict its interior design style and get palette recommendations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Globals ───────────────────────────────────────────────────────────────────

_predictor: Predictor | None = None
_tracer = get_tracer()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global _predictor
    weights_path = os.getenv("MODEL_WEIGHTS_PATH", "weights/best_model.pth")

    if not Path(weights_path).exists():
        print(
            f"WARNING: Model weights not found at '{weights_path}'. "
            "/predict will return 503 until weights are available. "
            "Train the model first: python model/train.py"
        )
        return

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    _predictor = Predictor(weights_path=weights_path, device=device)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(..., description="Room image (JPEG, PNG, or WebP)")):
    """
    Upload a room image and receive:
    - Predicted interior design style
    - Confidence scores for all styles
    - Complementary color palette recommendation
    - Langfuse trace ID for observability
    """
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first and set MODEL_WEIGHTS_PATH.",
        )

    # Validate content type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG, PNG, or WebP.",
        )

    image_bytes = await file.read()

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max 10 MB.")

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    with _tracer.trace("predict") as span:
        span.set_input({
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": len(image_bytes),
        })

        style, confidence, all_scores, inference_ms = _predictor.predict(image_bytes)
        palette = _predictor.get_palette(style)

        result = PredictionResponse(
            style=style,
            confidence=round(confidence, 4),
            all_scores=all_scores,
            palette=palette,
            trace_id=span.trace_id,
            inference_ms=inference_ms,
        )

        span.set_output({
            "style": style,
            "confidence": confidence,
            "inference_ms": inference_ms,
        })

    return result


@app.get("/styles", response_model=StyleListResponse)
async def list_styles():
    """Return the list of interior design styles the model can predict."""
    return StyleListResponse(styles=STYLE_CLASSES, count=len(STYLE_CLASSES))


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — returns whether the model is loaded."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HealthResponse(
        status="ok",
        model_loaded=_predictor is not None,
        device=device,
    )

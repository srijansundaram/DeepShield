"""
DeepShield — REST API
FastAPI wrapper around the detection pipeline.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /analyze        — upload image file → JSON verdict
    POST /analyze/url    — image URL → JSON verdict
    GET  /health         — model status
    GET  /docs           — auto-generated Swagger UI
"""

import io
import sys
from pathlib import Path
import requests

import torch
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

app = FastAPI(
    title="DeepShield API",
    description="Production deepfake detection — XceptionNet + EfficientNet-B4 ensemble",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = None

@app.on_event("startup")
def load_model():
    global detector
    from models.detector import EnsembleDetector
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = EnsembleDetector(device=device)
    print(f"✅ Model loaded on {device}")


class URLRequest(BaseModel):
    url: str

class DetectionResult(BaseModel):
    verdict: str
    is_fake: bool
    fake_probability: float
    real_probability: float
    confidence: float
    xception_score: float
    efficientnet_score: float
    face_count: int
    model_trained: bool


def analyze_pil(pil_image: Image.Image) -> dict:
    from utils.face_utils import analyze_faces
    img_bgr   = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    face_data = analyze_faces(img_bgr, detector)
    result    = face_data["overall"]
    return {
        "verdict":            result["verdict"],
        "is_fake":            result["is_fake"],
        "fake_probability":   round(result["fake_probability"], 4),
        "real_probability":   round(result["real_probability"], 4),
        "confidence":         round(result["confidence"], 4),
        "xception_score":     round(result.get("xception_score", 0), 4),
        "efficientnet_score": round(result.get("efficientnet_score", 0), 4),
        "face_count":         face_data["face_count"],
        "model_trained":      result.get("model_trained", False),
    }


@app.get("/health")
def health():
    return {
        "status":        "ok",
        "model_trained": getattr(detector, "is_trained", False),
        "device":        "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.post("/analyze", response_model=DetectionResult)
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        contents  = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image")
    try:
        return analyze_pil(pil_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/url", response_model=DetectionResult)
def analyze_url(body: URLRequest):
    try:
        resp      = requests.get(body.url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        pil_image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch image: {e}")
    try:
        return analyze_pil(pil_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {
        "name":    "DeepShield API",
        "version": "1.0.0",
        "endpoints": {
            "POST /analyze":     "Upload image file → JSON verdict",
            "POST /analyze/url": "Image URL → JSON verdict",
            "GET  /health":      "Model status",
            "GET  /docs":        "Swagger UI",
        }
    }
"""
DeepShield — Social Media Compression Handling (Tier 3, Item 3)
Detects heavy recompression (Instagram, Twitter/X, WhatsApp, Facebook)
and compensates before analysis so recompression artifacts aren't
mistaken for AI-generation artifacts.

Platforms re-encode images through their own JPEG pipelines, often
multiple times (upload -> CDN resize -> re-share -> re-download).
This produces block artifacts and stripped metadata that resemble
signals DeepShield's forensics tab flags — without compensation,
heavily-shared images get inflated false-positive forensic scores.
"""

import io
import numpy as np
import cv2
from PIL import Image
from typing import Dict


# Common export dimensions used by major platforms
_PLATFORM_DIMENSIONS = {
    (1080, 1080): "Instagram (square)",
    (1080, 1350): "Instagram (portrait)",
    (1080, 566):  "Instagram (landscape)",
    (1200, 675):  "Twitter/X",
    (1200, 630):  "Facebook",
    (800, 800):   "WhatsApp",
    (1024, 1024): "WhatsApp (HD)",
}


def _resave_mse(image: Image.Image, quality: int) -> float:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    reloaded = np.array(Image.open(buf).convert("RGB")).astype(float)
    orig = np.array(image.convert("RGB")).astype(float)
    return float(np.abs(orig - reloaded).mean())


def estimate_jpeg_quality(image: Image.Image) -> int:
    """Estimate the effective JPEG quality the image was last saved at."""
    best_q, best_diff = 95, float("inf")
    for q in [30, 50, 65, 75, 85, 95]:
        mse = _resave_mse(image, q)
        if mse < best_diff:
            best_diff, best_q = mse, q
    return best_q


def detect_platform_dimensions(image: Image.Image) -> str:
    """Check if image dimensions match a known social platform export size."""
    w, h = image.size
    if (w, h) in _PLATFORM_DIMENSIONS:
        return _PLATFORM_DIMENSIONS[(w, h)]
    for (pw, ph), name in _PLATFORM_DIMENSIONS.items():
        if abs(w - pw) <= 4 and abs(h - ph) <= 4:
            return name
    return ""


def classify_compression_level(image: Image.Image, has_exif: bool = None) -> Dict:
    """Classify how heavily an image has likely been recompressed."""
    est_quality = estimate_jpeg_quality(image)
    platform_match = detect_platform_dimensions(image)

    if has_exif is None:
        has_exif = hasattr(image, "_getexif") and image._getexif() is not None

    score = 0
    if est_quality <= 50:
        score += 2
    elif est_quality <= 70:
        score += 1
    if platform_match:
        score += 2
    if not has_exif:
        score += 1

    if score >= 4:
        level = "heavy"
    elif score >= 2:
        level = "moderate"
    elif score >= 1:
        level = "light"
    else:
        level = "none"

    return {
        "level": level,
        "estimated_quality": est_quality,
        "platform_match": platform_match or None,
        "has_exif": has_exif,
        "score": score,
    }


def preprocess_for_analysis(image: Image.Image, level: str) -> Image.Image:
    """
    Mild denoising + detail restoration to counteract recompression noise
    before the image hits the classifiers. Only applied for
    'moderate'/'heavy' — clean images pass through untouched.
    """
    if level not in ("moderate", "heavy"):
        return image

    arr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    strength = 9 if level == "heavy" else 5
    denoised = cv2.bilateralFilter(arr, d=strength, sigmaColor=50, sigmaSpace=50)

    blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(denoised, 1.3, blurred, -0.3, 0)

    rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def compression_adjusted_anomaly(raw_score: float, level: str) -> float:
    """
    Dampen a forensic anomaly score if it's likely caused by platform
    recompression rather than AI generation/manipulation, since both
    produce similar FFT/block signatures.
    """
    if level == "heavy":
        return round(raw_score * 0.55, 3)
    elif level == "moderate":
        return round(raw_score * 0.75, 3)
    return round(raw_score, 3)
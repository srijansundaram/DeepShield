"""
DeepShield — Forensics Utilities
FFT frequency analysis + EXIF metadata inspection + artifact signals
"""

import numpy as np
import cv2
from PIL import Image
from PIL.ExifTags import TAGS
import io
from typing import Dict, List, Optional, Tuple
import hashlib
import struct


# ──────────────────────────────────────────────
# Frequency Domain Analysis (FFT)
# ──────────────────────────────────────────────
def fft_analysis(image: Image.Image) -> Dict:
    """
    FFT-based analysis to detect unnatural high-frequency patterns.
    AI-generated images often show distinct spectral artifacts.
    Returns heatmap array and metrics.
    """
    img_rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # 2D FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    log_mag = np.log1p(magnitude)

    # Normalize for display
    log_mag_norm = (log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-8)

    # Radial frequency profile
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    dist_int = dist.astype(int)
    max_r = min(cx, cy)

    radial_profile = []
    for r in range(0, max_r, max(1, max_r // 64)):
        mask = (dist_int == r)
        if mask.any():
            radial_profile.append(float(magnitude[mask].mean()))

    # High-freq energy ratio
    low_mask = dist < max_r * 0.1
    high_mask = dist > max_r * 0.5
    low_energy = float(magnitude[low_mask].sum())
    high_energy = float(magnitude[high_mask].sum())
    hf_ratio = high_energy / (low_energy + 1e-8)

    # AI deepfakes tend to have elevated mid-band energy
    mid_mask = (dist > max_r * 0.15) & (dist < max_r * 0.4)
    mid_energy = float(magnitude[mid_mask].sum())
    total_energy = low_energy + mid_energy + high_energy
    mid_ratio = mid_energy / (total_energy + 1e-8)

    # Anomaly score based on energy distribution
    anomaly_score = min(1.0, (hf_ratio * 0.3 + mid_ratio * 0.7))

    return {
        "heatmap": log_mag_norm,
        "hf_ratio": round(hf_ratio, 4),
        "mid_energy_ratio": round(mid_ratio, 4),
        "anomaly_score": round(anomaly_score, 3),
        "radial_profile": radial_profile[:64],
    }


# ──────────────────────────────────────────────
# Compression Artifact Analysis
# ──────────────────────────────────────────────
def compression_analysis(image: Image.Image) -> Dict:
    """
    Detect re-encoding and compression artifacts.
    Multiple save/load cycles (common with generated images) leave traces.
    """
    img_rgb = np.array(image.convert("RGB"))

    # Save at multiple quality levels and compare
    scores = {}
    for q in [50, 75, 95]:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        reloaded = np.array(Image.open(buf).convert("RGB"))
        diff = np.abs(img_rgb.astype(float) - reloaded.astype(float))
        scores[f"q{q}_mse"] = round(float(diff.mean()), 4)

    # Block artifact detection (8x8 JPEG blocks)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(float)
    block_diffs = []
    for y in range(0, gray.shape[0] - 8, 8):
        for x in range(0, gray.shape[1] - 8, 8):
            right_edge = abs(float(gray[y:y+8, x+7].mean()) - float(gray[y:y+8, min(x+8, gray.shape[1]-1)].mean()))
            bottom_edge = abs(float(gray[y+7, x:x+8].mean()) - float(gray[min(y+8, gray.shape[0]-1), x:x+8].mean()))
            block_diffs.extend([right_edge, bottom_edge])

    block_score = float(np.mean(block_diffs)) if block_diffs else 0.0

    return {
        **scores,
        "block_artifact_score": round(block_score, 4),
        "re_encoding_suspected": block_score > 8.0,
    }


# ──────────────────────────────────────────────
# EXIF Metadata Forensics
# ──────────────────────────────────────────────
def analyze_metadata(image: Image.Image, filename: str = "") -> Dict:
    """
    Inspect EXIF and metadata for signs of AI generation or manipulation.
    """
    findings = []
    flags = []
    metadata = {}

    try:
        exif_data = image._getexif() if hasattr(image, "_getexif") else None
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if isinstance(value, bytes):
                    try:
                        value = value.decode("utf-8", errors="replace")
                    except Exception:
                        value = str(value)
                metadata[str(tag)] = str(value)[:200]

            # Check for camera info
            has_camera = any(k in metadata for k in ["Make", "Model", "LensModel"])
            has_gps = "GPSInfo" in metadata
            has_datetime = "DateTime" in metadata or "DateTimeOriginal" in metadata

            if not has_camera:
                findings.append("No camera metadata (typical of AI-generated images)")
                flags.append("no_camera_info")
            if not has_datetime:
                findings.append("No capture timestamp found")
                flags.append("no_timestamp")

            software = metadata.get("Software", "")
            ai_tools = ["stable diffusion", "midjourney", "dall-e", "gan", "diffusion", "adobe firefly"]
            for tool in ai_tools:
                if tool.lower() in software.lower():
                    findings.append(f"AI tool detected in software tag: {software}")
                    flags.append("ai_software_tag")
                    break

        else:
            findings.append("No EXIF data present (stripped or AI-generated)")
            flags.append("no_exif")

    except Exception as e:
        findings.append(f"EXIF read error: {str(e)[:80]}")

    # File hash
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    file_hash = hashlib.md5(buf.getvalue()).hexdigest()

    # Extension check
    if filename:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext in ["png", "webp"] and not metadata:
            findings.append("PNG/WebP without metadata — common for AI outputs")

    risk_score = min(1.0, len(flags) * 0.25)

    return {
        "metadata": metadata,
        "findings": findings,
        "flags": flags,
        "risk_score": round(risk_score, 2),
        "file_hash": file_hash,
        "has_exif": bool(metadata),
    }


# ──────────────────────────────────────────────
# Skin Tone Consistency
# ──────────────────────────────────────────────
def skin_tone_analysis(image: Image.Image, face_bbox: Optional[Tuple] = None) -> Dict:
    """
    Check for unnatural skin tone variations — deepfakes often blend poorly.
    """
    img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    if face_bbox:
        x, y, w, h = face_bbox
        roi = img_bgr[y:y+h, x:x+w]
    else:
        h, w = img_bgr.shape[:2]
        roi = img_bgr[h//4:3*h//4, w//4:3*w//4]

    if roi.size == 0:
        return {"skin_variance": 0.0, "anomaly": False}

    # Convert to YCrCb for skin detection
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    # Skin range in YCrCb
    lower = np.array([0, 120, 70], dtype=np.uint8)
    upper = np.array([255, 185, 145], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower, upper)
    skin_pixels = roi[skin_mask > 0]

    if len(skin_pixels) < 50:
        return {"skin_variance": 0.0, "skin_pixel_count": len(skin_pixels), "anomaly": False}

    variance = float(np.std(skin_pixels.astype(float)))
    # High variance can indicate blending artifacts
    anomaly = variance > 45.0

    return {
        "skin_variance": round(variance, 2),
        "skin_pixel_count": len(skin_pixels),
        "skin_coverage": round(len(skin_pixels) / (roi.shape[0] * roi.shape[1] + 1e-8), 3),
        "anomaly": anomaly,
    }

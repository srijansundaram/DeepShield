"""
DeepShield — Face Detection & Multi-Face Analysis
Uses OpenCV Haar cascade (no external deps) + optional DNN face detector
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import os


# ──────────────────────────────────────────────
# Haar cascade (always available)
# ──────────────────────────────────────────────
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)


def detect_faces(image: np.ndarray, min_size: int = 60) -> List[Tuple[int, int, int, int]]:
    """
    Returns list of (x, y, w, h) bounding boxes for detected faces.
    image should be BGR numpy array.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_size, min_size),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return []
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def crop_face(image: np.ndarray, bbox: Tuple[int, int, int, int], padding: float = 0.2) -> np.ndarray:
    """Crop face from image with padding."""
    h_img, w_img = image.shape[:2]
    x, y, w, h = bbox
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_img, x + w + pad_x)
    y2 = min(h_img, y + h + pad_y)
    return image[y1:y2, x1:x2]


def annotate_image(
    image: np.ndarray,
    face_results: List[Dict],
) -> np.ndarray:
    """
    Draw bounding boxes and verdict labels on image.
    face_results: list of {bbox, verdict, confidence}
    """
    annotated = image.copy()
    for fr in face_results:
        x, y, w, h = fr["bbox"]
        is_fake = fr["verdict"] == "Deepfake"
        color = (50, 50, 220) if is_fake else (50, 200, 80)  # BGR
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        label = f"{fr['verdict']} {fr['confidence']*100:.1f}%"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        lw, lh = label_size
        cv2.rectangle(annotated, (x, y - lh - 10), (x + lw + 6, y), color, -1)
        cv2.putText(
            annotated, label,
            (x + 3, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
        )
    return annotated


def analyze_faces(image_bgr: np.ndarray, detector) -> Dict:
    """
    Detect all faces and run deepfake detection on each.
    Returns summary dict.
    """
    faces = detect_faces(image_bgr)
    face_results = []

    if not faces:
        # Analyze full image if no faces found
        full_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = detector.predict_image(Image.fromarray(full_rgb))
        return {
            "face_count": 0,
            "face_results": [],
            "overall": result,
            "annotated": image_bgr,
        }

    for bbox in faces:
        crop = crop_face(image_bgr, bbox)
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = detector.predict_image(Image.fromarray(rgb_crop))
        face_results.append({**result, "bbox": bbox})

    # Overall verdict: fake if any face is fake
    any_fake = any(fr["is_fake"] for fr in face_results)
    avg_conf = np.mean([fr["fake_probability"] for fr in face_results])

    annotated = annotate_image(image_bgr, face_results)

    return {
        "face_count": len(faces),
        "face_results": face_results,
        "overall": {
            "is_fake": any_fake,
            "fake_probability": float(avg_conf),
            "real_probability": float(1 - avg_conf),
            "confidence": float(avg_conf if any_fake else 1 - avg_conf),
            "verdict": "Deepfake" if any_fake else "Real",
        },
        "annotated": annotated,
    }

"""
DeepShield — Face Detection & Multi-Face Analysis
Uses InsightFace for robust real-world face detection
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import os
import insightface
from insightface.app import FaceAnalysis


# ── InsightFace detector (cached) ──
_face_app = None

def _get_face_app():
    global _face_app
    if _face_app is None:
        _face_app = FaceAnalysis(
            name="buffalo_sc",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
    return _face_app


def detect_faces(image: np.ndarray, min_size: int = 30) -> List[Tuple[int, int, int, int]]:
    app = _get_face_app()
    faces = app.get(image)
    results = []
    if faces:
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            w, h = x2 - x1, y2 - y1
            if w >= min_size and h >= min_size:
                results.append((x1, y1, w, h))
    
    # Fallback to Haar cascade if InsightFace finds nothing
    if not results:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        haar_faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(haar_faces) > 0:
            # Keep only largest face
            haar_faces = sorted(haar_faces, key=lambda f: f[2]*f[3], reverse=True)
            x, y, w, h = haar_faces[0]
            results.append((int(x), int(y), int(w), int(h)))
    
    return results


def crop_face(image: np.ndarray, bbox: Tuple[int, int, int, int], padding: float = 0.1) -> np.ndarray:
    """Crop face from image with padding."""
    h_img, w_img = image.shape[:2]
    x, y, w, h = bbox
    # Reduce padding for large faces
    face_area = w * h
    img_area = w_img * h_img
    face_ratio = face_area / img_area
    # Smaller padding for larger faces
    adaptive_padding = max(0.05, 0.2 * (1 - face_ratio * 2))
    pad_x = int(w * adaptive_padding)
    pad_y = int(h * adaptive_padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_img, x + w + pad_x)
    y2 = min(h_img, y + h + pad_y)
    return image[y1:y2, x1:x2]


def annotate_image(image: np.ndarray, face_results: List[Dict]) -> np.ndarray:
    """Draw bounding boxes and verdict labels on image."""
    annotated = image.copy()
    for fr in face_results:
        x, y, w, h = fr["bbox"]
        is_fake = fr["verdict"] == "Deepfake"
        color = (50, 50, 220) if is_fake else (50, 200, 80)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        label = f"{fr['verdict']} {fr['confidence']*100:.1f}%"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        lw, lh = label_size
        cv2.rectangle(annotated, (x, y - lh - 10), (x + lw + 6, y), color, -1)
        cv2.putText(annotated, label, (x + 3, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return annotated


def analyze_faces(image_bgr: np.ndarray, detector) -> Dict:
    """
    Detect all faces and run deepfake detection on each.
    Returns summary dict.
    """
    faces = detect_faces(image_bgr)
    face_results = []

    if not faces:
        # No face detected — analyze full image
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
        if crop.size == 0:
            continue
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = detector.predict_image(Image.fromarray(rgb_crop))
        face_results.append({**result, "bbox": bbox})

    if not face_results:
        full_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = detector.predict_image(Image.fromarray(full_rgb))
        return {
            "face_count": 0,
            "face_results": [],
            "overall": result,
            "annotated": image_bgr,
        }

    # Average probability across all faces
    avg_conf = float(np.mean([fr["fake_probability"] for fr in face_results]))
    is_fake = avg_conf > 0.60

    annotated = annotate_image(image_bgr, face_results)

    return {
        "face_count": len(face_results),
        "face_results": face_results,
        "overall": {
            "is_fake": is_fake,
            "fake_probability": avg_conf,
            "real_probability": 1 - avg_conf,
            "confidence": avg_conf if is_fake else 1 - avg_conf,
            "verdict": "Deepfake" if is_fake else "Real",
            "xception_score": float(np.mean([fr.get("xception_score", 0) for fr in face_results])),
            "efficientnet_score": float(np.mean([fr.get("efficientnet_score", 0) for fr in face_results])),
        },
        "annotated": annotated,
    }
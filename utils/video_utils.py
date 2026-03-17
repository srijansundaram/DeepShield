"""
DeepShield — Video Analysis Pipeline
Frame-by-frame detection with temporal consistency analysis
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Generator, Optional, Tuple
import tempfile
import os


# ──────────────────────────────────────────────
# Frame Extraction
# ──────────────────────────────────────────────
def extract_frames(
    video_path: str,
    sample_rate: int = 10,
    max_frames: int = 60,
) -> Tuple[List[np.ndarray], float, int]:
    """
    Extract frames from video at given sample_rate (every Nth frame).
    Returns (frames_list_BGR, fps, total_frame_count).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    idx = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_rate == 0:
            frames.append(frame)
        idx += 1

    cap.release()
    return frames, fps, total


# ──────────────────────────────────────────────
# Temporal Consistency
# ──────────────────────────────────────────────
def temporal_consistency_score(fake_probs: List[float]) -> Dict:
    """
    Measure frame-to-frame variance — deepfakes often show erratic confidence jumps.
    Returns consistency metrics.
    """
    if len(fake_probs) < 2:
        return {"variance": 0.0, "consistency": 1.0, "spikes": []}

    arr = np.array(fake_probs)
    diffs = np.abs(np.diff(arr))
    variance = float(np.var(arr))
    mean_diff = float(np.mean(diffs))

    # Spikes: frames where confidence jumps > 0.25
    spike_threshold = 0.25
    spikes = [int(i + 1) for i, d in enumerate(diffs) if d > spike_threshold]

    # Consistency: lower variance = more consistent (real tends to be stable)
    consistency = float(max(0.0, 1.0 - variance * 4))

    return {
        "variance": round(variance, 4),
        "mean_frame_diff": round(mean_diff, 4),
        "consistency": round(consistency, 3),
        "spikes": spikes,
        "spike_count": len(spikes),
    }


# ──────────────────────────────────────────────
# Eye Blink Analysis
# ──────────────────────────────────────────────
_EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def estimate_blink_pattern(frames: List[np.ndarray]) -> Dict:
    """
    Estimate eye blink rate and pattern across frames.
    Deepfakes often have abnormal or missing blinks.
    """
    eye_counts = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = _EYE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
        eye_counts.append(len(eyes))

    if not eye_counts:
        return {"avg_eyes_per_frame": 0, "blink_events": 0, "blink_anomaly": False}

    avg_eyes = float(np.mean(eye_counts))
    # Blink events: frames where eyes drop to 0 after > 0
    blinks = sum(
        1 for i in range(1, len(eye_counts))
        if eye_counts[i] == 0 and eye_counts[i - 1] > 0
    )

    # Expected: ~15-20 blinks/min → ~0.25-0.33/sec
    duration_sec = len(frames) / 5  # assume ~5fps sampled
    expected_blinks = duration_sec * 0.28
    anomaly = blinks < expected_blinks * 0.3 or blinks > expected_blinks * 3.0

    return {
        "avg_eyes_per_frame": round(avg_eyes, 2),
        "blink_events": blinks,
        "expected_blinks": round(expected_blinks, 1),
        "blink_anomaly": anomaly,
    }


# ──────────────────────────────────────────────
# Full Video Analysis
# ──────────────────────────────────────────────
def analyze_video(
    video_path: str,
    detector,
    sample_rate: int = 8,
    max_frames: int = 50,
    progress_callback=None,
) -> Dict:
    """
    Full video deepfake analysis pipeline.
    Returns comprehensive results dict.
    """
    frames, fps, total_frames = extract_frames(video_path, sample_rate, max_frames)
    if not frames:
        return {"error": "Could not extract frames from video."}

    fake_probs = []
    frame_results = []

    for i, frame in enumerate(frames):
        if progress_callback:
            progress_callback(i / len(frames))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.predict_image(Image.fromarray(rgb))
        fake_probs.append(result["fake_probability"])
        frame_results.append({
            "frame_idx": i * sample_rate,
            "timestamp": round(i * sample_rate / fps, 2),
            **result,
        })

    avg_fake_prob = float(np.mean(fake_probs))
    is_fake = avg_fake_prob > 0.5

    temporal = temporal_consistency_score(fake_probs)
    blink = estimate_blink_pattern(frames)

    # Peak manipulation frame
    peak_idx = int(np.argmax(fake_probs))
    peak_frame = frames[peak_idx] if frames else None

    video_duration = total_frames / fps

    return {
        "is_fake": is_fake,
        "verdict": "Deepfake" if is_fake else "Real",
        "fake_probability": avg_fake_prob,
        "real_probability": 1 - avg_fake_prob,
        "confidence": avg_fake_prob if is_fake else 1 - avg_fake_prob,
        "frames_analyzed": len(frames),
        "total_frames": total_frames,
        "video_duration": round(video_duration, 1),
        "fps": round(fps, 1),
        "fake_probs_timeline": fake_probs,
        "frame_results": frame_results,
        "temporal": temporal,
        "blink_analysis": blink,
        "peak_frame": peak_frame,
        "peak_frame_idx": peak_idx,
    }

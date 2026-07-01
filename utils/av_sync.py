"""
DeepShield — Audio-Visual Sync Detection (Tier 2, Item 4)
Detects lip-audio desync — a strong signal in deepfake videos.

How it works:
  1. Extract audio from video → compute RMS energy per frame
  2. Extract lip/lower-face region per frame
  3. Compute lip movement magnitude per frame
  4. Correlate audio energy with lip movement
  5. Low correlation = likely deepfake

Standalone usage:
    from utils.av_sync import analyze_av_sync
    result = analyze_av_sync("video.mp4")
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List


def _extract_audio_energy(video_path: str, fps: float, n_frames: int) -> Optional[np.ndarray]:
    """Extract per-frame audio RMS energy using ffmpeg + librosa."""
    try:
        import librosa
        import tempfile
        import os
        import subprocess

        tmp_wav = tempfile.mktemp(suffix=".wav")
        ret = subprocess.run(
            [
                "ffmpeg", "-i", video_path,
                "-vn", "-ar", "16000", "-ac", "1",
                tmp_wav, "-y", "-loglevel", "quiet"
            ],
            capture_output=True
        )
        if ret.returncode != 0 or not os.path.exists(tmp_wav):
            return None

        audio, sr = librosa.load(tmp_wav, sr=16000, mono=True)
        os.unlink(tmp_wav)

        hop = max(1, int(sr / fps))
        energies = []
        for i in range(n_frames):
            start = i * hop
            end   = start + hop
            chunk = audio[start:end] if end <= len(audio) else audio[start:]
            rms   = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0
            energies.append(rms)

        return np.array(energies)

    except Exception as e:
        print(f"  AV sync — audio extraction failed: {e}")
        return None


def _extract_lip_movement(frames: List[np.ndarray]) -> np.ndarray:
    """
    Estimate lip movement per frame using lower-face ROI pixel variance.
    Falls back gracefully if dlib landmarks unavailable.
    """
    movements = []

    # Try dlib landmarks first (more accurate)
    try:
        import dlib
        predictor_path = Path.home() / ".deepshield" / "shape_predictor_68_face_landmarks.dat"
        if predictor_path.exists():
            detector  = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(str(predictor_path))
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                dets = detector(gray, 0)
                if not dets:
                    movements.append(0.0)
                    continue
                shape   = predictor(gray, dets[0])
                lip_pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)])
                lip_h   = float(np.std(lip_pts[:, 1]))
                movements.append(lip_h)
            return np.array(movements)
    except ImportError:
        pass

    # Fallback: lower-face ROI variance (no dlib needed)
    for frame in frames:
        h, w = frame.shape[:2]
        # Lower third of face, center horizontal band
        roi = frame[int(h * 0.60):int(h * 0.85), int(w * 0.25):int(w * 0.75)]
        if roi.size == 0:
            movements.append(0.0)
            continue
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(float)
        movements.append(float(np.std(gray_roi)))

    return np.array(movements)


def analyze_av_sync(
    video_path: str,
    sample_rate: int = 4,
    max_frames: int = 60,
) -> Dict:
    """
    Main entry point. Analyzes audio-visual sync in a video file.

    Args:
        video_path   : path to video file
        sample_rate  : sample every Nth frame
        max_frames   : max frames to analyze

    Returns dict with:
        sync_score   : 0–1, higher = better sync (more likely real)
        correlation  : Pearson r between audio energy and lip movement
        is_desynced  : True if sync score below threshold
        confidence   : abs(correlation) — how strong the signal is
    """
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    idx   = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_rate == 0:
            frames.append(frame)
        idx += 1
    cap.release()

    if len(frames) < 4:
        return {
            "sync_score":  0.5,
            "correlation": None,
            "is_desynced": False,
            "confidence":  0.0,
            "note":        "Too few frames to analyze",
        }

    n = len(frames)
    audio_energy = _extract_audio_energy(video_path, fps / sample_rate, n)
    lip_movement = _extract_lip_movement(frames)

    if audio_energy is None or len(audio_energy) == 0:
        return {
            "sync_score":  0.5,
            "correlation": None,
            "is_desynced": False,
            "confidence":  0.0,
            "note":        "Audio extraction failed — is ffmpeg installed?",
        }

    # Align lengths
    min_len      = min(len(audio_energy), len(lip_movement))
    audio_energy = audio_energy[:min_len]
    lip_movement = lip_movement[:min_len]

    def normalize(x):
        s = x.std()
        return (x - x.mean()) / (s + 1e-8)

    a_norm = normalize(audio_energy)
    l_norm = normalize(lip_movement)

    # Base correlation
    correlation = float(np.corrcoef(a_norm, l_norm)[0, 1])
    if np.isnan(correlation):
        correlation = 0.0

    # Try small time shifts ±3 frames to find best lag
    best_corr = correlation
    for lag in range(-3, 4):
        if lag == 0:
            continue
        a_shift = np.roll(a_norm, lag)
        c = float(np.corrcoef(a_shift, l_norm)[0, 1])
        if not np.isnan(c) and c > best_corr:
            best_corr = c

    sync_score  = float(np.clip((best_corr + 1) / 2, 0, 1))
    is_desynced = sync_score < 0.35
    confidence  = float(abs(best_corr))

    return {
        "sync_score":      round(sync_score, 3),
        "correlation":     round(best_corr, 3),
        "is_desynced":     is_desynced,
        "confidence":      round(confidence, 3),
        "frames_analyzed": n,
        "note": "Low sync score suggests audio-visual mismatch (deepfake signal)" if is_desynced else "Sync within normal range",
    }
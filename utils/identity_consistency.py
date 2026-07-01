"""
DeepShield — Face Identity Consistency (Tier 3, Item 2)
Tracks whether the same face identity persists across video frames.
Deepfakes (especially face-swaps) often show subtle identity drift
between frames that a single-frame classifier won't catch.

Requires the InsightFace 'buffalo_l' model pack (includes ArcFace
recognition embeddings) — downloaded automatically on first run.
"""

import numpy as np
from typing import List, Dict, Optional

_embed_app = None


def _get_embedding_app():
    global _embed_app
    if _embed_app is None:
        from insightface.app import FaceAnalysis
        _embed_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _embed_app.prepare(ctx_id=0, det_size=(320, 320))
    return _embed_app


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def analyze_identity_consistency(
    frames: List[np.ndarray],
    drift_threshold: float = 0.45,
) -> Dict:
    """
    Extract a face embedding per frame and measure identity drift
    against a running reference embedding.

    Returns:
        identity_score : 0-1, higher = more consistent identity (more likely real)
        drift_events   : frame indices where identity shifted sharply
        similarities   : per-frame similarity to reference (None if no face found)
        faces_found    : count of frames where an embedding was extracted
    """
    if len(frames) < 2:
        return {
            "identity_score": 1.0, "drift_events": [], "similarities": [],
            "faces_found": 0, "note": "Too few frames to analyze",
        }

    try:
        app = _get_embedding_app()
    except Exception as e:
        return {
            "identity_score": 1.0, "drift_events": [], "similarities": [],
            "faces_found": 0, "note": f"Identity model unavailable: {e}",
        }

    embeddings = []
    for frame in frames:
        try:
            faces = app.get(frame)
        except Exception:
            faces = []
        if not faces:
            embeddings.append(None)
            continue
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            emb = getattr(face, "embedding", None)
        embeddings.append(emb)

    valid = [e for e in embeddings if e is not None]
    if len(valid) < 2:
        return {
            "identity_score": 1.0, "drift_events": [], "similarities": [],
            "faces_found": len(valid), "note": "Not enough faces detected to track identity",
        }

    # Reference = median embedding of the first few valid frames (robust anchor)
    anchor_pool = valid[:max(3, len(valid) // 4)]
    reference = np.median(np.stack(anchor_pool), axis=0)

    similarities, drift_events = [], []
    for i, emb in enumerate(embeddings):
        if emb is None:
            similarities.append(None)
            continue
        sim = _cosine_sim(emb, reference)
        similarities.append(round(sim, 4))
        if sim < drift_threshold:
            drift_events.append(i)

    valid_sims = [s for s in similarities if s is not None]
    avg_sim = float(np.mean(valid_sims))
    identity_score = float(np.clip((avg_sim + 1) / 2, 0, 1))

    return {
        "identity_score": round(identity_score, 3),
        "avg_similarity": round(avg_sim, 3),
        "drift_events": drift_events,
        "drift_count": len(drift_events),
        "similarities": similarities,
        "faces_found": len(valid),
        "identity_anomaly": len(drift_events) > max(1, len(valid) * 0.15),
    }
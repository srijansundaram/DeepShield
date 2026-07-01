"""
DeepShield — Explainability Utilities (Tier 3, Item 1)
Normalizes model scores + forensic signals onto a common scale
so their relative contribution to the verdict can be visualized.
"""

from typing import Dict, List


def compute_signal_contributions(result: Dict) -> List[Dict]:
    """
    Normalize all detection signals onto a common 0-100 'signal strength' scale
    so relative influence can be visually compared.

    Model scores are weighted by the same coefficients used in the ensemble
    (0.40/0.35/0.25 or 0.55/0.45 fallback) and reflect what actually drove
    the mathematical probability.

    Forensic scores (FFT, compression, metadata, skin tone) are advisory —
    they inform the artifact_signals tags but are NOT blended into
    fake_probability. They're included here for transparency, clearly tagged.
    """
    signals = []

    has_vit = result.get("vit_score") is not None
    ensemble_weights = {
        "xception": 0.40 if has_vit else 0.55,
        "efficientnet": 0.35 if has_vit else 0.45,
        "vit": 0.25,
    }

    xc = result.get("xception_score")
    if xc is not None:
        signals.append({
            "name": "XceptionNet",
            "category": "model",
            "raw": round(xc, 4),
            "strength": abs(xc - 0.5) * 2 * ensemble_weights["xception"],
        })

    eff = result.get("efficientnet_score")
    if eff is not None:
        signals.append({
            "name": "EfficientNet-B4",
            "category": "model",
            "raw": round(eff, 4),
            "strength": abs(eff - 0.5) * 2 * ensemble_weights["efficientnet"],
        })

    vit = result.get("vit_score")
    if vit is not None:
        signals.append({
            "name": "Vision Transformer",
            "category": "model",
            "raw": round(vit, 4),
            "strength": abs(vit - 0.5) * 2 * ensemble_weights["vit"],
        })

    forensics = result.get("forensics", {})

    fft_r = forensics.get("fft_analysis", {})
    if fft_r:
        adjusted = result.get("fft_anomaly_adjusted", fft_r.get("anomaly_score", 0))
        signals.append({
            "name": "FFT frequency anomaly",
            "category": "forensic",
            "raw": fft_r.get("anomaly_score", 0),
            "strength": adjusted,
        })

    comp_r = forensics.get("compression_analysis", {})
    if comp_r:
        block = comp_r.get("block_artifact_score", 0)
        adjusted_block = result.get("block_score_adjusted", block)
        signals.append({
            "name": "Compression artifacts",
            "category": "forensic",
            "raw": block,
            "strength": min(1.0, adjusted_block / 20.0),
        })

    meta_r = forensics.get("metadata_analysis", {})
    if meta_r:
        signals.append({
            "name": "Metadata risk",
            "category": "forensic",
            "raw": meta_r.get("risk_score", 0),
            "strength": meta_r.get("risk_score", 0),
        })

    skin_r = forensics.get("skin_analysis", {})
    if skin_r and skin_r.get("skin_pixel_count", 0) > 100:
        var = skin_r.get("skin_variance", 0)
        signals.append({
            "name": "Skin tone consistency",
            "category": "forensic",
            "raw": var,
            "strength": min(1.0, var / 90.0),
        })

    total = sum(s["strength"] for s in signals) or 1e-8
    for s in signals:
        s["contribution_pct"] = round(s["strength"] / total * 100, 1)

    signals.sort(key=lambda s: s["strength"], reverse=True)
    return signals
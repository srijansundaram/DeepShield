"""
DeepShield — Visualization Utilities
GradCAM overlays, heatmaps, charts, comparison views
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, List, Tuple, Dict
import io


# ──────────────────────────────────────────────
# Colormap
# ──────────────────────────────────────────────
HEATMAP_CMAP = plt.cm.RdYlGn_r  # Green (safe) → Red (manipulated)


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _bgr_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# ──────────────────────────────────────────────
# GradCAM Overlay
# ──────────────────────────────────────────────
def overlay_gradcam(
    original: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
    size: Tuple[int, int] = (400, 400),
) -> Image.Image:
    """
    Overlay Grad-CAM heatmap on original image.
    Returns PIL Image.
    """
    orig_resized = original.convert("RGB").resize(size, Image.LANCZOS)
    orig_arr = np.array(orig_resized, dtype=np.float32) / 255.0

    cam_resized = cv2.resize(cam, size, interpolation=cv2.INTER_LINEAR)
    cam_resized = np.clip(cam_resized, 0, 1)

    # Apply colormap
    colored = HEATMAP_CMAP(cam_resized)[:, :, :3]  # drop alpha

    # Blend
    blended = (1 - alpha) * orig_arr + alpha * colored
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


# ──────────────────────────────────────────────
# Side-by-side Comparison
# ──────────────────────────────────────────────
def make_comparison_image(
    original: Image.Image,
    cam_overlay: Image.Image,
    size: Tuple[int, int] = (400, 400),
) -> Image.Image:
    """Create side-by-side original vs GradCAM comparison."""
    left = original.convert("RGB").resize(size, Image.LANCZOS)
    right = cam_overlay.convert("RGB").resize(size, Image.LANCZOS)

    combined = Image.new("RGB", (size[0] * 2 + 10, size[1]), color=(30, 30, 30))
    combined.paste(left, (0, 0))
    combined.paste(right, (size[0] + 10, 0))
    return combined


# ──────────────────────────────────────────────
# FFT Heatmap Visualization
# ──────────────────────────────────────────────
def render_fft_heatmap(fft_result: Dict, size: Tuple[int, int] = (300, 300)) -> Image.Image:
    """Render FFT magnitude spectrum as PIL Image."""
    heatmap = fft_result.get("heatmap", np.zeros((300, 300)))
    colored = plt.cm.inferno(heatmap)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    pil = Image.fromarray(colored).resize(size, Image.LANCZOS)
    return pil


# ──────────────────────────────────────────────
# Timeline Chart (frame-by-frame confidence)
# ──────────────────────────────────────────────
def render_timeline_chart(
    fake_probs: List[float],
    timestamps: Optional[List[float]] = None,
    width: int = 700,
    height: int = 200,
) -> Image.Image:
    """
    Render a frame-confidence timeline chart.
    Returns PIL Image.
    """
    if not fake_probs:
        return Image.new("RGB", (width, height), (245, 245, 245))

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor("#F8F8F8")
    ax.set_facecolor("#F8F8F8")

    x = timestamps if timestamps else list(range(len(fake_probs)))
    y = np.array(fake_probs)

    # Color each bar
    colors = []
    for p in y:
        if p < 0.35:
            colors.append("#1D9E75")
        elif p < 0.55:
            colors.append("#EF9F27")
        else:
            colors.append("#E24B4A")

    ax.bar(x, y, color=colors, width=max(0.5, len(x) / 80), alpha=0.9)
    ax.axhline(0.5, color="#888780", linestyle="--", linewidth=0.8, alpha=0.7, label="Threshold")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fake probability", fontsize=8, color="#5F5E5A")
    ax.set_xlabel("Time (s)" if timestamps else "Frame", fontsize=8, color="#5F5E5A")
    ax.tick_params(labelsize=7, colors="#5F5E5A")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color("#D3D1C7")
    ax.set_title("Frame-by-frame deepfake confidence", fontsize=9, color="#444441", pad=6)

    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="PNG", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ──────────────────────────────────────────────
# Confidence Gauge
# ──────────────────────────────────────────────
def render_gauge(confidence: float, is_fake: bool, size: int = 220) -> Image.Image:
    """Render a semicircular confidence gauge."""
    fig, ax = plt.subplots(figsize=(size / 100, size / 100), dpi=100, subplot_kw=dict(aspect="equal"))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    color = "#E24B4A" if is_fake else "#1D9E75"
    label = f"{'DEEPFAKE' if is_fake else 'REAL'}\n{confidence*100:.1f}%"

    # Background arc
    theta = np.linspace(0, np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), color="#E8E8E8", linewidth=14, solid_capstyle="round")

    # Filled arc
    fill_end = confidence * np.pi
    theta_fill = np.linspace(0, fill_end, 100)
    ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=color, linewidth=14, solid_capstyle="round")

    ax.text(0, 0.15, label, ha="center", va="center", fontsize=11, fontweight="bold",
            color=color, fontfamily="monospace")

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.axis("off")
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    plt.savefig(buf, format="PNG", bbox_inches="tight", transparent=True, dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ──────────────────────────────────────────────
# Model Score Bar
# ──────────────────────────────────────────────
def render_model_bars(xception_score: float, efficientnet_score: float) -> Image.Image:
    """Bar chart comparing model scores."""
    fig, ax = plt.subplots(figsize=(4.5, 1.8), dpi=100)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    models = ["XceptionNet", "EfficientNet-B4"]
    scores = [xception_score, efficientnet_score]
    colors = ["#E24B4A" if s > 0.5 else "#1D9E75" for s in scores]

    bars = ax.barh(models, scores, color=colors, height=0.45, alpha=0.9)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="#888780", linestyle="--", linewidth=0.8)

    for bar, score in zip(bars, scores):
        ax.text(min(score + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                f"{score*100:.1f}%", va="center", fontsize=8, color="#444441")

    ax.tick_params(labelsize=8, colors="#5F5E5A")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#D3D1C7")
    ax.spines["left"].set_color("#D3D1C7")
    plt.tight_layout(pad=0.4)

    buf = io.BytesIO()
    plt.savefig(buf, format="PNG", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()

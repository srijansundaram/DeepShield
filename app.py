"""
DeepShield — Forensic Deepfake Detection System
Clean light UI with professional forensic aesthetic
"""

import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import sys
import time
import io
from PIL import Image
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="DeepShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #F7F8FA !important;
    color: #1A1D23 !important;
  }
  .stApp { background-color: #F7F8FA !important; }

  section[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E5E7EB !important;
  }
  section[data-testid="stSidebar"] * { color: #1A1D23 !important; }

  /* Logo */
  .ds-logo { font-family: 'JetBrains Mono', monospace; font-size: 18px; font-weight: 700; color: #1A1D23 !important; letter-spacing: -0.02em; }
  .ds-logo span { color: #2563EB; }
  .ds-version { font-size: 11px; color: #9CA3AF; letter-spacing: 0.05em; margin-top: 2px; }
  .ds-nav-section { font-size: 10px; font-weight: 600; color: #9CA3AF; letter-spacing: 0.1em; text-transform: uppercase; margin: 20px 0 6px; }

  /* Verdicts */
  .verdict-fake { display: inline-flex; align-items: center; gap: 6px; background: #FEF2F2; color: #DC2626; border: 1px solid #FCA5A5; padding: 6px 16px; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 13px; }
  .verdict-real { display: inline-flex; align-items: center; gap: 6px; background: #F0FDF4; color: #16A34A; border: 1px solid #86EFAC; padding: 6px 16px; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 13px; }

  /* Tags */
  .tag-warn { display: inline-block; background: #FFFBEB; color: #D97706; border: 1px solid #FCD34D; padding: 2px 10px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 11px; margin: 2px; }
  .tag-ok { display: inline-block; background: #F0FDF4; color: #16A34A; border: 1px solid #86EFAC; padding: 2px 10px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 11px; margin: 2px; }
  .tag-model { display: inline-block; background: #EFF6FF; color: #2563EB; border: 1px solid #BFDBFE; padding: 2px 10px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 11px; margin: 2px; }
  .tag-forensic { display: inline-block; background: #FAF5FF; color: #9333EA; border: 1px solid #E9D5FF; padding: 2px 10px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 11px; margin: 2px; }

  /* Section labels */
  .section-label { font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #6B7280; margin: 16px 0 8px; }

  /* Page header */
  .scan-header { font-size: 24px; font-weight: 700; color: #1A1D23; letter-spacing: -0.02em; margin-bottom: 4px; }
  .scan-sub { font-size: 14px; color: #6B7280; margin-bottom: 24px; }

  /* Banners */
  .demo-banner { background: #FFFBEB; border: 1px solid #FCD34D; border-left: 3px solid #F59E0B; border-radius: 8px; padding: 10px 14px; font-size: 12px; color: #92400E; margin-bottom: 16px; }
  .trained-banner { background: #F0FDF4; border: 1px solid #86EFAC; border-left: 3px solid #16A34A; border-radius: 8px; padding: 10px 14px; font-size: 12px; color: #14532D; margin-bottom: 16px; }

  /* Stat cards */
  .stat-row { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
  .stat-cell { flex: 1; min-width: 100px; background: #FFFFFF; border: 1px solid #E5E7EB; padding: 16px; border-radius: 10px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
  .stat-cell .sl { font-size: 10px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #9CA3AF; margin-bottom: 6px; }
  .stat-cell .sv { font-family: 'JetBrains Mono', monospace; font-size: 22px; font-weight: 700; }

  /* Buttons */
  .stButton > button { background: #2563EB !important; color: #FFFFFF !important; border: none !important; border-radius: 8px !important; font-family: 'Inter', sans-serif !important; font-size: 13px !important; font-weight: 600 !important; padding: 8px 20px !important; }
  .stButton > button:hover { background: #1D4ED8 !important; }
  .stButton > button[kind="secondary"] { background: #FFFFFF !important; color: #374151 !important; border: 1px solid #D1D5DB !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #FFFFFF !important; border-bottom: 1px solid #E5E7EB !important; border-radius: 0 !important; padding: 0 4px !important; }
  .stTabs [data-baseweb="tab"] { font-size: 12px !important; font-weight: 500 !important; letter-spacing: 0.04em !important; color: #6B7280 !important; padding: 10px 16px !important; }
  .stTabs [aria-selected="true"] { color: #2563EB !important; border-bottom: 2px solid #2563EB !important; }

  /* Inputs */
  .stTextInput input { background: #FFFFFF !important; border: 1px solid #D1D5DB !important; border-radius: 8px !important; color: #1A1D23 !important; font-size: 14px !important; }
  .stTextInput input:focus { border-color: #2563EB !important; box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important; }

  /* Progress */
  .stProgress [data-baseweb="progress-bar"] > div { background: #2563EB !important; }
  .stProgress [data-baseweb="progress-bar"] { background: #E5E7EB !important; }

  /* Expander */
  .stExpander { border: 1px solid #E5E7EB !important; border-radius: 10px !important; background: #FFFFFF !important; }

  /* File uploader */
  [data-testid="stFileUploader"] { background: #FFFFFF; border: 2px dashed #D1D5DB; border-radius: 10px; padding: 8px; }
  [data-testid="stFileUploader"]:hover { border-color: #2563EB; }

  /* Divider */
  hr { border: none; border-top: 1px solid #E5E7EB; margin: 20px 0; }

  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  header { background: transparent !important; }
  header [data-testid="stToolbar"] { visibility: hidden; }
  section[data-testid="stSidebar"] {
    min-width: 240px !important;
    max-width: 240px !important;
  }

  /* Monospace data blocks */
  .data-block { font-family: 'JetBrains Mono', monospace; font-size: 12px; line-height: 2; background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 8px; padding: 12px 16px; }

  /* Sidebar nav buttons */
  section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #374151 !important;
    border: none !important;
    text-align: left !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    padding: 6px 10px !important;
    width: 100% !important;
  }
  section[data-testid="stSidebar"] .stButton > button:hover {
    background: #F3F4F6 !important;
    color: #111827 !important;
  }

  /* Remove the sidebar collapse/expand control entirely — cover all known variants */
  [data-testid="collapsedControl"],
  [data-testid="stSidebarCollapsedControl"],
  [data-testid="stSidebarCollapseButton"],
  [data-testid="stSidebarNavCollapseButton"],
  [data-testid="stSidebarHeader"] button,
  section[data-testid="stSidebar"] button[kind="header"],
  section[data-testid="stSidebar"] div[data-testid="stSidebarHeader"],
  section[data-testid="stSidebar"] [data-testid*="ollaps"],
  button[title="Collapse sidebar"],
  button[title="Expand sidebar"],
  button[aria-label="Collapse sidebar"],
  button[aria-label="Expand sidebar"] {
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
    width: 0 !important;
    height: 0 !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Model Loader ──

@st.cache_resource(show_spinner="Loading detection models...")
def load_detector():
    import torch, random
    from models.detector import EnsembleDetector
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return EnsembleDetector(device=device)


def model_status_banner():
    detector = load_detector()
    is_trained = getattr(detector, 'is_trained', False)
    if not is_trained:
        st.markdown("""
        <div class="demo-banner">
          ⚠ Demo mode — No trained checkpoint loaded. Drop .pth files into checkpoints/ to enable real detection.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="trained-banner">
          ✓ Trained model active — Checkpoint loaded. Real predictions enabled.
        </div>""", unsafe_allow_html=True)


# ── Core Analysis Functions ──

def run_image_analysis(pil_image: Image.Image, filename: str = "", detector=None) -> Dict:
    from utils.face_utils import analyze_faces
    from utils.forensics import fft_analysis, compression_analysis, analyze_metadata, skin_tone_analysis
    from utils.visualize import overlay_gradcam, render_fft_heatmap, render_model_bars, render_explainability_chart
    from utils.history import add_to_history
    from utils.social_compression import classify_compression_level, preprocess_for_analysis, compression_adjusted_anomaly
    from utils.explainability import compute_signal_contributions

    compression_info = classify_compression_level(pil_image)
    analysis_image    = preprocess_for_analysis(pil_image, compression_info["level"])

    img_bgr = cv2.cvtColor(np.array(analysis_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    face_data = analyze_faces(img_bgr, detector)
    result = face_data["overall"]
    result["compression_info"] = compression_info
    result["face_count"] = face_data["face_count"]
    result["face_results"] = face_data["face_results"]
    result["annotated_image"] = Image.fromarray(cv2.cvtColor(face_data["annotated"], cv2.COLOR_BGR2RGB))

    try:
        cam = detector.get_gradcam(analysis_image, class_idx=1 if result["is_fake"] else 0)
        result["gradcam"] = overlay_gradcam(pil_image, cam)
        result["cam_raw"] = cam
    except Exception:
        result["gradcam"] = None
        result["cam_raw"] = None

    fft_r  = fft_analysis(pil_image)
    comp_r = compression_analysis(pil_image)
    meta_r = analyze_metadata(pil_image, filename)
    skin_r = skin_tone_analysis(pil_image)
    result["forensics"] = {
        "fft_analysis": fft_r,
        "compression_analysis": comp_r,
        "metadata_analysis": meta_r,
        "skin_analysis": skin_r,
    }

    fake_prob_val = result.get("fake_probability", 0.5)
    comp_level = compression_info["level"]

    adj_fft_anomaly = compression_adjusted_anomaly(fft_r["anomaly_score"], comp_level)
    adj_block_score = comp_r["block_artifact_score"] * (
        0.55 if comp_level == "heavy" else 0.75 if comp_level == "moderate" else 1.0
    )

    signals = []
    if adj_fft_anomaly > 0.72:
        signals.append(("FREQ_ANOMALY", True))
    elif adj_fft_anomaly > 0.60 and fake_prob_val > 0.45:
        signals.append(("MILD_FREQ_ANOMALY", False))
    if adj_block_score > 14.0:
        signals.append(("REENCODING_DETECTED", True))
    if meta_r["risk_score"] > 0.6:
        signals.append(("METADATA_RISK", True))
    elif meta_r["risk_score"] > 0.4:
        signals.append(("MISSING_METADATA", False))
    skin_var   = skin_r.get("skin_variance", 0)
    skin_count = skin_r.get("skin_pixel_count", 0)
    if skin_count > 100:
        if skin_var > 60.0 and fake_prob_val > 0.45:
            signals.append(("SKIN_INCONSISTENCY", True))
        elif skin_var > 48.0:
            signals.append(("MILD_SKIN_VARIATION", False))
        else:
            signals.append(("SKIN_TONE_NORMAL", False))
    if comp_level in ("moderate", "heavy"):
        signals.append((f"COMPRESSION_COMPENSATED_{comp_level.upper()}", False))
    if not signals:
        signals.append(("ALL_CLEAR", False))

    result["fft_anomaly_adjusted"] = adj_fft_anomaly
    result["block_score_adjusted"] = round(adj_block_score, 3)

    result["artifact_signals"] = signals
    result["fft_image"]   = render_fft_heatmap(fft_r)
    result["model_bars"]  = render_model_bars(
        result.get("xception_score", 0.5),
        result.get("efficientnet_score", 0.5),
    )
    result["signal_contributions"] = compute_signal_contributions(result)
    result["explainability_chart"] = render_explainability_chart(result["signal_contributions"])

    thumb = pil_image.copy()
    thumb.thumbnail((80, 80))
    add_to_history(filename or "image", "image", result, thumb)
    return result


def run_video_analysis(video_path: str, filename: str = "", detector=None) -> Dict:
    from utils.video_utils import analyze_video
    from utils.forensics import fft_analysis
    from utils.visualize import render_timeline_chart, render_fft_heatmap, overlay_gradcam, render_identity_chart
    from utils.history import add_to_history

    progress_bar = st.progress(0, text="Scanning frames...")

    def update_progress(p):
        progress_bar.progress(min(p, 0.99), text=f"Scanning frames... {int(p*100)}%")

    result = analyze_video(video_path, detector, progress_callback=update_progress)
    progress_bar.progress(1.0, text="Scan complete")
    time.sleep(0.3)
    progress_bar.empty()

    fake_probs   = result.get("fake_probs_timeline", [])
    frame_results = result.get("frame_results", [])
    timestamps   = [fr["timestamp"] for fr in frame_results]
    result["timeline_image"] = render_timeline_chart(fake_probs, timestamps)
    identity = result.get("identity_consistency", {})
    if identity.get("similarities"):
        result["identity_chart"] = render_identity_chart(identity["similarities"], timestamps)

    peak = result.get("peak_frame")
    if peak is not None:
        pil_peak = Image.fromarray(cv2.cvtColor(peak, cv2.COLOR_BGR2RGB))
        try:
            cam = detector.get_gradcam(pil_peak)
            result["peak_gradcam"] = overlay_gradcam(pil_peak, cam)
            result["peak_pil"] = pil_peak
        except Exception:
            result["peak_gradcam"] = None
        fft_r = fft_analysis(pil_peak)
        result["fft_image"] = render_fft_heatmap(fft_r)
        result["forensics"] = {"fft_analysis": fft_r}

    add_to_history(filename or "video", "video", result)
    return result


# ── Result Renderers ──

def render_image_results(result: Dict, original: Image.Image, filename: str):
    from reports.pdf_report import generate_report

    is_fake     = result["is_fake"]
    confidence  = result["confidence"]
    fake_prob   = result["fake_probability"]
    verdict     = result["verdict"]
    v_color     = "#DC2626" if is_fake else "#16A34A"
    badge       = f'<span class="verdict-{"fake" if is_fake else "real"}">{"⚠ Deepfake detected" if is_fake else "✓ Authentic"}</span>'

    model_status_banner()
    st.markdown(f"<div style='margin-bottom:20px'>{badge}</div>", unsafe_allow_html=True)

    xc  = result.get("xception_score", 0)
    eff = result.get("efficientnet_score", 0)
    vit = result.get("vit_score")
    xc_color  = "#DC2626" if xc  > 0.5 else "#16A34A"
    eff_color = "#DC2626" if eff > 0.5 else "#16A34A"

    vit_cell = ""
    if vit is not None:
        vit_color = "#DC2626" if vit > 0.5 else "#16A34A"
        vit_cell = f'<div class="stat-cell"><div class="sl">ViT</div><div class="sv" style="color:{vit_color}">{vit*100:.1f}%</div></div>'

    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-cell"><div class="sl">Confidence</div><div class="sv" style="color:{v_color}">{confidence*100:.1f}%</div></div>
      <div class="stat-cell"><div class="sl">Fake probability</div><div class="sv" style="color:{v_color}">{fake_prob*100:.1f}%</div></div>
      <div class="stat-cell"><div class="sl">Faces detected</div><div class="sv">{result.get("face_count", 0)}</div></div>
      <div class="stat-cell"><div class="sl">XceptionNet</div><div class="sv" style="color:{xc_color}">{xc*100:.1f}%</div></div>
      <div class="stat-cell"><div class="sl">EfficientNet</div><div class="sv" style="color:{eff_color}">{eff*100:.1f}%</div></div>
      {vit_cell}
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Visual scan", "Forensics", "Model scores", "Face analysis", "Explainability", "Export"
    ])

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="section-label">Original</div>', unsafe_allow_html=True)
            st.image(original, use_container_width=True)
        with c2:
            st.markdown('<div class="section-label">Face detection</div>', unsafe_allow_html=True)
            annotated = result.get("annotated_image")
            if annotated:
                st.image(annotated, use_container_width=True)
        with c3:
            st.markdown('<div class="section-label">Grad-CAM heatmap</div>', unsafe_allow_html=True)
            gradcam = result.get("gradcam")
            if gradcam:
                st.image(gradcam, use_container_width=True)
                st.caption("Red = manipulation signal · Green = authentic")

        st.markdown('<div class="section-label">Artifact signals</div>', unsafe_allow_html=True)
        signals   = result.get("artifact_signals", [])
        tags_html = ""
        for sig, is_warn in signals:
            cls  = "tag-warn" if is_warn else "tag-ok"
            icon = "⚠" if is_warn else "✓"
            tags_html += f'<span class="{cls}">{icon} {sig}</span> '
        st.markdown(tags_html, unsafe_allow_html=True)

    with tab2:
        forensics = result.get("forensics", {})
        fft_r     = forensics.get("fft_analysis", {})
        comp_r    = forensics.get("compression_analysis", {})
        meta_r    = forensics.get("metadata_analysis", {})
        skin_r    = forensics.get("skin_analysis", {})

        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown('<div class="section-label">FFT frequency analysis</div>', unsafe_allow_html=True)
            fft_img = result.get("fft_image")
            if fft_img:
                st.image(fft_img, use_container_width=True)
            if fft_r:
                score = fft_r.get("anomaly_score", 0)
                s_color = "#DC2626" if score > 0.55 else "#D97706" if score > 0.35 else "#16A34A"
                st.markdown(f"""
                <div class="data-block">
                  HF ratio: {fft_r.get('hf_ratio','—')}<br>
                  Mid-band: {fft_r.get('mid_energy_ratio','—')}<br>
                  Anomaly score: <span style="color:{s_color};font-weight:700">{score:.3f}</span>
                </div>""", unsafe_allow_html=True)

        with fc2:
            st.markdown('<div class="section-label">Metadata forensics</div>', unsafe_allow_html=True)
            findings = meta_r.get("findings", [])
            if findings:
                for f in findings:
                    st.markdown(f'<div class="tag-warn" style="display:block;margin-bottom:4px">⚠ {f}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="tag-ok">✓ No metadata anomalies</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-label">Compression analysis</div>', unsafe_allow_html=True)
            block_score = comp_r.get("block_artifact_score", 0)
            re_enc      = comp_r.get("re_encoding_suspected", False)
            st.markdown(f"""
            <div class="data-block">
              Block artifact: <span style="color:{'#DC2626' if re_enc else '#16A34A'};font-weight:700">{block_score:.3f}</span><br>
              {'<span style="color:#DC2626">⚠ Re-encoding suspected</span>' if re_enc else '<span style="color:#16A34A">✓ Clean compression</span>'}
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-label">Skin tone analysis</div>', unsafe_allow_html=True)
            skin_var  = skin_r.get("skin_variance", 0)
            skin_anom = skin_r.get("anomaly", False)
            st.markdown(f"""
            <div class="data-block">
              Variance: <span style="color:{'#DC2626' if skin_anom else '#16A34A'};font-weight:700">{skin_var:.2f}</span><br>
              {'<span style="color:#DC2626">⚠ Abnormal blending detected</span>' if skin_anom else '<span style="color:#16A34A">✓ Skin tone consistent</span>'}
            </div>""", unsafe_allow_html=True)

            raw_meta = meta_r.get("metadata", {})
            if raw_meta:
                with st.expander("View raw EXIF"):
                    for k, v in list(raw_meta.items())[:20]:
                        st.markdown(f'<span style="font-family:JetBrains Mono,monospace;font-size:11px;color:#6B7280">{k}:</span> <span style="font-family:JetBrains Mono,monospace;font-size:11px">{v}</span>', unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:16px">Recompression analysis</div>', unsafe_allow_html=True)
        comp_info   = result.get("compression_info", {})
        level       = comp_info.get("level", "none")
        level_color = {"none": "#16A34A", "light": "#65A30D", "moderate": "#D97706", "heavy": "#DC2626"}.get(level, "#6B7280")
        platform    = comp_info.get("platform_match")
        st.markdown(f"""
        <div class="data-block">
          Recompression level: <span style="color:{level_color};font-weight:700">{level.upper()}</span><br>
          Estimated JPEG quality: {comp_info.get('estimated_quality','—')}<br>
          {"Platform match: " + platform + "<br>" if platform else ""}
          EXIF present: {"Yes" if comp_info.get('has_exif') else "No"}
        </div>""", unsafe_allow_html=True)
        if level in ("moderate", "heavy"):
            reduction = "45%" if level == "heavy" else "25%"
            st.markdown(f'<div class="tag-warn" style="margin-top:6px">⚠ Recompression detected — FFT/compression anomaly scores reduced by {reduction} to avoid false positives from platform re-encoding</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="tag-ok" style="margin-top:6px">✓ No significant recompression detected</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-label">Ensemble model breakdown</div>', unsafe_allow_html=True)
        model_bars = result.get("model_bars")
        if model_bars:
            st.image(model_bars, use_container_width=True)

        vit_line = f"ViT × 0.25 + " if vit is not None else ""
        st.markdown(f"""
        <div class="data-block" style="margin-top:12px">
          Ensemble → XceptionNet × {"0.40" if vit is not None else "0.55"} + EfficientNet-B4 × {"0.35" if vit is not None else "0.45"} + {vit_line if vit is not None else ""}...<br>
          Threshold → 50% fake probability<br>
          Verdict → {verdict.upper()} @ {confidence*100:.1f}%
        </div>""", unsafe_allow_html=True)

    with tab4:
        face_results = result.get("face_results", [])
        fc = result.get("face_count", 0)
        if fc == 0:
            st.markdown('<div class="tag-ok">No faces detected — full image analyzed</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="data-block" style="margin-top:12px">
              Fake probability: <span style="color:{v_color}">{fake_prob*100:.2f}%</span><br>
              Real probability: {(1-fake_prob)*100:.2f}%
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="section-label">{fc} face(s) detected</div>', unsafe_allow_html=True)
            for i, fr in enumerate(face_results):
                f_color = "#DC2626" if fr["verdict"] == "Deepfake" else "#16A34A"
                with st.expander(f"Face #{i+1} — {fr['verdict']} ({fr['confidence']*100:.1f}%)"):
                    vit_s = fr.get("vit_score")
                    vit_face_cell = f'<div class="stat-cell"><div class="sl">ViT</div><div class="sv" style="color:{f_color}">{vit_s*100:.1f}%</div></div>' if vit_s is not None else ""
                    st.markdown(f"""
                    <div class="stat-row">
                      <div class="stat-cell"><div class="sl">Verdict</div><div class="sv" style="color:{f_color};font-size:14px">{fr['verdict']}</div></div>
                      <div class="stat-cell"><div class="sl">Fake prob</div><div class="sv" style="color:{f_color}">{fr['fake_probability']*100:.1f}%</div></div>
                      <div class="stat-cell"><div class="sl">XceptionNet</div><div class="sv">{fr.get('xception_score',0)*100:.1f}%</div></div>
                      <div class="stat-cell"><div class="sl">EfficientNet</div><div class="sv">{fr.get('efficientnet_score',0)*100:.1f}%</div></div>
                      {vit_face_cell}
                    </div>""", unsafe_allow_html=True)

    with tab5:
        st.markdown('<div class="section-label">What drove this verdict</div>', unsafe_allow_html=True)
        contributions = result.get("signal_contributions", [])
        chart = result.get("explainability_chart")
        if chart:
            st.image(chart, use_container_width=True)

        if contributions:
            top = contributions[0]
            top_desc = "Ensemble model score" if top["category"] == "model" else "Forensic signal (advisory — not blended into probability)"
            st.markdown(f"""
            <div class="data-block" style="margin-top:8px">
              Strongest signal: <b>{top['name']}</b> ({top['contribution_pct']}% of total signal weight)<br>
              Category: {top_desc}
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-label" style="margin-top:16px">Full breakdown</div>', unsafe_allow_html=True)
            tags_html = ""
            for s in contributions:
                cls  = "tag-model" if s["category"] == "model" else "tag-forensic"
                icon = "🧠" if s["category"] == "model" else "🔍"
                tags_html += f'<span class="{cls}">{icon} {s["name"]} — {s["contribution_pct"]}%</span> '
            st.markdown(tags_html, unsafe_allow_html=True)

            st.caption(
                "Model scores are mathematically blended into the final probability. "
                "Forensic signals (FFT, metadata, compression, skin tone) are advisory "
                "corroborating evidence shown here for transparency — they inform the "
                "artifact tags but do not change the ensemble probability."
            )
        else:
            st.info("No signal data available for this result.")
    
    with tab6:
        st.markdown('<div class="section-label">Export forensic report</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        include_gradcam = col_a.checkbox("Include Grad-CAM", value=True)
        include_fft     = col_b.checkbox("Include FFT heatmap", value=True)
        if st.button("Generate PDF report", type="primary"):
            with st.spinner("Compiling report..."):
                try:
                    pdf_bytes = generate_report(
                        result=result,
                        original_image=original,
                        gradcam_image=result.get("gradcam") if include_gradcam else None,
                        fft_image=result.get("fft_image") if include_fft else None,
                        filename=filename,
                        analysis_type="image",
                    )
                    st.download_button("Download report", pdf_bytes,
                        file_name=f"deepshield_{filename}.pdf", mime="application/pdf")
                except Exception as e:
                    st.error(f"Report failed: {e}")


# ── Video Results Renderer ──

def render_video_results(result: Dict, filename: str):
    from reports.pdf_report import generate_report

    if "error" in result:
        st.error(result["error"])
        return

    is_fake    = result["is_fake"]
    confidence = result["confidence"]
    verdict    = result["verdict"]
    v_color    = "#DC2626" if is_fake else "#16A34A"
    badge      = f'<span class="verdict-{"fake" if is_fake else "real"}">{"⚠ Deepfake detected" if is_fake else "✓ Authentic video"}</span>'

    model_status_banner()
    st.markdown(f"<div style='margin-bottom:20px'>{badge}</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-cell"><div class="sl">Verdict</div><div class="sv" style="color:{v_color}">{verdict}</div></div>
      <div class="stat-cell"><div class="sl">Confidence</div><div class="sv" style="color:{v_color}">{confidence*100:.1f}%</div></div>
      <div class="stat-cell"><div class="sl">Frames</div><div class="sv">{result.get('frames_analyzed','—')}</div></div>
      <div class="stat-cell"><div class="sl">Duration</div><div class="sv">{result.get('video_duration',0):.1f}s</div></div>
      <div class="stat-cell"><div class="sl">FPS</div><div class="sv">{result.get('fps','—')}</div></div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Timeline", "Temporal analysis", "Peak frame", "Export"])

    with tab1:
        st.markdown('<div class="section-label">Frame-by-frame confidence</div>', unsafe_allow_html=True)
        timeline_img = result.get("timeline_image")
        if timeline_img:
            st.image(timeline_img, use_container_width=True)
        st.markdown("""
        <div style="margin-top:8px">
          <span class="tag-ok">< 35% Real</span>
          <span class="tag-warn">35–55% Uncertain</span>
          <span class="verdict-fake" style="font-size:11px;padding:2px 10px">> 55% Deepfake</span>
        </div>""", unsafe_allow_html=True)

        frame_results = result.get("frame_results", [])
        if frame_results:
            with st.expander("Per-frame data"):
                import pandas as pd
                df = pd.DataFrame([{
                    "Frame":        fr["frame_idx"],
                    "Time (s)":     fr["timestamp"],
                    "Verdict":      fr["verdict"],
                    "Fake %":       round(fr["fake_probability"] * 100, 1),
                    "Confidence %": round(fr["confidence"] * 100, 1),
                } for fr in frame_results])
                st.dataframe(df, use_container_width=True)

    with tab2:
        temporal = result.get("temporal", {})
        blink    = result.get("blink_analysis", {})
        av       = result.get("av_sync")

        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown('<div class="section-label">Temporal consistency</div>', unsafe_allow_html=True)
            consistency = temporal.get("consistency", 0)
            c_color = "#16A34A" if consistency > 0.7 else "#D97706" if consistency > 0.4 else "#DC2626"
            st.markdown(f"""
            <div class="data-block">
              Variance: {temporal.get('variance','—')}<br>
              Mean shift: {temporal.get('mean_frame_diff','—')}<br>
              Consistency: <span style="color:{c_color};font-weight:700">{consistency:.2f}</span>
            </div>""", unsafe_allow_html=True)
            spikes = temporal.get("spikes", [])
            if spikes:
                st.markdown(f'<div class="tag-warn" style="margin-top:8px">⚠ {len(spikes)} spike(s) at frames: {spikes[:8]}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="tag-ok" style="margin-top:8px">✓ No confidence spikes</div>', unsafe_allow_html=True)

        with tc2:
            st.markdown('<div class="section-label">Blink pattern</div>', unsafe_allow_html=True)
            blink_anom = blink.get("blink_anomaly", False)
            st.markdown(f"""
            <div class="data-block">
              Blink events: {blink.get('blink_events','—')}<br>
              Expected: {blink.get('expected_blinks','—')}<br>
              Avg eyes/frame: {blink.get('avg_eyes_per_frame','—')}
            </div>""", unsafe_allow_html=True)
            if blink_anom:
                st.markdown('<div class="tag-warn" style="margin-top:8px">⚠ Abnormal blink pattern</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="tag-ok" style="margin-top:8px">✓ Blink pattern normal</div>', unsafe_allow_html=True)

        # Face identity consistency
        identity = result.get("identity_consistency", {})
        if identity.get("similarities"):
            st.markdown('<div class="section-label">Face identity consistency</div>', unsafe_allow_html=True)
            id_score = identity.get("identity_score", 1.0)
            id_anom  = identity.get("identity_anomaly", False)
            id_color = "#DC2626" if id_anom else "#16A34A"
            chart    = result.get("identity_chart")
            ic1, ic2 = st.columns([2, 1])
            with ic1:
                if chart:
                    st.image(chart, use_container_width=True)
            with ic2:
                st.markdown(f"""
                <div class="data-block">
                  Identity score: <span style="color:{id_color};font-weight:700">{id_score:.2f}</span><br>
                  Avg similarity: {identity.get('avg_similarity','—')}<br>
                  Drift events: {identity.get('drift_count', 0)}<br>
                  Faces tracked: {identity.get('faces_found', 0)}
                </div>""", unsafe_allow_html=True)
                if id_anom:
                    st.markdown('<div class="tag-warn" style="margin-top:8px">⚠ Identity drift detected — possible face-swap</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="tag-ok" style="margin-top:8px">✓ Identity stable across frames</div>', unsafe_allow_html=True)
        elif identity.get("note"):
            st.caption(f"Identity consistency: {identity['note']}")

        # AV Sync
        if av and av.get("correlation") is not None:
            st.markdown('<div class="section-label">Audio-visual sync</div>', unsafe_allow_html=True)
            sync   = av.get("sync_score", 0.5)
            desync = av.get("is_desynced", False)
            s_color = "#DC2626" if desync else "#16A34A"
            st.markdown(f"""
            <div class="data-block">
              Sync score: <span style="color:{s_color};font-weight:700">{sync:.2f}</span><br>
              Correlation: {av.get('correlation','—')}<br>
              Frames analyzed: {av.get('frames_analyzed','—')}<br>
              {'<span style="color:#DC2626">⚠ Audio-visual desync detected</span>' if desync else '<span style="color:#16A34A">✓ Audio-visual sync normal</span>'}
            </div>""", unsafe_allow_html=True)
        elif av and av.get("note"):
            st.caption(f"AV sync: {av['note']}")

    with tab3:
        peak_pil    = result.get("peak_pil")
        peak_gradcam = result.get("peak_gradcam")
        peak_idx    = result.get("peak_frame_idx", 0)
        fake_probs  = result.get("fake_probs_timeline", [])
        peak_prob   = fake_probs[min(peak_idx, len(fake_probs)-1)] if fake_probs else 0
        st.markdown(f'<div class="section-label">Peak manipulation — frame #{peak_idx * 8} — {peak_prob*100:.1f}% fake probability</div>', unsafe_allow_html=True)
        pc1, pc2 = st.columns(2)
        if peak_pil:
            pc1.image(peak_pil, caption="Peak frame", use_container_width=True)
        if peak_gradcam:
            pc2.image(peak_gradcam, caption="Grad-CAM overlay", use_container_width=True)
        fft_img = result.get("fft_image")
        if fft_img:
            st.markdown('<div class="section-label">FFT — peak frame</div>', unsafe_allow_html=True)
            st.image(fft_img, width=350)

    with tab4:
        st.markdown('<div class="section-label">Export report</div>', unsafe_allow_html=True)
        if st.button("Generate PDF report", type="primary", key="video_pdf"):
            with st.spinner("Compiling report..."):
                try:
                    pdf_bytes = generate_report(
                        result=result,
                        original_image=result.get("peak_pil"),
                        gradcam_image=result.get("peak_gradcam"),
                        timeline_image=result.get("timeline_image"),
                        fft_image=result.get("fft_image"),
                        filename=filename,
                        analysis_type="video",
                    )
                    st.download_button("Download report", pdf_bytes,
                        file_name=f"deepshield_video_{filename}.pdf", mime="application/pdf")
                except Exception as e:
                    st.error(f"Report failed: {e}")


# ── Pages ──

def page_image():
    st.markdown('<div class="scan-header">Image analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="scan-sub">Upload an image to run XceptionNet + EfficientNet-B4 + ViT ensemble detection with Grad-CAM forensics.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp", "bmp"], label_visibility="collapsed")

    if uploaded:
        file_id = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("last_image_id") != file_id:
            st.session_state.pop("image_result", None)
            st.session_state.pop("image_pil", None)
            st.session_state["last_image_id"] = file_id

        pil_img = Image.open(uploaded).convert("RGB")
        st.session_state["image_pil"] = pil_img
        st.image(pil_img, width=360)

        if st.button("Run scan", type="primary"):
            detector = load_detector()
            with st.spinner("Running detection pipeline..."):
                result = run_image_analysis(pil_img, uploaded.name, detector)
            st.session_state["image_result"] = result
            st.session_state["image_filename"] = uploaded.name

    if st.session_state.get("image_result") and st.session_state.get("image_pil"):
        st.markdown("---")
        render_image_results(
            st.session_state["image_result"],
            st.session_state["image_pil"],
            st.session_state.get("image_filename", "image"),
        )


def page_url_analyzer():
    st.markdown('<div class="scan-header">URL analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="scan-sub">Paste a direct image URL to fetch and analyze.</div>', unsafe_allow_html=True)

    url = st.text_input("Image URL", placeholder="https://example.com/face.jpg", label_visibility="collapsed")

    if url and st.button("Fetch & scan", type="primary"):
        import requests
        try:
            with st.spinner("Fetching image..."):
                resp = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                resp.raise_for_status()
                pil_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            st.image(pil_img, caption=f"Fetched: {url[:60]}", width=360)
            detector = load_detector()
            fname = url.split("/")[-1][:40] or "url_image"
            with st.spinner("Running detection..."):
                result = run_image_analysis(pil_img, fname, detector)
            st.session_state["url_result"]   = result
            st.session_state["url_pil"]      = pil_img
            st.session_state["url_filename"] = fname
        except Exception as e:
            st.error(f"Could not fetch image: {e}")
            st.caption("Make sure the URL points directly to an image file (.jpg/.png/.webp)")

    if st.session_state.get("url_result") and st.session_state.get("url_pil"):
        st.markdown("---")
        render_image_results(
            st.session_state["url_result"],
            st.session_state["url_pil"],
            st.session_state.get("url_filename", "url_image"),
        )


def page_video():
    st.markdown('<div class="scan-header">Video analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="scan-sub">Frame-by-frame detection with temporal consistency, blink pattern, and audio-visual sync analysis.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv", "webm"], label_visibility="collapsed")
    col1, col2 = st.columns(2)
    sample_rate = col1.slider("Sample every Nth frame", 4, 20, 8)
    max_frames  = col2.slider("Max frames to analyze", 20, 100, 50)

    if uploaded:
        file_id = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("last_video_id") != file_id:
            st.session_state.pop("video_result", None)
            st.session_state["last_video_id"] = file_id

        st.markdown(f'<div class="tag-ok" style="margin-bottom:12px">✓ Loaded: {uploaded.name} ({uploaded.size // 1024} KB)</div>', unsafe_allow_html=True)

        if st.button("Run video scan", type="primary"):
            detector = load_detector()
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            try:
                result = run_video_analysis(tmp_path, uploaded.name, detector)
                st.session_state["video_result"]   = result
                st.session_state["video_filename"] = uploaded.name
            finally:
                os.unlink(tmp_path)

    if st.session_state.get("video_result"):
        st.markdown("---")
        render_video_results(
            st.session_state["video_result"],
            st.session_state.get("video_filename", "video"),
        )


def page_batch():
    st.markdown('<div class="scan-header">Batch scan</div>', unsafe_allow_html=True)
    st.markdown('<div class="scan-sub">Analyze up to 20 images simultaneously. Results exported as CSV.</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload images (up to 20)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and st.button("Run batch scan", type="primary"):
        if len(uploaded_files) > 20:
            st.warning("Max 20 images — processing first 20.")
            uploaded_files = uploaded_files[:20]

        detector = load_detector()
        rows     = []
        progress = st.progress(0, text="Initializing...")

        for i, f in enumerate(uploaded_files):
            progress.progress((i + 1) / len(uploaded_files), text=f"Scanning {f.name}... [{i+1}/{len(uploaded_files)}]")
            pil_img = Image.open(f).convert("RGB")
            try:
                result = run_image_analysis(pil_img, f.name, detector)
                rows.append({
                    "File":          f.name,
                    "Verdict":       result["verdict"],
                    "Confidence %":  round(result["confidence"] * 100, 1),
                    "Fake %":        round(result["fake_probability"] * 100, 1),
                    "Faces":         result.get("face_count", 0),
                    "XceptionNet %": round(result.get("xception_score", 0) * 100, 1),
                    "EfficientNet %":round(result.get("efficientnet_score", 0) * 100, 1),
                })
            except Exception:
                rows.append({"File": f.name, "Verdict": "ERROR", "Confidence %": 0,
                    "Fake %": 0, "Faces": 0, "XceptionNet %": 0, "EfficientNet %": 0})

        progress.progress(1.0, text="Batch scan complete")

        import pandas as pd
        df         = pd.DataFrame(rows)
        fake_count = (df["Verdict"] == "Deepfake").sum()
        real_count = (df["Verdict"] == "Real").sum()

        st.markdown(f"""
        <div class="stat-row" style="margin-top:16px">
          <div class="stat-cell"><div class="sl">Total</div><div class="sv">{len(rows)}</div></div>
          <div class="stat-cell"><div class="sl">Deepfakes</div><div class="sv" style="color:#DC2626">{fake_count}</div></div>
          <div class="stat-cell"><div class="sl">Authentic</div><div class="sv" style="color:#16A34A">{real_count}</div></div>
          <div class="stat-cell"><div class="sl">Detection rate</div><div class="sv" style="color:#D97706">{fake_count/len(rows)*100:.0f}%</div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)
        csv_bytes = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv_bytes, file_name="deepshield_batch.csv", mime="text/csv")


def page_gradcam():
    st.markdown('<div class="scan-header">Grad-CAM studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="scan-sub">Visualize which facial regions triggered the detection signal.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed", key="gradcam_upload")

    if uploaded:
        from utils.visualize import overlay_gradcam, make_comparison_image
        pil_img      = Image.open(uploaded).convert("RGB")
        col_s1, col_s2 = st.columns(2)
        alpha        = col_s1.slider("Overlay intensity", 0.2, 0.8, 0.45, 0.05)
        target_class = col_s2.selectbox("Target class", ["Fake regions", "Real regions"])
        class_idx    = 1 if target_class == "Fake regions" else 0

        if st.button("Generate Grad-CAM", type="primary"):
            detector = load_detector()
            with st.spinner("Computing activation maps..."):
                cam        = detector.get_gradcam(pil_img, class_idx=class_idx)
                overlay    = overlay_gradcam(pil_img, cam, alpha=alpha)
                comparison = make_comparison_image(pil_img, overlay)

            st.markdown('<div class="section-label">Original vs Grad-CAM</div>', unsafe_allow_html=True)
            st.image(comparison, use_container_width=True)
            st.caption("Left: Original · Right: Activation heatmap (red = high manipulation signal)")

            col_d1, col_d2 = st.columns(2)
            buf1 = io.BytesIO()
            overlay.save(buf1, format="PNG")
            col_d1.download_button("Download heatmap", buf1.getvalue(),
                file_name="gradcam.png", mime="image/png")
            buf2 = io.BytesIO()
            comparison.save(buf2, format="PNG")
            col_d2.download_button("Download comparison", buf2.getvalue(),
                file_name="gradcam_comparison.png", mime="image/png")


def page_history():
    from utils.history import render_history_panel
    st.markdown('<div class="scan-header">Analysis history</div>', unsafe_allow_html=True)
    st.markdown('<div class="scan-sub">All analyses from the current session.</div>', unsafe_allow_html=True)
    render_history_panel()


def page_about():
    st.markdown('<div class="scan-header">About DeepShield</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="max-width: 680px">
      <div class="section-label">Architecture</div>
      <div class="data-block">
        Primary model &nbsp;&nbsp;&nbsp; XceptionNet (transfer learning)<br>
        Secondary model &nbsp;&nbsp; EfficientNet-B4<br>
        Third model &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Vision Transformer (ViT-B/16)<br>
        Ensemble &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 40% Xception + 35% EfficientNet + 25% ViT<br>
        Calibration &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Temperature scaling (ECE ~0.005)<br>
        Explainability &nbsp;&nbsp; Grad-CAM<br>
        Face detection &nbsp;&nbsp; InsightFace + Haar cascade fallback<br>
        Forensics &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; FFT · EXIF · Compression · Skin tone
      </div>

      <div class="section-label" style="margin-top:20px">Capabilities</div>
      <div class="data-block">
        ✓ Image deepfake detection (GAN, diffusion, face-swap)<br>
        ✓ Video frame-by-frame analysis<br>
        ✓ Audio-visual sync detection for video<br>
        ✓ URL image fetching + analysis<br>
        ✓ Multi-face detection with individual scoring<br>
        ✓ Grad-CAM visualization studio<br>
        ✓ FFT frequency domain forensics<br>
        ✓ EXIF metadata forensics<br>
        ✓ Compression artifact detection<br>
        ✓ Skin tone consistency analysis<br>
        ✓ Eye blink anomaly detection (video)<br>
        ✓ Temporal consistency scoring (video)<br>
        ✓ Batch processing up to 20 images<br>
        ✓ REST API (FastAPI, /docs for Swagger UI)<br>
        ✓ PDF forensic report export<br>
        ✓ CSV batch results export
      </div>

      <div class="section-label" style="margin-top:20px">Performance</div>
      <div class="data-block">
        Val accuracy (own dataset) &nbsp; 97–98%<br>
        Diffusion faces (held-out) &nbsp;&nbsp; 100% (200 samples)<br>
        DFDC face-swap (unseen) &nbsp;&nbsp;&nbsp; 90.1% accuracy, AUC 0.9517<br>
        Confidence calibration ECE &nbsp;&nbsp; ~0.005<br>
        GPU inference (RTX 3050) &nbsp;&nbsp;&nbsp; ~40ms/image
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Main Navigation ──

DETECTION_PAGES = ["Image analysis", "Video analysis", "URL analyzer", "Batch scan", "Grad-CAM studio"]
TOOL_PAGES      = ["History log", "About"]
PAGE_KEYS       = {p.upper(): p for p in DETECTION_PAGES + TOOL_PAGES}

PAGE_ICONS = {
    "Image analysis":  "🖼️",
    "Video analysis":  "🎥",
    "URL analyzer":    "🔗",
    "Batch scan":      "📊",
    "Grad-CAM studio": "🔥",
    "History log":     "🕘",
    "About":           "ℹ️",
}


def main():
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Image analysis"

    current = st.session_state["current_page"]

    with st.sidebar:
        st.markdown('<div class="ds-logo">Deep<span>Shield</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="ds-version">Forensic Detection System v3.0</div>', unsafe_allow_html=True)
        st.markdown("---")

        st.markdown('<div class="ds-nav-section">Detection</div>', unsafe_allow_html=True)
        for p in DETECTION_PAGES:
            is_active = current == p
            icon      = PAGE_ICONS.get(p, "")
            label     = f"{'▸ ' if is_active else '  '}{icon}  {p}"
            if st.sidebar.button(label, key=f"nav_{p}", use_container_width=True):
                st.session_state["current_page"] = p
                st.rerun()

        st.markdown('<div class="ds-nav-section">Tools</div>', unsafe_allow_html=True)
        for p in TOOL_PAGES:
            is_active = current == p
            icon      = PAGE_ICONS.get(p, "")
            label     = f"{'▸ ' if is_active else '  '}{icon}  {p}"
            if st.sidebar.button(label, key=f"nav_{p}", use_container_width=True):
                st.session_state["current_page"] = p
                st.rerun()

        st.markdown("---")
        import torch
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        try:
            _det     = load_detector()
            _trained = getattr(_det, 'is_trained', False)
            _vit     = getattr(_det, 'vit_loaded', False)
        except Exception:
            _trained = False
            _vit     = False

        status_color = "#16A34A" if _trained else "#D97706"
        status_text  = "Trained" if _trained else "Demo mode"
        vit_status   = "Loaded" if _vit else "Not trained"
        vit_color    = "#16A34A" if _vit else "#9CA3AF"

        st.markdown(f"""
        <div style="font-size:11px;line-height:2.2;color:#6B7280">
          <div style="font-size:10px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px">System status</div>
          Device &nbsp;&nbsp;&nbsp;&nbsp; <span style="color:#1A1D23">{device}</span><br>
          Model &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:{status_color};font-weight:600">{status_text}</span><br>
          XceptionNet &nbsp; <span style="color:#16A34A">Loaded</span><br>
          EfficientNet &nbsp; <span style="color:#16A34A">Loaded</span><br>
          ViT &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:{vit_color}">{vit_status}</span>
        </div>""", unsafe_allow_html=True)

    page_map = {
        "Image analysis":  page_image,
        "Video analysis":  page_video,
        "URL analyzer":    page_url_analyzer,
        "Batch scan":      page_batch,
        "Grad-CAM studio": page_gradcam,
        "History log":     page_history,
        "About":           page_about,
    }

    page_fn = page_map.get(current)
    if page_fn:
        page_fn()


if __name__ == "__main__":
    main()
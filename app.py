"""
DeepShield — Forensic Deepfake Detection System
Redesigned UI: dark forensic terminal aesthetic
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
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0A0E14 !important;
    color: #C9D1D9 !important;
  }
  .stApp { background-color: #0A0E14 !important; }

  section[data-testid="stSidebar"] {
    background: #0D1117 !important;
    border-right: 1px solid #1E2A38 !important;
  }
  section[data-testid="stSidebar"] * { color: #C9D1D9 !important; }

  .ds-logo { font-family: 'Space Mono', monospace; font-size: 20px; font-weight: 700; color: #00FF9C !important; letter-spacing: 0.05em; margin-bottom: 4px; }
  .ds-version { font-family: 'Space Mono', monospace; font-size: 10px; color: #4A6274 !important; letter-spacing: 0.1em; }
  .ds-nav-section { font-family: 'Space Mono', monospace; font-size: 10px; color: #4A6274 !important; letter-spacing: 0.15em; text-transform: uppercase; margin: 20px 0 6px; }

  .verdict-fake { display: inline-block; background: #1A0A0A; color: #FF3B3B; border: 1px solid #FF3B3B; padding: 4px 18px; border-radius: 4px; font-family: 'Space Mono', monospace; font-weight: 700; font-size: 13px; letter-spacing: 0.1em; }
  .verdict-real { display: inline-block; background: #0A1A12; color: #00FF9C; border: 1px solid #00FF9C; padding: 4px 18px; border-radius: 4px; font-family: 'Space Mono', monospace; font-weight: 700; font-size: 13px; letter-spacing: 0.1em; }

  .tag-warn { display: inline-block; background: #1A1200; color: #FFB800; border: 1px solid #FFB800; padding: 2px 10px; border-radius: 4px; font-family: 'Space Mono', monospace; font-size: 11px; margin: 2px; }
  .tag-ok { display: inline-block; background: #0A1A12; color: #00FF9C; border: 1px solid #1E3A2A; padding: 2px 10px; border-radius: 4px; font-family: 'Space Mono', monospace; font-size: 11px; margin: 2px; }

  .section-label { font-family: 'Space Mono', monospace; font-size: 10px; letter-spacing: 0.15em; text-transform: uppercase; color: #4A6274; margin: 16px 0 8px; border-bottom: 1px solid #1E2A38; padding-bottom: 6px; }
  .scan-header { font-family: 'Space Mono', monospace; font-size: 22px; font-weight: 700; color: #C9D1D9; letter-spacing: 0.02em; margin-bottom: 4px; }
  .scan-sub { font-family: 'Inter', sans-serif; font-size: 13px; color: #4A6274; margin-bottom: 20px; }

  .demo-banner { background: #1A1200; border: 1px solid #FFB800; border-left: 3px solid #FFB800; border-radius: 6px; padding: 10px 14px; font-family: 'Space Mono', monospace; font-size: 11px; color: #FFB800; margin-bottom: 16px; }
  .trained-banner { background: #0A1A12; border: 1px solid #00FF9C; border-left: 3px solid #00FF9C; border-radius: 6px; padding: 10px 14px; font-family: 'Space Mono', monospace; font-size: 11px; color: #00FF9C; margin-bottom: 16px; }

  .stat-row { display: flex; gap: 1px; background: #1E2A38; border-radius: 8px; overflow: hidden; margin-bottom: 16px; }
  .stat-cell { flex: 1; background: #0D1117; padding: 14px 16px; text-align: center; }
  .stat-cell .sl { font-family: 'Space Mono', monospace; font-size: 9px; letter-spacing: 0.15em; color: #4A6274; text-transform: uppercase; margin-bottom: 4px; }
  .stat-cell .sv { font-family: 'Space Mono', monospace; font-size: 20px; font-weight: 700; }

  .stButton > button { background: #0A1A12 !important; color: #00FF9C !important; border: 1px solid #00FF9C !important; border-radius: 6px !important; font-family: 'Space Mono', monospace !important; font-size: 12px !important; letter-spacing: 0.08em !important; }
  .stButton > button:hover { background: #00FF9C !important; color: #0A0E14 !important; }
  .stButton > button[kind="primary"] { background: #00FF9C !important; color: #0A0E14 !important; font-weight: 700 !important; }

  .stTabs [data-baseweb="tab-list"] { background: #0D1117 !important; border-bottom: 1px solid #1E2A38 !important; }
  .stTabs [data-baseweb="tab"] { font-family: 'Space Mono', monospace !important; font-size: 11px !important; letter-spacing: 0.08em !important; color: #4A6274 !important; }
  .stTabs [aria-selected="true"] { color: #00FF9C !important; border-bottom: 2px solid #00FF9C !important; }

  .stTextInput input { background: #0D1117 !important; border-color: #1E2A38 !important; color: #C9D1D9 !important; font-family: 'Space Mono', monospace !important; }
  .stProgress [data-baseweb="progress-bar"] > div { background: #00FF9C !important; }
  .stProgress [data-baseweb="progress-bar"] { background: #1E2A38 !important; }
  .stExpander { border: 1px solid #1E2A38 !important; border-radius: 8px !important; background: #0D1117 !important; }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { background: transparent !important; }
    header [data-testid="stToolbar"] { visibility: hidden; }
    [data-testid="collapsedControl"],
    button[kind="header"] { 
        display: none !important; 
        pointer-events: none !important;
    }
    section[data-testid="stSidebar"] {
        min-width: 240px !important;
        max-width: 240px !important;
        transform: none !important;
    }
</style>
""", unsafe_allow_html=True)
# ── Model Loader ──

@st.cache_resource(show_spinner="Initializing detection models...")
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
          ⚠ DEMO MODE — No trained checkpoint loaded. Predictions are random (~50%).
          Drop .pth files into checkpoints/ to enable real detection.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="trained-banner">
          ✓ TRAINED MODEL ACTIVE — Checkpoint loaded. Real predictions enabled.
        </div>""", unsafe_allow_html=True)


# ── Core Analysis Functions ──

def run_image_analysis(pil_image: Image.Image, filename: str = "", detector=None) -> Dict:
    from utils.face_utils import analyze_faces
    from utils.forensics import fft_analysis, compression_analysis, analyze_metadata, skin_tone_analysis
    from utils.visualize import overlay_gradcam, render_fft_heatmap, render_model_bars
    from utils.history import add_to_history

    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    face_data = analyze_faces(img_bgr, detector)
    result = face_data["overall"]
    result["face_count"] = face_data["face_count"]
    result["face_results"] = face_data["face_results"]
    result["annotated_image"] = Image.fromarray(cv2.cvtColor(face_data["annotated"], cv2.COLOR_BGR2RGB))

    try:
        cam = detector.get_gradcam(pil_image, class_idx=1 if result["is_fake"] else 0)
        result["gradcam"] = overlay_gradcam(pil_image, cam)
        result["cam_raw"] = cam
    except Exception:
        result["gradcam"] = None
        result["cam_raw"] = None

    fft_r = fft_analysis(pil_image)
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
    signals = []
    if fft_r["anomaly_score"] > 0.72:
        signals.append(("FREQ_ANOMALY", True))
    elif fft_r["anomaly_score"] > 0.60 and fake_prob_val > 0.45:
        signals.append(("MILD_FREQ_ANOMALY", False))
    if comp_r["block_artifact_score"] > 14.0:
        signals.append(("REENCODING_DETECTED", True))
    if meta_r["risk_score"] > 0.6:
        signals.append(("METADATA_RISK", True))
    elif meta_r["risk_score"] > 0.4:
        signals.append(("MISSING_METADATA", False))
    skin_var = skin_r.get("skin_variance", 0)
    skin_count = skin_r.get("skin_pixel_count", 0)
    if skin_count > 100:
        if skin_var > 60.0 and fake_prob_val > 0.45:
            signals.append(("SKIN_INCONSISTENCY", True))
        elif skin_var > 48.0:
            signals.append(("MILD_SKIN_VARIATION", False))
        else:
            signals.append(("SKIN_TONE_NORMAL", False))
    if not signals:
        signals.append(("ALL_CLEAR", False))

    result["artifact_signals"] = signals
    result["fft_image"] = render_fft_heatmap(fft_r)
    result["model_bars"] = render_model_bars(
        result.get("xception_score", 0.5),
        result.get("efficientnet_score", 0.5),
    )

    thumb = pil_image.copy()
    thumb.thumbnail((80, 80))
    add_to_history(filename or "image", "image", result, thumb)
    return result


def run_video_analysis(video_path: str, filename: str = "", detector=None) -> Dict:
    from utils.video_utils import analyze_video
    from utils.forensics import fft_analysis
    from utils.visualize import render_timeline_chart, render_fft_heatmap, overlay_gradcam
    from utils.history import add_to_history

    progress_bar = st.progress(0, text="SCANNING FRAMES...")

    def update_progress(p):
        progress_bar.progress(min(p, 0.99), text=f"SCANNING FRAMES... {int(p*100)}%")

    result = analyze_video(video_path, detector, progress_callback=update_progress)
    progress_bar.progress(1.0, text="SCAN COMPLETE")
    time.sleep(0.4)
    progress_bar.empty()

    fake_probs = result.get("fake_probs_timeline", [])
    frame_results = result.get("frame_results", [])
    timestamps = [fr["timestamp"] for fr in frame_results]
    result["timeline_image"] = render_timeline_chart(fake_probs, timestamps)

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

    is_fake = result["is_fake"]
    confidence = result["confidence"]
    fake_prob = result["fake_probability"]
    verdict = result["verdict"]
    verdict_color = "#FF3B3B" if is_fake else "#00FF9C"
    badge = f'<span class="verdict-{"fake" if is_fake else "real"}">{"⚠ DEEPFAKE DETECTED" if is_fake else "✓ AUTHENTIC"}</span>'

    model_status_banner()
    st.markdown(f"<div style='margin-bottom:16px'>{badge}</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-cell"><div class="sl">Confidence</div><div class="sv" style="color:{verdict_color}">{confidence*100:.1f}%</div></div>
      <div class="stat-cell"><div class="sl">Fake Probability</div><div class="sv" style="color:{verdict_color}">{fake_prob*100:.1f}%</div></div>
      <div class="stat-cell"><div class="sl">Faces Detected</div><div class="sv">{result.get("face_count", 0)}</div></div>
      <div class="stat-cell"><div class="sl">XceptionNet</div><div class="sv" style="color:{'#FF3B3B' if result.get('xception_score',0)>0.5 else '#00FF9C'}">{result.get('xception_score',0)*100:.1f}%</div></div>
      <div class="stat-cell"><div class="sl">EfficientNet</div><div class="sv" style="color:{'#FF3B3B' if result.get('efficientnet_score',0)>0.5 else '#00FF9C'}">{result.get('efficientnet_score',0)*100:.1f}%</div></div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "VISUAL SCAN", "FORENSICS", "MODEL SCORES", "FACE ANALYSIS", "EXPORT"
    ])

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="section-label">Original</div>', unsafe_allow_html=True)
            st.image(original, use_container_width=True)
        with c2:
            st.markdown('<div class="section-label">Face Detection</div>', unsafe_allow_html=True)
            annotated = result.get("annotated_image")
            if annotated:
                st.image(annotated, use_container_width=True)
        with c3:
            st.markdown('<div class="section-label">Grad-CAM Heatmap</div>', unsafe_allow_html=True)
            gradcam = result.get("gradcam")
            if gradcam:
                st.image(gradcam, use_container_width=True)
                st.caption("RED = manipulation · GREEN = authentic")

        st.markdown('<div class="section-label">Artifact Signals</div>', unsafe_allow_html=True)
        signals = result.get("artifact_signals", [])
        tags_html = ""
        for sig, is_warn in signals:
            cls = "tag-warn" if is_warn else "tag-ok"
            icon = "⚠" if is_warn else "✓"
            tags_html += f'<span class="{cls}">{icon} {sig}</span> '
        st.markdown(tags_html, unsafe_allow_html=True)

    with tab2:
        forensics = result.get("forensics", {})
        fft_r = forensics.get("fft_analysis", {})
        comp_r = forensics.get("compression_analysis", {})
        meta_r = forensics.get("metadata_analysis", {})
        skin_r = forensics.get("skin_analysis", {})

        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown('<div class="section-label">FFT Frequency Analysis</div>', unsafe_allow_html=True)
            fft_img = result.get("fft_image")
            if fft_img:
                st.image(fft_img, use_container_width=True)
            if fft_r:
                score = fft_r.get("anomaly_score", 0)
                color = "#FF3B3B" if score > 0.55 else "#FFB800" if score > 0.35 else "#00FF9C"
                st.markdown(f"""
                <div style="font-family:'Space Mono',monospace;font-size:12px;margin-top:8px;line-height:2">
                  HF RATIO: <span style="color:#C9D1D9">{fft_r.get('hf_ratio','—')}</span><br>
                  MID-BAND: <span style="color:#C9D1D9">{fft_r.get('mid_energy_ratio','—')}</span><br>
                  ANOMALY SCORE: <span style="color:{color};font-weight:700">{score:.3f}</span>
                </div>""", unsafe_allow_html=True)

        with fc2:
            st.markdown('<div class="section-label">Metadata Forensics</div>', unsafe_allow_html=True)
            findings = meta_r.get("findings", [])
            if findings:
                for f in findings:
                    st.markdown(f'<div class="tag-warn" style="display:block;margin-bottom:4px">⚠ {f}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="tag-ok">✓ No metadata anomalies</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-label">Compression Analysis</div>', unsafe_allow_html=True)
            block_score = comp_r.get("block_artifact_score", 0)
            re_enc = comp_r.get("re_encoding_suspected", False)
            st.markdown(f"""
            <div style="font-family:'Space Mono',monospace;font-size:12px;line-height:2">
              BLOCK ARTIFACT: <span style="color:{'#FF3B3B' if re_enc else '#00FF9C'};font-weight:700">{block_score:.3f}</span><br>
              {'<span style="color:#FF3B3B">⚠ RE-ENCODING SUSPECTED</span>' if re_enc else '<span style="color:#00FF9C">✓ Clean compression</span>'}
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-label">Skin Tone Analysis</div>', unsafe_allow_html=True)
            skin_var = skin_r.get("skin_variance", 0)
            skin_anom = skin_r.get("anomaly", False)
            st.markdown(f"""
            <div style="font-family:'Space Mono',monospace;font-size:12px;line-height:2">
              VARIANCE: <span style="color:{'#FF3B3B' if skin_anom else '#00FF9C'};font-weight:700">{skin_var:.2f}</span><br>
              {'<span style="color:#FF3B3B">⚠ Abnormal blending detected</span>' if skin_anom else '<span style="color:#00FF9C">✓ Skin tone consistent</span>'}
            </div>""", unsafe_allow_html=True)

            raw_meta = meta_r.get("metadata", {})
            if raw_meta:
                with st.expander("VIEW RAW EXIF"):
                    for k, v in list(raw_meta.items())[:20]:
                        st.markdown(f'<span style="font-family:Space Mono,monospace;font-size:11px;color:#4A6274">{k}:</span> <span style="font-family:Space Mono,monospace;font-size:11px;color:#C9D1D9">{v}</span>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-label">Ensemble Model Breakdown</div>', unsafe_allow_html=True)
        model_bars = result.get("model_bars")
        if model_bars:
            st.image(model_bars, use_container_width=True)
        st.markdown(f"""
        <div style="font-family:'Space Mono',monospace;font-size:11px;color:#4A6274;margin-top:12px;line-height:2">
          ENSEMBLE → XceptionNet × 0.55 + EfficientNet-B4 × 0.45<br>
          THRESHOLD → 50% fake probability<br>
          VERDICT → {verdict.upper()} @ {confidence*100:.1f}%
        </div>""", unsafe_allow_html=True)

    with tab4:
        face_results = result.get("face_results", [])
        fc = result.get("face_count", 0)
        if fc == 0:
            st.markdown('<div class="tag-ok">No faces detected — full image analyzed</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Space Mono',monospace;font-size:13px;margin-top:12px">
              FAKE PROB: <span style="color:{verdict_color}">{fake_prob*100:.2f}%</span> &nbsp;&nbsp;
              REAL PROB: <span style="color:#4A6274">{(1-fake_prob)*100:.2f}%</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="font-family:Space Mono,monospace;font-size:12px;color:#4A6274;margin-bottom:12px">{fc} FACE(S) DETECTED</div>', unsafe_allow_html=True)
            for i, fr in enumerate(face_results):
                f_color = "#FF3B3B" if fr["verdict"] == "Deepfake" else "#00FF9C"
                with st.expander(f"FACE #{i+1} — {fr['verdict'].upper()} ({fr['confidence']*100:.1f}%)"):
                    st.markdown(f"""
                    <div class="stat-row">
                      <div class="stat-cell"><div class="sl">Verdict</div><div class="sv" style="color:{f_color};font-size:14px">{fr['verdict'].upper()}</div></div>
                      <div class="stat-cell"><div class="sl">Fake Prob</div><div class="sv" style="color:{f_color}">{fr['fake_probability']*100:.1f}%</div></div>
                      <div class="stat-cell"><div class="sl">XceptionNet</div><div class="sv">{fr.get('xception_score',0)*100:.1f}%</div></div>
                      <div class="stat-cell"><div class="sl">EfficientNet</div><div class="sv">{fr.get('efficientnet_score',0)*100:.1f}%</div></div>
                    </div>""", unsafe_allow_html=True)

    with tab5:
        st.markdown('<div class="section-label">Export Forensic Report</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        include_gradcam = col_a.checkbox("Include Grad-CAM", value=True)
        include_fft = col_b.checkbox("Include FFT Heatmap", value=True)
        if st.button("GENERATE PDF REPORT", type="primary"):
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
                    st.download_button("DOWNLOAD REPORT", pdf_bytes,
                        file_name=f"deepshield_{filename}.pdf", mime="application/pdf")
                except Exception as e:
                    st.error(f"Report failed: {e}")
                    # ── Video Results Renderer ──

def render_video_results(result: Dict, filename: str):
    from reports.pdf_report import generate_report

    if "error" in result:
        st.error(result["error"])
        return

    is_fake = result["is_fake"]
    confidence = result["confidence"]
    verdict = result["verdict"]
    verdict_color = "#FF3B3B" if is_fake else "#00FF9C"
    badge = f'<span class="verdict-{"fake" if is_fake else "real"}">{"⚠ DEEPFAKE DETECTED" if is_fake else "✓ AUTHENTIC VIDEO"}</span>'

    model_status_banner()
    st.markdown(f"<div style='margin-bottom:16px'>{badge}</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-cell"><div class="sl">Verdict</div><div class="sv" style="color:{verdict_color}">{verdict.upper()}</div></div>
      <div class="stat-cell"><div class="sl">Confidence</div><div class="sv" style="color:{verdict_color}">{confidence*100:.1f}%</div></div>
      <div class="stat-cell"><div class="sl">Frames</div><div class="sv">{result.get('frames_analyzed','—')}</div></div>
      <div class="stat-cell"><div class="sl">Duration</div><div class="sv">{result.get('video_duration',0):.1f}s</div></div>
      <div class="stat-cell"><div class="sl">FPS</div><div class="sv">{result.get('fps','—')}</div></div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["TIMELINE", "TEMPORAL ANALYSIS", "PEAK FRAME", "EXPORT"])

    with tab1:
        st.markdown('<div class="section-label">Frame-by-frame Confidence</div>', unsafe_allow_html=True)
        timeline_img = result.get("timeline_image")
        if timeline_img:
            st.image(timeline_img, use_container_width=True)
        st.markdown("""
        <div style="margin-top:8px">
          <span class="tag-ok">< 35% REAL</span>
          <span class="tag-warn">35–55% UNCERTAIN</span>
          <span class="verdict-fake" style="font-size:11px;padding:2px 10px">> 55% DEEPFAKE</span>
        </div>""", unsafe_allow_html=True)

        frame_results = result.get("frame_results", [])
        if frame_results:
            with st.expander("VIEW PER-FRAME DATA"):
                import pandas as pd
                df = pd.DataFrame([{
                    "Frame": fr["frame_idx"],
                    "Time (s)": fr["timestamp"],
                    "Verdict": fr["verdict"],
                    "Fake %": round(fr["fake_probability"] * 100, 1),
                    "Confidence %": round(fr["confidence"] * 100, 1),
                } for fr in frame_results])
                st.dataframe(df, use_container_width=True)

    with tab2:
        temporal = result.get("temporal", {})
        blink = result.get("blink_analysis", {})
        tc1, tc2 = st.columns(2)

        with tc1:
            st.markdown('<div class="section-label">Temporal Consistency</div>', unsafe_allow_html=True)
            consistency = temporal.get("consistency", 0)
            c_color = "#00FF9C" if consistency > 0.7 else "#FFB800" if consistency > 0.4 else "#FF3B3B"
            st.markdown(f"""
            <div style="font-family:'Space Mono',monospace;font-size:12px;line-height:2">
              VARIANCE: <span style="color:#C9D1D9">{temporal.get('variance','—')}</span><br>
              MEAN SHIFT: <span style="color:#C9D1D9">{temporal.get('mean_frame_diff','—')}</span><br>
              CONSISTENCY: <span style="color:{c_color};font-weight:700">{consistency:.2f}</span>
            </div>""", unsafe_allow_html=True)
            spikes = temporal.get("spikes", [])
            if spikes:
                st.markdown(f'<div class="tag-warn" style="margin-top:8px">⚠ {len(spikes)} SPIKE(S) AT FRAMES: {spikes[:8]}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="tag-ok" style="margin-top:8px">✓ No confidence spikes</div>', unsafe_allow_html=True)

        with tc2:
            st.markdown('<div class="section-label">Blink Pattern Analysis</div>', unsafe_allow_html=True)
            blink_anom = blink.get("blink_anomaly", False)
            st.markdown(f"""
            <div style="font-family:'Space Mono',monospace;font-size:12px;line-height:2">
              BLINK EVENTS: <span style="color:#C9D1D9">{blink.get('blink_events','—')}</span><br>
              EXPECTED: <span style="color:#C9D1D9">{blink.get('expected_blinks','—')}</span><br>
              AVG EYES/FRAME: <span style="color:#C9D1D9">{blink.get('avg_eyes_per_frame','—')}</span>
            </div>""", unsafe_allow_html=True)
            if blink_anom:
                st.markdown('<div class="tag-warn" style="margin-top:8px">⚠ ABNORMAL BLINK PATTERN</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="tag-ok" style="margin-top:8px">✓ Blink pattern normal</div>', unsafe_allow_html=True)

    with tab3:
        peak_pil = result.get("peak_pil")
        peak_gradcam = result.get("peak_gradcam")
        peak_idx = result.get("peak_frame_idx", 0)
        fake_probs = result.get("fake_probs_timeline", [])
        peak_prob = fake_probs[min(peak_idx, len(fake_probs)-1)] if fake_probs else 0
        st.markdown(f'<div style="font-family:Space Mono,monospace;font-size:11px;color:#4A6274;margin-bottom:12px">PEAK MANIPULATION — Frame #{peak_idx * 8} — {peak_prob*100:.1f}% fake probability</div>', unsafe_allow_html=True)
        pc1, pc2 = st.columns(2)
        if peak_pil:
            pc1.image(peak_pil, caption="Peak Frame", use_container_width=True)
        if peak_gradcam:
            pc2.image(peak_gradcam, caption="Grad-CAM Overlay", use_container_width=True)
        fft_img = result.get("fft_image")
        if fft_img:
            st.markdown('<div class="section-label">FFT — Peak Frame</div>', unsafe_allow_html=True)
            st.image(fft_img, width=350)

    with tab4:
        st.markdown('<div class="section-label">Export Report</div>', unsafe_allow_html=True)
        if st.button("GENERATE PDF REPORT", type="primary", key="video_pdf"):
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
                    st.download_button("DOWNLOAD REPORT", pdf_bytes,
                        file_name=f"deepshield_video_{filename}.pdf", mime="application/pdf")
                except Exception as e:
                    st.error(f"Report failed: {e}")
                    # ── Pages ──

def page_image():
    st.markdown('<div class="scan-header">Image Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="scan-sub">Upload an image to run XceptionNet + EfficientNet-B4 ensemble detection with Grad-CAM forensics.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp", "bmp"], label_visibility="collapsed")

    if uploaded:
        file_id = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("last_image_id") != file_id:
            st.session_state.pop("image_result", None)
            st.session_state.pop("image_pil", None)
            st.session_state["last_image_id"] = file_id

        pil_img = Image.open(uploaded).convert("RGB")
        st.session_state["image_pil"] = pil_img
        st.image(pil_img, width=380)

        if st.button("RUN SCAN", type="primary"):
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
    st.markdown('<div class="scan-header">URL Image Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="scan-sub">Paste a direct image URL to fetch and analyze. Replaces webcam — more practical for real-world deepfake investigation.</div>', unsafe_allow_html=True)

    url = st.text_input("Image URL", placeholder="https://example.com/face.jpg", label_visibility="collapsed")

    if url and st.button("FETCH & SCAN", type="primary"):
        import requests
        try:
            with st.spinner("Fetching image..."):
                resp = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                resp.raise_for_status()
                pil_img = Image.open(io.BytesIO(resp.content)).convert("RGB")

            st.image(pil_img, caption=f"Fetched: {url[:60]}", width=380)
            detector = load_detector()
            fname = url.split("/")[-1][:40] or "url_image"
            with st.spinner("Running detection..."):
                result = run_image_analysis(pil_img, fname, detector)
            st.session_state["url_result"] = result
            st.session_state["url_pil"] = pil_img
            st.session_state["url_filename"] = fname

        except Exception as e:
            st.error(f"Could not fetch image: {e}")
            st.markdown("""
            <div style="font-family:'Space Mono',monospace;font-size:11px;color:#4A6274;margin-top:8px">
              TIPS: URL must point directly to an image file (.jpg/.png/.webp)<br>
              Try right-clicking an image in your browser → Copy Image Address
            </div>""", unsafe_allow_html=True)

    if st.session_state.get("url_result") and st.session_state.get("url_pil"):
        st.markdown("---")
        render_image_results(
            st.session_state["url_result"],
            st.session_state["url_pil"],
            st.session_state.get("url_filename", "url_image"),
        )


def page_video():
    st.markdown('<div class="scan-header">Video Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="scan-sub">Frame-by-frame detection with temporal consistency and blink pattern analysis.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv", "webm"], label_visibility="collapsed")
    col1, col2 = st.columns(2)
    sample_rate = col1.slider("Sample every Nth frame", 4, 20, 8)
    max_frames = col2.slider("Max frames to analyze", 20, 100, 50)

    if uploaded:
        file_id = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("last_video_id") != file_id:
            st.session_state.pop("video_result", None)
            st.session_state["last_video_id"] = file_id

        st.markdown(f'<div class="tag-ok" style="margin-bottom:12px">✓ Loaded: {uploaded.name} ({uploaded.size // 1024} KB)</div>', unsafe_allow_html=True)

        if st.button("RUN VIDEO SCAN", type="primary"):
            detector = load_detector()
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            try:
                result = run_video_analysis(tmp_path, uploaded.name, detector)
                st.session_state["video_result"] = result
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
    st.markdown('<div class="scan-header">Batch Scan</div>', unsafe_allow_html=True)
    st.markdown('<div class="scan-sub">Analyze up to 20 images simultaneously. Results exported as ranked CSV.</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload images (up to 20)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and st.button("RUN BATCH SCAN", type="primary"):
        if len(uploaded_files) > 20:
            st.warning("Max 20 images — processing first 20.")
            uploaded_files = uploaded_files[:20]

        detector = load_detector()
        rows = []
        progress = st.progress(0, text="INITIALIZING...")

        for i, f in enumerate(uploaded_files):
            progress.progress((i + 1) / len(uploaded_files), text=f"SCANNING {f.name}... [{i+1}/{len(uploaded_files)}]")
            pil_img = Image.open(f).convert("RGB")
            try:
                result = run_image_analysis(pil_img, f.name, detector)
                rows.append({
                    "File": f.name,
                    "Verdict": result["verdict"],
                    "Confidence %": round(result["confidence"] * 100, 1),
                    "Fake %": round(result["fake_probability"] * 100, 1),
                    "Faces": result.get("face_count", 0),
                    "XceptionNet %": round(result.get("xception_score", 0) * 100, 1),
                    "EfficientNet %": round(result.get("efficientnet_score", 0) * 100, 1),
                })
            except Exception:
                rows.append({"File": f.name, "Verdict": "ERROR", "Confidence %": 0,
                    "Fake %": 0, "Faces": 0, "XceptionNet %": 0, "EfficientNet %": 0})

        progress.progress(1.0, text="BATCH SCAN COMPLETE")

        import pandas as pd
        df = pd.DataFrame(rows)
        fake_count = (df["Verdict"] == "Deepfake").sum()
        real_count = (df["Verdict"] == "Real").sum()

        st.markdown(f"""
        <div class="stat-row" style="margin-top:16px">
          <div class="stat-cell"><div class="sl">Total</div><div class="sv">{len(rows)}</div></div>
          <div class="stat-cell"><div class="sl">Deepfakes</div><div class="sv" style="color:#FF3B3B">{fake_count}</div></div>
          <div class="stat-cell"><div class="sl">Authentic</div><div class="sv" style="color:#00FF9C">{real_count}</div></div>
          <div class="stat-cell"><div class="sl">Detection Rate</div><div class="sv" style="color:#FFB800">{fake_count/len(rows)*100:.0f}%</div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)
        csv_bytes = df.to_csv(index=False).encode()
        st.download_button("DOWNLOAD CSV", csv_bytes, file_name="deepshield_batch.csv", mime="text/csv")


def page_gradcam():
    st.markdown('<div class="scan-header">Grad-CAM Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="scan-sub">Visualize which facial regions triggered the detection signal.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed", key="gradcam_upload")

    if uploaded:
        from utils.visualize import overlay_gradcam, make_comparison_image
        pil_img = Image.open(uploaded).convert("RGB")
        col_s1, col_s2 = st.columns(2)
        alpha = col_s1.slider("Overlay intensity", 0.2, 0.8, 0.45, 0.05)
        target_class = col_s2.selectbox("Target class", ["Fake regions", "Real regions"])
        class_idx = 1 if target_class == "Fake regions" else 0

        if st.button("GENERATE GRAD-CAM", type="primary"):
            detector = load_detector()
            with st.spinner("Computing activation maps..."):
                cam = detector.get_gradcam(pil_img, class_idx=class_idx)
                overlay = overlay_gradcam(pil_img, cam, alpha=alpha)
                comparison = make_comparison_image(pil_img, overlay)

            st.markdown('<div class="section-label">Original vs Grad-CAM</div>', unsafe_allow_html=True)
            st.image(comparison, use_container_width=True)
            st.caption("LEFT: Original · RIGHT: Activation heatmap (RED = high manipulation signal)")

            col_d1, col_d2 = st.columns(2)
            buf1 = io.BytesIO()
            overlay.save(buf1, format="PNG")
            col_d1.download_button("DOWNLOAD HEATMAP", buf1.getvalue(),
                file_name="gradcam.png", mime="image/png")
            buf2 = io.BytesIO()
            comparison.save(buf2, format="PNG")
            col_d2.download_button("DOWNLOAD COMPARISON", buf2.getvalue(),
                file_name="gradcam_comparison.png", mime="image/png")


def page_history():
    from utils.history import render_history_panel
    st.markdown('<div class="scan-header">Analysis History</div>', unsafe_allow_html=True)
    st.markdown('<div class="scan-sub">All analyses from the current session.</div>', unsafe_allow_html=True)
    render_history_panel()


def page_about():
    st.markdown('<div class="scan-header">About DeepShield</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:12px;line-height:2;color:#4A6274">
    <div style="color:#00FF9C;font-size:14px;margin-bottom:12px">ARCHITECTURE</div>
    PRIMARY MODEL &nbsp;&nbsp;&nbsp; XceptionNet (transfer learning)<br>
    SECONDARY MODEL &nbsp;&nbsp; EfficientNet-B4<br>
    ENSEMBLE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 55% Xception + 45% EfficientNet<br>
    EXPLAINABILITY &nbsp;&nbsp; Grad-CAM<br>
    FACE DETECTION &nbsp;&nbsp; OpenCV Haar Cascade<br>
    FORENSICS &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; FFT · EXIF · Compression · Skin Tone<br>
    <div style="color:#00FF9C;font-size:14px;margin:16px 0 12px">CAPABILITIES</div>
    ✓ Image deepfake detection<br>
    ✓ Video frame-by-frame analysis<br>
    ✓ URL image fetching + analysis<br>
    ✓ Multi-face detection & individual scoring<br>
    ✓ Grad-CAM visualization studio<br>
    ✓ FFT frequency domain forensics<br>
    ✓ EXIF metadata forensics<br>
    ✓ Compression artifact detection<br>
    ✓ Skin tone consistency analysis<br>
    ✓ Eye blink anomaly detection (video)<br>
    ✓ Temporal consistency scoring (video)<br>
    ✓ Batch processing up to 20 images<br>
    ✓ PDF forensic report export<br>
    ✓ CSV batch results export<br>
    </div>
    """, unsafe_allow_html=True)


# ── Main Navigation ──

DETECTION_PAGES = ["IMAGE ANALYSIS", "VIDEO ANALYSIS", "URL ANALYZER", "BATCH SCAN", "GRAD-CAM STUDIO"]
TOOL_PAGES = ["HISTORY LOG", "ABOUT"]


def main():
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "IMAGE ANALYSIS"

    current = st.session_state["current_page"]

    with st.sidebar:
        st.markdown('<div class="ds-logo">DeepShield</div>', unsafe_allow_html=True)
        st.markdown('<div class="ds-version">FORENSIC DETECTION SYSTEM v2.0</div>', unsafe_allow_html=True)
        st.markdown("---")

        st.markdown('<div class="ds-nav-section">Detection Modules</div>', unsafe_allow_html=True)
        for p in DETECTION_PAGES:
            is_active = current == p
            if st.sidebar.button(
                f"{'▶ ' if is_active else '  '}{p}",
                key=f"nav_{p}", use_container_width=True,
            ):
                st.session_state["current_page"] = p
                st.rerun()

        st.markdown('<div class="ds-nav-section">Tools</div>', unsafe_allow_html=True)
        for p in TOOL_PAGES:
            is_active = current == p
            if st.sidebar.button(
                f"{'▶ ' if is_active else '  '}{p}",
                key=f"nav_{p}", use_container_width=True,
            ):
                st.session_state["current_page"] = p
                st.rerun()

        st.markdown("---")
        import torch
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        try:
            _det = load_detector()
            _trained = getattr(_det, 'is_trained', False)
        except Exception:
            _trained = False
        status_color = "#00FF9C" if _trained else "#FFB800"
        status_text = "TRAINED" if _trained else "DEMO MODE"
        st.markdown(f"""
        <div style="font-family:'Space Mono',monospace;font-size:10px;line-height:2">
          <div style="color:#4A6274">SYSTEM STATUS</div>
          DEVICE &nbsp; <span style="color:#C9D1D9">{device}</span><br>
          MODEL &nbsp;&nbsp; <span style="color:{status_color}">{status_text}</span><br>
          XCEPTION &nbsp; <span style="color:#00FF9C">LOADED</span><br>
          EFFICIENT &nbsp; <span style="color:#00FF9C">LOADED</span>
        </div>""", unsafe_allow_html=True)

    page_map = {
        "IMAGE ANALYSIS": page_image,
        "VIDEO ANALYSIS": page_video,
        "URL ANALYZER": page_url_analyzer,
        "BATCH SCAN": page_batch,
        "GRAD-CAM STUDIO": page_gradcam,
        "HISTORY LOG": page_history,
        "ABOUT": page_about,
    }

    page_fn = page_map.get(current)
    if page_fn:
        page_fn()


if __name__ == "__main__":
    main()
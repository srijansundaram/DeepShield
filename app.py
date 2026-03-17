"""
DeepShield — Main Streamlit Application
Full deepfake detection system with all features
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
from typing import Optional, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ──────────────────────────────────────────────
# Page Config (must be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="DeepShield — Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #F8F7F4;
    border-right: 1px solid #E5E3DC;
  }
  section[data-testid="stSidebar"] .stRadio label {
    font-size: 14px !important;
    padding: 6px 0;
  }

  /* Cards */
  .ds-card {
    background: #fff;
    border: 1px solid #E5E3DC;
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 14px;
  }
  .ds-metric {
    background: #F8F7F4;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
  }
  .ds-metric .label {
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #888780;
    margin-bottom: 4px;
  }
  .ds-metric .value {
    font-size: 26px;
    font-weight: 600;
    line-height: 1.1;
  }
  .ds-metric .sub {
    font-size: 12px;
    color: #888780;
    margin-top: 2px;
  }

  /* Verdict badges */
  .verdict-fake {
    display: inline-block;
    background: #FCEBEB;
    color: #A32D2D;
    border: 1px solid #F09595;
    padding: 5px 16px;
    border-radius: 99px;
    font-weight: 600;
    font-size: 14px;
  }
  .verdict-real {
    display: inline-block;
    background: #E1F5EE;
    color: #0F6E56;
    border: 1px solid #5DCAA5;
    padding: 5px 16px;
    border-radius: 99px;
    font-weight: 600;
    font-size: 14px;
  }
  .tag-warn {
    display: inline-block;
    background: #FAEEDA;
    color: #854F0B;
    border: 1px solid #EF9F27;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 12px;
    margin: 2px;
  }
  .tag-ok {
    display: inline-block;
    background: #E1F5EE;
    color: #0F6E56;
    border: 1px solid #5DCAA5;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 12px;
    margin: 2px;
  }
  .section-header {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #888780;
    margin: 18px 0 8px;
  }
  .confidence-bar-track {
    background: #F1EFE8;
    border-radius: 4px;
    height: 8px;
    margin-top: 8px;
    overflow: hidden;
  }

  /* Upload area */
  [data-testid="stFileUploader"] {
    border: 1.5px dashed #C8C5BC !important;
    border-radius: 12px !important;
    padding: 10px !important;
  }

  /* Streamlit branding hide */
  #MainMenu, footer, header { visibility: hidden; }

  /* Tab styling */
  .stTabs [data-baseweb="tab"] {
    font-size: 13px;
    font-weight: 500;
  }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Lazy imports (avoid loading heavy deps on every page)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading detection models…")
def load_detector():
    import torch
    import random
    import numpy as np
    from models.detector import EnsembleDetector
    # Fix random seed so untrained EfficientNet weights are always identical
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        device = "cuda"
    else:
        device = "cpu"
    print(f"Loading detector on {device}")
    return EnsembleDetector(device=device)


# ──────────────────────────────────────────────
# Helper: Run full analysis pipeline on image
# ──────────────────────────────────────────────
def run_image_analysis(pil_image: Image.Image, filename: str = "", detector=None) -> Dict:
    from utils.face_utils import analyze_faces
    from utils.forensics import fft_analysis, compression_analysis, analyze_metadata, skin_tone_analysis
    from utils.visualize import overlay_gradcam, render_fft_heatmap, render_model_bars, render_gauge
    from utils.history import add_to_history

    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

    # Face analysis + deepfake detection
    face_data = analyze_faces(img_bgr, detector)
    result = face_data["overall"]
    result["face_count"] = face_data["face_count"]
    result["face_results"] = face_data["face_results"]
    result["annotated_image"] = Image.fromarray(cv2.cvtColor(face_data["annotated"], cv2.COLOR_BGR2RGB))

    # Grad-CAM
    try:
        cam = detector.get_gradcam(pil_image, class_idx=1 if result["is_fake"] else 0)
        result["gradcam"] = overlay_gradcam(pil_image, cam)
        result["cam_raw"] = cam
    except Exception as e:
        result["gradcam"] = None
        result["cam_raw"] = None

    # Forensics
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

    # Artifact signals — correlated with model confidence
    is_fake_result = result.get("is_fake", False)
    fake_prob_val  = result.get("fake_probability", 0.5)
    signals = []

    # FFT: only flag if score is high AND model also suspects fake
    if fft_r["anomaly_score"] > 0.72:
        signals.append(("Frequency anomaly", True))
    elif fft_r["anomaly_score"] > 0.60 and fake_prob_val > 0.45:
        signals.append(("Mild frequency anomaly", False))

    # Compression
    if comp_r["block_artifact_score"] > 14.0:
        signals.append(("Re-encoding detected", True))

    # Metadata
    if meta_r["risk_score"] > 0.6:
        signals.append(("Metadata risk", True))
    elif meta_r["risk_score"] > 0.4:
        signals.append(("Missing metadata", False))

    # Skin tone — only flag when model also suspects manipulation
    skin_var = skin_r.get("skin_variance", 0)
    skin_count = skin_r.get("skin_pixel_count", 0)
    if skin_count > 100:  # only analyze if enough skin pixels detected
        if skin_var > 60.0 and fake_prob_val > 0.45:
            signals.append(("Skin tone inconsistency", True))
        elif skin_var > 48.0 and fake_prob_val > 0.55:
            signals.append(("Mild skin variation", False))
        else:
            signals.append(("Skin tone: normal", False))

    # Clean bill if no warnings
    warn_count = sum(1 for _, w in signals if w)
    if warn_count == 0 and not signals:
        signals.append(("No forensic anomalies", False))
    elif warn_count == 0:
        signals.append(("Forensics: clean", False))

    result["artifact_signals"] = signals

    # Visualizations
    result["fft_image"] = render_fft_heatmap(fft_r)
    result["model_bars"] = render_model_bars(
        result.get("xception_score", 0.5),
        result.get("efficientnet_score", 0.5),
    )

    # History
    thumb = pil_image.copy()
    thumb.thumbnail((80, 80))
    add_to_history(filename or "image", "image", result, thumb)

    return result


def run_video_analysis(video_path: str, filename: str = "", detector=None) -> Dict:
    from utils.video_utils import analyze_video
    from utils.forensics import fft_analysis
    from utils.visualize import render_timeline_chart, render_fft_heatmap, overlay_gradcam
    from utils.history import add_to_history

    progress_bar = st.progress(0, text="Analyzing frames…")

    def update_progress(p):
        progress_bar.progress(min(p, 0.99), text=f"Analyzing frames… {int(p*100)}%")

    result = analyze_video(video_path, detector, progress_callback=update_progress)
    progress_bar.progress(1.0, text="Done!")
    time.sleep(0.4)
    progress_bar.empty()

    # Timeline chart
    fake_probs = result.get("fake_probs_timeline", [])
    frame_results = result.get("frame_results", [])
    timestamps = [fr["timestamp"] for fr in frame_results]
    result["timeline_image"] = render_timeline_chart(fake_probs, timestamps)

    # Grad-CAM on peak frame
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


# ──────────────────────────────────────────────
# Render results — Image
# ──────────────────────────────────────────────
def render_image_results(result: Dict, original: Image.Image, filename: str):
    from reports.pdf_report import generate_report

    is_fake = result["is_fake"]
    confidence = result["confidence"]
    fake_prob = result["fake_probability"]
    verdict = result["verdict"]
    badge = f'<span class="verdict-{"fake" if is_fake else "real"}">{verdict}</span>'

    # ── Untrained model warning ──
    from models.detector import MODEL_IS_TRAINED
    if not MODEL_IS_TRAINED:
        st.warning(
            "⚠️ **Untrained model** — No checkpoint is loaded. "
            "The model has random weights so predictions are meaningless (random ~50%). "
            "Train the model on FaceForensics++ or load a checkpoint into `checkpoints/` "
            "before relying on any results. See the README for instructions.",
            icon="🧪",
        )

    st.markdown(f"### Result &nbsp;&nbsp; {badge}", unsafe_allow_html=True)
    st.markdown("---")

    # Top metrics
    m1, m2, m3, m4 = st.columns(4)
    verdict_color = "#E24B4A" if is_fake else "#1D9E75"
    for col, label, val, sub, color in [
        (m1, "Verdict", verdict, f"{'Manipulation found' if is_fake else 'No manipulation'}", verdict_color),
        (m2, "Confidence", f"{confidence*100:.1f}%", "Ensemble model", verdict_color),
        (m3, "Fake probability", f"{fake_prob*100:.1f}%", "Weighted average", "#444441"),
        (m4, "Faces detected", str(result.get("face_count", 0)), "In image", "#444441"),
    ]:
        col.markdown(f"""
        <div class="ds-metric">
          <div class="label">{label}</div>
          <div class="value" style="color:{color}">{val}</div>
          <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🖼️ Visual Analysis", "🔬 Forensics", "📊 Model Scores", "👤 Face Results", "📄 Export Report"]
    )

    with tab1:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.markdown('<div class="section-header">Original Image</div>', unsafe_allow_html=True)
            st.image(original, use_container_width=True)
        with c2:
            st.markdown('<div class="section-header">Annotated (Multi-face)</div>', unsafe_allow_html=True)
            annotated = result.get("annotated_image")
            if annotated:
                st.image(annotated, use_container_width=True)
        with c3:
            st.markdown('<div class="section-header">Grad-CAM Heatmap</div>', unsafe_allow_html=True)
            gradcam = result.get("gradcam")
            if gradcam:
                st.image(gradcam, use_container_width=True)
                st.caption("🔴 Red = high manipulation probability · 🟢 Green = authentic regions")
            else:
                st.info("GradCAM not available for this image.")

        # Artifact signals
        st.markdown('<div class="section-header">Artifact Signals</div>', unsafe_allow_html=True)
        signals = result.get("artifact_signals", [])
        tags_html = ""
        for sig, is_warn in signals:
            cls = "tag-warn" if is_warn else "tag-ok"
            icon = "⚠️" if is_warn else "✅"
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
            st.markdown("**🌊 FFT Frequency Analysis**")
            fft_img = result.get("fft_image")
            if fft_img:
                st.image(fft_img, use_container_width=True)
            if fft_r:
                st.metric("High-freq energy ratio", fft_r.get("hf_ratio", "—"))
                st.metric("Mid-band ratio", fft_r.get("mid_energy_ratio", "—"))
                score = fft_r.get("anomaly_score", 0)
                color = "🔴" if score > 0.55 else "🟡" if score > 0.35 else "🟢"
                st.markdown(f"Anomaly score: {color} **{score:.3f}**")

        with fc2:
            st.markdown("**🗂️ Metadata Forensics**")
            findings = meta_r.get("findings", [])
            if findings:
                for f in findings:
                    st.warning(f"⚠️ {f}")
            else:
                st.success("✅ No metadata anomalies detected")

            st.markdown("**🧱 Compression Analysis**")
            block_score = comp_r.get("block_artifact_score", 0)
            re_enc = comp_r.get("re_encoding_suspected", False)
            st.metric("Block artifact score", f"{block_score:.3f}")
            if re_enc:
                st.warning("⚠️ Re-encoding suspected — possible manipulation")
            else:
                st.success("✅ No re-encoding artifacts detected")

            st.markdown("**🎨 Skin Tone Analysis**")
            skin_var = skin_r.get("skin_variance", 0)
            skin_anom = skin_r.get("anomaly", False)
            st.metric("Skin tone variance", f"{skin_var:.2f}")
            if skin_anom:
                st.warning("⚠️ Abnormal skin tone variation detected")
            else:
                st.success("✅ Skin tone appears consistent")

            raw_meta = meta_r.get("metadata", {})
            if raw_meta:
                with st.expander("View raw EXIF metadata"):
                    for k, v in list(raw_meta.items())[:20]:
                        st.text(f"{k}: {v}")

    with tab3:
        st.markdown("**Model Ensemble Scores**")
        model_bars = result.get("model_bars")
        if model_bars:
            st.image(model_bars, use_container_width=True)

        mc1, mc2 = st.columns(2)
        mc1.metric("XceptionNet", f"{result.get('xception_score', 0)*100:.2f}%",
                   delta="Fake" if result.get('xception_score', 0) > 0.5 else "Real",
                   delta_color="inverse")
        mc2.metric("EfficientNet-B4", f"{result.get('efficientnet_score', 0)*100:.2f}%",
                   delta="Fake" if result.get('efficientnet_score', 0) > 0.5 else "Real",
                   delta_color="inverse")

        st.markdown("---")
        st.markdown("**Ensemble weighting:** XceptionNet × 0.55 + EfficientNet-B4 × 0.45")
        st.markdown("**Decision threshold:** 50% fake probability")

    with tab4:
        face_results = result.get("face_results", [])
        fc = result.get("face_count", 0)

        if fc == 0:
            st.info("No faces detected. Whole-image analysis was performed.")
            mc1, mc2 = st.columns(2)
            mc1.metric("Fake probability", f"{fake_prob*100:.2f}%")
            mc2.metric("Real probability", f"{(1-fake_prob)*100:.2f}%")
        else:
            st.markdown(f"**{fc} face(s) detected and analyzed individually:**")
            for i, fr in enumerate(face_results):
                with st.expander(f"Face #{i+1} — {fr['verdict']} ({fr['confidence']*100:.1f}%)"):
                    fc1, fc2, fc3 = st.columns(3)
                    fc1.metric("Verdict", fr["verdict"])
                    fc2.metric("Fake probability", f"{fr['fake_probability']*100:.1f}%")
                    fc3.metric("Confidence", f"{fr['confidence']*100:.1f}%")
                    fc4, fc5 = st.columns(2)
                    fc4.metric("XceptionNet", f"{fr.get('xception_score', 0)*100:.1f}%")
                    fc5.metric("EfficientNet", f"{fr.get('efficientnet_score', 0)*100:.1f}%")

    with tab5:
        st.markdown("**Generate forensic analysis report**")
        st.markdown("Export a complete PDF report including all detection scores, Grad-CAM heatmap, forensics findings, and metadata analysis.")

        col_a, col_b = st.columns(2)
        include_gradcam = col_a.checkbox("Include Grad-CAM visualization", value=True)
        include_fft = col_b.checkbox("Include FFT heatmap", value=True)

        if st.button("📄 Generate PDF Report", type="primary"):
            with st.spinner("Generating report…"):
                try:
                    pdf_bytes = generate_report(
                        result=result,
                        original_image=original,
                        gradcam_image=result.get("gradcam") if include_gradcam else None,
                        fft_image=result.get("fft_image") if include_fft else None,
                        filename=filename,
                        analysis_type="image",
                    )
                    st.download_button(
                        label="⬇️ Download Report PDF",
                        data=pdf_bytes,
                        file_name=f"deepshield_report_{filename or 'image'}.pdf",
                        mime="application/pdf",
                    )
                    st.success("✅ Report ready! Click above to download.")
                except Exception as e:
                    st.error(f"Report generation failed: {e}")


# ──────────────────────────────────────────────
# Render results — Video
# ──────────────────────────────────────────────
def render_video_results(result: Dict, filename: str):
    from reports.pdf_report import generate_report

    if "error" in result:
        st.error(result["error"])
        return

    is_fake = result["is_fake"]
    confidence = result["confidence"]
    verdict = result["verdict"]
    badge = f'<span class="verdict-{"fake" if is_fake else "real"}">{verdict}</span>'

    from models.detector import MODEL_IS_TRAINED
    if not MODEL_IS_TRAINED:
        st.warning(
            "⚠️ **Untrained model** — predictions are random until a checkpoint is loaded into `checkpoints/`.",
            icon="🧪",
        )

    st.markdown(f"### Result &nbsp;&nbsp; {badge}", unsafe_allow_html=True)
    st.markdown("---")

    m1, m2, m3, m4, m5 = st.columns(5)
    verdict_color = "#E24B4A" if is_fake else "#1D9E75"
    for col, label, val, color in [
        (m1, "Verdict", verdict, verdict_color),
        (m2, "Confidence", f"{confidence*100:.1f}%", verdict_color),
        (m3, "Frames analyzed", str(result.get("frames_analyzed", "—")), "#444441"),
        (m4, "Duration", f"{result.get('video_duration', 0):.1f}s", "#444441"),
        (m5, "FPS", str(result.get("fps", "—")), "#444441"),
    ]:
        col.markdown(f"""<div class="ds-metric">
          <div class="label">{label}</div>
          <div class="value" style="color:{color}">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Timeline", "🕵️ Temporal Analysis", "🖼️ Peak Frame", "📄 Export Report"]
    )

    with tab1:
        st.markdown("**Frame-by-frame deepfake confidence**")
        timeline_img = result.get("timeline_image")
        if timeline_img:
            st.image(timeline_img, use_container_width=True)

        st.markdown('<div class="section-header">Color key</div>', unsafe_allow_html=True)
        st.markdown(
            '<span class="tag-ok">🟢 &lt;35% — Real</span> '
            '<span class="tag-warn">🟡 35–55% — Uncertain</span> '
            '<span class="verdict-fake" style="font-size:12px">🔴 &gt;55% — Deepfake</span>',
            unsafe_allow_html=True
        )

        # Show frame table
        frame_results = result.get("frame_results", [])
        if frame_results:
            with st.expander("View per-frame data"):
                import pandas as pd
                df = pd.DataFrame([{
                    "Frame": fr["frame_idx"],
                    "Timestamp (s)": fr["timestamp"],
                    "Verdict": fr["verdict"],
                    "Fake prob %": round(fr["fake_probability"] * 100, 1),
                    "Confidence %": round(fr["confidence"] * 100, 1),
                } for fr in frame_results])
                st.dataframe(df, use_container_width=True)

    with tab2:
        temporal = result.get("temporal", {})
        blink = result.get("blink_analysis", {})

        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown("**Temporal Consistency Analysis**")
            st.metric("Frame-to-frame variance", temporal.get("variance", "—"))
            st.metric("Mean confidence shift", temporal.get("mean_frame_diff", "—"))
            consistency = temporal.get("consistency", 0)
            c_color = "🟢" if consistency > 0.7 else "🟡" if consistency > 0.4 else "🔴"
            st.markdown(f"Consistency score: {c_color} **{consistency:.2f}**")
            spikes = temporal.get("spikes", [])
            if spikes:
                st.warning(f"⚠️ {len(spikes)} confidence spike(s) at frames: {spikes[:8]}")
            else:
                st.success("✅ No erratic confidence spikes detected")

        with tc2:
            st.markdown("**Eye Blink Pattern Analysis**")
            st.metric("Blink events detected", blink.get("blink_events", "—"))
            st.metric("Expected blinks", blink.get("expected_blinks", "—"))
            st.metric("Avg eyes per frame", blink.get("avg_eyes_per_frame", "—"))
            if blink.get("blink_anomaly"):
                st.warning("⚠️ Abnormal blink pattern — possible deepfake indicator")
            else:
                st.success("✅ Blink pattern appears normal")

    with tab3:
        peak_pil = result.get("peak_pil")
        peak_gradcam = result.get("peak_gradcam")
        peak_idx = result.get("peak_frame_idx", 0)
        fake_probs = result.get("fake_probs_timeline", [])
        peak_prob = fake_probs[min(peak_idx, len(fake_probs)-1)] if fake_probs else 0

        st.markdown(f"**Highest-confidence manipulation frame** (frame #{peak_idx * 8}, prob: {peak_prob*100:.1f}%)")
        pc1, pc2 = st.columns(2)
        if peak_pil:
            pc1.image(peak_pil, caption="Peak frame", use_container_width=True)
        if peak_gradcam:
            pc2.image(peak_gradcam, caption="Grad-CAM overlay", use_container_width=True)

        fft_img = result.get("fft_image")
        if fft_img:
            st.markdown("**FFT Frequency Analysis (peak frame)**")
            st.image(fft_img, width=350)

    with tab4:
        st.markdown("**Generate video forensic report**")
        if st.button("📄 Generate PDF Report", type="primary", key="video_pdf"):
            with st.spinner("Generating report…"):
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
                    st.download_button(
                        label="⬇️ Download Report PDF",
                        data=pdf_bytes,
                        file_name=f"deepshield_video_{filename}.pdf",
                        mime="application/pdf",
                    )
                    st.success("✅ Report ready!")
                except Exception as e:
                    st.error(f"Report generation failed: {e}")


# ──────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────
def page_image():
    st.markdown("## 🖼️ Image Deepfake Detection")
    st.markdown("Upload a photo to analyze with our XceptionNet + EfficientNet-B4 ensemble. "
                "Supports JPG, PNG, WEBP, BMP.")

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp", "bmp"],
                                label_visibility="collapsed")

    # Clear cached result if a new file is uploaded
    if uploaded:
        file_id = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("last_image_id") != file_id:
            st.session_state.pop("image_result", None)
            st.session_state.pop("image_pil", None)
            st.session_state["last_image_id"] = file_id

        pil_img = Image.open(uploaded).convert("RGB")
        st.session_state["image_pil"] = pil_img
        st.image(pil_img, caption=f"📎 {uploaded.name}", use_container_width=False, width=400)

        if st.button("🔍 Analyze Image", type="primary"):
            detector = load_detector()
            with st.spinner("Running detection pipeline…"):
                result = run_image_analysis(pil_img, uploaded.name, detector)
            st.session_state["image_result"] = result
            st.session_state["image_filename"] = uploaded.name

    # Render results from session state (persists across button clicks)
    if st.session_state.get("image_result") and st.session_state.get("image_pil"):
        render_image_results(
            st.session_state["image_result"],
            st.session_state["image_pil"],
            st.session_state.get("image_filename", "image"),
        )


def page_batch():
    st.markdown("## 📦 Batch Image Analysis")
    st.markdown("Upload multiple images at once. Results are displayed as a summary table.")

    uploaded_files = st.file_uploader(
        "Upload images (up to 20)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and st.button("🔍 Analyze All", type="primary"):
        if len(uploaded_files) > 20:
            st.warning("Maximum 20 images at once. Only first 20 will be processed.")
            uploaded_files = uploaded_files[:20]

        detector = load_detector()
        rows = []
        progress = st.progress(0, text="Starting batch analysis…")
        results_store = []

        for i, f in enumerate(uploaded_files):
            progress.progress((i + 1) / len(uploaded_files), text=f"Analyzing {f.name}…")
            pil_img = Image.open(f).convert("RGB")
            try:
                result = run_image_analysis(pil_img, f.name, detector)
                rows.append({
                    "Filename": f.name,
                    "Verdict": result["verdict"],
                    "Confidence %": round(result["confidence"] * 100, 1),
                    "Fake prob %": round(result["fake_probability"] * 100, 1),
                    "Faces": result.get("face_count", 0),
                    "XceptionNet %": round(result.get("xception_score", 0) * 100, 1),
                    "EfficientNet %": round(result.get("efficientnet_score", 0) * 100, 1),
                })
                results_store.append((f.name, pil_img, result))
            except Exception as e:
                rows.append({"Filename": f.name, "Verdict": "Error", "Confidence %": 0,
                             "Fake prob %": 0, "Faces": 0, "XceptionNet %": 0, "EfficientNet %": 0})

        progress.progress(1.0, text="Done!")

        import pandas as pd
        df = pd.DataFrame(rows)
        fake_count = (df["Verdict"] == "Deepfake").sum()
        real_count = (df["Verdict"] == "Real").sum()

        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("Total analyzed", len(rows))
        bc2.metric("Deepfakes detected", fake_count, delta_color="inverse",
                   delta=f"{fake_count/len(rows)*100:.0f}%" if rows else "0%")
        bc3.metric("Authentic", real_count)

        st.markdown("### Batch Results")

        def color_verdict(val):
            if val == "Deepfake":
                return "background-color: #FCEBEB; color: #A32D2D; font-weight: 600"
            elif val == "Real":
                return "background-color: #E1F5EE; color: #0F6E56; font-weight: 600"
            return ""

        styled_df = df.style.applymap(color_verdict, subset=["Verdict"])
        st.dataframe(styled_df, use_container_width=True)

        # Export CSV
        csv_bytes = df.to_csv(index=False).encode()
        st.download_button("⬇️ Download CSV", csv_bytes,
                           file_name="deepshield_batch_results.csv", mime="text/csv")


def page_video():
    st.markdown("## 🎬 Video Deepfake Detection")
    st.markdown("Upload a video for frame-by-frame analysis with temporal consistency checking.")

    uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv", "webm"],
                                label_visibility="collapsed")

    col1, col2 = st.columns(2)
    sample_rate = col1.slider("Frame sample rate (every Nth frame)", 4, 20, 8,
                              help="Lower = more thorough but slower")
    max_frames = col2.slider("Max frames to analyze", 20, 100, 50)

    if uploaded:
        file_id = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("last_video_id") != file_id:
            st.session_state.pop("video_result", None)
            st.session_state["last_video_id"] = file_id

        st.success(f"✅ Uploaded: **{uploaded.name}** ({uploaded.size // 1024} KB)")

        if st.button("🔍 Analyze Video", type="primary"):
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
        render_video_results(
            st.session_state["video_result"],
            st.session_state.get("video_filename", "video"),
        )


def page_webcam():
    st.markdown("## 📷 Live Webcam Detection")
    st.markdown("Real-time deepfake analysis from your webcam — frame is captured and analyzed every second.")

    # Session state for webcam
    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False
    if "webcam_result" not in st.session_state:
        st.session_state.webcam_result = None

    col_start, col_stop, col_snap = st.columns([1, 1, 2])

    if col_start.button("▶ Start Webcam", type="primary", disabled=st.session_state.webcam_running):
        st.session_state.webcam_running = True
        st.rerun()

    if col_stop.button("⏹ Stop", disabled=not st.session_state.webcam_running):
        st.session_state.webcam_running = False
        st.rerun()

    st.markdown("---")

    if st.session_state.webcam_running:
        detector = load_detector()

        # Live feed placeholder
        frame_placeholder = st.empty()
        result_placeholder = st.empty()
        status = st.empty()

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.session_state.webcam_running = False
            st.error("❌ Could not access webcam. Please check camera permissions in your browser/OS.")
            st.info("💡 Make sure no other application is using the camera, then try again.")
            return

        status.success("✅ Webcam connected — analyzing live…")

        frame_count = 0
        analyze_every = 5  # analyze every 5th frame for speed

        try:
            while st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Lost webcam feed.")
                    break

                frame_count += 1
                display_frame = frame.copy()

                # Run detection every N frames
                if frame_count % analyze_every == 0:
                    try:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result = detector.predict_image(Image.fromarray(rgb))
                        st.session_state.webcam_result = result
                    except Exception:
                        pass

                # Draw overlay on frame
                res = st.session_state.webcam_result
                if res:
                    is_fake = res["is_fake"]
                    conf = res["confidence"] * 100
                    verdict = res["verdict"]
                    color = (50, 50, 220) if is_fake else (50, 200, 80)  # BGR

                    # Background banner
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 52), (20, 20, 20), -1)
                    cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)

                    # Verdict text
                    cv2.putText(display_frame, f"{verdict}  {conf:.1f}%",
                                (12, 34), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

                    # Colored border
                    h, w = display_frame.shape[:2]
                    cv2.rectangle(display_frame, (0, 0), (w - 1, h - 1), color, 4)

                    # Mini bar in corner
                    bar_w = int((display_frame.shape[1] - 24) * res["fake_probability"])
                    cv2.rectangle(display_frame, (12, h - 18), (display_frame.shape[1] - 12, h - 8),
                                  (60, 60, 60), -1)
                    bar_color = (50, 50, 220) if res["fake_probability"] > 0.5 else (50, 200, 80)
                    cv2.rectangle(display_frame, (12, h - 18), (12 + bar_w, h - 8), bar_color, -1)

                # Show frame in Streamlit
                rgb_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_display, channels="RGB", use_container_width=True)

                # Show result metrics below
                if res:
                    result_placeholder.markdown(
                        f"""
                        <div style="display:flex;gap:16px;margin-top:8px">
                          <div class="ds-metric" style="flex:1">
                            <div class="label">Verdict</div>
                            <div class="value" style="color:{'#E24B4A' if res['is_fake'] else '#1D9E75'}">
                              {res['verdict']}
                            </div>
                          </div>
                          <div class="ds-metric" style="flex:1">
                            <div class="label">Confidence</div>
                            <div class="value">{res['confidence']*100:.1f}%</div>
                          </div>
                          <div class="ds-metric" style="flex:1">
                            <div class="label">Fake prob</div>
                            <div class="value">{res['fake_probability']*100:.1f}%</div>
                          </div>
                          <div class="ds-metric" style="flex:1">
                            <div class="label">Frame</div>
                            <div class="value">{frame_count}</div>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                time.sleep(0.04)  # ~25fps display

        finally:
            cap.release()
            status.empty()
            if not st.session_state.webcam_running:
                frame_placeholder.empty()
                result_placeholder.empty()

    else:
        # Not running — show last result or instructions
        st.markdown("""
        <div style="border:1.5px dashed #C8C5BC;border-radius:12px;padding:40px;text-align:center;color:#888780">
          <div style="font-size:48px;margin-bottom:12px">📷</div>
          <div style="font-size:16px;font-weight:500;color:#444441;margin-bottom:8px">
            Click <b>▶ Start Webcam</b> to begin live detection
          </div>
          <div style="font-size:13px">
            Allow camera access when prompted by your browser · Analysis runs every 5 frames
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Show snapshot option as fallback
        st.markdown("---")
        st.markdown("#### 📸 Or use single-frame capture")
        st.markdown("Take a photo from your camera for a one-time analysis.")
        img_file = st.camera_input("Capture photo", label_visibility="collapsed")
        if img_file:
            pil_img = Image.open(img_file).convert("RGB")
            if st.button("🔍 Analyze Captured Frame", type="primary"):
                detector = load_detector()
                with st.spinner("Analyzing…"):
                    result = run_image_analysis(pil_img, "webcam_capture.jpg", detector)
                render_image_results(result, pil_img, "webcam_capture.jpg")


def page_gradcam():
    st.markdown("## 🔬 Grad-CAM Visualization Studio")
    st.markdown("Upload an image to generate and customize Grad-CAM heatmaps showing manipulated regions.")

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"],
                                label_visibility="collapsed", key="gradcam_upload")

    if uploaded:
        from utils.visualize import overlay_gradcam, make_comparison_image

        pil_img = Image.open(uploaded).convert("RGB")
        col_s1, col_s2 = st.columns(2)
        alpha = col_s1.slider("Overlay intensity", 0.2, 0.8, 0.45, 0.05)
        target_class = col_s2.selectbox("Target class for visualization", ["Fake regions", "Real regions"])
        class_idx = 1 if target_class == "Fake regions" else 0

        if st.button("🎨 Generate Grad-CAM", type="primary"):
            detector = load_detector()
            with st.spinner("Computing Grad-CAM…"):
                cam = detector.get_gradcam(pil_img, class_idx=class_idx)
                overlay = overlay_gradcam(pil_img, cam, alpha=alpha)
                comparison = make_comparison_image(pil_img, overlay)

            st.markdown("**Side-by-side comparison:**")
            st.image(comparison, use_container_width=True)
            st.caption("Left: Original · Right: Grad-CAM overlay")

            col_d1, col_d2 = st.columns(2)
            buf1 = io.BytesIO()
            overlay.save(buf1, format="PNG")
            col_d1.download_button("⬇️ Download Heatmap", buf1.getvalue(),
                                   file_name="gradcam_heatmap.png", mime="image/png")

            buf2 = io.BytesIO()
            comparison.save(buf2, format="PNG")
            col_d2.download_button("⬇️ Download Comparison", buf2.getvalue(),
                                   file_name="gradcam_comparison.png", mime="image/png")


def page_history():
    from utils.history import render_history_panel
    st.markdown("## 📋 Analysis History")
    st.markdown("All analyses from the current session are logged here.")
    render_history_panel()


def page_about():
    st.markdown("## 🛡️ About DeepShield")
    st.markdown("""
DeepShield is an open-source deepfake detection system built with PyTorch and Streamlit.

### Architecture

| Component | Details |
|---|---|
| Primary model | XceptionNet (transfer learning on FaceForensics++) |
| Secondary model | EfficientNet-B4 |
| Ensemble strategy | Weighted average (55% Xception, 45% EfficientNet) |
| Explainability | Grad-CAM (gradient-weighted class activation mapping) |
| Face detection | OpenCV Haar cascade |
| Forensics | FFT analysis, EXIF inspection, compression artifacts |

### Features
- ✅ Image deepfake detection
- ✅ Video deepfake detection with temporal analysis
- ✅ Real-time webcam capture analysis
- ✅ Multi-face detection & individual face scoring
- ✅ Grad-CAM visualization studio
- ✅ FFT frequency domain forensics
- ✅ Metadata & EXIF forensics
- ✅ Compression artifact detection
- ✅ Skin tone consistency analysis
- ✅ Eye blink anomaly detection (video)
- ✅ Temporal consistency scoring (video)
- ✅ Batch image processing
- ✅ PDF forensic report export
- ✅ CSV batch export
- ✅ Session audit log

### References
- Rössler et al., *FaceForensics++*, ICCV 2019
- Chollet, *Xception: Deep Learning with Depthwise Separable Convolutions*, CVPR 2017
- Tan & Le, *EfficientNet: Rethinking Model Scaling*, ICML 2019
- Selvaraju et al., *Grad-CAM*, ICCV 2017
    """)


# ──────────────────────────────────────────────
# Sidebar + Navigation
# ──────────────────────────────────────────────

DETECTION_PAGES = [
    "🖼️ Image Analysis",
    "🎬 Video Analysis",
    "📷 Webcam",
    "📦 Batch Upload",
    "🔬 Grad-CAM Studio",
]
TOOL_PAGES = [
    "📋 History Log",
    "ℹ️ About",
]
ALL_PAGES = DETECTION_PAGES + TOOL_PAGES


def nav_button(label: str, current: str):
    """Render a sidebar nav button that updates session_state."""
    is_active = current == label
    bg = "#E1F5EE" if is_active else "transparent"
    border = "1px solid #9FE1CB" if is_active else "1px solid transparent"
    weight = "600" if is_active else "400"
    color = "#0F6E56" if is_active else "#5F5E5A"
    if st.sidebar.button(
        label,
        key=f"nav_{label}",
        use_container_width=True,
    ):
        st.session_state["current_page"] = label
        st.rerun()


def main():
    # Init session page
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "🖼️ Image Analysis"

    current = st.session_state["current_page"]

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px">
          <div style="width:34px;height:34px;border-radius:8px;background:#1D9E75;
                      display:flex;align-items:center;justify-content:center;font-size:18px">🛡️</div>
          <div>
            <div style="font-size:16px;font-weight:600;color:#2C2C2A">DeepShield</div>
            <div style="font-size:11px;color:#888780">v1.0 · Deepfake Detection</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Detection</div>', unsafe_allow_html=True)
        for p in DETECTION_PAGES:
            is_active = current == p
            # Style active button via CSS trick
            if is_active:
                st.markdown(
                    f'<div style="background:#E1F5EE;border:1px solid #9FE1CB;border-radius:8px;'
                    f'padding:1px 0;margin-bottom:2px">'
                    f'<style>div[data-testid="stButton"] button[kind="secondary"][id="nav_{p}"] '
                    f'{{ color:#0F6E56 !important; font-weight:600 !important; }}</style></div>',
                    unsafe_allow_html=True,
                )
            nav_button(p, current)

        st.markdown('<div class="section-header">Tools</div>', unsafe_allow_html=True)
        for p in TOOL_PAGES:
            nav_button(p, current)

        st.markdown("---")
        st.markdown("""
        <div style="font-size:12px;color:#888780">
          <div style="margin-bottom:4px">
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
            background:#1D9E75;margin-right:6px"></span>XceptionNet ready
          </div>
          <div>
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
            background:#1D9E75;margin-right:6px"></span>EfficientNet-B4 ready
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Checkpoint warning
        from models.detector import MODEL_IS_TRAINED
        if not MODEL_IS_TRAINED:
            st.markdown("""
            <div style="margin-top:12px;background:#FAEEDA;border:1px solid #EF9F27;
                        border-radius:8px;padding:10px 12px;font-size:11.5px;color:#854F0B">
              🧪 <b>Demo mode</b><br>
              No trained checkpoint loaded.<br>
              Results are not meaningful.<br>
              Add weights to <code>checkpoints/</code>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="margin-top:12px;background:#E1F5EE;border:1px solid #5DCAA5;
                        border-radius:8px;padding:10px 12px;font-size:11.5px;color:#0F6E56">
              ✅ <b>Trained model</b><br>
              Checkpoint loaded and ready.
            </div>
            """, unsafe_allow_html=True)

    # ── Route ──
    if current == "🖼️ Image Analysis":
        page_image()
    elif current == "🎬 Video Analysis":
        page_video()
    elif current == "📷 Webcam":
        page_webcam()
    elif current == "📦 Batch Upload":
        page_batch()
    elif current == "🔬 Grad-CAM Studio":
        page_gradcam()
    elif current == "📋 History Log":
        page_history()
    elif current == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()

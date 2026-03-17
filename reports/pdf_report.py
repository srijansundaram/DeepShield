"""
DeepShield — PDF Report Generator
Produces a detailed forensic analysis report using ReportLab
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from PIL import Image as PILImage
import io
import datetime
from typing import Dict, Optional
import tempfile
import os


# ─── Colors ───
TEAL = colors.HexColor("#1D9E75")
DANGER = colors.HexColor("#E24B4A")
AMBER = colors.HexColor("#EF9F27")
GRAY_DARK = colors.HexColor("#444441")
GRAY_MID = colors.HexColor("#888780")
GRAY_LIGHT = colors.HexColor("#F1EFE8")
WHITE = colors.white
BLACK = colors.HexColor("#2C2C2A")


def _pil_to_rl(pil_img: PILImage.Image, max_width: float = 400, max_height: float = 300) -> RLImage:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    w, h = pil_img.size
    scale = min(max_width / w, max_height / h, 1.0)
    return RLImage(buf, width=w * scale, height=h * scale)


def generate_report(
    result: Dict,
    original_image: Optional[PILImage.Image] = None,
    gradcam_image: Optional[PILImage.Image] = None,
    timeline_image: Optional[PILImage.Image] = None,
    fft_image: Optional[PILImage.Image] = None,
    filename: str = "",
    analysis_type: str = "image",
) -> bytes:
    """
    Generate a comprehensive PDF forensic report.
    Returns PDF as bytes.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2.2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    # Custom styles
    title_style = ParagraphStyle("Title", parent=styles["Normal"],
        fontSize=22, textColor=BLACK, fontName="Helvetica-Bold",
        spaceAfter=4, alignment=TA_LEFT)
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"],
        fontSize=11, textColor=GRAY_MID, spaceAfter=16, alignment=TA_LEFT)
    h2_style = ParagraphStyle("H2", parent=styles["Normal"],
        fontSize=13, textColor=BLACK, fontName="Helvetica-Bold",
        spaceBefore=16, spaceAfter=6)
    body_style = ParagraphStyle("Body", parent=styles["Normal"],
        fontSize=9.5, textColor=GRAY_DARK, leading=14, spaceAfter=4)
    label_style = ParagraphStyle("Label", parent=styles["Normal"],
        fontSize=8, textColor=GRAY_MID, spaceBefore=2)
    code_style = ParagraphStyle("Code", parent=styles["Normal"],
        fontSize=8, fontName="Courier", textColor=GRAY_DARK,
        backColor=GRAY_LIGHT, leading=12)

    is_fake = result.get("is_fake", False)
    verdict = result.get("verdict", "Unknown")
    confidence = result.get("confidence", 0.0)
    fake_prob = result.get("fake_probability", 0.0)
    verdict_color = DANGER if is_fake else TEAL
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    story = []

    # ── Header ──
    story.append(Paragraph("DeepShield", title_style))
    story.append(Paragraph("Deepfake Forensic Analysis Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY_MID, spaceAfter=12))

    # ── Meta table ──
    meta_data = [
        ["Analysis type", analysis_type.capitalize()],
        ["Filename", filename or "—"],
        ["Analysis date", now],
        ["Model", "XceptionNet + EfficientNet-B4 Ensemble"],
    ]
    meta_table = Table(meta_data, colWidths=[4 * cm, 12 * cm])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (0, -1), GRAY_MID),
        ("TEXTCOLOR", (1, 0), (1, -1), GRAY_DARK),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, GRAY_LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.5 * cm))

    # ── Verdict Banner ──
    verdict_text = f"VERDICT: {verdict.upper()}  —  {confidence*100:.1f}% confidence"
    banner_data = [[Paragraph(verdict_text, ParagraphStyle("V", parent=styles["Normal"],
        fontSize=14, textColor=WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER))]]
    banner = Table(banner_data, colWidths=[17 * cm])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), verdict_color),
        ("TOPPADDING", (0, 0), (0, 0), 10),
        ("BOTTOMPADDING", (0, 0), (0, 0), 10),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(banner)
    story.append(Spacer(1, 0.4 * cm))

    # ── Probability scores ──
    story.append(Paragraph("Detection Scores", h2_style))
    score_data = [
        ["Metric", "Score"],
        ["Fake probability", f"{fake_prob*100:.2f}%"],
        ["Real probability", f"{(1-fake_prob)*100:.2f}%"],
        ["XceptionNet score", f"{result.get('xception_score', 0)*100:.2f}%"],
        ["EfficientNet-B4 score", f"{result.get('efficientnet_score', 0)*100:.2f}%"],
    ]
    if analysis_type == "video":
        score_data += [
            ["Frames analyzed", str(result.get("frames_analyzed", "—"))],
            ["Video duration", f"{result.get('video_duration', 0):.1f}s"],
            ["Temporal variance", str(result.get("temporal", {}).get("variance", "—"))],
            ["Blink anomaly", str(result.get("blink_analysis", {}).get("blink_anomaly", "—"))],
        ]

    score_table = Table(score_data, colWidths=[9 * cm, 8 * cm])
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), GRAY_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GRAY_LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.3, GRAY_MID),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 0.4 * cm))

    # ── Face Results (if multi-face) ──
    face_results = result.get("face_results", [])
    if face_results:
        story.append(Paragraph(f"Face Analysis ({len(face_results)} face(s) detected)", h2_style))
        face_data = [["Face #", "Verdict", "Fake Prob", "XceptionNet", "EfficientNet"]]
        for i, fr in enumerate(face_results):
            face_data.append([
                str(i + 1),
                fr.get("verdict", "—"),
                f"{fr.get('fake_probability', 0)*100:.1f}%",
                f"{fr.get('xception_score', 0)*100:.1f}%",
                f"{fr.get('efficientnet_score', 0)*100:.1f}%",
            ])
        face_table = Table(face_data, colWidths=[2*cm, 3.5*cm, 3.5*cm, 4*cm, 4*cm])
        face_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), GRAY_DARK),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GRAY_LIGHT]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.3, GRAY_MID),
        ]))
        story.append(face_table)
        story.append(Spacer(1, 0.4 * cm))

    # ── Forensics ──
    forensics = result.get("forensics", {})
    metadata_r = forensics.get("metadata_analysis", {})
    fft_r = forensics.get("fft_analysis", {})
    comp_r = forensics.get("compression_analysis", {})
    skin_r = forensics.get("skin_analysis", {})

    if forensics:
        story.append(Paragraph("Forensic Analysis", h2_style))
        findings = metadata_r.get("findings", [])
        if findings:
            story.append(Paragraph("Metadata findings:", label_style))
            for f in findings:
                story.append(Paragraph(f"• {f}", body_style))

        forensics_data = [["Signal", "Value", "Anomaly"]]
        if fft_r:
            forensics_data.append(["FFT high-freq ratio", str(fft_r.get("hf_ratio", "—")),
                "Yes" if fft_r.get("anomaly_score", 0) > 0.6 else "No"])
        if comp_r:
            forensics_data.append(["Block artifact score", str(comp_r.get("block_artifact_score", "—")),
                "Yes" if comp_r.get("re_encoding_suspected") else "No"])
        if skin_r:
            forensics_data.append(["Skin tone variance", str(skin_r.get("skin_variance", "—")),
                "Yes" if skin_r.get("anomaly") else "No"])
        if metadata_r:
            forensics_data.append(["Metadata risk score", str(metadata_r.get("risk_score", "—")),
                "Yes" if metadata_r.get("risk_score", 0) > 0.5 else "No"])

        if len(forensics_data) > 1:
            f_table = Table(forensics_data, colWidths=[7*cm, 6*cm, 4*cm])
            f_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), GRAY_DARK),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GRAY_LIGHT]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.3, GRAY_MID),
            ]))
            story.append(f_table)

    # ── Images ──
    if original_image or gradcam_image or fft_image or timeline_image:
        story.append(Spacer(1, 0.4 * cm))
        story.append(Paragraph("Visual Evidence", h2_style))

    if original_image and gradcam_image:
        img_data = [[
            _pil_to_rl(original_image.resize((250, 250)), 250, 250),
            _pil_to_rl(gradcam_image.resize((250, 250)), 250, 250),
        ]]
        img_label = [[
            Paragraph("Original Image", label_style),
            Paragraph("Grad-CAM Heatmap", label_style),
        ]]
        for row in [img_data, img_label]:
            t = Table(row, colWidths=[8.5 * cm, 8.5 * cm])
            t.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
            story.append(t)
        story.append(Spacer(1, 0.3 * cm))

    if fft_image:
        story.append(Paragraph("FFT Frequency Analysis", label_style))
        story.append(_pil_to_rl(fft_image, 300, 200))
        story.append(Spacer(1, 0.3 * cm))

    if timeline_image:
        story.append(Paragraph("Frame-by-frame Confidence Timeline", label_style))
        story.append(_pil_to_rl(timeline_image, 500, 160))

    # ── Footer ──
    story.append(Spacer(1, 0.6 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY_MID))
    story.append(Paragraph(
        "Generated by DeepShield v1.0 · This report is for informational purposes only. "
        "Results should be interpreted by qualified professionals.",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=7.5,
                       textColor=GRAY_MID, alignment=TA_CENTER, spaceBefore=6),
    ))

    doc.build(story)
    return buf.getvalue()

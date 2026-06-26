# 🛡️ DeepShield — Deepfake Detection System

A production-grade deepfake detection system using deep learning, built with PyTorch and Streamlit.

---

## Features

### Detection Modes

| Mode | Description |
|------|-------------|
| **Image Analysis** | Single image detection with full forensics suite |
| **Video Analysis** | Frame-by-frame analysis with temporal consistency |
| **URL Analyzer** | Fetch and analyze any public image URL |
| **Batch Scan** | Analyze up to 20 images simultaneously |
| **Grad-CAM Studio** | Interactive activation heatmap visualization |

### ML Pipeline

- **XceptionNet** — Transfer learning backbone (55% ensemble weight)
- **EfficientNet-B4** — Secondary model (45% ensemble weight)
- **InsightFace** — Production-grade face detection with Haar cascade fallback
- **Grad-CAM** — Gradient-weighted class activation mapping for explainability
- **Ensemble** — Weighted combination for robust predictions

### Training

- Trained on **59,841 images** across 3 datasets:
  - **140k Real and Fake Faces** — GAN-generated face pairs
  - **Celeb-DF v2** — Celebrity deepfake videos (frame extracted)
  - **Wild Faces** — Real-world diverse human photos
- **XceptionNet val accuracy: 99.36%**
- **EfficientNet-B4 val accuracy: 98.63%**
- Training: AdamW + Cosine LR + Label Smoothing + fp16 mixed precision

### Forensics Suite

| Analysis | What it detects |
|----------|-----------------|
| **FFT Frequency Analysis** | Unnatural high/mid-band spectral patterns |
| **EXIF Metadata Forensics** | Missing/suspicious camera metadata |
| **Compression Analysis** | Re-encoding and JPEG block artifacts |
| **Skin Tone Analysis** | Unnatural blending in face regions |
| **Eye Blink Detection** | Abnormal blink patterns (video) |
| **Temporal Consistency** | Erratic frame-to-frame confidence changes (video) |

### Output & Reporting

- Confidence scores per model and ensemble
- Annotated images with face bounding boxes
- Grad-CAM heatmap overlays
- Frame-by-frame confidence timeline (video)
- PDF forensic report export
- CSV batch results export
- Session audit/history log

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/srijansundaram/DeepShield.git
cd DeepShield
pip install -r requirements.txt
```

### 2. Add trained weights

Place `.pth` files in `checkpoints/`:
```
DeepShield/
  checkpoints/
    xception_deepfake.pth
    efficientnet_deepfake.pth
```

Download weights from [Hugging Face](https://huggingface.co/srijansundaram/deepshield).

### 3. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Training Your Own Model

### Datasets Used

| Dataset | Type | Size |
|---------|------|------|
| 140k Real and Fake Faces | GAN faces | ~2.5GB |
| Celeb-DF v2 | Video deepfakes | ~2GB |
| Wild Faces | Real-world photos | ~500MB |

### Prepare dataset

```bash
python prepare_combined.py
```

### Train

```bash
# XceptionNet
python train.py --model xception --batch_size 16 --epochs 20 --data_dir ./data

# EfficientNet-B4
python train.py --model efficientnet_b4 --batch_size 16 --epochs 20 --data_dir ./data
```

---

## Project Structure

```
DeepShield/
├── app.py                    # Main Streamlit application
├── train.py                  # Training script
├── prepare_combined.py       # Multi-dataset preparation
├── requirements.txt
├── checkpoints/              # Trained model weights (not in repo)
├── models/
│   ├── __init__.py
│   └── detector.py           # XceptionNet + EfficientNet ensemble + GradCAM
├── utils/
│   ├── __init__.py
│   ├── face_utils.py         # InsightFace + Haar cascade fallback
│   ├── video_utils.py        # Frame extraction, temporal analysis, blink detection
│   ├── forensics.py          # FFT, metadata, compression, skin tone analysis
│   ├── visualize.py          # GradCAM overlay, charts
│   └── history.py            # Session audit log
└── reports/
    ├── __init__.py
    └── pdf_report.py         # PDF forensic report generator
```

---

## Performance

| Metric | Score |
|--------|-------|
| XceptionNet Val Accuracy | 99.36% |
| EfficientNet-B4 Val Accuracy | 98.63% |
| Unseen Celeb-DF Detection | ~60-70% |
| Real photo false positive rate | ~10% fake prob |
| GPU inference (RTX 3050) | ~40-80ms/image |

**Known limitations:**
- PGGAN-style fakes (thispersondoesnotexist.com) may be missed — not in training distribution
- Heavily compressed or low-resolution images reduce accuracy
- Diffusion-based fakes (Stable Diffusion, Midjourney) not yet supported

---

## Requirements

```
Python >= 3.11
PyTorch >= 2.0
CUDA (recommended, auto-detected)
```

---

## References

1. Rössler et al., _FaceForensics++_, ICCV 2019
2. Chollet, _Xception_, CVPR 2017
3. Tan & Le, _EfficientNet_, ICML 2019
4. Selvaraju et al., _Grad-CAM_, ICCV 2017
5. Li et al., _Celeb-DF_, CVPR 2020

---

## License

MIT License. See LICENSE for details.

---

## Disclaimer

DeepShield is a research tool. Results should be interpreted by qualified professionals.
No deepfake detector achieves 100% accuracy. Do not use as sole evidence in legal proceedings.


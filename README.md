# 🛡️ DeepShield — Deepfake Detection System

A production-grade deepfake detection system using deep learning, built with PyTorch and Streamlit. Detects GAN-generated faces (StyleGAN/PGGAN), diffusion-generated faces (Stable Diffusion/FLUX/SDXL), and face-swap deepfakes (DFDC-style), with a three-model calibrated ensemble, REST API, audio-visual sync detection, face identity tracking, and a full forensic analysis suite.

---

## Features

### Detection Modes

| Mode | Description |
|------|-------------|
| **Image Analysis** | Single image detection with full forensics suite + explainability dashboard |
| **Video Analysis** | Frame-by-frame analysis with temporal consistency, blink pattern, audio-visual sync, and face identity tracking |
| **URL Analyzer** | Fetch and analyze any public image URL |
| **Batch Scan** | Analyze up to 20 images simultaneously |
| **Grad-CAM Studio** | Interactive activation heatmap visualization |
| **REST API** | Programmatic access via FastAPI — `POST /analyze` |

### ML Pipeline

- **XceptionNet** — Primary transfer learning backbone
- **EfficientNet-B4** — Secondary model
- **Vision Transformer (ViT-B/16)** — Third ensemble model, added for better generalization to unseen fake types. Loads optionally — if `vit_deepfake.pth` isn't present, the app runs fine on the two-model ensemble automatically (no breaking change)
- **Ensemble weights** — 40% Xception + 35% EfficientNet + 25% ViT (when ViT is loaded); falls back to 55%/45% Xception/EfficientNet otherwise
- **InsightFace** — Production-grade face detection with Haar cascade fallback (GPU acceleration currently blocked on Python 3.14 — see Known Limitations)
- **Grad-CAM** — Gradient-weighted class activation mapping for explainability
- **Temperature scaling** — Post-hoc confidence calibration for trustworthy probability outputs (handles all three models automatically)
- **Ensemble** — Weighted combination for robust predictions

### Training Data

Trained on **~140,000+ images** across 6 dataset sources:

| Dataset | Type | Purpose |
|---------|------|---------|
| 140k Real and Fake Faces | GAN faces | Base real/fake pairs |
| Celeb-DF v2 | Celebrity deepfake videos (frames) | Face-swap coverage |
| Wild Faces | Real-world diverse photos | Real-world generalization |
| StyleGAN3 (troykueh) | StyleGAN3-generated faces | GAN coverage (thispersondoesnotexist-style) |
| Stable Diffusion faces (bwandowando) + 130k-real-vs-fake-face (FLUX/SDXL) | Diffusion-generated faces | Modern diffusion fake coverage |
| DFDC train split | Face-swap deepfakes | Face-swap manipulation coverage |

**Diffusion coverage scope:** training data covers **Stable Diffusion 1.4/1.5, SDXL, and FLUX DEV/PRO**. Proprietary commercial generators (Midjourney, veri.ai, Adobe Firefly) are **not guaranteed to be detected**, since training data for these closed systems isn't publicly available.

- **XceptionNet val accuracy: 97–98%**
- **EfficientNet-B4 val accuracy: 97–98%**
- **100% accuracy on 200 held-out diffusion (SD/FLUX/SDXL) validation images**
- Training: AdamW + Cosine LR + Label Smoothing + fp16 mixed precision
- Calibration: Temperature scaling reduced ECE from ~0.09 to ~0.005 across all models

### Forensics Suite

| Analysis | What it detects |
|----------|-----------------|
| **FFT Frequency Analysis** | Unnatural high/mid-band spectral patterns |
| **EXIF Metadata Forensics** | Missing/suspicious camera metadata |
| **Compression Analysis** | Re-encoding and JPEG block artifacts |
| **Skin Tone Analysis** | Unnatural blending in face regions |
| **Eye Blink Detection** | Abnormal blink patterns (video) |
| **Temporal Consistency** | Erratic frame-to-frame confidence changes (video) |
| **Audio-Visual Sync Detection** | Lip movement vs audio energy correlation mismatch (video) — strong face-swap signal, no other free tool covers this well |
| **Face Identity Consistency** | ArcFace embedding drift across frames — flags identity shifts typical of face-swap manipulation (video) |
| **Social Media Recompression Handling** | Detects heavy platform re-encoding (Instagram/WhatsApp/Twitter/Facebook) and dampens FFT/compression anomaly scores accordingly to avoid false positives on heavily-shared real photos |
| **Explainability Dashboard** | Ranks every detection signal (model scores + forensic signals) by relative contribution to the verdict, clearly separating mathematically-weighted ensemble scores from advisory forensic evidence |

### Output & Reporting

- Confidence scores per model and ensemble (temperature-calibrated)
- Annotated images with face bounding boxes
- Grad-CAM heatmap overlays
- Explainability signal-contribution chart
- Frame-by-frame confidence timeline (video)
- Face identity consistency timeline (video)
- Audio-visual sync score (video)
- Recompression level + platform-match detection
- PDF forensic report export
- CSV batch results export
- Session audit/history log
- JSON API responses for programmatic integration

---

## Quick Start (using pretrained weights)

### 1. Clone & install

```bash
git clone https://github.com/srijansundaram/DeepShield.git
cd DeepShield
pip install -r requirements.txt
```

### 2. Add trained weights

Place `.pth` files and `calibration.json` in `checkpoints/`:
```
DeepShield/
  checkpoints/
    xception_deepfake.pth
    efficientnet_deepfake.pth
    vit_deepfake.pth          # optional — ensemble degrades gracefully without it
    calibration.json
```

Download weights from [Hugging Face](https://huggingface.co/srijansundaram/deepshield).

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### 4. Run the REST API (optional)

```bash
pip install fastapi uvicorn python-multipart
uvicorn api:app --host 0.0.0.0 --port 8000
```

Swagger UI available at `http://localhost:8000/docs`.

```bash
# Test it
curl -X POST http://localhost:8000/analyze -F "file=@your_image.jpg"
```

---

## Full Reproduction Guide — Training From Scratch

If you want to retrain DeepShield yourself instead of using pretrained weights, follow this exact sequence. This rebuilds the full dataset and trains all three models from zero.

### Step 1 — Get a Kaggle API key

Go to [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create New Token. This downloads `kaggle.json`.

```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
pip install kaggle
```

### Step 2 — Download base datasets

```bash
mkdir -p ~/datasets/deepshield

# 140k Real and Fake Faces (GAN base pairs)
kaggle datasets download -d xhlulu/140k-real-and-fake-faces \
  -p ~/datasets/deepshield/140k --unzip -q

# Celeb-DF v2 (face-swap video frames) — request access first if needed
# Place manually at ~/datasets/deepshield/celebdf/

# Wild real-world faces — any small diverse face photo set works
# Place manually at ~/datasets/deepshield/wildfaces/
```

### Step 3 — Build the base dataset

```bash
python prepare_combined.py
```
This merges 140k + Celeb-DF + wildfaces into `~/datasets/deepshield/prepared/{train,val}/{real,fake}`.

### Step 4 — Download Tier 1 expansion datasets (StyleGAN + diffusion coverage)

```bash
# StyleGAN3 fakes
kaggle datasets download -d troykueh/real-vs-fake-faces-stylegan3 \
  -p ~/datasets/deepshield/stylegan3 --unzip -q

# Diffusion + GAN faces (real/photoshop/gan/diffusion subfolders)
kaggle datasets download -d fatimasalman/real-photoshop-gan-diffusion-faces \
  -p ~/datasets/deepshield/diffusion_gan --unzip -q

# 130k-real-vs-fake-face — has FLUX_DEV/FLUX_PRO/SDXL fakes (best diffusion coverage)
kaggle datasets download -d shreyanshpatel1/130k-real-vs-fake-face \
  -p ~/datasets/deepshield/130k_rvf --unzip -q
```

Run:
```bash
python prepare_tier1.py
```
This merges StyleGAN3 + diffusion fakes into the prepared dataset, balances classes, and reports final counts.

### Step 5 — Download face-swap coverage (DFDC)

```bash
kaggle datasets download -d itamargr/dfdc-faces-of-the-train-sample \
  -p ~/datasets/deepshield/benchmark/dfdc --unzip -q
```

Use the **train** split only for training data (keep **validation** split untouched for honest benchmarking). Add ~15,000 face-swap fakes and matching reals from `dfdc/train/{fake,real}` into the prepared dataset's train/val folders, then balance train/fake vs train/real counts (see `prepare_tier1.py` pattern — a similar one-off script handles this; not yet folded into the main script).

### Step 6 — Train all three models

```bash
python train.py --model xception --data_dir ~/datasets/deepshield/prepared \
  --output_dir ./checkpoints --epochs 15 --batch_size 16 --lr 5e-5 --patience 5

python train.py --model efficientnet_b4 --data_dir ~/datasets/deepshield/prepared \
  --output_dir ./checkpoints --epochs 15 --batch_size 8 --lr 5e-5 --patience 5

python train.py --model vit --data_dir ~/datasets/deepshield/prepared \
  --output_dir ./checkpoints --epochs 15 --batch_size 16 --lr 5e-5 --patience 5
```
Lower batch size for EfficientNet-B4 if you hit CUDA OOM (380px inputs are memory-heavy; 8 works on 4GB VRAM cards). ViT is optional — skip it and the app runs on the two-model ensemble.

### Step 7 — Calibrate confidence

```bash
python calibrate.py
```
Outputs `checkpoints/calibration.json` with temperature values for all loaded models (ViT included automatically if its checkpoint exists), no retraining needed.

### Step 8 — Validate

```bash
streamlit run app.py
```
Test against: a real photo, a StyleGAN face (thispersondoesnotexist.com), a Stable Diffusion face, and a face-swap deepfake clip. Expect calibrated confidence (not 99%/1% extremes) and correct verdicts across all four categories.

### Step 9 (optional) — Benchmark against external datasets

```bash
python benchmark.py
```
Reports accuracy, AUC, F1, and false positive rate on DFDC validation split (untouched from training). Edit `DATASETS` in `benchmark.py` to point at any other labeled real/fake dataset you want to benchmark against.

---

## Project Structure

```
DeepShield/
├── app.py                    # Main Streamlit application
├── api.py                    # FastAPI REST API
├── train.py                  # Training script (Xception, EfficientNet-B4, ViT)
├── calibrate.py              # Temperature scaling calibration
├── benchmark.py              # Cross-dataset benchmark evaluation
├── prepare_combined.py       # Base dataset preparation (140k + Celeb-DF + wild)
├── prepare_tier1.py          # Tier 1 expansion (StyleGAN3 + diffusion fakes)
├── requirements.txt
├── checkpoints/              # Trained model weights + calibration.json (not in repo)
├── models/
│   ├── __init__.py
│   └── detector.py           # XceptionNet + EfficientNet + ViT ensemble + GradCAM + calibration
├── utils/
│   ├── __init__.py
│   ├── face_utils.py           # InsightFace + Haar cascade fallback
│   ├── video_utils.py          # Frame extraction, temporal analysis, blink detection
│   ├── forensics.py            # FFT, metadata, compression, skin tone analysis
│   ├── visualize.py            # GradCAM overlay, charts, explainability + identity charts
│   ├── history.py              # Session audit log
│   ├── av_sync.py              # Audio-visual lip-sync desync detection (video)
│   ├── identity_consistency.py # ArcFace-based face identity drift tracking (video)
│   ├── explainability.py       # Signal-contribution scoring for the explainability dashboard
│   └── social_compression.py   # Social media recompression detection + forensic score adjustment
└── reports/
    ├── __init__.py
    └── pdf_report.py         # PDF forensic report generator
```

---

## REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/analyze` | POST | Upload an image file → JSON verdict |
| `/analyze/url` | POST | Provide an image URL → JSON verdict |
| `/health` | GET | Model load status, device (CPU/CUDA) |
| `/docs` | GET | Interactive Swagger UI |

Example response:
```json
{
  "verdict": "Deepfake",
  "is_fake": true,
  "fake_probability": 0.988,
  "real_probability": 0.012,
  "confidence": 0.988,
  "xception_score": 0.985,
  "efficientnet_score": 0.992,
  "vit_score": 0.979,
  "face_count": 1,
  "model_trained": true
}
```
`vit_score` is only present when the ViT checkpoint is loaded.

---

## Performance

### Core metrics

| Metric | Score |
|--------|-------|
| XceptionNet Val Accuracy | ~97–98% |
| EfficientNet-B4 Val Accuracy | ~97–98% |
| Diffusion (SD/FLUX/SDXL) held-out accuracy | 100% (200 samples) |
| StyleGAN/PGGAN detection | Correctly flagged after Tier 1 retrain |
| Confidence calibration (ECE) | ~0.09 → ~0.005 after temperature scaling |
| GPU inference (RTX 3050, CUDA) | ~40–80ms/image |
| CPU face detection | 3–4x slower than GPU (see Known Limitations) |

### Face-swap detection (DFDC) — before vs after adding ViT

| Metric | Two-model ensemble | Three-model ensemble (+ViT) | Change |
|--------|--------------------:|------------------------------:|:------:|
| Accuracy | 88.6% | **90.1%** | +1.5% |
| AUC-ROC | 0.931 | **0.9517** | +0.021 |
| F1 | 0.8834 | **0.9003** | +0.017 |

Adding ViT as a third ensemble member improved every metric on unseen DFDC face-swap validation data, confirming ViT generalizes better to manipulation types the CNNs weren't as tuned to.

| Face-swap (DFDC train split, sanity check) | 98.9% accuracy, AUC 0.997, zero false positives |
|---|---|

**Known limitations:**
- **InsightFace GPU acceleration is currently blocked** — `onnxruntime-gpu` does not yet ship CUDA-enabled wheels for Python 3.14. Face detection and identity-consistency tracking fall back to CPU automatically; the app remains fully functional, just slower per-image/frame. Will be resolved once onnxruntime publishes 3.14 CUDA wheels.
- Heavily compressed or low-resolution images reduce accuracy (partially mitigated by social media recompression handling — see Forensics Suite).
- Diffusion detection covers SD 1.4/1.5, SDXL, and FLUX DEV/PRO only. Proprietary commercial generators (Midjourney, veri.ai, Adobe Firefly) are not guaranteed to be detected, as training data for these closed systems isn't publicly available.
- Face-swap deepfakes (DFDC-style) initially showed near-0% detection rate on the two-model ensemble trained without face-swap-specific data. Resolved by adding DFDC training data and a third ViT ensemble model: **90.1% accuracy on DFDC validation** (AUC 0.9517, F1 0.9003), 9.2%-ish false positive rate on real faces.
- Audio-visual sync detection requires `ffmpeg` installed and accessible on PATH; degrades gracefully with a note if unavailable.

---

## Requirements

```
Python >= 3.11 (note: 3.14 currently blocks onnxruntime-gpu CUDA support)
PyTorch >= 2.0
CUDA (recommended, auto-detected)
ffmpeg (required for audio-visual sync detection)
Kaggle API key (only needed for retraining from scratch)
```

Note: `utils/identity_consistency.py` automatically downloads the InsightFace `buffalo_l` model pack (~300MB) on first video scan.

---

## Roadmap

- [ ] **Docker deployment** — one-command containerized deployment for enterprise environments (in progress)

---

## References

1. Rössler et al., _FaceForensics++_, ICCV 2019
2. Chollet, _Xception_, CVPR 2017
3. Tan & Le, _EfficientNet_, ICML 2019
4. Dosovitskiy et al., _An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale_ (ViT), ICLR 2021
5. Selvaraju et al., _Grad-CAM_, ICCV 2017
6. Li et al., _Celeb-DF_, CVPR 2020
7. Dolhansky et al., _The DeepFake Detection Challenge (DFDC) Dataset_, 2020
8. Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q., _On Calibration of Modern Neural Networks_, ICML 2017 (temperature scaling)

---

## License

MIT License. See LICENSE for details.

---

## Disclaimer

DeepShield is a research tool. Results should be interpreted by qualified professionals.
No deepfake detector achieves 100% accuracy. Do not use as sole evidence in legal proceedings.

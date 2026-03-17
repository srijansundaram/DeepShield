# 🛡️ DeepShield — Deepfake Detection System

A production-grade deepfake detection system using deep learning, built with PyTorch and Streamlit.

---

## Features

### Detection Modes

| Mode                | Description                                       |
| ------------------- | ------------------------------------------------- |
| **Image Analysis**  | Single image detection with full forensics        |
| **Video Analysis**  | Frame-by-frame analysis with temporal consistency |
| **Webcam**          | Capture & analyze from webcam                     |
| **Batch Upload**    | Analyze up to 20 images at once                   |
| **Grad-CAM Studio** | Interactive heatmap visualization                 |

### ML Pipeline

- **XceptionNet** — Transfer learning backbone (55% ensemble weight)
- **EfficientNet-B4** — Secondary model (45% ensemble weight)
- **Multi-face detection** — OpenCV Haar cascade, each face analyzed independently
- **Grad-CAM** — Gradient-weighted class activation mapping for explainability

### Forensics Suite

| Analysis                    | What it detects                                   |
| --------------------------- | ------------------------------------------------- |
| **FFT Frequency Analysis**  | Unnatural high/mid-band spectral patterns         |
| **EXIF Metadata Forensics** | Missing/suspicious camera metadata                |
| **Compression Analysis**    | Re-encoding and JPEG block artifacts              |
| **Skin Tone Analysis**      | Unnatural blending in face regions                |
| **Eye Blink Detection**     | Abnormal blink patterns (video)                   |
| **Temporal Consistency**    | Erratic frame-to-frame confidence changes (video) |

### Output & Reporting

- Confidence scores per model and ensemble
- Annotated images with face bounding boxes
- Frame-by-frame confidence timeline chart
- PDF forensic report (download)
- CSV batch results export
- Session audit/history log

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/your-repo/deepshield
cd deepshield
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Training Your Own Model

The system ships with **randomly initialized weights** for demonstration purposes.
For production use, fine-tune on a deepfake dataset:

### Recommended Datasets

| Dataset         | Faces                | Notes                            |
| --------------- | -------------------- | -------------------------------- |
| FaceForensics++ | 1,000 videos         | Most widely used benchmark       |
| Celeb-DF v2     | 590 videos           | High-quality celebrity deepfakes |
| DFDC (Facebook) | 100,000+ videos      | Most diverse                     |
| WildDeepfake    | 7,314 face sequences | In-the-wild                      |

### Training Script

```python
# train.py — example fine-tuning snippet
from models.detector import XceptionDetector
import torch, torch.nn as nn
from torch.optim import AdamW
from torchvision import datasets
from models.detector import TRANSFORM

model = XceptionDetector()
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# Load dataset
dataset = datasets.ImageFolder("data/train", transform=TRANSFORM)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(20):
    for imgs, labels in loader:
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

# Save
torch.save(model.state_dict(), "checkpoints/xception_deepfake.pth")
```

### Loading a Checkpoint

In `models/detector.py`, add to `XceptionDetector.__init__`:

```python
checkpoint_path = "checkpoints/xception_deepfake.pth"
if os.path.exists(checkpoint_path):
    self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
```

---

## Project Structure

```
deepshield/
├── app.py                    # Main Streamlit application
├── requirements.txt
├── .streamlit/
│   └── config.toml           # UI theme
├── models/
│   ├── __init__.py
│   └── detector.py           # XceptionNet + EfficientNet ensemble + GradCAM
├── utils/
│   ├── __init__.py
│   ├── face_utils.py         # Face detection & multi-face analysis
│   ├── video_utils.py        # Frame extraction, temporal analysis, blink detection
│   ├── forensics.py          # FFT, metadata, compression, skin tone analysis
│   ├── visualize.py          # GradCAM overlay, charts, gauges
│   └── history.py            # Session audit log
├── reports/
│   ├── __init__.py
│   └── pdf_report.py         # PDF forensic report generator
└── README.md
```

---

## Requirements

```
Python >= 3.9
PyTorch >= 2.0
CUDA (optional, auto-detected)
```

Key dependencies: `streamlit`, `torch`, `torchvision`, `timm`, `opencv-python`, `Pillow`, `reportlab`, `plotly`, `pandas`

---

## Configuration

Edit `.streamlit/config.toml` to change theme colors, max upload size, etc.

---

## Performance Notes

- On CPU: ~200–400ms per image
- On GPU (CUDA): ~40–80ms per image
- Video (50 frames): ~15–30s on CPU, ~3–6s on GPU
- Batch (20 images): ~5–8s on CPU

---

## 🚀 Future Improvements & Feature Roadmap

DeepShield is designed as a modular deepfake forensics platform.  
The following improvements are planned for future releases:

---

### 🧠 Advanced AI Detection

- Integration of **Vision Transformers (ViT / Swin Transformer)** for stronger generalization
- Expansion of ensemble with **temporal transformer-based video models**
- Self-supervised pretraining on large unlabeled real-world media datasets
- Domain adaptation to handle heavy **social-media compression artifacts**
- Cross-dataset robustness benchmarking

---

### 🧬 Diffusion-Edited Image Detection (Planned Feature)

Modern fake media increasingly comes from **diffusion-based editing tools** rather than classical face-swap deepfakes.

A future release will introduce:

- A **dedicated diffusion-artifact classifier**
- Training on datasets such as:
  - **CIFAKE**
  - **GenImage**
- Detection capabilities for:
  - Stable Diffusion edits
  - Midjourney / DALL-E synthetic artifacts
  - AI inpainting & outpainting traces
  - Generative upscaling hallucination patterns

> ⚠️ This module will be added **after initial deployment**, as it requires  
> a separate training pipeline, curated datasets, and real-world calibration.

---

### 🎥 Video Intelligence Upgrades

- Face tracking with identity persistence across frames
- Lip-sync consistency analysis
- Head-pose trajectory modelling
- Audio-visual synchronization detection
- Temporal anomaly scoring

---

### 📊 Forensics & Explainability Enhancements

- Layer-wise attention visualization
- Patch-level authenticity scoring
- Frequency heatmap overlays
- Confidence calibration using temperature scaling
- Forensic decision fusion dashboard

---

### 🌐 Platform & Deployment Features

- REST API for third-party integrations
- Docker container deployment
- Real-time streaming deepfake detection
- Cloud inference pipeline (AWS / GCP / Azure)
- User authentication & forensic case management
- Model auto-update and version control

---

### 📱 UI / UX Improvements

- Drag-and-drop video timeline navigation
- Multi-report comparison dashboard
- Authenticity risk score indicator
- Professional dark forensic theme
- Exportable investigation workspace

---

## References

1. Rössler et al., _FaceForensics++: Learning to Detect Manipulated Facial Images_, ICCV 2019
2. Chollet, _Xception: Deep Learning with Depthwise Separable Convolutions_, CVPR 2017
3. Tan & Le, _EfficientNet: Rethinking Model Scaling for CNNs_, ICML 2019
4. Selvaraju et al., _Grad-CAM: Visual Explanations from Deep Networks_, ICCV 2017
5. Li et al., _Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics_, CVPR 2020

---

## License

MIT License. See LICENSE for details.

---

## Disclaimer

DeepShield is a research tool. Results should be interpreted by qualified professionals.
No deepfake detector achieves 100% accuracy. Do not use as sole evidence in legal proceedings.

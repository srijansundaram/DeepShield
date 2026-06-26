"""
DeepShield — Combined Dataset Preparation
Merges 140k + CIFAKE + Celeb-DF into unified train/val structure
"""

import os
import cv2
import shutil
import random
from pathlib import Path
from tqdm import tqdm
from PIL import Image

random.seed(42)

OUTPUT_DIR = Path("/home/srijansundaram/datasets/deepshield/prepared")
TRAIN_REAL = OUTPUT_DIR / "train" / "real"
TRAIN_FAKE = OUTPUT_DIR / "train" / "fake"
VAL_REAL   = OUTPUT_DIR / "val" / "real"
VAL_FAKE   = OUTPUT_DIR / "val" / "fake"

for d in [TRAIN_REAL, TRAIN_FAKE, VAL_REAL, VAL_FAKE]:
    d.mkdir(parents=True, exist_ok=True)

VAL_SPLIT = 0.15
CELEB_MAX_FRAMES = 20
IMG_SIZE = 299

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def extract_face(frame_bgr, size=299):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        h, w = frame_bgr.shape[:2]
        cx, cy = w//2, h//2
        s = min(w, h) // 2
        crop = frame_bgr[cy-s//2:cy+s//2, cx-s//2:cx+s//2]
    else:
        x, y, w, h = faces[0]
        pad = int(max(w, h) * 0.3)
        H, W = frame_bgr.shape[:2]
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(W, x+w+pad), min(H, y+h+pad)
        crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (size, size))
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def copy_images(src_dir, label, tag, max_images=None):
    src = Path(src_dir)
    images = list(src.rglob("*.jpg")) + list(src.rglob("*.png")) + list(src.rglob("*.jpeg"))
    random.shuffle(images)
    if max_images:
        images = images[:max_images]

    n_val = max(1, int(len(images) * VAL_SPLIT))
    val_set = set(range(n_val))
    copied = 0

    dst_train = TRAIN_REAL if label == "real" else TRAIN_FAKE
    dst_val   = VAL_REAL   if label == "real" else VAL_FAKE

    for i, img_path in enumerate(tqdm(images, desc=f"  {tag}/{label}")):
        dst = dst_val if i < n_val else dst_train
        fname = f"{tag}_{label}_{i:06d}.jpg"
        try:
            img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            img.save(dst / fname, quality=95)
            copied += 1
        except Exception:
            continue
    return copied


def extract_video_frames(video_path, label, tag, idx):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // CELEB_MAX_FRAMES)
    saved = 0

    dst_train = TRAIN_REAL if label == "real" else TRAIN_FAKE
    dst_val   = VAL_REAL   if label == "real" else VAL_FAKE

    for frame_idx in range(0, total, step):
        if saved >= CELEB_MAX_FRAMES:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        face = extract_face(frame)
        if face is None:
            continue
        dst = dst_val if idx % 7 == 0 else dst_train
        fname = f"{tag}_{label}_{idx:04d}_{saved:03d}.jpg"
        Image.fromarray(face).save(dst / fname, quality=95)
        saved += 1

    cap.release()
    return saved


# ── Dataset 1: 140k Real and Fake Faces ──
print("\n[1/3] Processing 140k Real and Fake Faces...")
base_140k = Path("../datasets/deepshield/140k/real_vs_fake/real-vs-fake")

real_140k = copy_images(base_140k / "train" / "real", "real", "140k", max_images=30000)
fake_140k = copy_images(base_140k / "train" / "fake", "fake", "140k", max_images=30000)
print(f"  ✓ 140k: {real_140k} real, {fake_140k} fake")


# ── Dataset 2: CIFAKE ──
print("\n[2/3] Processing CIFAKE...")
base_cifake = Path("../datasets/deepshield/cifake")

real_cifake = copy_images(base_cifake / "train" / "REAL", "real", "cifake", max_images=15000)
fake_cifake = copy_images(base_cifake / "train" / "FAKE", "fake", "cifake", max_images=15000)
print(f"  ✓ CIFAKE: {real_cifake} real, {fake_cifake} fake")


# ── Dataset 3: Celeb-DF ──
print("\n[3/3] Processing Celeb-DF v2 (extracting frames)...")
base_celeb = Path("../datasets/deepshield/celebdf")

real_vids = list((base_celeb / "Celeb-real").rglob("*.mp4")) + \
            list((base_celeb / "YouTube-real").rglob("*.mp4"))
fake_vids  = list((base_celeb / "Celeb-synthesis").rglob("*.mp4"))

random.shuffle(real_vids)
random.shuffle(fake_vids)

celeb_real = 0
for i, v in enumerate(tqdm(real_vids[:80], desc="  celebdf/real")):
    celeb_real += extract_video_frames(v, "real", "celeb", i)

celeb_fake = 0
for i, v in enumerate(tqdm(fake_vids[:80], desc="  celebdf/fake")):
    celeb_fake += extract_video_frames(v, "fake", "celeb", i)

print(f"  ✓ Celeb-DF: {celeb_real} real frames, {celeb_fake} fake frames")

# ── Dataset 4: Wild Real Faces ──
print("\n[4/4] Processing real-world faces...")
base_wild = Path("/home/srijansundaram/datasets/deepshield/wildfaces")
wild_real = copy_images(base_wild, "real", "wild", max_images=10000)
print(f"  ✓ Wild faces: {wild_real} real")

# ── Summary ──
def count(folder):
    return len(list(Path(folder).rglob("*.jpg")))

print(f"""
╔══════════════════════════════════════╗
║     DATASET PREPARATION COMPLETE     ║
╠══════════════════════════════════════╣
║  train/real : {count(TRAIN_REAL):>6} images          ║
║  train/fake : {count(TRAIN_FAKE):>6} images          ║
║  val/real   : {count(VAL_REAL):>6} images          ║
║  val/fake   : {count(VAL_FAKE):>6} images          ║
╚══════════════════════════════════════╝
""")
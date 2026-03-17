"""
DeepShield — Dataset Preparation
Extracts face crops from FaceForensics++ or any video/image dataset.

Usage:
  # From FF++ videos:
  python prepare_dataset.py --source ./ff++ --output ./data --mode video

  # From already extracted frames:
  python prepare_dataset.py --source ./frames --output ./data --mode images

  # Quick test with synthetic data (no dataset needed):
  python prepare_dataset.py --mode synthetic --output ./data --n_samples 200
"""

import os
import cv2
import argparse
import shutil
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
from tqdm import tqdm


FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def extract_face(frame_bgr, padding=0.3, target_size=299):
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    ph, pw = int(h * padding), int(w * padding)
    H, W   = frame_bgr.shape[:2]
    x1 = max(0, x - pw); y1 = max(0, y - ph)
    x2 = min(W, x + w + pw); y2 = min(H, y + h + ph)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (target_size, target_size))
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def extract_from_video(video_path, out_dir, label, max_frames=30, target_size=299):
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, total // max_frames)
    saved = 0
    idx   = 0

    stem  = Path(video_path).stem
    while cap.isOpened() and saved < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        face = extract_face(frame, target_size=target_size)
        if face is not None:
            fname = os.path.join(out_dir, label, f"{stem}_{idx:05d}.jpg")
            Image.fromarray(face).save(fname, quality=95)
            saved += 1
        idx += step

    cap.release()
    return saved


# ── Synthetic dataset for testing ─────────────────────
def make_synthetic_dataset(output_dir, n_samples=200, target_size=299):
    """
    Creates fake training images for testing the pipeline WITHOUT a real dataset.
    Real images = clean face-like blobs
    Fake images = same but with visible blending artifacts
    WARNING: This does NOT train a useful model — only for code testing.
    """
    print(f"\n  Generating {n_samples} synthetic samples per class…")
    random.seed(42)
    np.random.seed(42)

    for split in ["train", "val"]:
        for label in ["real", "fake"]:
            os.makedirs(os.path.join(output_dir, split, label), exist_ok=True)

    n_train = int(n_samples * 0.8)
    n_val   = n_samples - n_train

    for split, count in [("train", n_train), ("val", n_val)]:
        for label in ["real", "fake"]:
            out = os.path.join(output_dir, split, label)
            for i in tqdm(range(count), desc=f"  {split}/{label}", leave=False):
                img = _make_synthetic_face(target_size, is_fake=(label == "fake"))
                img.save(os.path.join(out, f"{label}_{i:05d}.jpg"), quality=92)

    print(f"  ✅ Synthetic dataset created at: {output_dir}")
    print(f"  ⚠  This dataset is for PIPELINE TESTING only — not for real deepfake detection.")


def _make_synthetic_face(size=299, is_fake=False):
    """Generate a simple synthetic face image."""
    img = Image.new("RGB", (size, size), _rand_bg())
    draw = ImageDraw.Draw(img)

    # Face oval
    cx, cy = size // 2, size // 2
    fw, fh = int(size * 0.42), int(size * 0.52)
    skin = (
        random.randint(160, 220),
        random.randint(120, 170),
        random.randint(90, 130),
    )
    draw.ellipse([cx - fw, cy - fh, cx + fw, cy + fh], fill=skin)

    # Eyes
    for ex in [cx - fw // 3, cx + fw // 3]:
        ey = cy - fh // 6
        draw.ellipse([ex - 12, ey - 7, ex + 12, ey + 7], fill=(30, 20, 15))
        draw.ellipse([ex - 5, ey - 3, ex + 5, ey + 3], fill=(255, 255, 240))

    # Nose
    draw.polygon([
        (cx, cy - 5), (cx - 8, cy + 18), (cx + 8, cy + 18)
    ], fill=tuple(max(0, c - 30) for c in skin))

    # Mouth
    draw.arc([cx - 22, cy + fh // 4, cx + 22, cy + fh // 3 + 10],
             start=0, end=180, fill=(160, 60, 60), width=3)

    # Fake artifacts: visible blending seam + color shift in patches
    if is_fake:
        arr = np.array(img).astype(np.float32)
        # Blending seam — horizontal line across forehead
        seam_y = cy - fh // 2 + random.randint(-10, 10)
        arr[seam_y - 2: seam_y + 2, cx - fw: cx + fw] *= 0.6
        arr[seam_y - 2: seam_y + 2, cx - fw: cx + fw] += 60

        # Color patch artifact in eye region
        patch_x = cx + random.randint(-fw // 2, fw // 2)
        patch_y = cy + random.randint(-fh // 3, fh // 3)
        r = random.randint(12, 25)
        Y, X = np.ogrid[:size, :size]
        mask = (X - patch_x) ** 2 + (Y - patch_y) ** 2 < r ** 2
        arr[mask, 0] = np.clip(arr[mask, 0] * random.uniform(0.5, 1.5), 0, 255)

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # Slight blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
    return img


def _rand_bg():
    c = random.randint(160, 230)
    return (c, c, c)


# ── Main ──────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source",    type=str, default=None)
    p.add_argument("--output",    type=str, default="./data")
    p.add_argument("--mode",      type=str, default="synthetic",
                   choices=["video", "images", "synthetic"])
    p.add_argument("--n_samples", type=int, default=500,
                   help="Samples per class (synthetic mode)")
    p.add_argument("--max_frames",type=int, default=30,
                   help="Frames to extract per video")
    p.add_argument("--img_size",  type=int, default=299)
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.mode == "synthetic":
        make_synthetic_dataset(args.output, args.n_samples, args.img_size)

    elif args.mode == "video":
        if not args.source:
            print("❌ --source required for video mode")
            return

        # Expect source/real/*.mp4 and source/fake/*.mp4
        for split in ["train", "val"]:
            for label in ["real", "fake"]:
                os.makedirs(os.path.join(args.output, split, label), exist_ok=True)

        for label in ["real", "fake"]:
            vid_dir = os.path.join(args.source, label)
            if not os.path.isdir(vid_dir):
                print(f"⚠ Skipping {vid_dir} — not found")
                continue
            videos = [f for f in Path(vid_dir).rglob("*") if f.suffix in {".mp4", ".avi", ".mov"}]
            print(f"\n  Processing {len(videos)} {label} videos…")

            random.shuffle(videos)
            n_val = max(1, int(len(videos) * 0.2))
            splits = {"val": videos[:n_val], "train": videos[n_val:]}

            for split, vids in splits.items():
                total = 0
                for v in tqdm(vids, desc=f"  {split}/{label}"):
                    out_dir = os.path.join(args.output, split)
                    n = extract_from_video(str(v), out_dir, label, args.max_frames, args.img_size)
                    total += n
                print(f"  {split}/{label}: {total} face crops saved")

    elif args.mode == "images":
        if not args.source:
            print("❌ --source required for images mode")
            return
        # Expect source/real/ and source/fake/ with images
        for split in ["train", "val"]:
            for label in ["real", "fake"]:
                os.makedirs(os.path.join(args.output, split, label), exist_ok=True)

        for label in ["real", "fake"]:
            src = os.path.join(args.source, label)
            imgs = list(Path(src).rglob("*.jpg")) + list(Path(src).rglob("*.png"))
            random.shuffle(imgs)
            n_val = max(1, int(len(imgs) * 0.2))
            for i, img_path in enumerate(tqdm(imgs, desc=f"  {label}")):
                split = "val" if i < n_val else "train"
                dst   = os.path.join(args.output, split, label, img_path.name)
                shutil.copy2(img_path, dst)
            print(f"  {label}: {len(imgs)} images split into train/val")

    print(f"\n  Dataset ready at: {args.output}")


if __name__ == "__main__":
    main()

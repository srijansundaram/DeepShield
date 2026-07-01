"""
DeepShield — Tier 1 Dataset Preparation
Verified Kaggle datasets as of June 2026.

Usage:
    python prepare_tier1.py                # download + prepare
    python prepare_tier1.py --skip_download  # just prepare from existing folders
"""

import os
import random
import shutil
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

random.seed(42)

BASE_DIR   = Path("/home/srijansundaram/datasets/deepshield")
OUTPUT_DIR = BASE_DIR / "prepared"
VAL_SPLIT  = 0.15
IMG_SIZE   = 299

STYLEGAN_DIR  = BASE_DIR / "stylegan3"
DIFFUSION_DIR = BASE_DIR / "diffusion_gan"
FACES_140K    = BASE_DIR / "140k" / "real_vs_fake" / "real-vs-fake"


def check_kaggle_auth():
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print(f"  ❌ kaggle.json not found at {kaggle_json}")
        raise SystemExit(1)


def kaggle_download(slug, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {slug} → {dest}")
    ret = os.system(f"kaggle datasets download -d {slug} -p {dest} --unzip -q")
    if ret != 0:
        raise RuntimeError(f"Failed: {slug}")


def collect_images(folder, exts=(".jpg", ".jpeg", ".png", ".webp")):
    paths = []
    for ext in exts:
        paths += list(Path(folder).rglob(f"*{ext}"))
    return paths


def save_split(img_paths, label, tag, max_images=None):
    random.shuffle(img_paths)
    if max_images:
        img_paths = img_paths[:max_images]
    n_val     = max(1, int(len(img_paths) * VAL_SPLIT))
    dst_train = OUTPUT_DIR / "train" / label
    dst_val   = OUTPUT_DIR / "val"   / label
    dst_train.mkdir(parents=True, exist_ok=True)
    dst_val.mkdir(parents=True, exist_ok=True)
    copied = 0
    for i, p in enumerate(tqdm(img_paths, desc=f"  {tag}/{label}")):
        dst   = dst_val if i < n_val else dst_train
        fname = f"{tag}_{label}_{i:06d}.jpg"
        try:
            Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE)).save(dst / fname, quality=95)
            copied += 1
        except Exception:
            continue
    return copied


def download_datasets():
    print("\n[Download] Starting...")
    check_kaggle_auth()

    if not any(STYLEGAN_DIR.rglob("*.jpg")) and not any(STYLEGAN_DIR.rglob("*.png")):
        kaggle_download("troykueh/real-vs-fake-faces-stylegan3", STYLEGAN_DIR)
    else:
        print(f"  ✓ StyleGAN already exists: {STYLEGAN_DIR}")

    if not any(DIFFUSION_DIR.rglob("*.jpg")) and not any(DIFFUSION_DIR.rglob("*.png")):
        kaggle_download("fatimasalman/real-photoshop-gan-diffusion-faces", DIFFUSION_DIR)
    else:
        print(f"  ✓ Diffusion already exists: {DIFFUSION_DIR}")


def prepare_stylegan(max_images=15000):
    print("\n[1/3] StyleGAN3 fakes...")
    if not STYLEGAN_DIR.exists():
        print(f"  ❌ Not found: {STYLEGAN_DIR}")
        return 0

    # Dataset structure: has 'fake' and 'real' subdirs
    fake_paths = collect_images(STYLEGAN_DIR / "fake")
    if not fake_paths:
        # flat structure — try filtering by folder name
        all_paths = collect_images(STYLEGAN_DIR)
        fake_paths = [p for p in all_paths if "fake" in str(p).lower()]
    if not fake_paths:
        # no label in path — take everything as fakes (dataset is StyleGAN3 generated)
        fake_paths = collect_images(STYLEGAN_DIR)

    print(f"  Found {len(fake_paths)} StyleGAN fake images")
    n = save_split(fake_paths, "fake", "stylegan3", max_images)
    print(f"  ✓ Added {n}")
    return n


def prepare_diffusion(max_images=15000):
    print("\n[2/3] Diffusion + GAN fakes...")
    if not DIFFUSION_DIR.exists():
        print(f"  ❌ Not found: {DIFFUSION_DIR}")
        return 0

    # fatimasalman dataset has: real/, photoshop/, gan/, diffusion/ subdirs
    fake_paths = []
    for subdir in ["diffusion", "gan", "Diffusion", "GAN", "fake", "Fake"]:
        d = DIFFUSION_DIR / subdir
        if d.exists():
            fake_paths += collect_images(d)

    if not fake_paths:
        # fallback — everything not in 'real' folder
        all_paths = collect_images(DIFFUSION_DIR)
        fake_paths = [p for p in all_paths if "real" not in str(p).lower()]

    print(f"  Found {len(fake_paths)} diffusion/GAN fake images")
    n = save_split(fake_paths, "fake", "diffusion", max_images)

    # Also grab reals from this dataset to balance
    real_paths = []
    for subdir in ["real", "Real"]:
        d = DIFFUSION_DIR / subdir
        if d.exists():
            real_paths += collect_images(d)
    if real_paths:
        nr = save_split(real_paths, "real", "diffusion_real", min(max_images, len(real_paths)))
        print(f"  ✓ Also added {nr} real faces from this dataset")

    print(f"  ✓ Added {n} fakes")
    return n


def prepare_existing_140k(max_images=10000):
    """Pull more fakes from your existing 140k dataset."""
    print("\n[3/3] Extra fakes from existing 140k dataset...")
    fake_dir = FACES_140K / "train" / "fake"
    if not fake_dir.exists():
        fake_dir = FACES_140K / "test" / "fake"
    if not fake_dir.exists():
        print(f"  ❌ 140k fake dir not found at {FACES_140K}")
        return 0

    paths = collect_images(fake_dir)
    # These are already in training — sample a fresh subset tagged differently
    paths = [p for p in paths if "140k" not in p.stem]
    print(f"  Found {len(paths)} additional 140k fakes")
    n = save_split(paths, "fake", "140k_extra", max_images)
    print(f"  ✓ Added {n}")
    return n


def count(folder):
    f = Path(folder)
    return len(list(f.rglob("*.jpg"))) if f.exists() else 0


def main(skip_download, max_per_class):
    print(f"\nDeepShield Tier 1 prep")
    print(f"Output: {OUTPUT_DIR}")

    if not skip_download:
        download_datasets()
    else:
        print("\n[Download] Skipped")

    n1 = prepare_stylegan(max_per_class)
    n2 = prepare_diffusion(max_per_class)
    n3 = prepare_existing_140k(max_per_class)

    tr = count(OUTPUT_DIR / "train" / "real")
    tf = count(OUTPUT_DIR / "train" / "fake")
    vr = count(OUTPUT_DIR / "val"   / "real")
    vf = count(OUTPUT_DIR / "val"   / "fake")

    print(f"""
╔══════════════════════════════════════════╗
║      TIER 1 DATASET PREP COMPLETE        ║
╠══════════════════════════════════════════╣
║  StyleGAN3 fakes : {n1:<6}              ║
║  Diffusion fakes : {n2:<6}              ║
║  140k extra      : {n3:<6}              ║
╠══════════════════════════════════════════╣
║  train/real : {tr:<6}                   ║
║  train/fake : {tf:<6}                   ║
║  val/real   : {vr:<6}                   ║
║  val/fake   : {vf:<6}                   ║
╚══════════════════════════════════════════╝

Next:
  python train.py --model xception         --data_dir {OUTPUT_DIR} --epochs 15
  python train.py --model efficientnet_b4  --data_dir {OUTPUT_DIR} --epochs 15
  python calibrate.py
""")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skip_download", action="store_true")
    p.add_argument("--max_per_class", type=int, default=15000)
    args = p.parse_args()
    main(args.skip_download, args.max_per_class)
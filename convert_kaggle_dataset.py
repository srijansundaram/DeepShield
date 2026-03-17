"""
DeepShield — Kaggle Dataset Converter
Converts the '140k Real and Fake Faces' Kaggle dataset
into the format expected by train.py

Kaggle structure:
  real_vs_fake/
    train/
      real/   ← images
      fake/   ← images
    valid/
      real/
      fake/
    test/
      real/
      fake/

DeepShield expected structure:
  data/
    train/
      real/
      fake/
    val/
      real/
      fake/

Usage:
  python convert_kaggle_dataset.py --kaggle_dir "C:/path/to/real_vs_fake" --output_dir "./data"
"""

import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


def convert(kaggle_dir: str, output_dir: str):
    kaggle_dir  = Path(kaggle_dir)
    output_dir  = Path(output_dir)

    # Mapping: kaggle folder name → deepshield folder name
    split_map = {
        "train": "train",
        "valid": "val",   # kaggle calls it 'valid', we need 'val'
        # 'test' is ignored for training
    }

    total_copied = 0

    for kaggle_split, ds_split in split_map.items():
        for label in ["real", "fake"]:
            src = kaggle_dir / kaggle_split / label
            dst = output_dir / ds_split / label

            if not src.exists():
                print(f"  ⚠  Not found, skipping: {src}")
                continue

            dst.mkdir(parents=True, exist_ok=True)

            images = list(src.glob("*.jpg")) + list(src.glob("*.png")) + list(src.glob("*.jpeg"))

            print(f"  Copying {len(images):,} images  {kaggle_split}/{label}  →  {ds_split}/{label}")

            for img_path in tqdm(images, desc=f"  {ds_split}/{label}", leave=False):
                shutil.copy2(img_path, dst / img_path.name)

            total_copied += len(images)

    print(f"\n  ✅ Done! {total_copied:,} images copied to: {output_dir}")
    print(f"\n  Final structure:")

    for split in ["train", "val"]:
        for label in ["real", "fake"]:
            folder = output_dir / split / label
            if folder.exists():
                count = len(list(folder.glob("*")))
                print(f"    {output_dir}/{split}/{label}/  →  {count:,} images")

    print(f"\n  ✅ Ready to train! Run:")
    print(f"     python train.py --data_dir {output_dir} --model xception --epochs 20 --batch_size 32")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kaggle_dir",  type=str, required=True,
                   help="Path to extracted real_vs_fake folder from Kaggle")
    p.add_argument("--output_dir",  type=str, default="./data",
                   help="Where to put the converted dataset (default: ./data)")
    args = p.parse_args()

    print(f"\n  DeepShield — Kaggle Dataset Converter")
    print(f"  Source : {args.kaggle_dir}")
    print(f"  Output : {args.output_dir}\n")

    convert(args.kaggle_dir, args.output_dir)

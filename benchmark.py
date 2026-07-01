"""
DeepShield — Cross-dataset Benchmark
Evaluates trained models on FaceForensics++ and DFDC test sets.

Usage:
    python benchmark.py
"""

import sys
import json
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

BENCHMARK_DIR = Path("/home/srijansundaram/datasets/deepshield/benchmark")
OUTPUT_FILE   = Path("/home/srijansundaram/DeepShield/benchmark_results.json")
REPORT_FILE   = Path("/home/srijansundaram/DeepShield/benchmark_report.txt")

DATASETS = [
    ("DFDC Validation", BENCHMARK_DIR / "dfdc" / "validation" / "real", BENCHMARK_DIR / "dfdc" / "validation" / "fake"),
    ("DFDC Train",      BENCHMARK_DIR / "dfdc" / "train" / "real",      BENCHMARK_DIR / "dfdc" / "train" / "fake"),
]


def collect_images(folder, max_images=500):
    folder = Path(folder)
    if not folder.exists():
        return []
    paths = []
    for ext in [".jpg", ".jpeg", ".png"]:
        paths += list(folder.rglob(f"*{ext}"))
    import random
    random.shuffle(paths)
    return paths[:max_images]


def run_benchmark(detector, name, real_paths, fake_paths):
    print(f"\n  [{name}]")
    print(f"  Real: {len(real_paths)} | Fake: {len(fake_paths)}")

    all_labels, all_probs, errors = [], [], 0
    start = time.time()

    for label, paths in [(0, real_paths), (1, fake_paths)]:
        for p in tqdm(paths, desc=f"  {'Real' if label==0 else 'Fake'}"):
            try:
                img    = Image.open(p).convert("RGB")
                result = detector.predict_image(img)
                all_probs.append(result["fake_probability"])
                all_labels.append(label)
            except Exception:
                errors += 1
                continue

    if not all_labels:
        return None

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = (all_probs > 0.5).astype(int)
    elapsed    = time.time() - start
    n          = len(all_labels)

    metrics = {
        "dataset":               name,
        "n_samples":             n,
        "n_real":                int((all_labels == 0).sum()),
        "n_fake":                int((all_labels == 1).sum()),
        "accuracy":              round(accuracy_score(all_labels, all_preds) * 100, 2),
        "auc":                   round(roc_auc_score(all_labels, all_probs), 4),
        "precision":             round(precision_score(all_labels, all_preds, zero_division=0), 4),
        "recall":                round(recall_score(all_labels, all_preds, zero_division=0), 4),
        "f1":                    round(f1_score(all_labels, all_preds, zero_division=0), 4),
        "false_positive_rate":   round(((all_preds == 1) & (all_labels == 0)).sum() / max((all_labels == 0).sum(), 1), 4),
        "errors":                errors,
        "inference_ms_per_image": round(elapsed / n * 1000, 1),
    }
    metrics["confusion_matrix"] = confusion_matrix(all_labels, all_preds).tolist()

    print(f"  Accuracy: {metrics['accuracy']}% | AUC: {metrics['auc']} | F1: {metrics['f1']} | FPR: {metrics['false_positive_rate']}")
    print(f"  Speed: {metrics['inference_ms_per_image']}ms/image")
    return metrics


def generate_report(all_results):
    lines = ["=" * 60, "  DeepShield — Cross-dataset Benchmark Report", "=" * 60, ""]
    for r in all_results:
        if r is None:
            continue
        lines += [
            f"  Dataset   : {r['dataset']}",
            f"  Samples   : {r['n_samples']} ({r['n_real']} real, {r['n_fake']} fake)",
            f"  Accuracy  : {r['accuracy']}%",
            f"  AUC-ROC   : {r['auc']}",
            f"  F1 Score  : {r['f1']}",
            f"  Precision : {r['precision']}",
            f"  Recall    : {r['recall']}",
            f"  FP Rate   : {r['false_positive_rate']}",
            f"  Speed     : {r['inference_ms_per_image']}ms/image",
            "",
            "  Confusion Matrix (rows=actual, cols=predicted):",
            "               Pred Real  Pred Fake",
            f"  Actual Real  {r['confusion_matrix'][0][0]:<10} {r['confusion_matrix'][0][1]}",
            f"  Actual Fake  {r['confusion_matrix'][1][0]:<10} {r['confusion_matrix'][1][1]}",
            "", "-" * 60, "",
        ]
    report = "\n".join(lines)
    REPORT_FILE.write_text(report)
    print(f"\n  Report saved → {REPORT_FILE}\n\n{report}")


def main():
    from models.detector import EnsembleDetector
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    detector = EnsembleDetector(device=device)

    if not detector.is_trained:
        print("⚠ No trained checkpoint found.")

    all_results = []
    for name, real_dir, fake_dir in DATASETS:
        real_paths = collect_images(real_dir, max_images=500)
        fake_paths = collect_images(fake_dir, max_images=500)
        if not real_paths and not fake_paths:
            print(f"\n  ⚠ Skipping {name} — no images at {real_dir} / {fake_dir}")
            continue
        result = run_benchmark(detector, name, real_paths, fake_paths)
        if result:
            all_results.append(result)

    if all_results:
        with open(OUTPUT_FILE, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  JSON saved → {OUTPUT_FILE}")
        generate_report(all_results)
    else:
        print("\n  No datasets found.")


if __name__ == "__main__":
    main()
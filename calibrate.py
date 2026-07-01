"""
DeepShield — Temperature Scaling Calibration
Run once after training. Saves calibration.json to checkpoints/.
No retraining required.

Usage:
    python calibrate.py \
        --val_dir ~/datasets/deepshield/prepared/val \
        --checkpoint_dir ~/DeepShield/checkpoints
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm


VAL_TF_XC = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

VAL_TF_EFF = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def collect_logits(model, val_dir, transform, device, batch_size=32):
    dataset = datasets.ImageFolder(val_dir, transform=transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)
    all_logits, all_labels = [], []
    model.eval()
    for imgs, labels in tqdm(loader, desc="  Collecting logits"):
        all_logits.append(model(imgs.to(device)).cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def nll_with_temp(logits, labels, T):
    return F.nll_loss(F.log_softmax(logits / T, dim=1), labels).item()


def find_temperature(logits, labels, lo=0.1, hi=10.0, steps=500):
    best_T, best_nll = 1.0, float("inf")
    for T in torch.linspace(lo, hi, steps):
        nll = nll_with_temp(logits, labels, T.item())
        if nll < best_nll:
            best_nll, best_T = nll, T.item()
    return round(best_T, 4)


def compute_ece(logits, labels, T, n_bins=15):
    probs = F.softmax(logits / T, dim=1)
    confs, preds = probs.max(dim=1)
    correct = preds.eq(labels)
    ece = 0.0
    for lo, hi in zip(torch.linspace(0,1,n_bins+1), torch.linspace(0,1,n_bins+1)[1:]):
        mask = (confs >= lo.item()) & (confs < hi.item())
        if mask.sum() == 0:
            continue
        ece += mask.sum().item() * abs(correct[mask].float().mean().item() - confs[mask].mean().item())
    return ece / len(labels)


def calibrate(val_dir, checkpoint_dir, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    results = {}

    configs = [
        {"name": "xception",       "ckpt": "xception_deepfake.pth",     "arch": "xception",        "transform": VAL_TF_XC,  "key": "xception_T"},
        {"name": "efficientnet_b4","ckpt": "efficientnet_deepfake.pth",  "arch": "efficientnet_b4", "transform": VAL_TF_EFF, "key": "efficientnet_T"},
    ]

    for cfg in configs:
        ckpt_path = os.path.join(checkpoint_dir, cfg["ckpt"])
        if not os.path.exists(ckpt_path):
            print(f"\n  ⚠ Skipping {cfg['name']} — checkpoint not found")
            results[cfg["key"]] = 1.0
            continue

        print(f"\n  ── Calibrating {cfg['name']} ──")
        model = timm.create_model(cfg["arch"], pretrained=False, num_classes=2).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)

        logits, labels = collect_logits(model, val_dir, cfg["transform"], device, batch_size)

        ece_before = compute_ece(logits, labels, 1.0)
        T_opt      = find_temperature(logits, labels)
        ece_after  = compute_ece(logits, labels, T_opt)

        print(f"  Optimal T         : {T_opt}")
        print(f"  ECE before → after: {ece_before:.4f} → {ece_after:.4f}")

        results[cfg["key"]]                    = T_opt
        results[f"{cfg['name']}_ece_before"]   = round(ece_before, 4)
        results[f"{cfg['name']}_ece_after"]    = round(ece_after,  4)

    out_path = os.path.join(checkpoint_dir, "calibration.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  ✅ Saved → {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--val_dir",        default=os.path.expanduser("~/datasets/deepshield/prepared/val"))
    p.add_argument("--checkpoint_dir", default=os.path.expanduser("~/DeepShield/checkpoints"))
    p.add_argument("--batch_size",     type=int, default=32)
    args = p.parse_args()
    calibrate(args.val_dir, args.checkpoint_dir, args.batch_size)
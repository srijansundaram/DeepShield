"""
DeepShield — Training Pipeline
Trains XceptionNet + EfficientNet-B4 on deepfake datasets.

Supports:
  - FaceForensics++ (FF++)
  - Celeb-DF v2
  - Custom dataset (any folder with real/ and fake/ subfolders)

Usage:
  python train.py --data_dir ./data --model xception --epochs 20 --batch_size 32

Google Colab quick start:
  !python train.py --data_dir /content/data --model xception --epochs 20 --batch_size 32 --colab
"""

import os
import sys
import time
import argparse
import json
import shutil
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from torchvision.utils import make_grid
import timm
import numpy as np
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════
# 1. ARGUMENT PARSER
# ══════════════════════════════════════════════════════
def get_args():
    p = argparse.ArgumentParser(description="DeepShield Training Script")

    # Paths
    p.add_argument("--data_dir",    type=str, default="./data",
                   help="Root dataset directory. Expects train/real, train/fake, val/real, val/fake")
    p.add_argument("--output_dir",  type=str, default="./checkpoints",
                   help="Where to save model checkpoints")
    p.add_argument("--log_dir",     type=str, default="./training_logs",
                   help="Where to save training plots and metrics")

    # Model
    p.add_argument("--model",       type=str, default="xception",
                   choices=["xception", "efficientnet_b4", "both"],
                   help="Which model to train")
    p.add_argument("--pretrained",  action="store_true", default=True,
                   help="Use ImageNet pretrained weights (recommended)")

    # Training
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--patience",    type=int,   default=5,
                   help="Early stopping patience (epochs)")
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--seed",        type=int,   default=42)

    # Data
    p.add_argument("--img_size_xception",    type=int, default=299)
    p.add_argument("--img_size_efficientnet",type=int, default=380)
    p.add_argument("--val_split",   type=float, default=0.2,
                   help="Validation split if no val/ folder exists")
    p.add_argument("--max_per_class", type=int, default=None,
                   help="Cap images per class (useful for quick testing)")

    # Misc
    p.add_argument("--colab",       action="store_true",
                   help="Colab mode: mount drive, adjust paths")
    p.add_argument("--resume",      type=str,   default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--fp16",        action="store_true",
                   help="Use mixed precision training (faster on GPU)")

    return p.parse_args()


# ══════════════════════════════════════════════════════
# 2. DATASET
# ══════════════════════════════════════════════════════
class DeepfakeDataset(Dataset):
    """
    Expects folder structure:
        data/
          train/
            real/   ← jpg/png face images
            fake/   ← jpg/png face images
          val/
            real/
            fake/

    Also supports flat structure:
        data/
          real/
          fake/
    (will auto-split into train/val)
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        root: str,
        split: str = "train",        # "train" or "val"
        img_size: int = 299,
        augment: bool = True,
        max_per_class: Optional[int] = None,
        val_split: float = 0.2,
    ):
        self.img_size = img_size
        self.augment  = augment and split == "train"

        # Build transforms
        if self.augment:
            self.transform = T.Compose([
                T.Resize((img_size + 32, img_size + 32)),
                T.RandomCrop(img_size),
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

        # Find images
        self.samples = []  # list of (path, label)  0=real, 1=fake

        split_dir = os.path.join(root, split)
        if os.path.isdir(split_dir):
            # Proper train/val split exists
            real_dir = os.path.join(split_dir, "real")
            fake_dir = os.path.join(split_dir, "fake")
        else:
            # Flat structure — do manual split
            real_dir = os.path.join(root, "real")
            fake_dir = os.path.join(root, "fake")
            if not os.path.isdir(real_dir) or not os.path.isdir(fake_dir):
                raise FileNotFoundError(
                    f"Cannot find dataset at {root}.\n"
                    f"Expected: {root}/train/real, {root}/train/fake\n"
                    f"      or: {root}/real, {root}/fake"
                )

        real_imgs = self._scan(real_dir, max_per_class)
        fake_imgs = self._scan(fake_dir, max_per_class)

        # If using flat structure, do val split
        if not os.path.isdir(split_dir):
            real_imgs, fake_imgs = self._manual_split(
                real_imgs, fake_imgs, split, val_split
            )

        self.samples = [(p, 0) for p in real_imgs] + [(p, 1) for p in fake_imgs]
        self.class_counts = {0: len(real_imgs), 1: len(fake_imgs)}

        print(f"  [{split}] real={len(real_imgs):,}  fake={len(fake_imgs):,}  total={len(self.samples):,}")

    def _scan(self, folder: str, limit: Optional[int]) -> List[str]:
        paths = []
        if not os.path.isdir(folder):
            print(f"  ⚠ Folder not found: {folder}")
            return paths
        for root, _, files in os.walk(folder):
            for f in files:
                if Path(f).suffix.lower() in self.IMG_EXTS:
                    paths.append(os.path.join(root, f))
        paths.sort()
        if limit:
            paths = paths[:limit]
        return paths

    def _manual_split(self, real, fake, split, val_split):
        def split_list(lst):
            n_val = max(1, int(len(lst) * val_split))
            if split == "val":
                return lst[-n_val:]
            return lst[:-n_val]
        return split_list(real), split_list(fake)

    def get_sampler(self) -> WeightedRandomSampler:
        """Balanced sampler to handle class imbalance."""
        n_real = self.class_counts[0]
        n_fake = self.class_counts[1]
        total  = n_real + n_fake
        w_real = total / (2 * n_real + 1e-8)
        w_fake = total / (2 * n_fake + 1e-8)
        weights = [w_real if lbl == 0 else w_fake for _, lbl in self.samples]
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))
        return self.transform(img), label


# ══════════════════════════════════════════════════════
# 3. MODEL BUILDER
# ══════════════════════════════════════════════════════
def build_model(model_name: str, pretrained: bool = True) -> nn.Module:
    print(f"\n  Building {model_name} (pretrained={pretrained})…")
    if model_name == "xception":
        model = timm.create_model("xception", pretrained=pretrained, num_classes=2)
    elif model_name == "efficientnet_b4":
        model = timm.create_model("efficientnet_b4", pretrained=pretrained, num_classes=2)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")
    return model


# ══════════════════════════════════════════════════════
# 4. TRAINING ENGINE
# ══════════════════════════════════════════════════════
class Trainer:
    def __init__(self, model, model_name, args, device):
        self.model      = model.to(device)
        self.model_name = model_name
        self.args       = args
        self.device     = device

        self.criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer  = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        self.scheduler  = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
        self.scaler     = torch.cuda.amp.GradScaler() if args.fp16 and device.type == "cuda" else None

        # History
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
            "val_auc":    [],
        }
        self.best_val_auc  = 0.0
        self.best_epoch    = 0
        self.patience_cnt  = 0

        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.log_dir,    exist_ok=True)

        # Resume
        self.start_epoch = 0
        if args.resume and os.path.exists(args.resume):
            self._load_checkpoint(args.resume)

    # ── One epoch ──────────────────────────────
    def _run_epoch(self, loader: DataLoader, train: bool) -> Tuple[float, float, float]:
        self.model.train() if train else self.model.eval()
        total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for imgs, labels in tqdm(loader, desc="  train" if train else "  val  ", leave=False):
                imgs   = imgs.to(self.device)
                labels = labels.to(self.device)

                if train:
                    self.optimizer.zero_grad()

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        logits = self.model(imgs)
                        loss   = self.criterion(logits, labels)
                    if train:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                else:
                    logits = self.model(imgs)
                    loss   = self.criterion(logits, labels)
                    if train:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()

                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                preds = (probs > 0.5).astype(int)

                total_loss  += loss.item() * imgs.size(0)
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                all_probs.extend(probs.tolist())

        n    = len(all_labels)
        loss = total_loss / n
        acc  = accuracy_score(all_labels, all_preds)
        auc  = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
        return loss, acc, auc

    # ── Full training loop ──────────────────────
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        print(f"\n{'═'*60}")
        print(f"  Training {self.model_name.upper()}")
        print(f"  Device: {self.device} | Epochs: {self.args.epochs} | BS: {self.args.batch_size}")
        print(f"  LR: {self.args.lr} | FP16: {self.scaler is not None}")
        print(f"{'═'*60}\n")

        for epoch in range(self.start_epoch, self.args.epochs):
            t0 = time.time()

            train_loss, train_acc, _ = self._run_epoch(train_loader, train=True)
            val_loss,   val_acc, val_auc = self._run_epoch(val_loader, train=False)

            self.scheduler.step()
            elapsed = time.time() - t0

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["val_auc"].append(val_auc)

            # Log
            lr_now = self.optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch+1:02d}/{self.args.epochs}  "
                f"| train loss {train_loss:.4f}  acc {train_acc*100:.2f}%"
                f"  | val loss {val_loss:.4f}  acc {val_acc*100:.2f}%  AUC {val_auc:.4f}"
                f"  | lr {lr_now:.2e}  [{elapsed:.0f}s]"
            )

            # Save best
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_epoch   = epoch + 1
                self.patience_cnt = 0
                self._save_checkpoint(epoch, val_auc, is_best=True)
                print(f"  ✅ New best AUC: {val_auc:.4f} — checkpoint saved")
            else:
                self.patience_cnt += 1
                self._save_checkpoint(epoch, val_auc, is_best=False)

            # Early stopping
            if self.patience_cnt >= self.args.patience:
                print(f"\n  ⏹ Early stopping at epoch {epoch+1} (patience={self.args.patience})")
                break

            # Save plots every 5 epochs
            if (epoch + 1) % 5 == 0:
                self._save_plots()

        print(f"\n  Training complete. Best AUC: {self.best_val_auc:.4f} at epoch {self.best_epoch}")
        self._save_plots()
        self._save_history()
        self._final_evaluation(val_loader)

    # ── Final evaluation ──────────────────────
    def _final_evaluation(self, val_loader: DataLoader):
        print(f"\n  {'─'*50}")
        print(f"  Final Evaluation on Validation Set")
        print(f"  {'─'*50}")

        # Load best checkpoint
        best_path = os.path.join(self.args.output_dir, f"{self.model_name}_best.pth")
        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path, map_location=self.device))

        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="  evaluating", leave=False):
                imgs = imgs.to(self.device)
                probs = torch.softmax(self.model(imgs), dim=1)[:, 1].cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.numpy().tolist())
                all_probs.extend(probs.tolist())

        acc = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        cm  = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds,
                                       target_names=["Real", "Fake"], digits=4)

        print(f"\n  Accuracy : {acc*100:.2f}%")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"  {'':8}  Pred Real  Pred Fake")
        print(f"  True Real  {cm[0,0]:9d}  {cm[0,1]:9d}")
        print(f"  True Fake  {cm[1,0]:9d}  {cm[1,1]:9d}")
        print(f"\n  Classification Report:\n{report}")

        # ROC curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"{self.model_name} — Final Evaluation", fontsize=13)

        axes[0].plot(fpr, tpr, color="#1D9E75", lw=2, label=f"AUC = {auc:.4f}")
        axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("ROC Curve")
        axes[0].legend()

        axes[1].imshow(cm, cmap="Greens", aspect="auto")
        axes[1].set_xticks([0, 1]); axes[1].set_yticks([0, 1])
        axes[1].set_xticklabels(["Pred Real", "Pred Fake"])
        axes[1].set_yticklabels(["True Real", "True Fake"])
        for i in range(2):
            for j in range(2):
                axes[1].text(j, i, str(cm[i, j]), ha="center", va="center",
                             fontsize=16, color="black")
        axes[1].set_title("Confusion Matrix")

        plt.tight_layout()
        out_path = os.path.join(self.args.log_dir, f"{self.model_name}_final_eval.png")
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"\n  Evaluation plot saved → {out_path}")

        # Save metrics JSON
        metrics = {"accuracy": acc, "auc": auc,
                   "confusion_matrix": cm.tolist(), "best_epoch": self.best_epoch}
        with open(os.path.join(self.args.log_dir, f"{self.model_name}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    # ── Checkpoint helpers ─────────────────────
    def _save_checkpoint(self, epoch: int, val_auc: float, is_best: bool):
        state = {
            "epoch":     epoch + 1,
            "model":     self.model_name,
            "state_dict":self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "val_auc":   val_auc,
            "history":   self.history,
        }
        last_path = os.path.join(self.args.output_dir, f"{self.model_name}_last.pth")
        torch.save(state["state_dict"], last_path)
        if is_best:
            best_path = os.path.join(self.args.output_dir, f"{self.model_name}_best.pth")
            torch.save(state["state_dict"], best_path)
            # Also save the app-ready name
            app_path = os.path.join(self.args.output_dir, f"{self.model_name}_deepfake.pth")
            torch.save(state["state_dict"], app_path)

    def _load_checkpoint(self, path: str):
        print(f"  Resuming from {path}")
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            self.model.load_state_dict(state["state_dict"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.start_epoch = state["epoch"]
            self.history     = state.get("history", self.history)
            self.best_val_auc= max(state.get("history", {}).get("val_auc", [0]))
        else:
            self.model.load_state_dict(state)

    def _save_plots(self):
        epochs = range(1, len(self.history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"{self.model_name} Training Progress", fontsize=12)

        axes[0].plot(epochs, self.history["train_loss"], label="Train", color="#378ADD")
        axes[0].plot(epochs, self.history["val_loss"],   label="Val",   color="#E24B4A")
        axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(epochs, [a*100 for a in self.history["train_acc"]], label="Train", color="#378ADD")
        axes[1].plot(epochs, [a*100 for a in self.history["val_acc"]],   label="Val",   color="#E24B4A")
        axes[1].set_title("Accuracy (%)"); axes[1].legend(); axes[1].grid(alpha=0.3)

        axes[2].plot(epochs, self.history["val_auc"], color="#1D9E75", label="Val AUC")
        axes[2].axhline(self.best_val_auc, color="#888780", linestyle="--", alpha=0.5)
        axes[2].set_title("Validation AUC"); axes[2].legend(); axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.args.log_dir, f"{self.model_name}_progress.png"), dpi=100)
        plt.close()

    def _save_history(self):
        with open(os.path.join(self.args.log_dir, f"{self.model_name}_history.json"), "w") as f:
            json.dump(self.history, f, indent=2)


# ══════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════
def main():
    args = get_args()

    # Colab setup
    if args.colab:
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            print("✓ Google Drive mounted")
        except ImportError:
            pass

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═'*60}")
    print(f"  DeepShield Training Pipeline")
    print(f"  Device  : {device}")
    if device.type == "cuda":
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Data    : {args.data_dir}")
    print(f"  Model(s): {args.model}")
    print(f"{'═'*60}\n")

    models_to_train = (
        ["xception", "efficientnet_b4"] if args.model == "both"
        else [args.model]
    )

    for model_name in models_to_train:
        img_size = (
            args.img_size_xception if model_name == "xception"
            else args.img_size_efficientnet
        )

        print(f"\n{'─'*60}")
        print(f"  Preparing dataset for {model_name} (img_size={img_size})")
        print(f"{'─'*60}")

        try:
            train_ds = DeepfakeDataset(
                args.data_dir, split="train", img_size=img_size,
                augment=True, max_per_class=args.max_per_class,
                val_split=args.val_split,
            )
            val_ds = DeepfakeDataset(
                args.data_dir, split="val", img_size=img_size,
                augment=False, max_per_class=args.max_per_class,
                val_split=args.val_split,
            )
        except FileNotFoundError as e:
            print(f"\n❌ Dataset error:\n  {e}")
            print("\n  Please prepare your dataset first. See README for instructions.")
            sys.exit(1)

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=train_ds.get_sampler(),
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        model   = build_model(model_name, args.pretrained)
        trainer = Trainer(model, model_name, args, device)
        trainer.train(train_loader, val_loader)

        print(f"\n  ✅ {model_name} checkpoint saved to: {args.output_dir}/")
        print(f"     → {model_name}_deepfake.pth  (use this in DeepShield app)")

    print(f"\n{'═'*60}")
    print(f"  All training complete!")
    print(f"  Copy checkpoint files to your deepshield/checkpoints/ folder")
    print(f"  Then restart the app — it will load them automatically.")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()

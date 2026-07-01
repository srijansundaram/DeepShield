"""
DeepShield — Training Script v2
XceptionNet + EfficientNet-B4 with:
- Multi-dataset support
- Aggressive augmentation
- Label smoothing
- Cosine LR scheduler
- Early stopping
- fp16 mixed precision
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm
import numpy as np


# ── Label Smoothing Loss ──
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_prob = F.log_softmax(pred, dim=1)
        smooth_target = torch.full_like(log_prob, self.smoothing / (n_classes - 1))
        smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return -(smooth_target * log_prob).sum(dim=1).mean()


# ── Augmentation ──
def get_transforms(img_size=299, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size + 20, img_size + 20)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomRotation(10),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.RandomErasing(p=0.1),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


def get_transforms_eff(img_size=380, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size + 20, img_size + 20)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomRotation(10),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


# ── Training Loop ──
def train_model(
    model_name="xception",
    data_dir="./data",
    output_dir="./checkpoints",
    epochs=20,
    batch_size=16,
    lr=1e-4,
    weight_decay=1e-4,
    patience=5,
    fp16=True,
    pretrained=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    os.makedirs(output_dir, exist_ok=True)

    # Model + transforms
    if model_name == "xception":
        model = timm.create_model("xception", pretrained=pretrained, num_classes=2)
        img_size = 299
        train_tf = get_transforms(img_size, is_train=True)
        val_tf   = get_transforms(img_size, is_train=False)
        ckpt_name = "xception_deepfake.pth"
    elif model_name == "efficientnet_b4":
        model = timm.create_model("efficientnet_b4", pretrained=pretrained, num_classes=2)
        img_size = 380
        train_tf = get_transforms_eff(img_size, is_train=True)
        val_tf   = get_transforms_eff(img_size, is_train=False)
        ckpt_name = "efficientnet_deepfake.pth"
    else:
        model = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=2)
        img_size = 224
        train_tf = get_transforms(img_size, is_train=True)
        val_tf   = get_transforms(img_size, is_train=False)
        ckpt_name = "vit_deepfake.pth"

    model = model.to(device)

    # Datasets
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_dataset   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=val_tf)

    print(f"\n  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")
    print(f"  Classes: {train_dataset.classes}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    criterion = LabelSmoothingLoss(smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=fp16 and device.type == "cuda")

    best_val_acc = 0.0
    patience_counter = 0
    history = []

    print(f"\n  Starting training: {model_name} for {epochs} epochs\n")
    print(f"  {'Epoch':<8} {'Train Loss':<14} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<10} {'LR'}")
    print(f"  {'-'*70}")

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for imgs, labels in tqdm(train_loader, desc=f"  Epoch {epoch}/{epochs}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=fp16 and device.type == "cuda"):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += imgs.size(0)

        scheduler.step()

        # ── Validate ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.amp.autocast('cuda', enabled=fp16 and device.type == "cuda"):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)

        t_loss = train_loss / train_total
        t_acc  = train_correct / train_total * 100
        v_loss = val_loss / val_total
        v_acc  = val_correct / val_total * 100
        cur_lr = scheduler.get_last_lr()[0]

        print(f"  {epoch:<8} {t_loss:<14.4f} {t_acc:<12.2f} {v_loss:<12.4f} {v_acc:<10.2f} {cur_lr:.2e}")

        history.append({
            "epoch": epoch, "train_loss": t_loss, "train_acc": t_acc,
            "val_loss": v_loss, "val_acc": v_acc, "lr": cur_lr
        })

        # Save best
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            patience_counter = 0
            ckpt_path = os.path.join(output_dir, ckpt_name)
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved best model → {ckpt_path} (val_acc={v_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch} — no improvement for {patience} epochs")
                break

    # Save history
    with open(os.path.join(output_dir, f"{model_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  ✅ Training complete — Best val accuracy: {best_val_acc:.2f}%")
    return best_val_acc


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",        type=str,   default="xception",
                   choices=["xception", "efficientnet_b4", "vit"])
    p.add_argument("--data_dir",     type=str,   default="./data")
    p.add_argument("--output_dir",   type=str,   default="./checkpoints")
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience",     type=int,   default=5)
    p.add_argument("--no_fp16",      action="store_true")
    p.add_argument("--no_pretrained",action="store_true")
    args = p.parse_args()

    train_model(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        fp16=not args.no_fp16,
        pretrained=not args.no_pretrained,
    )
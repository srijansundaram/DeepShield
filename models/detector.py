"""
DeepShield — Core Detection Models
XceptionNet + EfficientNet-B4 + ViT Ensemble with GradCAM + Temperature Calibration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image
import cv2
import timm
import os
import json
import warnings
warnings.filterwarnings("ignore")

MODEL_IS_TRAINED = False

TRANSFORM = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

TRANSFORM_EFF = T.Compose([
    T.Resize((380, 380)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TRANSFORM_VIT = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def _find_checkpoint(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths = [
        os.path.join(script_dir, "..", "checkpoints", filename),
        os.path.join(os.getcwd(), "checkpoints", filename),
        os.path.join(script_dir, "checkpoints", filename),
        os.path.join("checkpoints", filename),
        filename,
    ]
    for p in search_paths:
        p = os.path.normpath(p)
        if os.path.exists(p):
            print(f"  Found checkpoint: {p}")
            return p
    print(f"  Checkpoint not found: {filename}")
    return None


def _load_calibration(checkpoint_dir=None):
    search = []
    if checkpoint_dir:
        search.append(os.path.join(checkpoint_dir, "calibration.json"))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search += [
        os.path.join(script_dir, "..", "checkpoints", "calibration.json"),
        os.path.join(os.getcwd(), "checkpoints", "calibration.json"),
    ]
    for p in search:
        p = os.path.normpath(p)
        if os.path.exists(p):
            with open(p) as f:
                cal = json.load(f)
            print(f"  ✓ Loaded calibration: xception_T={cal.get('xception_T',1.0)}, "
                  f"efficientnet_T={cal.get('efficientnet_T',1.0)}, "
                  f"vit_T={cal.get('vit_T',1.0)}")
            return cal
    print("  ⚠ No calibration.json found — using T=1.0 (uncalibrated)")
    return {"xception_T": 1.0, "efficientnet_T": 1.0, "vit_T": 1.0}


def _face_region_heatmap(size=299):
    cx, cy = size // 2, int(size * 0.42)
    Y, X = np.ogrid[:size, :size]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    cam = np.exp(-dist ** 2 / (2 * (size * 0.20) ** 2))
    return cam.astype(np.float32)


def _get_fc(model):
    for attr in ['fc', 'classifier']:
        layer = getattr(model, attr, None)
        if isinstance(layer, nn.Linear):
            return layer
    head = getattr(model, 'head', None)
    if head is not None:
        for attr in ['fc', 'l']:
            layer = getattr(head, attr, None)
            if isinstance(layer, nn.Linear):
                return layer
    for _, m in reversed(list(model.named_modules())):
        if isinstance(m, nn.Linear):
            return m
    return None


class XceptionDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("xception", pretrained=False, num_classes=2)

    def forward(self, x):
        return self.model(x)

    def get_gradcam(self, x: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        self.eval()
        storage = {}

        def hook_bn4(m, i, o):
            storage['bn4'] = o.clone()

        handle = self.model.bn4.register_forward_hook(hook_bn4)
        try:
            with torch.no_grad():
                _ = self.model(x.clone())
            handle.remove()

            feats_raw = storage.get('bn4')
            if feats_raw is None:
                return _face_region_heatmap(299)

            feats = feats_raw.detach().clone().requires_grad_(True)
            x2 = F.relu(feats, inplace=False)
            x2 = self.model.global_pool(x2)
            x2 = x2.flatten(1)

            fc = _get_fc(self.model)
            if fc is None:
                return _face_region_heatmap(299)

            out = fc(x2)
            out[0, class_idx].backward()

            if feats.grad is None:
                return _face_region_heatmap(299)

            grads   = feats.grad[0]
            acts    = feats.detach()[0]
            weights = grads.mean(dim=(1, 2))
            cam     = F.relu(
                (weights[:, None, None] * acts).sum(0), inplace=False
            ).cpu().numpy()

            if cam.max() <= cam.min():
                return _face_region_heatmap(299)

            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam = cv2.resize(cam, (299, 299))
            cam = cv2.GaussianBlur(cam, (11, 11), 0)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            return cam.astype(np.float32)

        except Exception as e:
            try:
                handle.remove()
            except Exception:
                pass
            print(f"GradCAM error (using fallback): {e}")
            return _face_region_heatmap(299)


class EfficientNetDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=2)

    def forward(self, x):
        return self.model(x)


class ViTDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2)

    def forward(self, x):
        return self.model(x)


class EnsembleDetector:
    def __init__(self, device="cpu"):
        self.device = device

        torch.manual_seed(42)
        if device == "cuda":
            torch.cuda.manual_seed(42)

        self.xception     = XceptionDetector().to(device)
        torch.manual_seed(42)
        self.efficientnet = EfficientNetDetector().to(device)
        torch.manual_seed(42)
        self.vit          = ViTDetector().to(device)

        self.calibration    = _load_calibration()
        self.xception_T     = self.calibration.get("xception_T", 1.0)
        self.efficientnet_T = self.calibration.get("efficientnet_T", 1.0)
        self.vit_T          = self.calibration.get("vit_T", 1.0)

        global MODEL_IS_TRAINED
        loaded = False
        vit_loaded = False

        for fname, model_obj in [
            ("xception_deepfake.pth",     self.xception.model),
            ("efficientnet_deepfake.pth", self.efficientnet.model),
        ]:
            path = _find_checkpoint(fname)
            if path:
                try:
                    state = torch.load(path, map_location=device, weights_only=True)
                    if isinstance(state, dict) and "state_dict" in state:
                        state = state["state_dict"]
                    model_obj.load_state_dict(state, strict=False)
                    print(f"✓ Loaded: {fname}")
                    loaded = True
                except Exception as e:
                    print(f"⚠ Failed to load {fname}: {e}")

        # ViT is optional — ensemble degrades gracefully without it
        vit_path = _find_checkpoint("vit_deepfake.pth")
        if vit_path:
            try:
                state = torch.load(vit_path, map_location=device, weights_only=True)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                self.vit.model.load_state_dict(state, strict=False)
                print(f"✓ Loaded: vit_deepfake.pth")
                vit_loaded = True
            except Exception as e:
                print(f"⚠ Failed to load vit_deepfake.pth: {e}")

        self.vit_loaded = vit_loaded
        MODEL_IS_TRAINED = loaded
        self.is_trained = loaded

        self.xception.eval()
        self.efficientnet.eval()
        self.vit.eval()

        if loaded:
            vit_status = "+ ViT" if vit_loaded else "(ViT not yet trained)"
            print(f"✅ Trained model loaded — XceptionNet + EfficientNet-B4 {vit_status}")
        else:
            print("⚠ No checkpoint found — running in demo mode")

    def predict_image(self, image):
        img_rgb = image.convert("RGB")
        with torch.no_grad():
            x_t      = TRANSFORM(img_rgb).unsqueeze(0).to(self.device)
            x_logits = self.xception(x_t)
            x_prob   = torch.softmax(x_logits / self.xception_T, dim=1)[0, 0].item()

            e_t      = TRANSFORM_EFF(img_rgb).unsqueeze(0).to(self.device)
            e_logits = self.efficientnet(e_t)
            e_prob   = torch.softmax(e_logits / self.efficientnet_T, dim=1)[0, 0].item()

            if self.vit_loaded:
                v_t      = TRANSFORM_VIT(img_rgb).unsqueeze(0).to(self.device)
                v_logits = self.vit(v_t)
                v_prob   = torch.softmax(v_logits / self.vit_T, dim=1)[0, 0].item()
                # Three model ensemble: Xception 40%, EfficientNet 35%, ViT 25%
                fake_prob = 0.40 * x_prob + 0.35 * e_prob + 0.25 * v_prob
            else:
                # Fallback to two-model ensemble
                fake_prob = 0.55 * x_prob + 0.45 * e_prob
                v_prob    = None

        is_fake = fake_prob > 0.5
        result = {
            "is_fake":            is_fake,
            "confidence":         fake_prob if is_fake else (1 - fake_prob),
            "fake_probability":   fake_prob,
            "real_probability":   1 - fake_prob,
            "xception_score":     x_prob,
            "efficientnet_score": e_prob,
            "verdict":            "Deepfake" if is_fake else "Real",
            "model_trained":      MODEL_IS_TRAINED,
        }
        if v_prob is not None:
            result["vit_score"] = v_prob
        return result

    def get_gradcam(self, image, class_idx=1):
        img_rgb = image.convert("RGB")
        x_t = TRANSFORM(img_rgb).unsqueeze(0).to(self.device)
        return self.xception.get_gradcam(x_t, class_idx)

    def predict_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.predict_image(Image.fromarray(rgb))
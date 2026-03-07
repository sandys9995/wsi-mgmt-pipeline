# inference/mine_topk.py

import os
import heapq
import numpy as np
import torch
import openslide

from PIL import Image
from tqdm import tqdm

import cv2  # <-- NEW (pip install opencv-python)

from stain.macenko_norm import pil_to_tensor255_chw  # keep your existing helper
from torchstain.torch.normalizers.macenko import TorchMacenkoNormalizer
from utils.slide_io import get_mpp
from utils.patch_filters import is_bad_patch

from torchvision import models
import torch.nn as nn


# ========================
# CONFIG
# ========================

TOPK = 5000
TARGET_MPP = 0.50
OUT_SIZE = 224
BATCH_SIZE = 256

THUMB_STRIDE = 16      # sampling step in thumbnail pixels
MASK_WIN = 3           # window radius for mask check
MIN_MASK_FRAC = 0.10   # require >=10% tissue in window

LOW_TEXTURE_STD = 5.0  # reject flat tiles

# Color-agnostic pen/marker detection on thumbnail
PEN_S_THR = 140        # saturation threshold (0-255)
PEN_V_MIN = 40         # avoid very dark
PEN_V_MAX = 245        # avoid pure white
PEN_DIL_K = 11         # dilation kernel
PEN_DIL_IT = 2         # dilation iterations

REFERENCE_TILE = "ref_image/target_reference_clean.png"
MODEL_PATH = "results/model_patch/best.pt"

OUT_ROOT = "data_test/outputs/top5k"

SLIDE_LIST = [
    "/Volumes/Expansion/data1/GR/DigitalSlide_A2M_10S_1.mrxs",
    "/Volumes/Expansion/data1/GR/DigitalSlide_A2M_16S_1.mrxs",
    "/Volumes/Expansion/data1/DUS/106.svs",
    "/Volumes/Expansion/data1/CPTAC-GBM/GBM/C3L-02955-21.svs"
]


# ========================
# DEVICE
# ========================

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = get_device()
print("Using device:", device)

# Macenko uses eigh -> not supported on MPS
norm_device = "cpu" if device == "mps" else device
if device == "mps":
    print("[warn] Macenko uses eigh; forcing normalizer to CPU (model stays on MPS).")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


# ========================
# SLIDE BOUNDS
# ========================

def get_bounds(slide: openslide.OpenSlide):
    props = slide.properties
    try:
        bx = int(float(props.get("openslide.bounds-x", 0)))
        by = int(float(props.get("openslide.bounds-y", 0)))
        bw = int(float(props.get("openslide.bounds-width", slide.dimensions[0])))
        bh = int(float(props.get("openslide.bounds-height", slide.dimensions[1])))
    except Exception:
        bx, by = 0, 0
        bw, bh = slide.dimensions

    W, H = slide.dimensions
    bx = max(0, min(bx, W - 1))
    by = max(0, min(by, H - 1))
    bw = max(1, min(bw, W - bx))
    bh = max(1, min(bh, H - by))
    return bx, by, bw, bh


# ========================
# MODEL
# ========================

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

# safer load
state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
model.load_state_dict(state)

model = model.to(device)
model.eval()


# ========================
# MACENKO
# ========================

ref_img = Image.open(REFERENCE_TILE).convert("RGB")
ref_tensor = pil_to_tensor255_chw(ref_img).to(norm_device)

normalizer = TorchMacenkoNormalizer()
if hasattr(normalizer, "to"):
    normalizer = normalizer.to(norm_device)
normalizer.fit(ref_tensor)


def to_model_tensor(x: torch.Tensor) -> torch.Tensor:
    if isinstance(x, (tuple, list)):
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got {tuple(x.shape)}")

    # allow HWC or CHW
    if x.shape[0] != 3 and x.shape[-1] == 3:
        x = x.permute(2, 0, 1)
    if x.shape[0] != 3:
        raise ValueError(f"Expected 3 channels, got {tuple(x.shape)}")

    x = x.float()
    x_max = float(x.max().item()) if x.numel() else 1.0
    if x_max > 5.0:
        x = x / 255.0
    else:
        x = x.clamp(0, 1)

    mean = IMAGENET_MEAN.to(x.device)
    std  = IMAGENET_STD.to(x.device)
    return (x - mean) / std


def tensor_to_pil_safe(x: torch.Tensor) -> Image.Image:
    """
    FIXES your squeezed 3x224 issue:
    - accepts CHW or HWC
    - accepts float [0,1] or [0,255]
    """
    if isinstance(x, (tuple, list)):
        x = x[0]
    x = x.detach().cpu()

    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got {tuple(x.shape)}")

    # HWC -> CHW
    if x.shape[0] != 3 and x.shape[-1] == 3:
        x = x.permute(2, 0, 1)

    if x.shape[0] != 3:
        raise ValueError(f"Expected 3 channels, got {tuple(x.shape)}")

    x = x.float()
    mx = float(x.max().item()) if x.numel() else 1.0
    if mx <= 5.0:
        x = (x.clamp(0, 1) * 255.0).round()
    else:
        x = x.clamp(0, 255).round()

    x = x.to(torch.uint8)
    arr = x.permute(1, 2, 0).numpy()
    return Image.fromarray(arr, mode="RGB")


# ========================
# FILTERS
# ========================

def low_texture(rgb: np.ndarray, std_thr: float = 5.0) -> bool:
    return float(rgb.std()) < std_thr


# ========================
# COLOR-AGNOSTIC PEN MASK ON THUMBNAIL
# ========================

def pen_mask_color_agnostic(thumb_rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2HSV)
    S = hsv[..., 1].astype(np.uint8)
    V = hsv[..., 2].astype(np.uint8)

    pen = (S >= PEN_S_THR) & (V >= PEN_V_MIN) & (V <= PEN_V_MAX)

    # dilate to remove pen margins too
    u8 = (pen.astype(np.uint8) * 255)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (PEN_DIL_K, PEN_DIL_K))
    pen_dil = (cv2.dilate(u8, ker, iterations=PEN_DIL_IT) > 0)

    return pen_dil


# ========================
# CLEAN TISSUE MASK ON BOUNDS
# ========================

def compute_clean_mask_on_bounds(slide, bx, by, bw, bh, level=None, white_thr=230):
    if level is None:
        level = slide.level_count - 1

    down = float(slide.level_downsamples[level])
    tw = max(1, int(round(bw / down)))
    th = max(1, int(round(bh / down)))

    thumb = slide.read_region((bx, by), level, (tw, th)).convert("RGB")
    arr = np.asarray(thumb, dtype=np.uint8)

    nonwhite = np.any(arr < white_thr, axis=2)

    # remove pen/marker regardless of color
    pen_dil = pen_mask_color_agnostic(arr)

    clean = nonwhite & (~pen_dil)

    # cleanup: open/close to remove speckles
    clean_u8 = (clean.astype(np.uint8) * 255)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_u8 = cv2.morphologyEx(clean_u8, cv2.MORPH_OPEN, ker, iterations=1)
    clean_u8 = cv2.morphologyEx(clean_u8, cv2.MORPH_CLOSE, ker, iterations=2)

    clean = clean_u8 > 0
    return clean, level, down


def generate_candidates(mask: np.ndarray, down: float, bx: int, by: int, stride: int):
    coords = []
    h, w = mask.shape
    for yy in range(0, h, stride):
        y0 = max(0, yy - MASK_WIN); y1 = min(h, yy + MASK_WIN + 1)
        for xx in range(0, w, stride):
            x0 = max(0, xx - MASK_WIN); x1 = min(w, xx + MASK_WIN + 1)
            frac = float(mask[y0:y1, x0:x1].mean())
            if frac >= MIN_MASK_FRAC:
                cx = bx + int(round(xx * down))
                cy = by + int(round(yy * down))
                coords.append((cx, cy))
    return coords


# ========================
# PATCH READ (STRICT)
# ========================

def read_patch_centered_strict(slide, cx, cy, read_px):
    W, H = slide.dimensions
    x0 = int(cx - read_px // 2)
    y0 = int(cy - read_px // 2)

    if x0 < 0 or y0 < 0 or (x0 + read_px) > W or (y0 + read_px) > H:
        return None, None, None

    patch = slide.read_region((x0, y0), 0, (read_px, read_px)).convert("RGB")
    if read_px != OUT_SIZE:
        patch = patch.resize((OUT_SIZE, OUT_SIZE), resample=Image.BILINEAR)
    return patch, x0, y0


# ========================
# MAIN
# ========================

def mine_slide(slide_path: str):
    slide_id = os.path.splitext(os.path.basename(slide_path))[0]
    print("\nMining:", slide_id)

    slide = openslide.OpenSlide(slide_path)
    bx, by, bw, bh = get_bounds(slide)

    mpp = get_mpp(slide)
    if mpp is None or mpp <= 0:
        mpp = 0.25
        print(f"[warn] Missing/invalid MPP for {slide_id}; using fallback mpp={mpp}")

    read_px = int(round(OUT_SIZE * (TARGET_MPP / mpp)))
    read_px = max(read_px, OUT_SIZE)

    mask, mask_level, down = compute_clean_mask_on_bounds(slide, bx, by, bw, bh, white_thr=230)
    candidates = generate_candidates(mask, down, bx, by, stride=THUMB_STRIDE)

    print(f"Bounds: x={bx} y={by} w={bw} h={bh} | candidates={len(candidates)} | read_px={read_px}")

    heap = []  # (prob, (x0, y0, norm_pil))
    batch_imgs, batch_meta = [], []

    skipped_oob = 0
    skipped_bad = 0
    skipped_flat = 0
    skipped_norm_fail = 0

    for (cx, cy) in tqdm(candidates, desc=f"[mine] {slide_id}", unit="pt"):
        patch, x0, y0 = read_patch_centered_strict(slide, cx, cy, read_px)
        if patch is None:
            skipped_oob += 1
            continue

        patch_np = np.asarray(patch, dtype=np.uint8)

        if is_bad_patch(patch_np):
            skipped_bad += 1
            continue

        if low_texture(patch_np, std_thr=LOW_TEXTURE_STD):
            skipped_flat += 1
            continue

        # normalize (CPU if MPS), then score on model device
        try:
            tensor = pil_to_tensor255_chw(patch).to(norm_device)
            out = normalizer.normalize(tensor)
            norm_t = out[0] if isinstance(out, (tuple, list)) else out

            norm_pil = tensor_to_pil_safe(norm_t)   # <-- FIX squeezed images
            model_tensor = to_model_tensor(norm_t).to(device)
        except Exception:
            skipped_norm_fail += 1
            continue

        batch_imgs.append(model_tensor)
        batch_meta.append((x0, y0, norm_pil))

        if len(batch_imgs) == BATCH_SIZE:
            batch_tensor = torch.stack(batch_imgs)
            with torch.no_grad():
                probs = torch.softmax(model(batch_tensor), dim=1)[:, 1].detach().cpu().numpy()

            for prob, meta in zip(probs, batch_meta):
                prob = float(prob)
                if len(heap) < TOPK:
                    heapq.heappush(heap, (prob, meta))
                else:
                    if prob > heap[0][0]:
                        heapq.heapreplace(heap, (prob, meta))

            batch_imgs, batch_meta = [], []

    # flush remainder
    if batch_imgs:
        batch_tensor = torch.stack(batch_imgs)
        with torch.no_grad():
            probs = torch.softmax(model(batch_tensor), dim=1)[:, 1].detach().cpu().numpy()
        for prob, meta in zip(probs, batch_meta):
            prob = float(prob)
            if len(heap) < TOPK:
                heapq.heappush(heap, (prob, meta))
            else:
                if prob > heap[0][0]:
                    heapq.heapreplace(heap, (prob, meta))

    slide.close()
    heap.sort(reverse=True)

    print(
        f"[mine] kept={len(heap)} | "
        f"skipped_oob={skipped_oob} skipped_bad={skipped_bad} "
        f"skipped_flat={skipped_flat} skipped_norm_fail={skipped_norm_fail}"
    )

    save_topk(slide_id, heap)


def save_topk(slide_id, heap):
    out_dir = os.path.join(OUT_ROOT, slide_id)
    os.makedirs(out_dir, exist_ok=True)

    for rank, (prob, (x0, y0, patch)) in enumerate(heap):
        patch.save(os.path.join(out_dir, f"rank{rank:05d}__x{x0}__y{y0}__p{prob:.4f}.png"))

    print("Saved:", len(heap), "patches ->", out_dir)


if __name__ == "__main__":
    for slide_path in SLIDE_LIST:
        mine_slide(slide_path)
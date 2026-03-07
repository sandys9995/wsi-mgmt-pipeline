import os
import glob
import csv
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFile, PngImagePlugin
import torch
from tqdm import tqdm

from torchstain.torch.normalizers.macenko import TorchMacenkoNormalizer

ImageFile.LOAD_TRUNCATED_IMAGES = True
PngImagePlugin.MAX_TEXT_CHUNK = 10 * 1024 * 1024


@dataclass
class NormConfig:
    in_dir: str
    out_dir: str
    reference_tile_path: str
    device: str = "cpu"
    limit: Optional[int] = None

    # Robustness
    min_tissue_ratio: float = 0.20   # stricter for your tumor-core dataset
    white_thr: int = 230
    black_thr: int = 15
    max_black_frac: float = 0.95


def _safe_makedirs(p: str):
    os.makedirs(p, exist_ok=True)


def pil_to_tensor255_chw(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB"), dtype=np.float32)  # H,W,3
    return torch.from_numpy(arr).permute(2, 0, 1)         # 3,H,W


def tensor_chw_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().cpu()
    if t.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got {tuple(t.shape)}")

    if t.shape[0] != 3 and t.shape[-1] == 3:
        t = t.permute(2, 0, 1)

    if t.shape[0] != 3:
        raise ValueError(f"Expected 3 channels, got {tuple(t.shape)}")

    if torch.is_floating_point(t):
        t = t.float()
        t_max = float(t.max().item()) if t.numel() else 1.0
        if t_max <= 5.0:
            t = t.clamp(0, 1) * 255.0
        else:
            t = t.clamp(0, 255)
        t = t.round().to(torch.uint8)
    else:
        t = t.clamp(0, 255).to(torch.uint8)

    arr = t.permute(1, 2, 0).numpy()
    return Image.fromarray(arr, mode="RGB")


def tissue_ratio_fast(arr_rgb: np.ndarray, white_thr: int = 230) -> float:
    # tissue = any channel < white_thr
    nonwhite = np.any(arr_rgb < white_thr, axis=2)
    return float(nonwhite.mean())


def black_fraction(arr_rgb: np.ndarray, thr: int = 15) -> float:
    b = np.all(arr_rgb <= thr, axis=2)
    return float(b.mean())


@torch.no_grad()
def _normalize_one(
    normalizer: TorchMacenkoNormalizer,
    img: Image.Image,
    device: str,
    *,
    min_tissue_ratio: float,
    white_thr: int,
    black_thr: int,
    max_black_frac: float,
) -> Tuple[Optional[Image.Image], str, dict]:
    """
    Returns (normalized_img_or_None, status, stats)
    status ∈ {"ok", "skip_low_tissue", "skip_black", "exception"}
    """
    img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)

    stats = {
        "mean": float(arr.mean()),
        "tissue_ratio": tissue_ratio_fast(arr, white_thr=white_thr),
        "black_frac": black_fraction(arr, thr=black_thr),
    }

    if stats["black_frac"] >= max_black_frac or stats["mean"] <= 1.0:
        return None, "skip_black", stats

    if stats["tissue_ratio"] < min_tissue_ratio:
        return None, "skip_low_tissue", stats

    x = pil_to_tensor255_chw(img).to(device)

    try:
        out = normalizer.normalize(x)
        x_norm = out[0] if isinstance(out, (tuple, list)) else out
        return tensor_chw_to_pil(x_norm), "ok", stats
    except Exception:
        return None, "exception", stats


def normalize_patches(cfg: NormConfig):
    _safe_makedirs(cfg.out_dir)
    for lab in ["0", "1"]:
        _safe_makedirs(os.path.join(cfg.out_dir, lab))

    # logs
    log_dir = os.path.join(cfg.out_dir, "logs")
    _safe_makedirs(log_dir)
    log_csv = os.path.join(log_dir, "macenko_log.csv")
    skip_txt = os.path.join(log_dir, "skipped_paths.txt")

    # Device safety
    if cfg.device == "mps":
        print("[norm] WARNING: MPS not supported for Macenko (eigh). Using CPU.")
        device = "cpu"
    elif cfg.device == "cuda" and not torch.cuda.is_available():
        print("[norm] WARNING: CUDA not available. Using CPU.")
        device = "cpu"
    else:
        device = cfg.device

    # Fit on reference
    ref_img = Image.open(cfg.reference_tile_path).convert("RGB")
    ref_arr = np.asarray(ref_img, dtype=np.uint8)
    ref_tr = tissue_ratio_fast(ref_arr, white_thr=cfg.white_thr)
    if ref_tr < 0.20:
        raise RuntimeError(
            f"Reference tile has too little tissue (tissue_ratio={ref_tr:.3f}). "
            f"Pick a more tissue-rich reference patch."
        )

    ref_t = pil_to_tensor255_chw(ref_img).to(device)

    normalizer = TorchMacenkoNormalizer()
    if hasattr(normalizer, "to"):
        normalizer = normalizer.to(device)
    normalizer.fit(ref_t)

    # Gather patches
    patch_paths = []
    for lab in ["0", "1"]:
        patch_paths.extend(glob.glob(os.path.join(cfg.in_dir, lab, "*.png")))
    patch_paths = sorted(patch_paths)
    if cfg.limit is not None:
        patch_paths = patch_paths[: cfg.limit]

    print(f"[norm] Found {len(patch_paths)} patches to normalize.")

    n_ok = n_skip_tissue = n_skip_black = n_exc = 0

    # init logs
    if os.path.exists(skip_txt):
        os.remove(skip_txt)

    with open(log_csv, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(
            fcsv,
            fieldnames=["path", "label", "status", "mean", "tissue_ratio", "black_frac"]
        )
        w.writeheader()

        pbar = tqdm(patch_paths, desc="[norm] macenko", unit="patch")
        for p in pbar:
            lab = os.path.basename(os.path.dirname(p))

            img = Image.open(p).convert("RGB")
            img_norm, status, stats = _normalize_one(
                normalizer,
                img,
                device,
                min_tissue_ratio=cfg.min_tissue_ratio,
                white_thr=cfg.white_thr,
                black_thr=cfg.black_thr,
                max_black_frac=cfg.max_black_frac,
            )

            # log always
            w.writerow({
                "path": p,
                "label": lab,
                "status": status,
                "mean": f"{stats['mean']:.4f}",
                "tissue_ratio": f"{stats['tissue_ratio']:.6f}",
                "black_frac": f"{stats['black_frac']:.6f}",
            })

            if status == "ok":
                out_path = os.path.join(cfg.out_dir, lab, os.path.basename(p))
                img_norm.save(out_path, format="PNG", optimize=True)
                n_ok += 1
            else:
                # DO NOT contaminate normalized dataset
                with open(skip_txt, "a", encoding="utf-8") as f:
                    f.write(f"{status}\t{p}\n")
                if status == "skip_low_tissue":
                    n_skip_tissue += 1
                elif status == "skip_black":
                    n_skip_black += 1
                else:
                    n_exc += 1

            pbar.set_postfix({
                "ok": n_ok,
                "skip_tis": n_skip_tissue,
                "skip_blk": n_skip_black,
                "exc": n_exc,
            })

    print(f"[norm] Done. Output: {cfg.out_dir}")
    print(f"[norm] Summary: ok={n_ok}, skip_low_tissue={n_skip_tissue}, skip_black={n_skip_black}, exception={n_exc}")
    print(f"[norm] Log CSV: {log_csv}")
    print(f"[norm] Skipped list: {skip_txt}")
# scripts/make_masks.py
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import openslide
from PIL import Image
import pandas as pd
import cv2
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess.masking import compute_mask_status_fields, select_mask_with_fallback
from src.preprocess.strategy import (
    estimate_white_leak_fraction,
    infer_sparse_canvas_mode,
    non_black_bbox_fraction,
    resolve_stain_vectors,
    retry_acceptance,
)

WSI_EXTS = {".svs", ".ndpi", ".mrxs", ".tif", ".tiff"}

MASK_PROFILES: dict[str, dict] = {
    # Recommended for large-scale runs: avoids edge/background leakage while
    # keeping hematoxylin-rich tissue recovery active.
    "scale_balanced_v1": {
        "use_blood_as_background": True,
        "exclude_clot_from_tissue": True,
        "clot_h_max": 0.30,
        "clot_e_min": 0.52,
        "clot_red_dom_min": 0.12,
        "clot_sat_min": 55,
        "clot_gray_max": 205,
        "clot_edge_max": 0.55,
        "clot_min_component_area_ratio": 0.00035,
        "clot_use_hue_gate": True,
        "clot_hue_low_max": 14,
        "clot_hue_high_min": 170,
        "clot_dilate_kernel": 5,
        "h_rescue_min": 0.26,
        "h_rescue_sat_min": 6,
        "h_rescue_edge_min": 0.0,
        "h_rescue_od_min": 0.06,
        "soft_rescue_enabled": False,
        "supplement_enabled": True,
        "supplement_od_min": 0.10,
        "supplement_h_min": 0.18,
        "supplement_e_min": 0.12,
        "supplement_sat_min": 12,
        "supplement_gray_max": 238,
        "supplement_edge_min": 0.02,
        "supplement_touch_kernel": 15,
        "supplement_min_component_area_ratio": 0.00003,
        "supplement_white_like_max": 0.28,
        "supplement_max_added_frac_of_clean": 0.12,
        "soft_rescue_od_min": 0.08,
        "soft_rescue_h_min": 0.14,
        "soft_rescue_sat_min": 10,
        "soft_rescue_gray_max": 236,
        "soft_rescue_min_component_area_ratio": 0.0002,
        "soft_rescue_white_like_max": 0.50,
        "soft_rescue_touch_min": 0.0015,
        "soft_rescue_touch_kernel": 13,
        "soft_rescue_component_edge_min": 0.03,
        "min_component_area_ratio": 0.00015,
        "ink_magenta_h_min": 165,
        "ink_magenta_h_low_max": 8,
        "ink_magenta_s_min": 180,
        "ink_magenta_v_max": 220,
        "ink_dilate_kernel": 1,
        "enable_dark_pen_mask": False,
        "soft_relax_retry_enabled": False,
        "edge_guard_px": 128,
        "edge_guard_loose_od_max": 0.22,
        "edge_guard_loose_h_max": 0.20,
        "edge_guard_loose_score_max": 0.30,
        "border_component_h_max": 0.22,
        "border_component_od_max": 0.40,
        "border_component_white_min": 0.10,
        "final_edge_crop_px": 96,
    },
    "sparse_canvas_pure_v1": {
        "target_ds": 32.0,
        "use_blood_as_background": True,
        "exclude_clot_from_tissue": True,
        "clot_h_max": 0.32,
        "clot_e_min": 0.50,
        "clot_red_dom_min": 0.10,
        "clot_sat_min": 50,
        "clot_gray_max": 210,
        "clot_edge_max": 0.60,
        "clot_min_component_area_ratio": 0.00020,
        "clot_use_hue_gate": True,
        "clot_hue_low_max": 14,
        "clot_hue_high_min": 170,
        "clot_dilate_kernel": 3,
        "soft_relax_retry_enabled": False,
        "expected_frac_min_effective": 0.06,
        "od_q": 45,
        "fallback_od_q": 25,
        "score_thr": 0.30,
        "fallback_score_thr": 0.22,
        "w_edge": 0.08,
        "min_component_area_ratio": 0.00002,
        "h_rescue_min": 0.22,
        "h_rescue_sat_min": 6,
        "h_rescue_od_min": 0.04,
        "soft_rescue_enabled": True,
        "supplement_enabled": True,
        "supplement_od_min": 0.06,
        "supplement_h_min": 0.10,
        "supplement_e_min": 0.08,
        "supplement_sat_min": 8,
        "supplement_gray_max": 242,
        "supplement_edge_min": 0.0,
        "supplement_touch_kernel": 11,
        "supplement_min_component_area_ratio": 0.00001,
        "supplement_white_like_max": 0.45,
        "supplement_max_added_frac_of_clean": 0.15,
        "h_rescue_min_component_area_ratio": 0.000003,
        "soft_rescue_od_min": 0.04,
        "soft_rescue_h_min": 0.10,
        "soft_rescue_sat_min": 6,
        "soft_rescue_min_component_area_ratio": 0.00002,
        "soft_rescue_white_like_max": 0.70,
        "soft_rescue_touch_min": 0.0,
        "soft_rescue_touch_kernel": 9,
        "soft_rescue_component_edge_min": 0.005,
        "edge_guard_px": 64,
        "edge_guard_loose_od_max": 0.25,
        "edge_guard_loose_h_max": 0.25,
        "edge_guard_loose_score_max": 0.45,
        "final_edge_crop_px": 8,
        "roi_min_component_area_ratio": 0.00005,
        "low_tissue_retry_min_nonwhite_added_frac": 0.75,
        "low_tissue_retry_max_white_leak_increase": 0.02,
    },
}


def _resolve_path(project_root: Path, p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def _as_list(v):
    if isinstance(v, list):
        return v
    if v is None:
        return []
    return [v]


def list_wsis(wsi_dirs: list[Path], recursive: bool = True) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for wsi_dir in wsi_dirs:
        center = wsi_dir.name
        if recursive:
            files.extend([(p, center) for p in sorted(wsi_dir.rglob("*")) if p.is_file() and p.suffix.lower() in WSI_EXTS])
        else:
            files.extend([(p, center) for p in sorted(wsi_dir.iterdir()) if p.is_file() and p.suffix.lower() in WSI_EXTS])
    uniq: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for p, center in files:
        if p.stem not in seen:
            uniq.append((p, center))
            seen.add(p.stem)
    return uniq

def read_thumb_rgb(slide: openslide.OpenSlide, target_ds: float = 64.0) -> np.ndarray:
    # pick level whose downsample is closest to target_ds
    ds = np.array(slide.level_downsamples, dtype=np.float32)
    lvl = int(np.argmin(np.abs(ds - target_ds)))
    w, h = slide.level_dimensions[lvl]
    img = slide.read_region((0, 0), lvl, (w, h)).convert("RGB")
    return np.array(img, dtype=np.uint8), lvl, float(ds[lvl])


def read_region_rgb_at_ds(
    slide: openslide.OpenSlide,
    x0_l0: int,
    y0_l0: int,
    w_l0: int,
    h_l0: int,
    target_ds: float,
) -> tuple[np.ndarray, int, float]:
    ds = np.array(slide.level_downsamples, dtype=np.float32)
    lvl = int(np.argmin(np.abs(ds - target_ds)))
    lvl_ds = float(ds[lvl])
    w = int(max(1, np.ceil(w_l0 / lvl_ds)))
    h = int(max(1, np.ceil(h_l0 / lvl_ds)))
    img = slide.read_region((int(x0_l0), int(y0_l0)), lvl, (w, h)).convert("RGB")
    return np.array(img, dtype=np.uint8), lvl, lvl_ds

def save_preview(img_rgb: np.ndarray, mask: np.ndarray, out_png: Path) -> None:
    overlay = img_rgb.copy()
    overlay[mask > 0] = (0.7 * overlay[mask > 0] + 0.3 * np.array([0, 255, 0])).astype(np.uint8)
    Image.fromarray(overlay).save(out_png)


def _black_canvas_mask(rgb: np.ndarray, cfg: dict) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    black_gray = int(cfg.get("black_canvas_gray_max", 8))
    black_rgb = int(cfg.get("black_canvas_rgb_max", 12))
    return (gray <= black_gray) | (
        (rgb[:, :, 0] <= black_rgb) & (rgb[:, :, 1] <= black_rgb) & (rgb[:, :, 2] <= black_rgb)
    )


def _non_black_bbox(rgb: np.ndarray, cfg: dict) -> tuple[int, int, int, int] | None:
    black = _black_canvas_mask(rgb, cfg)
    non_black = (~black).astype(np.uint8)
    if int(non_black.sum()) == 0:
        return None
    n, labels, stats, _ = cv2.connectedComponentsWithStats(non_black, connectivity=8)
    if n <= 1:
        return None
    min_ratio = float(cfg.get("roi_min_component_area_ratio", 0.0002))
    min_area = int(max(1, round(min_ratio * non_black.size)))
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    merged = keep[labels]
    if int(merged.sum()) == 0:
        return None
    ys, xs = np.where(merged)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return x1, y1, x2, y2


def _expand_bbox(
    x1: int, y1: int, x2: int, y2: int, width: int, height: int, expand_px: int, expand_frac: float
) -> tuple[int, int, int, int]:
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = max(expand_px, int(round(bw * expand_frac)))
    pad_y = max(expand_px, int(round(bh * expand_frac)))
    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y)
    nx2 = min(width, x2 + pad_x)
    ny2 = min(height, y2 + pad_y)
    return nx1, ny1, nx2, ny2


def _od(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float32) / 255.0
    return -np.log(np.clip(x, 1e-6, 1.0))


def _rapid_channel(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    return np.clip(0.5 * (r + b) - g, 0.0, None)


def _vec_triplet(v: np.ndarray | list[float] | tuple[float, float, float]) -> tuple[float, float, float]:
    arr = np.asarray(v, dtype=np.float32).reshape(-1)
    if arr.size != 3 or np.any(~np.isfinite(arr)):
        return float(np.nan), float(np.nan), float(np.nan)
    return float(arr[0]), float(arr[1]), float(arr[2])


def _calibrate_from_reference_patches(mask_cfg: dict) -> dict:
    raw_ref = mask_cfg.get(
        "reference_patches_dir",
        "/Users/sandeepsharma/Downloads/WSI_PATCH_selection/Tumor patch extraction/src/results/patches_raw",
    )
    ref_dir = Path(raw_ref)
    if not ref_dir.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        candidate = (project_root / ref_dir).resolve()
        if candidate.exists():
            ref_dir = candidate
        else:
            ref_dir = ref_dir.resolve()
    sample_n = int(mask_cfg.get("reference_sample_patches", 400))
    seed = int(mask_cfg.get("reference_seed", 1337))

    if not ref_dir.exists():
        return {}

    files = sorted(ref_dir.rglob("*.png"))
    if not files:
        return {}

    rng = np.random.default_rng(seed)
    if len(files) > sample_n:
        idx = rng.choice(len(files), size=sample_n, replace=False)
        chosen = [files[i] for i in idx]
    else:
        chosen = files

    rapid_caps = []
    od_caps = []

    for fp in chosen:
        try:
            rgb = np.array(Image.open(fp).convert("RGB"), dtype=np.uint8)
        except Exception:
            continue
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            continue
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        valid = (gray > 8) & (gray < 252)
        if int(valid.sum()) < 64:
            continue
        rapid = _rapid_channel(rgb)[valid]
        od_sum = _od(rgb).sum(axis=2)[valid]
        rapid_caps.append(float(np.percentile(rapid, 70)))
        od_caps.append(float(np.percentile(od_sum, 70)))

    if not rapid_caps or not od_caps:
        return {}

    ref_rapid_cap = max(8.0, float(np.percentile(np.array(rapid_caps), 75)))
    ref_od_cap = max(0.25, float(np.percentile(np.array(od_caps), 75)))
    return {"ref_rapid_thr_cap": ref_rapid_cap, "ref_od_thr_cap": ref_od_cap}


def _apply_mask_profile(mask_cfg: dict, profile_override: str | None = None, quiet: bool = False) -> dict:
    profile = str(profile_override if profile_override is not None else mask_cfg.get("profile", "")).strip()
    if not profile:
        return mask_cfg
    overrides = MASK_PROFILES.get(profile)
    if overrides is None:
        if not quiet:
            print(f"Mask profile '{profile}' not found. Using explicit config values only.")
        return mask_cfg
    merged = dict(mask_cfg)
    merged.update(overrides)
    merged["profile"] = profile
    if not quiet:
        print(f"Using mask profile: {profile}")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Build tissue masks for available WSIs.")
    parser.add_argument("--config", default="configs/pilot.yaml", help="Path to config yaml.")
    parser.add_argument("--n-slides", type=int, default=None, help="Override number of slides. <=0 means all.")
    parser.add_argument("--slide-id", default=None, help="Optional single slide_id to run.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg_path = _resolve_path(project_root, args.config)

    mask_dir = _resolve_path(project_root, "data/masks")
    mask_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    default_mask_cfg = {
        "strategy": "auto_hybrid",
        "profile": "scale_balanced_v1",
        "sparse_canvas_profile": "sparse_canvas_pure_v1",
        "sparse_canvas_black_frac_min": 0.75,
        "sparse_canvas_bbox_frac_max": 0.15,
        "border_ignore_px": 10,
        "expected_frac_min": 0.12,
        "expected_frac_max": 0.62,
        "expected_frac_min_effective": 0.08,
        "expected_frac_max_effective": 0.80,
        "dynamic_stain_vectors_enabled": True,
        "stain_vector_fixed": False,
        "stain_estimation_od_min": 0.12,
        "stain_estimation_gray_max": 245,
        "stain_estimation_sat_min": 6,
        "stain_estimation_min_pixels": 1200,
        "stain_estimation_sample_max": 60000,
        "stain_estimation_angle_pct_low": 1.0,
        "stain_estimation_angle_pct_high": 99.0,
        "stain_estimation_max_he_dot": 0.985,
        "stain_estimation_min_pair_score": 1.20,
        "stain_vector_h": (0.651, 0.701, 0.290),
        "stain_vector_e": (0.216, 0.801, 0.558),
        "stain_vector_r": (0.316, -0.598, 0.737),
        "low_frac_trigger": 0.08,
        "high_frac_trigger": 0.70,
        "od_q": 55,
        "fallback_od_q": 35,
        "score_thr": 0.36,
        "fallback_score_thr": 0.28,
        "w_rapid": 0.55,
        "w_od": 0.30,
        "w_edge": 0.15,
        "use_blood_as_background": False,
        "exclude_clot_from_tissue": True,
        "clot_h_max": 0.30,
        "clot_e_min": 0.52,
        "clot_red_dom_min": 0.12,
        "clot_sat_min": 55,
        "clot_gray_max": 205,
        "clot_edge_max": 0.55,
        "clot_min_component_area_ratio": 0.00035,
        "clot_use_hue_gate": True,
        "clot_hue_low_max": 14,
        "clot_hue_high_min": 170,
        "clot_dilate_kernel": 5,
        "rbc_e_min": 0.58,
        "rbc_h_max": 0.42,
        "rbc_red_dom_min": 0.06,
        "fill_hole_max_area_ratio": 0.00025,
        "bg_white_gray_min": 244,
        "bg_white_sat_max": 20,
        "white_guard_gray_min": 242,
        "white_guard_sat_max": 20,
        "white_void_gray_min": 228,
        "white_void_sat_max": 24,
        "white_bg_h_max": 0.16,
        "white_bg_edge_max": 0.25,
        "pale_void_gray_min": 210,
        "pale_void_sat_max": 52,
        "pale_void_edge_max": 0.16,
        "pale_void_h_max": 0.24,
        "white_void_min_area_ratio": 0.00035,
        "white_void_min_tissue_overlap": 0.002,
        "white_void_min_near_tissue": 0.10,
        "white_void_edge_margin_px": 4,
        "h_rescue_min": 0.34,
        "h_rescue_sat_min": 14,
        "h_rescue_edge_min": 0.03,
        "h_rescue_od_min": 0.08,
        "soft_rescue_enabled": False,
        "supplement_enabled": True,
        "supplement_od_min": 0.10,
        "supplement_h_min": 0.18,
        "supplement_e_min": 0.12,
        "supplement_sat_min": 12,
        "supplement_gray_max": 238,
        "supplement_edge_min": 0.02,
        "supplement_touch_kernel": 15,
        "supplement_min_component_area_ratio": 0.00003,
        "supplement_white_like_max": 0.28,
        "supplement_max_added_frac_of_clean": 0.12,
        "h_rescue_min_component_area_ratio": 0.00003,
        "soft_rescue_od_min": 0.06,
        "soft_rescue_h_min": 0.14,
        "soft_rescue_sat_min": 10,
        "soft_rescue_gray_max": 242,
        "soft_rescue_min_component_area_ratio": 0.00012,
        "soft_rescue_white_like_max": 0.55,
        "soft_rescue_touch_min": 0.0025,
        "soft_rescue_touch_kernel": 11,
        "soft_rescue_allow_large_without_touch": False,
        "soft_rescue_large_component_factor": 4.0,
        "soft_rescue_component_edge_min": 0.02,
        "edge_guard_px": 96,
        "edge_guard_od_max": 0.12,
        "edge_guard_h_max": 0.18,
        "edge_guard_score_max": 0.22,
        "edge_guard_gray_min": 170,
        "edge_guard_sat_max": 120,
        "edge_guard_loose_od_max": 0.18,
        "edge_guard_loose_h_max": 0.20,
        "edge_guard_loose_score_max": 0.35,
        "border_component_margin_px": 8,
        "border_component_h_max": 0.20,
        "border_component_od_max": 0.22,
        "border_component_white_min": 0.25,
        "border_component_small_area_ratio": 0.0012,
        "final_edge_crop_px": 48,
        "ink_black_v_max": 70,
        "ink_black_s_min": 18,
        "ink_green_h_min": 25,
        "ink_green_h_max": 110,
        "ink_green_s_min": 22,
        "ink_green_v_max": 250,
        "green_core_v_min": 70,
        "green_outline_dilate_kernel": 31,
        "dark_outline_v_max": 145,
        "dark_pen_v_max": 165,
        "dark_pen_s_min": 12,
        "dark_pen_min_area_ratio": 0.00003,
        "dark_pen_edge_px": 96,
        "dark_pen_min_green_overlap_frac": 0.001,
        "dark_pen_dilate_kernel": 5,
        "edge_strip_px": 48,
        "edge_art_s_max": 30,
        "edge_art_v_min": 175,
        "stats_min_gray": 8,
        "stats_max_gray": 252,
        "ink_dilate_kernel": 3,
        "black_canvas_gray_max": 8,
        "black_canvas_rgb_max": 12,
        "close_kernel": 11,
        "open_kernel": 5,
        "close_iterations": 1,
        "open_iterations": 1,
        "min_component_area_ratio": 0.0002,
        "reference_patches_dir": "../Tumor patch extraction/src/results/patches_raw",
        "reference_sample_patches": 400,
        "reference_seed": 1337,
        "low_tissue_retry_enabled": True,
        "low_tissue_retry_ds": 32.0,
        "low_tissue_retry_min_black_frac": 0.70,
        "low_tissue_retry_min_gain": 0.0015,
        "low_tissue_retry_min_added_px": 3000,
        "low_tissue_retry_min_nonwhite_added_frac": 0.85,
        "low_tissue_retry_max_white_leak_increase": 0.01,
        "low_tissue_retry_max_low_frac": 0.10,
        "roi_expand_px": 24,
        "roi_expand_frac": 0.08,
        "roi_min_component_area_ratio": 0.0002,
        "soft_relax_retry_enabled": False,
        "soft_relax_max_mask_frac": 0.32,
        "soft_relax_min_pale_void_frac": 0.40,
        "soft_relax_min_gain": 0.0035,
        "soft_relax_min_added_nonwhite_frac": 0.80,
        "soft_relax_max_white_leak_increase": 0.02,
    }
    if cfg_path.exists():
        with cfg_path.open("r") as f:
            pilot_cfg = yaml.safe_load(f) or {}
        paths_cfg = pilot_cfg.get("paths", {})
        raw_wsi_dirs = paths_cfg.get("wsi_dir", "data/raw_wsi")
        wsi_recursive = bool(paths_cfg.get("wsi_recursive", True))
        wsi_dirs = [_resolve_path(project_root, p) for p in _as_list(raw_wsi_dirs)]
        raw_mask_dir = paths_cfg.get("mask_dir", "data/masks")
        mask_dir = _resolve_path(project_root, raw_mask_dir)
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_cfg = dict(default_mask_cfg)
        mask_cfg.update(pilot_cfg.get("mask", {}))
    else:
        wsi_dirs = [_resolve_path(project_root, "data/raw_wsi")]
        wsi_recursive = True
        mask_cfg = default_mask_cfg

    for wd in wsi_dirs:
        if not wd.exists():
            raise FileNotFoundError(f"WSI dir not found: {wd}")

    mask_cfg = _apply_mask_profile(mask_cfg)
    strategy = str(mask_cfg.get("strategy", "auto_hybrid")).strip().lower()

    ref_caps = _calibrate_from_reference_patches(mask_cfg)
    if ref_caps:
        mask_cfg.update(ref_caps)
        print(
            "Reference calibration: "
            f"ref_rapid_thr_cap={mask_cfg['ref_rapid_thr_cap']:.2f}, "
            f"ref_od_thr_cap={mask_cfg['ref_od_thr_cap']:.4f}"
        )
    else:
        print("Reference calibration: not applied (no valid reference patches found).")

    wsis = list_wsis(wsi_dirs, recursive=wsi_recursive)
    if len(wsis) == 0:
        raise RuntimeError(f"No WSIs found in {[str(p) for p in wsi_dirs]}")

    if args.slide_id:
        want = str(args.slide_id).strip()
        wsis = [(p, center) for p, center in wsis if p.stem == want]
        if not wsis:
            raise RuntimeError(f"--slide-id '{want}' not found in WSI dirs.")

    n_cfg = int(mask_cfg.get("n_slides", 0))
    n_sel = int(args.n_slides) if args.n_slides is not None else n_cfg
    if n_sel > 0:
        wsis = wsis[:n_sel]

    print(f"WSI dirs: {[str(p) for p in wsi_dirs]}")
    print(f"WSI recursive scan: {wsi_recursive}")
    print(f"Found {len(wsis)} WSIs")

    by_center_rows: dict[str, list[dict]] = {}

    for p, center in wsis:
        stem = p.stem
        center_mask_dir = mask_dir / str(center) / "mask"
        center_mask_dir.mkdir(parents=True, exist_ok=True)
        out_npy = center_mask_dir / f"{stem}.npy"
        out_png = center_mask_dir / f"{stem}.png"

        overwrite = bool(mask_cfg.get("overwrite", True))
        if out_npy.exists() and not overwrite:
            print(f"[skip] {stem} mask exists")
            continue

        print(f"[mask] {stem}")
        slide = openslide.OpenSlide(str(p))
        target_ds = float(mask_cfg.get("target_ds", 64.0))
        img, lvl, ds = read_thumb_rgb(slide, target_ds=target_ds)
        black_canvas_frac_diag = float(_black_canvas_mask(img, mask_cfg).mean())
        bbox_frac_diag = float(non_black_bbox_fraction(img, cfg=mask_cfg))

        slide_cfg = dict(mask_cfg)
        profile_used = str(slide_cfg.get("profile", ""))
        sparse_canvas_mode = bool(slide_cfg.get("sparse_canvas_mode", False))
        if strategy == "auto_hybrid":
            sparse_canvas_mode = infer_sparse_canvas_mode(
                slide_suffix=p.suffix,
                black_canvas_frac=black_canvas_frac_diag,
                non_black_bbox_frac=bbox_frac_diag,
                cfg=mask_cfg,
            )
            if sparse_canvas_mode:
                sparse_profile = str(mask_cfg.get("sparse_canvas_profile", "sparse_canvas_pure_v1")).strip()
                slide_cfg = _apply_mask_profile(dict(mask_cfg), profile_override=sparse_profile, quiet=True)
                profile_used = sparse_profile
            else:
                slide_cfg = dict(mask_cfg)
                profile_used = str(slide_cfg.get("profile", ""))

        slide_cfg["sparse_canvas_mode"] = bool(sparse_canvas_mode)

        final_target_ds = float(slide_cfg.get("target_ds", target_ds))
        if abs(final_target_ds - ds) > 1e-3:
            img, lvl, ds = read_thumb_rgb(slide, target_ds=final_target_ds)
            black_canvas_frac_diag = float(_black_canvas_mask(img, slide_cfg).mean())
            bbox_frac_diag = float(non_black_bbox_fraction(img, cfg=slide_cfg))

        stain_diag = resolve_stain_vectors(img, cfg=slide_cfg)
        slide_cfg["stain_vector_h"] = np.asarray(stain_diag["stain_vector_h"], dtype=np.float32)
        slide_cfg["stain_vector_e"] = np.asarray(stain_diag["stain_vector_e"], dtype=np.float32)
        slide_cfg["stain_vector_r"] = np.asarray(stain_diag["stain_vector_r"], dtype=np.float32)
        stain_h = _vec_triplet(slide_cfg["stain_vector_h"])
        stain_e = _vec_triplet(slide_cfg["stain_vector_e"])
        stain_r = _vec_triplet(slide_cfg["stain_vector_r"])

        print(
            f"  thumb level={lvl} ds={ds:.1f} size={img.shape[1]}x{img.shape[0]} "
            f"profile={profile_used or 'none'} sparse_canvas={sparse_canvas_mode} "
            f"stain={stain_diag['stain_vector_source']} ({stain_diag['stain_estimation_reason']})"
        )
        mask, stats, _ = select_mask_with_fallback(img_rgb=img, cfg=slide_cfg)

        retry_used = False
        retry_mode = "none"
        retry_ds = np.nan
        retry_gain = 0.0
        retry_added_px = 0
        retry_added_nonwhite_frac = np.nan
        retry_bbox_x = np.nan
        retry_bbox_y = np.nan
        retry_bbox_w = np.nan
        retry_bbox_h = np.nan
        status_for_retry = str(stats.get("mask_status_effective", stats.get("mask_status", "")))
        if bool(slide_cfg.get("low_tissue_retry_enabled", True)) and status_for_retry == "low_tissue":
            black_frac = float(stats.get("black_canvas_frac", 0.0))
            low_frac_key = "mask_frac_effective" if bool(slide_cfg.get("sparse_canvas_mode", False)) else "mask_frac"
            low_frac = float(stats.get(low_frac_key, stats.get("mask_frac", 0.0)))
            min_black = float(slide_cfg.get("low_tissue_retry_min_black_frac", 0.70))
            max_low = float(
                slide_cfg.get(
                    "low_tissue_retry_max_low_frac_effective"
                    if bool(slide_cfg.get("sparse_canvas_mode", False))
                    else "low_tissue_retry_max_low_frac",
                    0.08,
                )
            )
            want_retry = bool((black_frac >= min_black) or (low_frac <= max_low))
            if want_retry:
                bbox = _non_black_bbox(img, slide_cfg)
                if bbox is not None:
                    x1, y1, x2, y2 = _expand_bbox(
                        *bbox,
                        width=int(img.shape[1]),
                        height=int(img.shape[0]),
                        expand_px=int(slide_cfg.get("roi_expand_px", 24)),
                        expand_frac=float(slide_cfg.get("roi_expand_frac", 0.08)),
                    )
                    retry_bbox_x = int(x1)
                    retry_bbox_y = int(y1)
                    retry_bbox_w = int(x2 - x1)
                    retry_bbox_h = int(y2 - y1)

                    # Map coarse ROI bounds to level-0 for targeted finer read.
                    x0_l0 = int(round(x1 * ds))
                    y0_l0 = int(round(y1 * ds))
                    w0_l0 = int(max(1, round((x2 - x1) * ds)))
                    h0_l0 = int(max(1, round((y2 - y1) * ds)))
                    want_ds = float(slide_cfg.get("low_tissue_retry_ds", 32.0))
                    if want_ds < ds:
                        roi_img, roi_lvl, roi_ds = read_region_rgb_at_ds(
                            slide=slide,
                            x0_l0=x0_l0,
                            y0_l0=y0_l0,
                            w_l0=w0_l0,
                            h_l0=h0_l0,
                            target_ds=want_ds,
                        )
                        retry_ds = float(roi_ds)
                        roi_cfg = dict(slide_cfg)
                        roi_cfg.update(
                            {
                                "min_component_area_ratio": float(slide_cfg.get("roi_min_component_area_ratio", 0.0002)),
                                "open_kernel": min(5, int(slide_cfg.get("open_kernel", 5))),
                                "close_kernel": max(7, int(slide_cfg.get("close_kernel", 11))),
                            }
                        )
                        roi_mask, _, _ = select_mask_with_fallback(img_rgb=roi_img, cfg=roi_cfg)
                        roi_mask_coarse = cv2.resize(
                            roi_mask.astype(np.uint8),
                            (int(x2 - x1), int(y2 - y1)),
                            interpolation=cv2.INTER_NEAREST,
                        )

                        merged = mask.copy().astype(np.uint8)
                        prev = merged[y1:y2, x1:x2]
                        merged[y1:y2, x1:x2] = np.maximum(prev, roi_mask_coarse)
                        old_frac = float(mask.mean())
                        new_frac = float(merged.mean())
                        retry_gain = max(0.0, new_frac - old_frac)

                        added = (merged > 0) & (mask == 0)
                        retry_added_px = int(added.sum())
                        if int(added.sum()) > 0:
                            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                            nonwhite = (gray < 242) | (hsv[:, :, 1] > 18)
                            retry_added_nonwhite_frac = float((added & nonwhite).sum() / added.sum())
                        else:
                            retry_added_nonwhite_frac = 1.0

                        old_white_leak = estimate_white_leak_fraction(img, mask, cfg=slide_cfg)
                        new_white_leak = estimate_white_leak_fraction(img, merged, cfg=slide_cfg)
                        retry_white_leak_delta = float(new_white_leak - old_white_leak)
                        status_fields = compute_mask_status_fields(img_rgb=img, mask=merged.astype(np.uint8), cfg=slide_cfg)
                        new_status_effective = str(status_fields.get("mask_status_effective", "unknown"))
                        accept = retry_acceptance(
                            retry_gain=retry_gain,
                            retry_added_px=retry_added_px,
                            retry_added_nonwhite_frac=retry_added_nonwhite_frac,
                            retry_white_leak_delta=retry_white_leak_delta,
                            new_status_effective=new_status_effective,
                            cfg=slide_cfg,
                        )
                        if accept:
                            mask = merged.astype(np.uint8)
                            retry_used = True
                            retry_mode = "roi_ds"
                            stats = dict(stats)
                            stats.update(status_fields)
                            stats["low_tissue_retry_gain"] = retry_gain
                            stats["low_tissue_retry_added_nonwhite_frac"] = retry_added_nonwhite_frac
                            stats["low_tissue_retry_white_leak_delta"] = retry_white_leak_delta
                            print(
                                f"  low-tissue ROI retry accepted: roi_level={roi_lvl} ds={roi_ds:.1f} "
                                f"gain={retry_gain*100:.2f}pp added_px={retry_added_px} "
                                f"tissue%={new_frac*100:.2f} status={new_status_effective}"
                            )
                        else:
                            print(
                                f"  low-tissue ROI retry rejected: roi_level={roi_lvl} ds={roi_ds:.1f} "
                                f"gain={retry_gain*100:.2f}pp added_px={retry_added_px} "
                                f"nonwhite={retry_added_nonwhite_frac:.3f} leak_delta={retry_white_leak_delta:.3f}"
                            )

        soft_relax_used = False
        soft_relax_gain = 0.0
        soft_relax_added_nonwhite_frac = np.nan
        soft_relax_white_leak = np.nan
        if bool(slide_cfg.get("soft_relax_retry_enabled", True)):
            cur_frac = float(mask.mean())
            cur_pale = float(stats.get("pale_void_frac", 0.0))
            if (cur_frac <= float(slide_cfg.get("soft_relax_max_mask_frac", 0.32))) and (
                cur_pale >= float(slide_cfg.get("soft_relax_min_pale_void_frac", 0.40))
            ):
                relax_cfg = dict(slide_cfg)
                relax_cfg.update(
                    {
                        "soft_rescue_touch_min": 0.0,
                        "soft_rescue_min_component_area_ratio": 0.00005,
                        "soft_rescue_white_like_max": 0.72,
                        "soft_rescue_sat_min": 8,
                    }
                )
                relax_mask, relax_stats, _ = select_mask_with_fallback(img_rgb=img, cfg=relax_cfg)
                old = mask.astype(np.uint8)
                new = relax_mask.astype(np.uint8)
                gain = float(new.mean() - old.mean())
                added = (new > 0) & (old == 0)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                nonwhite = (gray < 242) | (hsv[:, :, 1] > 18)
                white_hole = (gray >= 228) & (hsv[:, :, 1] <= 24)
                added_nonwhite = float((added & nonwhite).sum() / max(1, added.sum()))
                old_white_leak = float((old.astype(bool) & white_hole).sum() / max(1, old.sum()))
                new_white_leak = float((new.astype(bool) & white_hole).sum() / max(1, new.sum()))
                leak_delta = new_white_leak - old_white_leak

                soft_relax_gain = gain
                soft_relax_added_nonwhite_frac = added_nonwhite
                soft_relax_white_leak = new_white_leak

                min_gain = float(slide_cfg.get("soft_relax_min_gain", 0.0035))
                min_nonwhite = float(slide_cfg.get("soft_relax_min_added_nonwhite_frac", 0.80))
                max_leak_inc = float(slide_cfg.get("soft_relax_max_white_leak_increase", 0.02))
                if (gain >= min_gain) and (added_nonwhite >= min_nonwhite) and (leak_delta <= max_leak_inc):
                    mask = new
                    stats = dict(relax_stats)
                    stats.update(compute_mask_status_fields(img_rgb=img, mask=mask.astype(np.uint8), cfg=slide_cfg))
                    soft_relax_used = True
                    print(
                        f"  soft-relax retry accepted: gain={gain*100:.2f}pp "
                        f"added_nonwhite={added_nonwhite:.3f} white_leak={new_white_leak:.3f}"
                    )
                else:
                    print(
                        f"  soft-relax retry rejected: gain={gain*100:.2f}pp "
                        f"added_nonwhite={added_nonwhite:.3f} white_leak_delta={leak_delta:.3f}"
                    )
        stats = dict(stats)
        stats.update(compute_mask_status_fields(img_rgb=img, mask=mask.astype(np.uint8), cfg=slide_cfg))
        white_leak_frac_est = estimate_white_leak_fraction(img, mask, cfg=slide_cfg)
        slide.close()

        np.save(out_npy, mask.astype(np.uint8))
        save_preview(img, mask, out_png)
        row = {
                "slide_id": stem,
                "center": str(center),
                "thumb_level": int(lvl),
                "thumb_downsample": float(ds),
                "thumb_width": int(img.shape[1]),
                "thumb_height": int(img.shape[0]),
                "low_tissue_retry_used": bool(retry_used),
                "low_tissue_retry_mode": str(retry_mode),
                "low_tissue_retry_downsample": float(retry_ds) if np.isfinite(retry_ds) else np.nan,
                "low_tissue_retry_gain": float(retry_gain),
                "low_tissue_retry_added_px": int(retry_added_px),
                "low_tissue_retry_added_nonwhite_frac": (
                    float(retry_added_nonwhite_frac) if np.isfinite(retry_added_nonwhite_frac) else np.nan
                ),
                "low_tissue_retry_bbox_x": int(retry_bbox_x) if np.isfinite(retry_bbox_x) else np.nan,
                "low_tissue_retry_bbox_y": int(retry_bbox_y) if np.isfinite(retry_bbox_y) else np.nan,
                "low_tissue_retry_bbox_w": int(retry_bbox_w) if np.isfinite(retry_bbox_w) else np.nan,
                "low_tissue_retry_bbox_h": int(retry_bbox_h) if np.isfinite(retry_bbox_h) else np.nan,
                "soft_relax_retry_used": bool(soft_relax_used),
                "soft_relax_retry_gain": float(soft_relax_gain),
                "soft_relax_retry_added_nonwhite_frac": (
                    float(soft_relax_added_nonwhite_frac) if np.isfinite(soft_relax_added_nonwhite_frac) else np.nan
                ),
                "soft_relax_retry_white_leak": (
                    float(soft_relax_white_leak) if np.isfinite(soft_relax_white_leak) else np.nan
                ),
                "sparse_canvas_mode": bool(sparse_canvas_mode),
                "profile_used": str(profile_used),
                "non_black_bbox_frac": float(bbox_frac_diag),
                "white_leak_frac_est": float(white_leak_frac_est),
                "mask_frac": float(stats["mask_frac"]),
                "mask_frac_effective": float(stats.get("mask_frac_effective", np.nan)),
                "fallback_used": bool(stats["fallback_used"]),
                "fallback_attempted": bool(stats.get("fallback_attempted", False)),
                "mask_status": str(stats["mask_status"]),
                "mask_status_effective": str(stats.get("mask_status_effective", stats["mask_status"])),
                "status_basis": str(stats.get("status_basis", "global")),
                "stain_vector_source": str(stain_diag.get("stain_vector_source", "")),
                "stain_estimation_reason": str(stain_diag.get("stain_estimation_reason", "")),
                "stain_estimation_pixels": int(stain_diag.get("stain_estimation_pixels", 0)),
                "stain_estimation_valid_frac": float(stain_diag.get("stain_estimation_valid_frac", np.nan)),
                "stain_estimation_pair_score": float(stain_diag.get("stain_estimation_pair_score", np.nan)),
                "stain_estimation_he_dot": float(stain_diag.get("stain_estimation_he_dot", np.nan)),
                "stain_vector_h_r": float(stain_h[0]),
                "stain_vector_h_g": float(stain_h[1]),
                "stain_vector_h_b": float(stain_h[2]),
                "stain_vector_e_r": float(stain_e[0]),
                "stain_vector_e_g": float(stain_e[1]),
                "stain_vector_e_b": float(stain_e[2]),
                "stain_vector_r_r": float(stain_r[0]),
                "stain_vector_r_g": float(stain_r[1]),
                "stain_vector_r_b": float(stain_r[2]),
                "od_thr": float(stats["od_thr"]),
                "sat_thr": float(stats["sat_thr"]) if np.isfinite(stats["sat_thr"]) else np.nan,
                "ink_frac": float(stats["ink_frac"]),
                "edge_art_frac": float(stats.get("edge_art_frac", np.nan)),
                "rbc_like_frac": float(stats.get("rbc_like_frac", np.nan)),
                "clot_like_frac": float(stats.get("clot_like_frac", np.nan)),
                "drop_blood_like_frac": float(stats.get("drop_blood_like_frac", np.nan)),
                "large_white_void_frac": float(stats.get("large_white_void_frac", np.nan)),
                "pale_void_frac": float(stats.get("pale_void_frac", np.nan)),
                "hard_white_bg_frac": float(stats.get("hard_white_bg_frac", np.nan)),
                "h_rescue_frac": float(stats.get("h_rescue_frac", np.nan)),
                "soft_rescue_frac": float(stats.get("soft_rescue_frac", np.nan)),
                "supplement_added_frac": float(stats.get("supplement_added_frac", np.nan)),
                "edge_false_frac": float(stats.get("edge_false_frac", np.nan)),
                "border_drop_frac": float(stats.get("border_drop_frac", np.nan)),
                "edge_crop_removed_frac": float(stats.get("edge_crop_removed_frac", np.nan)),
                "black_canvas_frac": float(stats.get("black_canvas_frac", np.nan)),
                "bg_white_frac": float(stats.get("bg_white_frac", np.nan)),
                "valid_stats_frac": float(stats.get("valid_stats_frac", np.nan)),
                "blood_like_frac": float(stats["blood_like_frac"]),
            }
        summary_rows.append(row)
        by_center_rows.setdefault(str(center), []).append(row)

        print(
            f"  saved: {out_npy.name}  (shape={mask.shape}, tissue%={mask.mean()*100:.2f}, "
            f"status={stats['mask_status']} eff={stats.get('mask_status_effective', stats['mask_status'])}, "
            f"fallback={stats['fallback_used']})"
        )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values("slide_id").reset_index(drop=True)
        summary_path = mask_dir / "mask_summary.csv"
        if summary_path.exists():
            prev_df = pd.read_csv(summary_path, dtype={"slide_id": "string"})
            prev_df["slide_id"] = prev_df["slide_id"].astype(str).str.strip()
            summary_df["slide_id"] = summary_df["slide_id"].astype(str).str.strip()
            summary_df = pd.concat([prev_df, summary_df], ignore_index=True, sort=False)
            summary_df = (
                summary_df.drop_duplicates(subset=["slide_id"], keep="last")
                .sort_values("slide_id")
                .reset_index(drop=True)
            )
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved mask summary -> {summary_path}")
        for center, rows in sorted(by_center_rows.items()):
            center_df = pd.DataFrame(rows).sort_values("slide_id").reset_index(drop=True)
            center_summary_path = mask_dir / center / "mask" / "mask_summary.csv"
            center_summary_path.parent.mkdir(parents=True, exist_ok=True)
            center_df.to_csv(center_summary_path, index=False)

    print("Done creating masks.")


if __name__ == "__main__":
    main()
    

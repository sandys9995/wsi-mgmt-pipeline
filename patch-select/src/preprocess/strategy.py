from __future__ import annotations

from typing import Any

import numpy as np


def _vec3_from_cfg(cfg: dict[str, Any], key: str, default: tuple[float, float, float]) -> np.ndarray:
    raw = cfg.get(key, default)
    try:
        arr = np.asarray(raw, dtype=np.float32).reshape(-1)
        if arr.size != 3 or np.any(~np.isfinite(arr)):
            raise ValueError("bad vec")
        return arr
    except Exception:
        return np.asarray(default, dtype=np.float32)


def _unit(v: np.ndarray, *, abs_value: bool = False) -> np.ndarray:
    vv = np.asarray(v, dtype=np.float32).reshape(3)
    if abs_value:
        vv = np.abs(vv)
    n = float(np.linalg.norm(vv))
    if n <= 1e-8:
        return vv.astype(np.float32)
    return (vv / n).astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = _unit(a)
    bb = _unit(b)
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if den <= 1e-8:
        return -1.0
    return float(np.dot(aa, bb) / den)


def black_canvas_mask(rgb: np.ndarray, cfg: dict[str, Any] | None = None) -> np.ndarray:
    import cv2

    cfg = cfg or {}
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    black_gray = int(cfg.get("black_canvas_gray_max", 8))
    black_rgb = int(cfg.get("black_canvas_rgb_max", 12))
    return (gray <= black_gray) | (
        (rgb[:, :, 0] <= black_rgb) & (rgb[:, :, 1] <= black_rgb) & (rgb[:, :, 2] <= black_rgb)
    )


def non_black_bbox_fraction(rgb: np.ndarray, cfg: dict[str, Any] | None = None) -> float:
    import cv2

    cfg = cfg or {}
    black = black_canvas_mask(rgb, cfg=cfg)
    non_black = (~black).astype(np.uint8)
    if int(non_black.sum()) == 0:
        return 0.0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(non_black, connectivity=8)
    if n <= 1:
        return 0.0
    min_ratio = float(cfg.get("roi_min_component_area_ratio", 0.0002))
    min_area = int(max(1, round(min_ratio * non_black.size)))
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    merged = keep[labels]
    if int(merged.sum()) == 0:
        return 0.0
    ys, xs = np.where(merged)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    return float(bbox_area / float(non_black.size))


def infer_sparse_canvas_mode(
    slide_suffix: str,
    black_canvas_frac: float,
    non_black_bbox_frac: float,
    cfg: dict[str, Any] | None = None,
) -> bool:
    cfg = cfg or {}
    black_min = float(cfg.get("sparse_canvas_black_frac_min", 0.75))
    bbox_max = float(cfg.get("sparse_canvas_bbox_frac_max", 0.15))
    suffix = str(slide_suffix).strip().lower()
    return bool(
        (black_canvas_frac >= black_min)
        or (non_black_bbox_frac <= bbox_max)
        or (suffix == ".mrxs")
    )


def resolve_stain_vectors(
    rgb: np.ndarray,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Resolve H/E/R stain vectors for a slide thumbnail.
    - When dynamic estimation is enabled, estimate vectors from thumbnail OD space.
    - On instability/low signal, safely fall back to configured/default vectors.
    """
    import cv2

    cfg = cfg or {}
    h_default = _unit(_vec3_from_cfg(cfg, "stain_vector_h", (0.651, 0.701, 0.290)), abs_value=True)
    e_default = _unit(_vec3_from_cfg(cfg, "stain_vector_e", (0.216, 0.801, 0.558)), abs_value=True)
    r_default = _unit(_vec3_from_cfg(cfg, "stain_vector_r", (0.316, -0.598, 0.737)))

    def _result(
        source: str,
        reason: str,
        n_pix: int,
        valid_frac: float,
        pair_score: float = float("nan"),
        he_dot: float = float("nan"),
    ) -> dict[str, Any]:
        return {
            "stain_vector_h": h_default.astype(np.float32),
            "stain_vector_e": e_default.astype(np.float32),
            "stain_vector_r": r_default.astype(np.float32),
            "stain_vector_source": source,
            "stain_estimation_reason": reason,
            "stain_estimation_pixels": int(n_pix),
            "stain_estimation_valid_frac": float(valid_frac),
            "stain_estimation_pair_score": float(pair_score),
            "stain_estimation_he_dot": float(he_dot),
        }

    if bool(cfg.get("stain_vector_fixed", False)):
        return _result("fixed", "fixed_cfg", 0, 0.0)

    if not bool(cfg.get("dynamic_stain_vectors_enabled", True)):
        return _result("static", "dynamic_disabled", 0, 0.0)

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return _result("fallback", "bad_image", 0, 0.0)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    black = black_canvas_mask(rgb, cfg=cfg)

    od = -np.log(np.clip(rgb.astype(np.float32) / 255.0, 1e-6, 1.0))
    od_sum = od.sum(axis=2)

    od_min = float(cfg.get("stain_estimation_od_min", 0.12))
    gray_max = int(cfg.get("stain_estimation_gray_max", 245))
    sat_min = int(cfg.get("stain_estimation_sat_min", 6))
    valid = (~black) & (gray <= gray_max) & ((od_sum >= od_min) | (hsv[:, :, 1] >= sat_min))
    valid_frac = float(valid.mean())

    min_pixels = int(cfg.get("stain_estimation_min_pixels", 1200))
    n_pix = int(valid.sum())
    if n_pix < min_pixels:
        return _result("fallback", "too_few_pixels", n_pix, valid_frac)

    px = od[valid].reshape(-1, 3).astype(np.float32)
    sample_max = int(cfg.get("stain_estimation_sample_max", 60000))
    if px.shape[0] > sample_max:
        idx = np.linspace(0, px.shape[0] - 1, num=sample_max, dtype=np.int64)
        px = px[idx]

    norms = np.linalg.norm(px, axis=1)
    good = norms > 1e-8
    if int(good.sum()) < min_pixels:
        return _result("fallback", "too_few_nonzero_od", int(good.sum()), valid_frac)
    px_u = px[good] / norms[good][:, None]

    cov = np.cov(px_u, rowvar=False)
    if cov.shape != (3, 3) or np.any(~np.isfinite(cov)):
        return _result("fallback", "cov_invalid", int(px_u.shape[0]), valid_frac)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)
    second = int(order[-2])
    first = int(order[-1])
    if float(eigvals[second]) <= 1e-8:
        return _result("fallback", "cov_rank_low", int(px_u.shape[0]), valid_frac)

    basis = eigvecs[:, [second, first]].astype(np.float32)  # (3,2)
    proj = px_u @ basis
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    if angles.size < min_pixels:
        return _result("fallback", "angle_samples_low", int(angles.size), valid_frac)

    p_low = float(cfg.get("stain_estimation_angle_pct_low", 1.0))
    p_high = float(cfg.get("stain_estimation_angle_pct_high", 99.0))
    p_low = max(0.0, min(100.0, p_low))
    p_high = max(0.0, min(100.0, p_high))
    if p_high <= p_low:
        p_low, p_high = 1.0, 99.0

    a_low = float(np.percentile(angles, p_low))
    a_high = float(np.percentile(angles, p_high))
    v1 = basis @ np.array([np.cos(a_low), np.sin(a_low)], dtype=np.float32)
    v2 = basis @ np.array([np.cos(a_high), np.sin(a_high)], dtype=np.float32)

    v1 = _unit(v1, abs_value=True)
    v2 = _unit(v2, abs_value=True)
    score_12 = _cosine(v1, h_default) + _cosine(v2, e_default)
    score_21 = _cosine(v2, h_default) + _cosine(v1, e_default)
    if score_21 > score_12:
        h_vec, e_vec = v2, v1
        pair_score = float(score_21)
    else:
        h_vec, e_vec = v1, v2
        pair_score = float(score_12)

    he_dot = float(abs(np.dot(_unit(h_vec), _unit(e_vec))))
    max_he_dot = float(cfg.get("stain_estimation_max_he_dot", 0.985))
    min_pair_score = float(cfg.get("stain_estimation_min_pair_score", 1.20))
    if (he_dot >= max_he_dot) or (pair_score < min_pair_score):
        return _result(
            "fallback",
            "he_unstable",
            int(px_u.shape[0]),
            valid_frac,
            pair_score=pair_score,
            he_dot=he_dot,
        )

    r_vec = np.cross(h_vec, e_vec).astype(np.float32)
    if float(np.linalg.norm(r_vec)) <= 1e-8:
        r_vec = r_default.copy()
    else:
        r_vec = _unit(r_vec, abs_value=False)
        if _cosine(r_vec, r_default) < 0.0:
            r_vec = (-r_vec).astype(np.float32)

    return {
        "stain_vector_h": h_vec.astype(np.float32),
        "stain_vector_e": e_vec.astype(np.float32),
        "stain_vector_r": r_vec.astype(np.float32),
        "stain_vector_source": "estimated",
        "stain_estimation_reason": "ok",
        "stain_estimation_pixels": int(px_u.shape[0]),
        "stain_estimation_valid_frac": float(valid_frac),
        "stain_estimation_pair_score": float(pair_score),
        "stain_estimation_he_dot": float(he_dot),
    }


def estimate_white_leak_fraction(
    rgb: np.ndarray,
    mask: np.ndarray,
    cfg: dict[str, Any] | None = None,
) -> float:
    import cv2

    cfg = cfg or {}
    if int(mask.sum()) == 0:
        return 0.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    white_gray = int(cfg.get("white_void_gray_min", 228))
    white_sat = int(cfg.get("white_void_sat_max", 24))
    white_like = (gray >= white_gray) & (hsv[:, :, 1] <= white_sat)
    mask_bool = mask.astype(bool)
    return float((mask_bool & white_like).sum() / max(1, mask_bool.sum()))


def retry_acceptance(
    *,
    retry_gain: float,
    retry_added_px: int,
    retry_added_nonwhite_frac: float,
    retry_white_leak_delta: float,
    new_status_effective: str,
    cfg: dict[str, Any] | None = None,
) -> bool:
    cfg = cfg or {}
    min_gain = float(cfg.get("low_tissue_retry_min_gain", 0.0015))
    min_added_px = int(cfg.get("low_tissue_retry_min_added_px", 3000))
    min_nonwhite = float(cfg.get("low_tissue_retry_min_nonwhite_added_frac", 0.85))
    max_white_leak_inc = float(cfg.get("low_tissue_retry_max_white_leak_increase", 0.01))
    return bool(
        ((retry_gain >= min_gain) or (retry_added_px >= min_added_px))
        and (retry_added_nonwhite_frac >= min_nonwhite)
        and (retry_white_leak_delta <= max_white_leak_inc)
        and (new_status_effective != "high_tissue")
    )

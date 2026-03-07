from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class MaskBuildResult:
    mask: np.ndarray
    stats: dict[str, Any]
    debug_layers: dict[str, np.ndarray]


def _cfg(cfg: dict | None, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    return cfg.get(key, default)


def _safe_percentile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, q))


def _od(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float32) / 255.0
    return -np.log(np.clip(x, 1e-6, 1.0))


def _vec3_from_cfg(cfg: dict | None, key: str, default: tuple[float, float, float]) -> np.ndarray:
    raw = _cfg(cfg, key, default)
    try:
        arr = np.asarray(raw, dtype=np.float32).reshape(-1)
        if arr.size != 3 or np.any(~np.isfinite(arr)):
            raise ValueError("bad vec")
        return arr
    except Exception:
        return np.asarray(default, dtype=np.float32)


def _he_stain_matrix() -> np.ndarray:
    h = np.array([0.650, 0.704, 0.286], dtype=np.float32)
    e = np.array([0.072, 0.990, 0.105], dtype=np.float32)
    h = h / np.linalg.norm(h)
    e = e / np.linalg.norm(e)
    return np.stack([h, e], axis=1)  # (3,2)


def _he_channels_from_od(od: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    he = _he_stain_matrix()
    he_pinv = np.linalg.pinv(he)  # (2,3)
    flat = od.reshape(-1, 3).T
    c = he_pinv @ flat
    c = np.clip(c, 0.0, None)
    h = c[0, :].reshape(od.shape[:2]).astype(np.float32)
    e = c[1, :].reshape(od.shape[:2]).astype(np.float32)
    return h, e


def _stain_channels_from_od(od: np.ndarray, cfg: dict | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Defaults can be overridden from config (vectors should be in OD space).
    h_vec = _vec3_from_cfg(cfg, "stain_vector_h", (0.651, 0.701, 0.290))
    e_vec = _vec3_from_cfg(cfg, "stain_vector_e", (0.216, 0.801, 0.558))
    r_vec = _vec3_from_cfg(cfg, "stain_vector_r", (0.316, -0.598, 0.737))

    def _norm(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n <= 1e-8:
            return v
        return (v / n).astype(np.float32)

    h_vec = _norm(h_vec)
    e_vec = _norm(e_vec)
    r_vec = _norm(r_vec)
    M = np.stack([h_vec, e_vec, r_vec], axis=1).astype(np.float32)  # (3,3)
    det = float(np.linalg.det(M))
    if abs(det) < 1e-4:
        # Fallback: classic H/E + residual proxy.
        h, e = _he_channels_from_od(od)
        r = np.clip(od.sum(axis=2).astype(np.float32) - (h + e), 0.0, None)
        return h, e, r

    pinv = np.linalg.pinv(M)
    flat = od.reshape(-1, 3).T
    c = pinv @ flat
    c = np.clip(c, 0.0, None)
    h = c[0, :].reshape(od.shape[:2]).astype(np.float32)
    e = c[1, :].reshape(od.shape[:2]).astype(np.float32)
    r = c[2, :].reshape(od.shape[:2]).astype(np.float32)
    return h, e, r


def _fill_holes(mask: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    inv = (mask == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    if n <= 1:
        return mask

    border_labels = set()
    border_labels.update(np.unique(labels[0, :]).tolist())
    border_labels.update(np.unique(labels[-1, :]).tolist())
    border_labels.update(np.unique(labels[:, 0]).tolist())
    border_labels.update(np.unique(labels[:, -1]).tolist())

    max_hole_ratio = float(_cfg(cfg, "fill_hole_max_area_ratio", 0.00025))
    max_hole_area = int(max(1, round(max_hole_ratio * mask.size)))

    holes = np.zeros(n, dtype=bool)
    holes[1:] = stats[1:, cv2.CC_STAT_AREA] <= max_hole_area
    for lb in border_labels:
        holes[int(lb)] = False
    holes[0] = False
    out = mask.copy().astype(np.uint8)
    out[holes[labels]] = 1
    return out


def _cleanup_mask(mask: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    close_k = max(3, int(_cfg(cfg, "close_kernel", 11)) | 1)
    open_k = max(3, int(_cfg(cfg, "open_kernel", 5)) | 1)
    close_it = int(_cfg(cfg, "close_iterations", 1))
    open_it = int(_cfg(cfg, "open_iterations", 1))
    min_area_ratio = float(_cfg(cfg, "min_component_area_ratio", 0.0002))

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))

    clean = mask.astype(np.uint8)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k_close, iterations=max(0, close_it))
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, k_open, iterations=max(0, open_it))
    clean = _fill_holes(clean, cfg=cfg)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    if n > 1:
        min_area = int(max(1, round(min_area_ratio * clean.size)))
        keep = np.zeros(n, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
        clean = keep[labels].astype(np.uint8)
    return clean


def _filter_components_by_area(mask_bool: np.ndarray, min_area_ratio: float) -> np.ndarray:
    mask_u8 = mask_bool.astype(np.uint8)
    if int(mask_u8.sum()) == 0:
        return np.zeros_like(mask_bool, dtype=bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return mask_u8.astype(bool)
    min_area = int(max(1, round(float(min_area_ratio) * mask_u8.size)))
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def _rapid_tissue_channel(rgb: np.ndarray) -> np.ndarray:
    """
    Rapid-style single-channel tissue emphasis:
    emphasize H&E tissue and suppress green pen/background.
    """
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    ch = 0.5 * (r + b) - g
    return np.clip(ch, 0.0, None)


def _normalize_by_percentiles(arr: np.ndarray, valid: np.ndarray, q_low: float, q_high: float) -> np.ndarray:
    vals = arr[valid] if np.any(valid) else arr.reshape(-1)
    lo = _safe_percentile(vals, q_low)
    hi = _safe_percentile(vals, q_high)
    if hi <= lo + 1e-6:
        return np.zeros_like(arr, dtype=np.float32), lo, hi
    out = (arr - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    return out, float(lo), float(hi)


def _artifact_masks(rgb: np.ndarray, hsv: np.ndarray, cfg: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)

    # Pen/marker colors
    ink_black = (v < int(_cfg(cfg, "ink_black_v_max", 70))) & (s > int(_cfg(cfg, "ink_black_s_min", 18)))
    ink_green = (
        (h >= int(_cfg(cfg, "ink_green_h_min", 25)))
        & (h <= int(_cfg(cfg, "ink_green_h_max", 110)))
        & (s > int(_cfg(cfg, "ink_green_s_min", 22)))
        & (v < int(_cfg(cfg, "ink_green_v_max", 250)))
    )
    ink_blue = (h >= 85) & (h <= 140) & (s > 22) & (v < 250)
    mag_hi = int(_cfg(cfg, "ink_magenta_h_min", 145))
    mag_lo = int(_cfg(cfg, "ink_magenta_h_low_max", 10))
    mag_s = int(_cfg(cfg, "ink_magenta_s_min", 150))
    mag_v = int(_cfg(cfg, "ink_magenta_v_max", 245))
    ink_magenta = (((h >= mag_hi) | (h <= mag_lo)) & (s > mag_s) & (v < mag_v))
    ink = ink_black | ink_green | ink_blue | ink_magenta

    # RBC-like regions: high red/eosin, low hematoxylin proxy.
    blood_like = (r > 140) & ((r - g) > 30) & ((r - b) > 25) & (s > 35)

    k = max(1, int(_cfg(cfg, "ink_dilate_kernel", 3)))
    if k > 1:
        k = max(3, k | 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        ink = cv2.dilate(ink.astype(np.uint8), ker, iterations=1) > 0
    return ink, blood_like


def _build_mask(img_rgb: np.ndarray, cfg: dict | None, permissive: bool = False) -> MaskBuildResult:
    border = int(_cfg(cfg, "border_ignore_px", 10))
    work = img_rgb.copy()
    if border > 0:
        work[:border, :, :] = 255
        work[-border:, :, :] = 255
        work[:, :border, :] = 255
        work[:, -border:, :] = 255

    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(work, cv2.COLOR_RGB2HSV)

    black_gray = int(_cfg(cfg, "black_canvas_gray_max", 8))
    black_rgb = int(_cfg(cfg, "black_canvas_rgb_max", 12))
    black_canvas = (gray <= black_gray) | (
        (work[:, :, 0] <= black_rgb) & (work[:, :, 1] <= black_rgb) & (work[:, :, 2] <= black_rgb)
    )

    # Very bright, low-saturation background that should not drive slide-wise stats.
    bg_white_gray = int(_cfg(cfg, "bg_white_gray_min", 244))
    bg_white_sat = int(_cfg(cfg, "bg_white_sat_max", 20))
    bg_white = (gray >= bg_white_gray) & (hsv[:, :, 1] <= bg_white_sat)

    stats_min_gray = int(_cfg(cfg, "stats_min_gray", 8))
    stats_max_gray = int(_cfg(cfg, "stats_max_gray", 252))
    valid_stats = (~black_canvas) & (~bg_white) & (gray >= stats_min_gray) & (gray <= stats_max_gray)

    rapid = _rapid_tissue_channel(work)
    rapid_vals = rapid[valid_stats] if np.any(valid_stats) else rapid.reshape(-1)
    rapid_u8 = np.clip(rapid_vals, 0, 255).astype(np.uint8).reshape(-1, 1)
    otsu_rapid_u8, _ = cv2.threshold(rapid_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rapid_thr = float(otsu_rapid_u8)

    # OD channel contributes when tissue is pale/low-chroma.
    od = _od(work)
    od_sum = od.sum(axis=2).astype(np.float32)
    h_ch, e_ch, r_ch = _stain_channels_from_od(od, cfg=cfg)
    od_vals = od_sum[valid_stats] if np.any(valid_stats) else od_sum.reshape(-1)
    if permissive:
        od_q = float(_cfg(cfg, "fallback_od_q", 35))
        score_thr = float(_cfg(cfg, "fallback_score_thr", 0.28))
    else:
        od_q = float(_cfg(cfg, "od_q", 55))
        score_thr = float(_cfg(cfg, "score_thr", 0.36))
    od_thr = _safe_percentile(od_vals, od_q)
    # Optional caps derived from reference good patches to avoid over-strict thresholds.
    od_thr_cap = _cfg(cfg, "ref_od_thr_cap", None)
    rapid_thr_cap = _cfg(cfg, "ref_rapid_thr_cap", None)
    if od_thr_cap is not None:
        od_thr = min(float(od_thr), float(od_thr_cap))
    if rapid_thr_cap is not None:
        rapid_thr = min(float(rapid_thr), float(rapid_thr_cap))

    # Sobel edge map for background removal robustness (PathQC-like).
    g_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(g_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g_blur, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)

    rapid_n, _, _ = _normalize_by_percentiles(rapid, valid_stats, 5, 98)
    od_n, _, _ = _normalize_by_percentiles(od_sum, valid_stats, 10, 98)
    edge_n, _, _ = _normalize_by_percentiles(edge, valid_stats, 30, 99)

    w_rapid = float(_cfg(cfg, "w_rapid", 0.55))
    w_od = float(_cfg(cfg, "w_od", 0.30))
    w_edge = float(_cfg(cfg, "w_edge", 0.15))
    score = w_rapid * rapid_n + w_od * od_n + w_edge * edge_n

    rapid_mask = rapid >= rapid_thr
    od_mask = od_sum >= od_thr
    score_mask = score >= score_thr
    # Guard against large bright-white voids being counted as tissue.
    white_guard_gray = int(_cfg(cfg, "white_guard_gray_min", 242))
    white_guard_sat = int(_cfg(cfg, "white_guard_sat_max", 20))
    white_guard = (gray >= white_guard_gray) & (hsv[:, :, 1] <= white_guard_sat)
    base_tissue = (rapid_mask | od_mask) & score_mask & valid_stats & (~white_guard)

    ink_mask, blood_like_mask = _artifact_masks(work, hsv, cfg=cfg)
    ink_mask[black_canvas] = False

    # Pen loops often have green/cyan core + dark outline: remove both.
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    green_core = (
        (h >= int(_cfg(cfg, "ink_green_h_min", 25)))
        & (h <= int(_cfg(cfg, "ink_green_h_max", 110)))
        & (s > int(_cfg(cfg, "ink_green_s_min", 22)))
        & (v > int(_cfg(cfg, "green_core_v_min", 70)))
    )
    green_core[black_canvas] = False
    k_outline = max(3, int(_cfg(cfg, "green_outline_dilate_kernel", 31)) | 1)
    ker_outline = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_outline, k_outline))
    near_green = cv2.dilate(green_core.astype(np.uint8), ker_outline, iterations=1) > 0
    dark_outline = near_green & (v < int(_cfg(cfg, "dark_outline_v_max", 145)))
    ink_mask |= green_core | near_green | dark_outline

    # Large dark marker/pen strokes (loops): select components by darkness + saturation,
    # then keep those that touch edge or are close to green marker regions.
    if bool(_cfg(cfg, "enable_dark_pen_mask", True)):
        dark_v_max = int(_cfg(cfg, "dark_pen_v_max", 165))
        dark_s_min = int(_cfg(cfg, "dark_pen_s_min", 12))
        dark_cc_min_ratio = float(_cfg(cfg, "dark_pen_min_area_ratio", 0.00003))
        dark_edge_px = int(_cfg(cfg, "dark_pen_edge_px", 96))
        green_overlap_min = float(_cfg(cfg, "dark_pen_min_green_overlap_frac", 0.001))
        dark_candidates = (hsv[:, :, 2] < dark_v_max) & (hsv[:, :, 1] > dark_s_min) & (~black_canvas)
        if np.any(dark_candidates):
            n_cc, labels_cc, stats_cc, _ = cv2.connectedComponentsWithStats(
                dark_candidates.astype(np.uint8), connectivity=8
            )
            if n_cc > 1:
                min_area = int(max(1, round(dark_cc_min_ratio * dark_candidates.size)))
                keep = np.zeros(n_cc, dtype=bool)
                for lb in range(1, n_cc):
                    area = int(stats_cc[lb, cv2.CC_STAT_AREA])
                    if area < min_area:
                        continue
                    x = int(stats_cc[lb, cv2.CC_STAT_LEFT])
                    y = int(stats_cc[lb, cv2.CC_STAT_TOP])
                    w = int(stats_cc[lb, cv2.CC_STAT_WIDTH])
                    h_cc = int(stats_cc[lb, cv2.CC_STAT_HEIGHT])
                    touch_edge = (
                        (x <= dark_edge_px)
                        or (y <= dark_edge_px)
                        or ((x + w) >= (work.shape[1] - dark_edge_px))
                        or ((y + h_cc) >= (work.shape[0] - dark_edge_px))
                    )
                    comp = labels_cc == lb
                    green_overlap = float((comp & near_green).mean())
                    if touch_edge or (green_overlap >= green_overlap_min):
                        keep[lb] = True
                dark_pen_mask = keep[labels_cc]
                dark_pen_dilate = max(1, int(_cfg(cfg, "dark_pen_dilate_kernel", 5)))
                if dark_pen_dilate > 1:
                    dark_pen_dilate = max(3, dark_pen_dilate | 1)
                    ker_dark = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (dark_pen_dilate, dark_pen_dilate)
                    )
                    dark_pen_mask = cv2.dilate(dark_pen_mask.astype(np.uint8), ker_dark, iterations=1) > 0
                ink_mask |= dark_pen_mask

    # Coverslip/blank-edge artifacts near slide boundary.
    edge_strip = int(_cfg(cfg, "edge_strip_px", 48))
    edge_art = np.zeros_like(black_canvas, dtype=bool)
    if edge_strip > 0:
        edge_strip = min(edge_strip, max(1, min(work.shape[0], work.shape[1]) // 3))
        edge_zone = np.zeros_like(black_canvas, dtype=bool)
        edge_zone[:edge_strip, :] = True
        edge_zone[-edge_strip:, :] = True
        edge_zone[:, : max(8, edge_strip // 2)] = True
        edge_zone[:, -max(8, edge_strip // 2):] = True
        edge_art = edge_zone & (hsv[:, :, 1] < int(_cfg(cfg, "edge_art_s_max", 30))) & (hsv[:, :, 2] > int(_cfg(cfg, "edge_art_v_min", 175)))

    # RBC mask (use blood as background): high eosin + low hematoxylin + red dominance.
    e_n, _, _ = _normalize_by_percentiles(e_ch, valid_stats, 5, 99)
    h_n, _, _ = _normalize_by_percentiles(h_ch, valid_stats, 5, 99)
    r_n, _, _ = _normalize_by_percentiles(r_ch, valid_stats, 5, 99)
    stain_guided_enabled = bool(_cfg(cfg, "stain_guided_enabled", True))
    r = work[:, :, 0].astype(np.float32)
    g = work[:, :, 1].astype(np.float32)
    b = work[:, :, 2].astype(np.float32)
    red_dom = (r - 0.5 * (g + b)) / 255.0
    rbc_like = (
        (e_n > float(_cfg(cfg, "rbc_e_min", 0.58)))
        & (h_n < float(_cfg(cfg, "rbc_h_max", 0.42)))
        & (red_dom > float(_cfg(cfg, "rbc_red_dom_min", 0.06)))
    )
    rbc_like &= (~black_canvas)

    # Hard clot suppression: red-dominant eosin-rich components are removed.
    exclude_clot_from_tissue = bool(_cfg(cfg, "exclude_clot_from_tissue", True))
    clot_like = np.zeros_like(rbc_like, dtype=bool)
    if exclude_clot_from_tissue:
        clot_h_max = float(_cfg(cfg, "clot_h_max", 0.30))
        clot_e_min = float(_cfg(cfg, "clot_e_min", 0.52))
        clot_red_dom_min = float(_cfg(cfg, "clot_red_dom_min", 0.12))
        clot_sat_min = int(_cfg(cfg, "clot_sat_min", 55))
        clot_gray_max = int(_cfg(cfg, "clot_gray_max", 205))
        clot_edge_max = float(_cfg(cfg, "clot_edge_max", 0.55))
        clot_r_max = float(_cfg(cfg, "clot_r_max", 0.82))
        clot_min_ratio = float(_cfg(cfg, "clot_min_component_area_ratio", 0.00035))
        clot_use_edge_gate = bool(_cfg(cfg, "clot_use_edge_gate", True))
        clot_use_hue_gate = bool(_cfg(cfg, "clot_use_hue_gate", True))
        clot_hue_low_max = int(_cfg(cfg, "clot_hue_low_max", 14))
        clot_hue_high_min = int(_cfg(cfg, "clot_hue_high_min", 170))
        hue = hsv[:, :, 0]
        red_hue = (hue <= clot_hue_low_max) | (hue >= clot_hue_high_min)
        clot_like = (
            blood_like_mask
            & (h_n <= clot_h_max)
            & (e_n >= clot_e_min)
            & (red_dom >= clot_red_dom_min)
            & (hsv[:, :, 1] >= clot_sat_min)
            & (gray <= clot_gray_max)
            & (~ink_mask)
            & (~black_canvas)
        )
        if stain_guided_enabled:
            clot_like &= (r_n <= clot_r_max)
        if clot_use_hue_gate:
            clot_like &= red_hue
        if clot_use_edge_gate:
            clot_like &= (edge_n <= clot_edge_max)
        clot_dilate = max(1, int(_cfg(cfg, "clot_dilate_kernel", 5)))
        if clot_dilate > 1:
            clot_dilate = max(3, clot_dilate | 1)
            ker_clot = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clot_dilate, clot_dilate))
            clot_like = cv2.dilate(clot_like.astype(np.uint8), ker_clot, iterations=1) > 0
        clot_like = _filter_components_by_area(clot_like, min_area_ratio=clot_min_ratio)

    use_blood_as_background = bool(_cfg(cfg, "use_blood_as_background", True))
    blood_background = rbc_like | clot_like
    if use_blood_as_background:
        drop_blood_like = blood_background
    elif exclude_clot_from_tissue:
        drop_blood_like = clot_like
    else:
        drop_blood_like = np.zeros_like(rbc_like, dtype=bool)

    tissue = base_tissue & (~ink_mask) & (~drop_blood_like)
    tissue &= (~edge_art)
    tissue[black_canvas] = False
    clean = _cleanup_mask(tissue.astype(np.uint8), cfg=cfg)
    clean[ink_mask] = 0
    clean[edge_art] = 0
    clean[drop_blood_like] = 0
    clean[black_canvas] = 0

    # Keep large white/pale low-texture voids as background (do not include in mask).
    white_void_gray = int(_cfg(cfg, "white_void_gray_min", 228))
    white_void_sat = int(_cfg(cfg, "white_void_sat_max", 24))
    white_bg_h_max = float(_cfg(cfg, "white_bg_h_max", 0.22))
    white_bg_edge_max = float(_cfg(cfg, "white_bg_edge_max", 0.25))
    pale_void_gray = int(_cfg(cfg, "pale_void_gray_min", 210))
    pale_void_sat = int(_cfg(cfg, "pale_void_sat_max", 52))
    pale_void_edge_max = float(_cfg(cfg, "pale_void_edge_max", 0.16))
    pale_void_h_max = float(_cfg(cfg, "pale_void_h_max", 0.24))
    white_void_min_ratio = float(_cfg(cfg, "white_void_min_area_ratio", 0.00035))
    hard_white_void = (gray >= white_void_gray) & (hsv[:, :, 1] <= white_void_sat)
    pale_void = (
        (gray >= pale_void_gray)
        & (hsv[:, :, 1] <= pale_void_sat)
        & (edge_n <= pale_void_edge_max)
        & (h_n <= pale_void_h_max)
    )
    white_void = (hard_white_void | pale_void) & (~black_canvas)
    hard_white_bg = hard_white_void & (h_n <= white_bg_h_max) & (edge_n <= white_bg_edge_max)
    clean[hard_white_bg] = 0
    if np.any(white_void):
        n_w, labels_w, stats_w, _ = cv2.connectedComponentsWithStats(
            white_void.astype(np.uint8), connectivity=8
        )
        if n_w > 1:
            min_void_area = int(max(1, round(white_void_min_ratio * white_void.size)))
            min_overlap = float(_cfg(cfg, "white_void_min_tissue_overlap", 0.002))
            min_near = float(_cfg(cfg, "white_void_min_near_tissue", 0.10))
            edge_margin = int(_cfg(cfg, "white_void_edge_margin_px", 4))
            large_white_void = np.zeros_like(clean, dtype=bool)
            k_void = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            tissue_neighborhood = cv2.dilate(clean.astype(np.uint8), k_void, iterations=1) > 0

            for lb in range(1, n_w):
                area = int(stats_w[lb, cv2.CC_STAT_AREA])
                if area < min_void_area:
                    continue
                x = int(stats_w[lb, cv2.CC_STAT_LEFT])
                y = int(stats_w[lb, cv2.CC_STAT_TOP])
                w = int(stats_w[lb, cv2.CC_STAT_WIDTH])
                h_cc = int(stats_w[lb, cv2.CC_STAT_HEIGHT])
                touches_edge = (
                    (x <= edge_margin)
                    or (y <= edge_margin)
                    or ((x + w) >= (work.shape[1] - edge_margin))
                    or ((y + h_cc) >= (work.shape[0] - edge_margin))
                )
                comp = labels_w == lb
                overlap_frac = float(clean[comp].mean())
                near_frac = float(tissue_neighborhood[comp].mean())
                # Remove whole component once identified as a cavity-like white void.
                if (overlap_frac >= min_overlap) or (near_frac >= min_near and not touches_edge):
                    large_white_void |= comp
            clean[large_white_void] = 0
        else:
            large_white_void = np.zeros_like(clean, dtype=bool)
    else:
        large_white_void = np.zeros_like(clean, dtype=bool)

    # Rescue hematoxylin-rich (purple) tissue that can be lost after aggressive void/background filtering.
    h_rescue_min = float(_cfg(cfg, "h_rescue_min", 0.34))
    h_rescue_sat_min = int(_cfg(cfg, "h_rescue_sat_min", 14))
    h_rescue_edge_min = float(_cfg(cfg, "h_rescue_edge_min", 0.03))
    h_rescue_od_min = float(_cfg(cfg, "h_rescue_od_min", 0.08))
    h_rescue_min_ratio = float(_cfg(cfg, "h_rescue_min_component_area_ratio", 0.00003))
    h_rescue = (
        (h_n >= h_rescue_min)
        & (hsv[:, :, 1] >= h_rescue_sat_min)
        & (edge_n >= h_rescue_edge_min)
        & (od_n >= h_rescue_od_min)
        & valid_stats
        & (~ink_mask)
        & (~edge_art)
        & (~black_canvas)
        & (~hard_white_bg)
    )
    h_rescue &= (~drop_blood_like)
    if np.any(h_rescue):
        n_h, labels_h, stats_h, _ = cv2.connectedComponentsWithStats(h_rescue.astype(np.uint8), connectivity=8)
        if n_h > 1:
            min_h_area = int(max(1, round(h_rescue_min_ratio * h_rescue.size)))
            keep_h = np.zeros(n_h, dtype=bool)
            keep_h[1:] = stats_h[1:, cv2.CC_STAT_AREA] >= min_h_area
            h_rescue = keep_h[labels_h]
        clean[h_rescue] = 1
        clean[ink_mask] = 0
        clean[edge_art] = 0
        clean[drop_blood_like] = 0
        clean[hard_white_bg] = 0

    # Secondary rescue for low-edge but valid tissue blocks (e.g., pale-purple stroma).
    soft_od_min = float(_cfg(cfg, "soft_rescue_od_min", 0.06))
    soft_h_min = float(_cfg(cfg, "soft_rescue_h_min", 0.14))
    soft_sat_min = int(_cfg(cfg, "soft_rescue_sat_min", 10))
    soft_gray_max = int(_cfg(cfg, "soft_rescue_gray_max", 242))
    soft_min_ratio = float(_cfg(cfg, "soft_rescue_min_component_area_ratio", 0.00012))
    soft_white_like_max = float(_cfg(cfg, "soft_rescue_white_like_max", 0.55))
    soft_touch_min = float(_cfg(cfg, "soft_rescue_touch_min", 0.001))
    soft_touch_kernel = max(3, int(_cfg(cfg, "soft_rescue_touch_kernel", 11)) | 1)
    soft_allow_large_untouched = bool(_cfg(cfg, "soft_rescue_allow_large_without_touch", False))
    soft_large_factor = float(_cfg(cfg, "soft_rescue_large_component_factor", 4.0))
    soft_component_edge_min = float(_cfg(cfg, "soft_rescue_component_edge_min", 0.02))
    soft_rescue_enabled = bool(_cfg(cfg, "soft_rescue_enabled", True))

    white_like = (gray >= 240) & (hsv[:, :, 1] <= 20)
    soft_signal = (
        ((od_n >= soft_od_min) & (gray <= soft_gray_max))
        | (h_n >= soft_h_min)
        | (hsv[:, :, 1] >= soft_sat_min)
    )
    soft_rescue = (
        valid_stats
        & soft_signal
        & (~ink_mask)
        & (~edge_art)
        & (~hard_white_bg)
        & (~black_canvas)
    )
    soft_rescue &= (~drop_blood_like)

    if soft_rescue_enabled and np.any(soft_rescue):
        n_s, labels_s, stats_s, _ = cv2.connectedComponentsWithStats(soft_rescue.astype(np.uint8), connectivity=8)
        if n_s > 1:
            min_s_area = int(max(1, round(soft_min_ratio * soft_rescue.size)))
            keep_s = np.zeros(n_s, dtype=bool)
            k_touch = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (soft_touch_kernel, soft_touch_kernel))
            clean_touch = cv2.dilate(clean.astype(np.uint8), k_touch, iterations=1) > 0
            for lb in range(1, n_s):
                area = int(stats_s[lb, cv2.CC_STAT_AREA])
                if area < min_s_area:
                    continue
                comp = labels_s == lb
                wfrac = float(white_like[comp].mean())
                tfrac = float(clean_touch[comp].mean())
                efrac = float(edge_n[comp].mean())
                large_ok = soft_allow_large_untouched and (area >= int(round(max(1.0, soft_large_factor) * min_s_area)))
                if (wfrac <= soft_white_like_max) and (efrac >= soft_component_edge_min) and (
                    tfrac >= soft_touch_min or large_ok
                ):
                    keep_s[lb] = True
            soft_rescue = keep_s[labels_s]
        clean[soft_rescue] = 1
        clean[ink_mask] = 0
        clean[edge_art] = 0
        clean[drop_blood_like] = 0
        clean[hard_white_bg] = 0
    else:
        soft_rescue = np.zeros_like(clean, dtype=bool)

    # Optional supplement: recover limited additional near-tissue pixels with strict purity gates.
    supplement_enabled = bool(_cfg(cfg, "supplement_enabled", False))
    supplement_added = np.zeros_like(clean, dtype=bool)
    if supplement_enabled and np.any(clean):
        supplement_od_min = float(_cfg(cfg, "supplement_od_min", 0.08))
        supplement_h_min = float(_cfg(cfg, "supplement_h_min", 0.16))
        supplement_e_min = float(_cfg(cfg, "supplement_e_min", 0.10))
        supplement_sat_min = int(_cfg(cfg, "supplement_sat_min", 10))
        supplement_gray_max = int(_cfg(cfg, "supplement_gray_max", 242))
        supplement_edge_min = float(_cfg(cfg, "supplement_edge_min", 0.01))
        supplement_touch_kernel = max(3, int(_cfg(cfg, "supplement_touch_kernel", 13)) | 1)
        supplement_min_ratio = float(_cfg(cfg, "supplement_min_component_area_ratio", 0.00003))
        supplement_white_like_max = float(_cfg(cfg, "supplement_white_like_max", 0.35))
        supplement_max_added_frac = float(_cfg(cfg, "supplement_max_added_frac_of_clean", 0.15))

        near_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (supplement_touch_kernel, supplement_touch_kernel))
        near_tissue = cv2.dilate(clean.astype(np.uint8), near_k, iterations=1) > 0
        supplement_candidate = (
            (clean == 0)
            & near_tissue
            & valid_stats
            & (gray <= supplement_gray_max)
            & (hsv[:, :, 1] >= supplement_sat_min)
            & (edge_n >= supplement_edge_min)
            & (od_n >= supplement_od_min)
            & ((h_n >= supplement_h_min) | (e_n >= supplement_e_min))
            & (~ink_mask)
            & (~edge_art)
            & (~drop_blood_like)
            & (~hard_white_bg)
            & (~black_canvas)
        )
        if np.any(supplement_candidate):
            n_sp, labels_sp, stats_sp, _ = cv2.connectedComponentsWithStats(
                supplement_candidate.astype(np.uint8), connectivity=8
            )
            if n_sp > 1:
                min_sp_area = int(max(1, round(supplement_min_ratio * supplement_candidate.size)))
                white_like = (gray >= 240) & (hsv[:, :, 1] <= 20)
                keep_sp = np.zeros(n_sp, dtype=bool)
                for lb in range(1, n_sp):
                    area = int(stats_sp[lb, cv2.CC_STAT_AREA])
                    if area < min_sp_area:
                        continue
                    comp = labels_sp == lb
                    wfrac = float(white_like[comp].mean())
                    if wfrac <= supplement_white_like_max:
                        keep_sp[lb] = True
                supplement_candidate = keep_sp[labels_sp]

        if np.any(supplement_candidate):
            current_area = int(clean.astype(bool).sum())
            max_add = int(max(1, round(supplement_max_added_frac * max(1, current_area))))
            cand_idx = np.flatnonzero(supplement_candidate.reshape(-1))
            if cand_idx.size > max_add:
                score_flat = score.reshape(-1)[cand_idx]
                k = int(max_add)
                if k > 0:
                    keep_rel = np.argpartition(score_flat, -k)[-k:]
                    keep_idx = cand_idx[keep_rel]
                    add_flat = np.zeros(clean.size, dtype=bool)
                    add_flat[keep_idx] = True
                    supplement_added = add_flat.reshape(clean.shape)
            else:
                supplement_added = supplement_candidate.astype(bool)

        if np.any(supplement_added):
            clean[supplement_added] = 1
            clean[ink_mask] = 0
            clean[edge_art] = 0
            clean[drop_blood_like] = 0
            clean[hard_white_bg] = 0

    # Final guardrail: suppress edge/background false tissue (coverslip glare, border waves).
    edge_guard_px = int(_cfg(cfg, "edge_guard_px", 96))
    edge_guard_od_max = float(_cfg(cfg, "edge_guard_od_max", 0.12))
    edge_guard_h_max = float(_cfg(cfg, "edge_guard_h_max", 0.18))
    edge_guard_score_max = float(_cfg(cfg, "edge_guard_score_max", 0.22))
    edge_guard_gray_min = int(_cfg(cfg, "edge_guard_gray_min", 170))
    edge_guard_sat_max = int(_cfg(cfg, "edge_guard_sat_max", 120))
    edge_guard_loose_od_max = float(_cfg(cfg, "edge_guard_loose_od_max", 0.18))
    edge_guard_loose_h_max = float(_cfg(cfg, "edge_guard_loose_h_max", 0.20))
    edge_guard_loose_score_max = float(_cfg(cfg, "edge_guard_loose_score_max", 0.35))
    edge_false = np.zeros_like(clean, dtype=bool)
    if edge_guard_px > 0:
        m = min(edge_guard_px, max(1, min(clean.shape[0], clean.shape[1]) // 3))
        edge_zone = np.zeros_like(clean, dtype=bool)
        edge_zone[:m, :] = True
        edge_zone[-m:, :] = True
        edge_zone[:, :m] = True
        edge_zone[:, -m:] = True
        edge_false_strict = (
            edge_zone
            & (od_n <= edge_guard_od_max)
            & (h_n <= edge_guard_h_max)
            & (score <= edge_guard_score_max)
            & (gray >= edge_guard_gray_min)
            & (hsv[:, :, 1] <= edge_guard_sat_max)
        )
        # Loose branch catches high-sat edge interference (e.g., coverslip rainbow/green waves).
        edge_false_loose = (
            edge_zone
            & (od_n <= edge_guard_loose_od_max)
            & (h_n <= edge_guard_loose_h_max)
            & (score <= edge_guard_loose_score_max)
        )
        edge_false = edge_false_strict | edge_false_loose
        clean[edge_false] = 0

    # Drop border-touching components that look like background artifacts.
    border_drop = np.zeros_like(clean, dtype=bool)
    border_margin = int(_cfg(cfg, "border_component_margin_px", 8))
    border_h_max = float(_cfg(cfg, "border_component_h_max", 0.20))
    border_od_max = float(_cfg(cfg, "border_component_od_max", 0.22))
    border_white_min = float(_cfg(cfg, "border_component_white_min", 0.25))
    border_small_ratio = float(_cfg(cfg, "border_component_small_area_ratio", 0.0012))
    white_like_for_border = (gray >= 232) & (hsv[:, :, 1] <= 35)
    if border_margin > 0 and np.any(clean):
        n_c, labels_c, stats_c, _ = cv2.connectedComponentsWithStats(clean.astype(np.uint8), connectivity=8)
        if n_c > 1:
            h_img, w_img = clean.shape
            small_area = int(max(1, round(border_small_ratio * clean.size)))
            drop = np.zeros(n_c, dtype=bool)
            for lb in range(1, n_c):
                x = int(stats_c[lb, cv2.CC_STAT_LEFT])
                y = int(stats_c[lb, cv2.CC_STAT_TOP])
                w = int(stats_c[lb, cv2.CC_STAT_WIDTH])
                h = int(stats_c[lb, cv2.CC_STAT_HEIGHT])
                area = int(stats_c[lb, cv2.CC_STAT_AREA])
                touches = (
                    (x <= border_margin)
                    or (y <= border_margin)
                    or ((x + w) >= (w_img - border_margin))
                    or ((y + h) >= (h_img - border_margin))
                )
                if not touches:
                    continue
                comp = labels_c == lb
                mean_h = float(h_n[comp].mean())
                mean_od = float(od_n[comp].mean())
                white_frac = float(white_like_for_border[comp].mean())
                if ((mean_h <= border_h_max) and (mean_od <= border_od_max) and (white_frac >= border_white_min)) or (
                    area <= small_area and mean_h <= (border_h_max + 0.05) and mean_od <= (border_od_max + 0.05)
                ):
                    drop[lb] = True
            border_drop = drop[labels_c]
            clean[border_drop] = 0

    # Hard crop on extreme borders to suppress persistent scanner/coverslip edge artifacts.
    edge_crop_removed = np.zeros_like(clean, dtype=bool)
    final_edge_crop = int(_cfg(cfg, "final_edge_crop_px", 16))
    if final_edge_crop > 0:
        m = min(final_edge_crop, max(1, min(clean.shape[0], clean.shape[1]) // 8))
        before = clean.astype(bool).copy()
        clean[:m, :] = 0
        clean[-m:, :] = 0
        clean[:, :m] = 0
        clean[:, -m:] = 0
        edge_crop_removed = before & (~clean.astype(bool))

    stats = {
        "mode": "fallback" if permissive else "primary",
        "mask_frac": float(clean.mean()),
        "od_thr": float(od_thr),
        "od_thr_otsu": float(np.nan),
        "od_low_guard": float(np.nan),
        "od_high_guard": float(np.nan),
        "sat_thr": float(np.nan),
        "rapid_thr": float(rapid_thr),
        "score_thr": float(score_thr),
        "ink_frac": float(ink_mask.mean()),
        "edge_art_frac": float(edge_art.mean()),
        "rbc_like_frac": float(rbc_like.mean()),
        "clot_like_frac": float(clot_like.mean()),
        "drop_blood_like_frac": float(drop_blood_like.mean()),
        "large_white_void_frac": float(large_white_void.mean()),
        "pale_void_frac": float(pale_void.mean()),
        "hard_white_bg_frac": float(hard_white_bg.mean()),
        "h_rescue_frac": float(h_rescue.mean()),
        "soft_rescue_frac": float(soft_rescue.mean()),
        "supplement_added_frac": float(supplement_added.mean()),
        "edge_false_frac": float(edge_false.mean()),
        "border_drop_frac": float(border_drop.mean()),
        "edge_crop_removed_frac": float(edge_crop_removed.mean()),
        "black_canvas_frac": float(black_canvas.mean()),
        "bg_white_frac": float(bg_white.mean()),
        "valid_stats_frac": float(valid_stats.mean()),
        "blood_like_frac": float(blood_like_mask.mean()),
    }
    debug = {
        "rapid": rapid,
        "od_sum": od_sum,
        "score": score,
        "ink_mask": ink_mask.astype(np.uint8),
        "base_tissue": base_tissue.astype(np.uint8),
    }
    return MaskBuildResult(mask=clean.astype(np.uint8), stats=stats, debug_layers=debug)


def build_tissue_mask(img_rgb: np.ndarray, cfg: dict | None = None) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    out = _build_mask(img_rgb=img_rgb, cfg=cfg, permissive=False)
    return out.mask, out.stats, out.debug_layers


def fallback_tissue_mask(img_rgb: np.ndarray, cfg: dict | None = None) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    out = _build_mask(img_rgb=img_rgb, cfg=cfg, permissive=True)
    return out.mask, out.stats, out.debug_layers


def _status_from_band(v: float, lo: float, hi: float) -> str:
    if v < lo:
        return "low_tissue"
    if v > hi:
        return "high_tissue"
    return "ok"


def _effective_mask_fraction(img_rgb: np.ndarray, mask: np.ndarray, cfg: dict | None = None) -> tuple[float, float]:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    black_gray = int(_cfg(cfg, "black_canvas_gray_max", 8))
    black_rgb = int(_cfg(cfg, "black_canvas_rgb_max", 12))
    black_canvas = (gray <= black_gray) | (
        (img_rgb[:, :, 0] <= black_rgb) & (img_rgb[:, :, 1] <= black_rgb) & (img_rgb[:, :, 2] <= black_rgb)
    )
    non_black_pixels = int((~black_canvas).sum())
    effective = float(mask.astype(bool).sum()) / float(max(1, non_black_pixels))
    non_black_frac = float((~black_canvas).mean())
    return effective, non_black_frac


def compute_mask_status_fields(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    cfg: dict | None = None,
) -> dict[str, Any]:
    expected_min = float(_cfg(cfg, "expected_frac_min", 0.15))
    expected_max = float(_cfg(cfg, "expected_frac_max", 0.60))
    expected_min_eff = float(_cfg(cfg, "expected_frac_min_effective", 0.08))
    expected_max_eff = float(_cfg(cfg, "expected_frac_max_effective", 0.80))
    sparse_canvas_mode = bool(_cfg(cfg, "sparse_canvas_mode", False))

    mask_frac = float(mask.mean())
    mask_frac_effective, non_black_frac = _effective_mask_fraction(img_rgb=img_rgb, mask=mask, cfg=cfg)
    global_status = _status_from_band(mask_frac, expected_min, expected_max)
    if sparse_canvas_mode:
        effective_status = _status_from_band(mask_frac_effective, expected_min_eff, expected_max_eff)
        status_basis = "effective"
    else:
        effective_status = global_status
        status_basis = "global"

    return {
        "mask_frac": mask_frac,
        "mask_frac_effective": float(mask_frac_effective),
        "non_black_frac": float(non_black_frac),
        "mask_status": global_status,
        "mask_status_effective": effective_status,
        "status_basis": status_basis,
        "expected_frac_min": expected_min,
        "expected_frac_max": expected_max,
        "expected_frac_min_effective": expected_min_eff,
        "expected_frac_max_effective": expected_max_eff,
    }


def select_mask_with_fallback(
    img_rgb: np.ndarray,
    cfg: dict | None = None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    expected_min = float(_cfg(cfg, "expected_frac_min", 0.15))
    expected_max = float(_cfg(cfg, "expected_frac_max", 0.60))
    expected_min_eff = float(_cfg(cfg, "expected_frac_min_effective", 0.08))
    expected_max_eff = float(_cfg(cfg, "expected_frac_max_effective", 0.80))
    sparse_canvas_mode = bool(_cfg(cfg, "sparse_canvas_mode", False))

    m1, s1, d1 = build_tissue_mask(img_rgb, cfg=cfg)
    s1 = dict(s1)
    s1.update(compute_mask_status_fields(img_rgb=img_rgb, mask=m1, cfg=cfg))

    if sparse_canvas_mode:
        frac1 = float(s1["mask_frac_effective"])
        low_trigger = float(_cfg(cfg, "low_frac_trigger_effective", expected_min_eff))
        high_trigger = float(_cfg(cfg, "high_frac_trigger_effective", expected_max_eff))
        target_min = expected_min_eff
        target_max = expected_max_eff
    else:
        frac1 = float(s1["mask_frac"])
        low_trigger = float(_cfg(cfg, "low_frac_trigger", expected_min))
        high_trigger = float(_cfg(cfg, "high_frac_trigger", expected_max))
        target_min = expected_min
        target_max = expected_max

    need_fallback = bool(frac1 < low_trigger or frac1 > high_trigger)

    fallback_used = False
    chosen = (m1, s1, d1)
    if need_fallback:
        m2, s2, d2 = fallback_tissue_mask(img_rgb, cfg=cfg)
        s2 = dict(s2)
        s2.update(compute_mask_status_fields(img_rgb=img_rgb, mask=m2, cfg=cfg))
        frac2 = float(s2["mask_frac_effective"] if sparse_canvas_mode else s2["mask_frac"])

        def dist_to_band(v: float) -> float:
            if v < target_min:
                return target_min - v
            if v > target_max:
                return v - target_max
            return 0.0

        if dist_to_band(frac2) < dist_to_band(frac1):
            chosen = (m2, s2, d2)
            fallback_used = True

    mask, stats, debug = chosen

    merged = dict(stats)
    merged.update(
        {
            "fallback_used": bool(fallback_used),
            "fallback_attempted": bool(need_fallback),
        }
    )
    return mask, merged, debug

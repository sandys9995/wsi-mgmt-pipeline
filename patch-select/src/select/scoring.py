from __future__ import annotations

import cv2
import numpy as np
import pandas as pd


def _od(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float32) / 255.0
    return -np.log(np.clip(x, 1e-6, 1.0))


def _he_stain_matrix() -> np.ndarray:
    # Common fixed H&E stain vectors (Ruifrok-like), normalized.
    h = np.array([0.650, 0.704, 0.286], dtype=np.float32)
    e = np.array([0.072, 0.990, 0.105], dtype=np.float32)
    h = h / np.linalg.norm(h)
    e = e / np.linalg.norm(e)
    return np.stack([h, e], axis=1)  # (3,2)


def _artifact_fraction_tile(tile: np.ndarray) -> float:
    hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    ink_dark = (v < 90) & (s > 30)
    ink_green = (h >= 35) & (h <= 105) & (s > 45) & (v < 245)
    ink_blue = (h >= 100) & (h <= 140) & (s > 45) & (v < 245)
    ink_magenta = ((h >= 145) | (h <= 10)) & (s > 140) & (v < 245)
    pen = ink_dark | ink_green | ink_blue | ink_magenta

    r = tile[:, :, 0].astype(np.float32)
    g = tile[:, :, 1].astype(np.float32)
    b = tile[:, :, 2].astype(np.float32)
    blood_pool = (r > 140) & ((r - g) > 35) & ((r - b) > 35)
    dark_line = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY) < 70
    return float((pen | blood_pool | dark_line).mean())


def _norm01(x: np.ndarray, q_low: float = 5, q_high: float = 99) -> np.ndarray:
    lo = float(np.percentile(x, q_low))
    hi = float(np.percentile(x, q_high))
    if hi <= lo + 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _rbc_fraction_tile(tile: np.ndarray, h_channel: np.ndarray, e_channel: np.ndarray, cfg: dict) -> float:
    hn = _norm01(h_channel)
    en = _norm01(e_channel)
    r = tile[:, :, 0].astype(np.float32)
    g = tile[:, :, 1].astype(np.float32)
    b = tile[:, :, 2].astype(np.float32)
    red_dom = (r - 0.5 * (g + b)) / 255.0

    e_min = float(cfg.get("qc", {}).get("rbc_e_min", 0.58))
    h_max = float(cfg.get("qc", {}).get("rbc_h_max", 0.42))
    red_dom_min = float(cfg.get("qc", {}).get("rbc_red_dom_min", 0.06))
    rbc = (en > e_min) & (hn < h_max) & (red_dom > red_dom_min)
    return float(rbc.mean())


def _blood_score_tile(tile: np.ndarray, h_channel: np.ndarray) -> float:
    r = tile[:, :, 0].astype(np.float32) / 255.0
    g = tile[:, :, 1].astype(np.float32) / 255.0
    b = tile[:, :, 2].astype(np.float32) / 255.0

    red_dom = np.maximum(0.0, r - 0.5 * (g + b))
    red_pixels = (r > 0.55) & ((r - g) > 0.12) & ((r - b) > 0.12)
    # Blood tends to have weaker nuclear stain than truly cell-rich purple regions.
    low_h = h_channel < np.percentile(h_channel, 35)

    score = (
        float(red_dom.mean())
        + 0.7 * float(red_pixels.mean())
        + 0.3 * float((red_pixels & low_h).mean())
    )
    return score


def _pct(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(pct=True).to_numpy(dtype=np.float32)


def compute_scores_and_types(tiles, tf, fs, cfg):
    """
    H&E-aware scoring with purity penalties.
    Returns dataframe with score features + type A/B/C/D.
    """
    n = len(tiles)
    h_mean = np.zeros(n, np.float32)
    h_p90 = np.zeros(n, np.float32)
    e_mean = np.zeros(n, np.float32)
    blood = np.zeros(n, np.float32)
    texture = np.zeros(n, np.float32)
    white_frac = np.zeros(n, np.float32)
    artifact_frac = np.zeros(n, np.float32)
    rbc_frac = np.zeros(n, np.float32)

    he = _he_stain_matrix()
    he_pinv = np.linalg.pinv(he)  # (2,3)

    for i, t in enumerate(tiles):
        od = _od(t)
        flat = od.reshape(-1, 3).T  # (3,P)
        c = he_pinv @ flat  # (2,P)
        c = np.clip(c, 0.0, None)
        h = c[0, :]
        e = c[1, :]
        h_img = h.reshape(t.shape[:2])
        e_img = e.reshape(t.shape[:2])

        h_mean[i] = float(h.mean())
        h_p90[i] = float(np.percentile(h, 90))
        e_mean[i] = float(e.mean())

        g = cv2.cvtColor(t, cv2.COLOR_RGB2GRAY)
        texture[i] = float(cv2.Laplacian(g, cv2.CV_32F).var())
        white_frac[i] = float((g > 235).mean())
        artifact_frac[i] = _artifact_fraction_tile(t)
        rbc_frac[i] = _rbc_fraction_tile(t, h_channel=h_img, e_channel=e_img, cfg=cfg)
        blood[i] = _blood_score_tile(t, h_channel=h_img)

    h_mean_p = _pct(h_mean)
    h_p90_p = _pct(h_p90)
    e_p = _pct(e_mean)
    tex_p = _pct(texture)
    white_p = _pct(white_frac)
    artifact_p = _pct(artifact_frac)
    rbc_p = _pct(rbc_frac)
    blood_p = _pct(blood)

    w = cfg.get("scoring", {}).get("cell_rich_weights", {})
    w_h_mean = float(w.get("h_mean", 0.30))
    w_h_p90 = float(w.get("h_p90", 0.30))
    w_tex = float(w.get("texture", 0.30))
    w_e = float(w.get("e_mean", 0.10))
    w_white = float(w.get("white_penalty", 0.40))
    w_art = float(w.get("artifact_penalty", 0.60))
    w_blood = float(w.get("blood_penalty", 0.30))
    w_rbc = float(w.get("rbc_penalty", 0.60))

    cell_rich = (
        w_h_mean * h_mean_p
        + w_h_p90 * h_p90_p
        + w_tex * tex_p
        + w_e * e_p
        - w_white * white_p
        - w_art * artifact_p
        - w_blood * blood_p
        - w_rbc * rbc_p
    ).astype(np.float32)
    cell_rich_p = _pct(cell_rich)

    types = np.array(["D"] * n, dtype=object)
    # A: cell-rich and not artifact-heavy
    types[(cell_rich_p >= 0.70) & (artifact_p <= 0.70)] = "A"
    # C: blood dominant, not strongly nuclear-rich
    types[((blood_p >= 0.85) | (rbc_p >= 0.85)) & (h_p90_p < 0.70)] = "C"
    # B: medium quality tissue with structure and moderate H signal
    types[(types == "D") & (cell_rich_p >= 0.45) & (artifact_p <= 0.80)] = "B"

    df = pd.DataFrame(
        {
            "tissue_frac": tf,
            "focus": fs,
            "h_mean": h_mean,
            "h_p90": h_p90,
            "e_mean": e_mean,
            "blood_score": blood,
            "texture": texture,
            "white_frac": white_frac,
            "artifact_frac": artifact_frac,
            "rbc_frac": rbc_frac,
            "h_mean_p": h_mean_p,
            "h_p90_p": h_p90_p,
            "e_p": e_p,
            "blood_p": blood_p,
            "tex_p": tex_p,
            "white_p": white_p,
            "artifact_p": artifact_p,
            "rbc_p": rbc_p,
            "cell_rich_score": cell_rich,
            "cell_rich_p": cell_rich_p,
            "type": types,
        }
    )
    return df

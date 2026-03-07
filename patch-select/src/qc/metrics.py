import numpy as np
import cv2


def _od(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float32) / 255.0
    return -np.log(np.clip(x, 1e-6, 1.0))


def _he_stain_matrix() -> np.ndarray:
    h = np.array([0.650, 0.704, 0.286], dtype=np.float32)
    e = np.array([0.072, 0.990, 0.105], dtype=np.float32)
    h = h / np.linalg.norm(h)
    e = e / np.linalg.norm(e)
    return np.stack([h, e], axis=1)


def _he_channels(tile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    od = _od(tile)
    he = _he_stain_matrix()
    pinv = np.linalg.pinv(he)
    c = pinv @ od.reshape(-1, 3).T
    c = np.clip(c, 0.0, None)
    h = c[0, :].reshape(tile.shape[:2]).astype(np.float32)
    e = c[1, :].reshape(tile.shape[:2]).astype(np.float32)
    return h, e


def _norm01(x: np.ndarray, q_low: float = 5, q_high: float = 99) -> np.ndarray:
    lo = float(np.percentile(x, q_low))
    hi = float(np.percentile(x, q_high))
    if hi <= lo + 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _rbc_mask(tile: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    h, e = _he_channels(tile)
    hn = _norm01(h, 5, 99)
    en = _norm01(e, 5, 99)
    r = tile[:, :, 0].astype(np.float32)
    g = tile[:, :, 1].astype(np.float32)
    b = tile[:, :, 2].astype(np.float32)
    red_dom = (r - 0.5 * (g + b)) / 255.0

    e_min = float((cfg or {}).get("rbc_e_min", 0.58))
    h_max = float((cfg or {}).get("rbc_h_max", 0.42))
    red_dom_min = float((cfg or {}).get("rbc_red_dom_min", 0.06))
    return (en > e_min) & (hn < h_max) & (red_dom > red_dom_min)


def _artifact_mask(tile: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Marker / pen families.
    ink_black = (v < 70) & (s > 18)
    ink_green = (h >= 25) & (h <= 110) & (s > 22) & (v < 250)
    ink_blue = (h >= 85) & (h <= 140) & (s > 22) & (v < 250)
    ink_magenta = ((h >= 145) | (h <= 10)) & (s > 150) & (v < 245)
    pen = ink_black | ink_green | ink_blue | ink_magenta

    # Dark stroke/shadow artifacts.
    g = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
    dark_line = g < 70
    return pen | dark_line


def tissue_fraction(tiles: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    """
    tiles: (N,H,W,3) uint8
    Tissue estimate from OD + chroma with optional RBC exclusion.
    """
    od_sum = _od(tiles).sum(axis=3)
    cmax = tiles.max(axis=3).astype(np.float32)
    cmin = tiles.min(axis=3).astype(np.float32)
    chroma = cmax - cmin
    gray = tiles.mean(axis=3)

    min_od = float((cfg or {}).get("tissue_od_min", 0.12))
    min_chroma = float((cfg or {}).get("tissue_chroma_min", 12))
    gray_max = float((cfg or {}).get("tissue_gray_max", 245))
    exclude_rbc = bool((cfg or {}).get("exclude_rbc_from_tissue", True))

    tissue = (od_sum > min_od) & ((chroma >= min_chroma) | (gray <= gray_max))
    if exclude_rbc:
        out = []
        for i in range(len(tiles)):
            rbc = _rbc_mask(tiles[i], cfg=cfg)
            out.append(float((tissue[i] & (~rbc)).mean()))
        return np.array(out, dtype=np.float32)
    return tissue.mean(axis=(1, 2))


def brightness_mean(tiles: np.ndarray) -> np.ndarray:
    gray = tiles.mean(axis=3)
    return gray.mean(axis=(1, 2))


def focus_score(tiles: np.ndarray) -> np.ndarray:
    scores = []
    for t in tiles:
        g = cv2.cvtColor(t, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = (gx * gx + gy * gy) ** 0.5
        scores.append(float(mag.mean()))
    return np.array(scores, dtype=np.float32)


def adipose_score(tiles: np.ndarray) -> np.ndarray:
    scores = []
    for t in tiles:
        g = cv2.cvtColor(t, cv2.COLOR_RGB2GRAY)
        white = (g > 240).mean()
        lap = cv2.Laplacian(g, cv2.CV_32F).var()
        scores.append(float(white - 0.001 * lap))
    return np.array(scores, dtype=np.float32)


def artifact_fraction(tiles: np.ndarray) -> np.ndarray:
    vals = []
    for t in tiles:
        vals.append(float(_artifact_mask(t).mean()))
    return np.array(vals, dtype=np.float32)


def rbc_fraction(tiles: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    vals = []
    for t in tiles:
        vals.append(float(_rbc_mask(t, cfg=cfg).mean()))
    return np.array(vals, dtype=np.float32)

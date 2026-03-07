import numpy as np


def black_fraction(rgb: np.ndarray, thr: int = 15) -> float:
    b = (rgb[..., 0] <= thr) & (rgb[..., 1] <= thr) & (rgb[..., 2] <= thr)
    return float(b.mean())


def white_fraction(rgb: np.ndarray, thr: int = 230) -> float:
    w = (rgb[..., 0] >= thr) & (rgb[..., 1] >= thr) & (rgb[..., 2] >= thr)
    return float(w.mean())


def tissue_fraction(rgb: np.ndarray, white_thr: int = 230) -> float:
    # tissue = not-white
    nonwhite = np.any(rgb < white_thr, axis=2)
    return float(nonwhite.mean())


def is_bad_patch(
    rgb: np.ndarray,
    *,
    min_tissue_frac: float = 0.20,
    white_thr: int = 230,
    max_white_frac: float = 0.95,
    black_thr: int = 15,
    max_black_frac: float = 0.95,
    min_mean: float = 10.0,
    max_mean: float = 245.0,
) -> bool:
    """
    General-purpose rejector for patch extraction.

    - rejects extreme mean (too dark / too bright)
    - rejects mostly-white background
    - rejects mostly-black tiles
    - rejects low tissue coverage
    """
    m = float(rgb.mean())
    if m < min_mean or m > max_mean:
        return True

    wf = white_fraction(rgb, thr=white_thr)
    if wf > max_white_frac:
        return True

    bf = black_fraction(rgb, thr=black_thr)
    if bf > max_black_frac:
        return True

    tf = tissue_fraction(rgb, white_thr=white_thr)
    if tf < min_tissue_frac:
        return True

    return False
# src/select/viz.py
from __future__ import annotations

import numpy as np
from PIL import Image


def save_montage(tiles: np.ndarray, out_path, n: int = 200, seed: int = 1337, tile_pad: int = 2) -> None:
    """
    tiles: (N,H,W,3)
    Saves a simple grid montage of randomly sampled tiles.
    """
    out_path = str(out_path)
    N = len(tiles)
    if N == 0:
        return

    rng = np.random.default_rng(seed)
    k = min(n, N)
    idx = rng.choice(N, size=k, replace=False)
    sample = tiles[idx]

    H, W = sample.shape[1], sample.shape[2]
    cols = int(np.ceil(np.sqrt(k)))
    rows = int(np.ceil(k / cols))

    canvas = Image.new("RGB", (cols * (W + tile_pad) + tile_pad, rows * (H + tile_pad) + tile_pad), (255, 255, 255))

    for i in range(k):
        r = i // cols
        c = i % cols
        x = tile_pad + c * (W + tile_pad)
        y = tile_pad + r * (H + tile_pad)
        canvas.paste(Image.fromarray(sample[i]), (x, y))

    canvas.save(out_path)
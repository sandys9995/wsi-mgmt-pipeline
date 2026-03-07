"""
Extract stain-reference tiles from your "perfect" large cut-out patches.

Assumptions
- You have 7–8 "perfect" cut-outs as PNG/JPG/TIF in a folder.
- Each cut-out is already at the correct effective magnification (20× equivalent).
- We will sample 3–5 tiles per cut-out (default=4) of size 224×224 (default).

What it does
1) For each cut-out image: randomly sample candidate tiles
2) Keep only tiles with enough tissue (reject mostly white/background)
3) Save tiles to an output folder
4) Also save a "target_reference.png" = the single best tile (highest tissue fraction),
   which you can use as the fixed reference image for stain normalization.

Run
python extract_ref_tiles.py \
  --in_dir perfect_h_e_colour_stain \
  --out_dir ref_tiles \
  --tile 224 \
  --tiles_per_img 9 \
  --candidates_per_tile 30
"""

import os
import glob
import random
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def is_tissue(tile_rgb: np.ndarray,
              white_thresh: int = 220,
              tissue_frac_min: float = 0.60) -> tuple[bool, float]:
    """
    Simple tissue filter: count pixels that are NOT near-white in all channels.

    tile_rgb: H×W×3 uint8
    white_thresh: pixels with R,G,B > white_thresh are considered background
    tissue_frac_min: minimum fraction of non-white pixels to accept
    """
    # background if all channels very bright
    bg = (tile_rgb[..., 0] > white_thresh) & (tile_rgb[..., 1] > white_thresh) & (tile_rgb[..., 2] > white_thresh)
    tissue_frac = 1.0 - bg.mean()
    return (tissue_frac >= tissue_frac_min), float(tissue_frac)


def sample_tiles_from_image(img: Image.Image,
                            tile: int,
                            tiles_per_img: int,
                            candidates_per_tile: int,
                            seed: int,
                            tissue_frac_min: float,
                            white_thresh: int) -> list[tuple[Image.Image, float, tuple[int, int]]]:
    """
    Randomly sample tiles that pass tissue filtering.
    Returns list of (tile_image, tissue_frac, (x,y)) in pixel coordinates of the cut-out.
    """
    rng = random.Random(seed)
    w, h = img.size

    if w < tile or h < tile:
        return []

    accepted = []
    used = 0

    # We try to get tiles_per_img accepted tiles.
    # For each tile, we attempt up to candidates_per_tile random positions.
    for _ in range(tiles_per_img):
        best = None  # keep best candidate even if none pass, then skip
        for _cand in range(candidates_per_tile):
            x = rng.randint(0, w - tile)
            y = rng.randint(0, h - tile)
            crop = img.crop((x, y, x + tile, y + tile))
            arr = np.asarray(crop, dtype=np.uint8)

            ok, frac = is_tissue(arr, white_thresh=white_thresh, tissue_frac_min=tissue_frac_min)

            # track best candidate by tissue fraction
            if best is None or frac > best[1]:
                best = (crop, frac, (x, y))

            if ok:
                accepted.append((crop, frac, (x, y)))
                used += 1
                break  # move to next tile

        # if none passed, we simply skip (better to have fewer but clean reference tiles)
        # you can relax tissue_frac_min if you want more tiles.

    return accepted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder of perfect cut-outs (png/jpg/tif)")
    ap.add_argument("--out_dir", required=True, help="Output folder for reference tiles")
    ap.add_argument("--tile", type=int, default=224, help="Tile size in pixels (e.g., 224 or 256)")
    ap.add_argument("--tiles_per_img", type=int, default=4, help="How many accepted tiles to save per cut-out")
    ap.add_argument("--candidates_per_tile", type=int, default=30, help="Random attempts per tile")
    ap.add_argument("--seed", type=int, default=7, help="Random seed")
    ap.add_argument("--tissue_frac_min", type=float, default=0.60, help="Min tissue fraction (non-white) to accept")
    ap.add_argument("--white_thresh", type=int, default=220, help="White threshold for background test")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(str(in_dir / e)))
    files = sorted(files)

    if not files:
        raise RuntimeError(f"No images found in {in_dir}")

    all_saved = []
    best_overall = None  # (tile_img, tissue_frac, src_name, (x,y))

    # deterministic but different per file
    for i, fp in enumerate(files):
        src_name = Path(fp).stem
        img = Image.open(fp).convert("RGB")

        tiles = sample_tiles_from_image(
            img=img,
            tile=args.tile,
            tiles_per_img=args.tiles_per_img,
            candidates_per_tile=args.candidates_per_tile,
            seed=args.seed + i * 1000,
            tissue_frac_min=args.tissue_frac_min,
            white_thresh=args.white_thresh
        )

        # save tiles
        for j, (timg, frac, (x, y)) in enumerate(tiles):
            out_name = f"{src_name}__t{j:02d}__x{x}_y{y}__tissue{frac:.3f}.png"
            out_path = out_dir / out_name
            timg.save(out_path)
            all_saved.append((str(out_path), src_name, x, y, frac))

            if best_overall is None or frac > best_overall[1]:
                best_overall = (timg, frac, src_name, (x, y))

        print(f"[{i+1:02d}/{len(files):02d}] {src_name}: saved {len(tiles)}/{args.tiles_per_img} tiles")

    # Save one fixed reference image (single best tile)
    if best_overall is not None:
        ref_img, frac, src_name, (x, y) = best_overall
        ref_path = out_dir / "target_reference.png"
        ref_img.save(ref_path)
        print(f"\nSaved fixed target reference tile: {ref_path} (from {src_name} at x={x}, y={y}, tissue={frac:.3f})")

    # Save a small manifest CSV for traceability
    import csv
    manifest_path = out_dir / "ref_tiles_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile_path", "source_cutout", "x", "y", "tissue_frac"])
        for row in all_saved:
            w.writerow(row)
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
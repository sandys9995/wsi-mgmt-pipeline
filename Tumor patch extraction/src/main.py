# main.py
# Run: python main.py
#
# Folder assumptions (relative to this file):
# - train_coords.csv
# - ref_image/target_reference_clean.png
# - dataset/, utils/, stain/
#
# Output:
# - results/patches_raw/{0,1}/*.png
# - results/patches_index.csv
# - results/patches_norm/{0,1}/*.png
# - results/patches_norm/logs/macenko_log.csv
# - results/patches_norm/logs/skipped_paths.txt

import os

from dataset.extract_patches import ExtractConfig, extract_patches
from stain.macenko_norm import NormConfig, normalize_patches

SLIDE_DIR = "/Volumes/Expansion/wsi-annotated"


def run():
    project_root = os.path.dirname(os.path.abspath(__file__))

    coords_csv = os.path.join(project_root, "train_coords.csv")
    results_dir = os.path.join(project_root, "results")
    ref_tile = os.path.join(project_root, "ref_image", "target_reference_clean.png")

    # ---- sanity checks ----
    if not os.path.exists(coords_csv):
        raise FileNotFoundError(f"coords_csv not found: {coords_csv}")
    if not os.path.exists(ref_tile):
        raise FileNotFoundError(f"reference tile not found: {ref_tile}")
    if not os.path.isdir(SLIDE_DIR):
        raise NotADirectoryError(f"SLIDE_DIR not found or not a directory: {SLIDE_DIR}")

    # 1) Extract (20x equivalent at target_mpp=0.50)
    extract_cfg = ExtractConfig(
        slide_dir=SLIDE_DIR,
        coords_csv=coords_csv,
        out_dir=results_dir,
        target_mpp=0.50,
        out_size=224,
        white_thr=230,
        limit=None,  # set e.g. 500 for a quick test
    )
    index_csv = extract_patches(extract_cfg)  # returns results/patches_index.csv

    # 2) Normalize (Macenko) - reads from patches_raw, writes to patches_norm
    norm_cfg = NormConfig(
        in_dir=os.path.join(results_dir, "patches_raw"),
        out_dir=os.path.join(results_dir, "patches_norm"),
        reference_tile_path=ref_tile,
        device="cpu",   # keep cpu for Macenko stability; if you pass "mps" it will auto-fallback to cpu
        limit=None
    )
    normalize_patches(norm_cfg)

    print("\n=== DONE ===")
    print("Index CSV:", index_csv)
    print("Raw patches:", os.path.join(results_dir, "patches_raw"))
    print("Normalized patches:", os.path.join(results_dir, "patches_norm"))
    print("Macenko logs:", os.path.join(results_dir, "patches_norm", "logs"))


if __name__ == "__main__":
    run()
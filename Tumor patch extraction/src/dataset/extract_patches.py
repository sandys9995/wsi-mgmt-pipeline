import os
import csv
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np
from PIL import Image
import openslide
from tqdm import tqdm

from utils.slide_io import build_slide_index, open_slide_by_id, get_mpp, get_bounds
from utils.patch_filters import is_bad_patch


@dataclass
class ExtractConfig:
    slide_dir: str
    coords_csv: str
    out_dir: str
    target_mpp: float = 0.50   # 20x equivalent
    out_size: int = 224        # final patch size
    white_thr: int = 230
    limit: Optional[int] = None
    log_every: int = 500       # print counters every N rows (in addition to tqdm)


def _safe_makedirs(p: str):
    os.makedirs(p, exist_ok=True)


def _read_region_level0(slide: openslide.OpenSlide, x: int, y: int, read_px: int) -> Image.Image:
    im = slide.read_region((int(x), int(y)), level=0, size=(int(read_px), int(read_px)))
    return im.convert("RGB")


def _normalize_header(h: str) -> str:
    return h.strip().lower()


def extract_patches(cfg: ExtractConfig) -> str:
    _safe_makedirs(cfg.out_dir)
    out_index_csv = os.path.join(cfg.out_dir, "patches_index.csv")

    slide_index = build_slide_index(cfg.slide_dir, recursive=True)
    if len(slide_index) == 0:
        raise RuntimeError(f"No slides found in {cfg.slide_dir}")

    raw_dir = os.path.join(cfg.out_dir, "patches_raw")
    for lab in ["0", "1"]:
        _safe_makedirs(os.path.join(raw_dir, lab))

    # --- counters (transparency) ---
    rows_total = 0
    rows_written = 0
    rows_skipped_missing_slide = 0
    rows_skipped_no_mpp = 0
    rows_skipped_oob = 0
    rows_skipped_bad = 0

    # read all rows once so tqdm has a total
    with open(cfg.coords_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError("CSV has no header.")
        fieldnames = reader.fieldnames
        rows = list(reader)

    if cfg.limit is not None:
        rows = rows[: cfg.limit]

    # Build a robust column map (your CSV uses x_qp/y_qp)
    cols = {_normalize_header(c): c for c in fieldnames}
    required = ["slide_id", "label"]
    for req in required:
        if req not in cols:
            raise KeyError(f"CSV header missing required column '{req}'. Found: {fieldnames}")

    # Accept either x/y OR x_qp/y_qp
    if "x" in cols and "y" in cols:
        xcol, ycol = cols["x"], cols["y"]
    elif "x_qp" in cols and "y_qp" in cols:
        xcol, ycol = cols["x_qp"], cols["y_qp"]
    else:
        raise KeyError(
            f"CSV header mismatch. Need (x,y) or (x_qp,y_qp).\nFound: {fieldnames}"
        )

    # Optional columns
    wsi_path_col = cols.get("wsi_path", None)
    mpp0_col = cols.get("mpp0", None)

    # bounds columns may be present but sometimes empty
    bounds_x_col = cols.get("bounds_x", None)
    bounds_y_col = cols.get("bounds_y", None)
    bounds_w_col = cols.get("bounds_w", None)
    bounds_h_col = cols.get("bounds_h", None)

    # write index csv
    with open(out_index_csv, "w", newline="") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=["slide_id", "x", "y", "label", "mpp", "read_px", "png_path"]
        )
        writer.writeheader()

        pbar = tqdm(rows, desc="[extract] patches", unit="row")
        for i, r in enumerate(pbar, 1):
            rows_total += 1

            slide_id = r[cols["slide_id"]].strip()
            label = int(float(r[cols["label"]]))

            # read coords from CSV
            x_qp = int(float(r[xcol]))
            y_qp = int(float(r[ycol]))

            # open slide (prefer path if given, else by id)
            try:
                slide = open_slide_by_id(slide_index, slide_id) if wsi_path_col is None else openslide.OpenSlide(r[wsi_path_col])
            except Exception:
                rows_skipped_missing_slide += 1
                continue

            # mpp from slide is the source of truth
            mpp = get_mpp(slide)
            if mpp is None:
                rows_skipped_no_mpp += 1
                slide.close()
                continue

            # bounds: prefer CSV if fully present, else from slide props
            def _get_int(colname, default=None):
                if colname is None:
                    return default
                v = str(r.get(colname, "")).strip()
                if v == "" or v.lower() == "nan":
                    return default
                try:
                    return int(float(v))
                except Exception:
                    return default

            bx = _get_int(bounds_x_col, None)
            by = _get_int(bounds_y_col, None)
            bw = _get_int(bounds_w_col, None)
            bh = _get_int(bounds_h_col, None)

            if bx is None or by is None or bw is None or bh is None:
                bx, by, bw, bh = get_bounds(slide)

            def inside_bounds(xx, yy):
                return (bx <= xx <= bx + bw - 1) and (by <= yy <= by + bh - 1)

            # Auto-detect whether x_qp/y_qp are canvas or bounds-relative
            if inside_bounds(x_qp, y_qp):
                x0, y0 = x_qp, y_qp
            elif inside_bounds(x_qp + bx, y_qp + by):
                x0, y0 = x_qp + bx, y_qp + by
            else:
                rows_skipped_oob += 1
                slide.close()
                continue

            # compute read_px at level 0 to match target mpp
            read_px = int(round(cfg.out_size * (cfg.target_mpp / mpp)))
            read_px = max(read_px, cfg.out_size)

            # clamp to slide dims
            W, H = slide.dimensions
            x0 = max(0, min(x0, W - read_px))
            y0 = max(0, min(y0, H - read_px))

            patch = _read_region_level0(slide, x0, y0, read_px)
            slide.close()

            patch_np = np.asarray(patch, dtype=np.uint8)
            if is_bad_patch(patch_np, white_thr=cfg.white_thr):
                rows_skipped_bad += 1
                continue

            if read_px != cfg.out_size:
                patch = patch.resize((cfg.out_size, cfg.out_size), resample=Image.BILINEAR)

            fname = f"{slide_id}__x{x0}__y{y0}__mpp{mpp:.3f}__r{read_px}.png"
            out_path = os.path.join(raw_dir, str(label), fname)
            patch.save(out_path, format="PNG", optimize=True)

            writer.writerow({
                "slide_id": slide_id,
                "x": x0,
                "y": y0,
                "label": label,
                "mpp": f"{mpp:.6f}",
                "read_px": read_px,
                "png_path": out_path
            })
            rows_written += 1

            # tqdm postfix = live transparency
            pbar.set_postfix({
                "written": rows_written,
                "bad": rows_skipped_bad,
                "oob": rows_skipped_oob,
                "no_mpp": rows_skipped_no_mpp,
                "missing": rows_skipped_missing_slide,
            })

            if cfg.log_every and (rows_total % cfg.log_every == 0):
                print(
                    f"[extract] rows={rows_total} written={rows_written} "
                    f"bad={rows_skipped_bad} oob={rows_skipped_oob} "
                    f"no_mpp={rows_skipped_no_mpp} missing={rows_skipped_missing_slide}"
                )

    print(f"[extract] DONE")
    print(f"[extract] total={rows_total} written={rows_written}")
    print(f"[extract] skipped: bad={rows_skipped_bad}, oob={rows_skipped_oob}, no_mpp={rows_skipped_no_mpp}, missing={rows_skipped_missing_slide}")
    print(f"[extract] index csv: {out_index_csv}")
    return out_index_csv
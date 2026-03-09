# src/select/pipeline.py
from __future__ import annotations

import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def _stem(p: str) -> str:
    return Path(p).stem


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _find_mask(mask_dir: Path, slide_path: str) -> Path:
    stem = Path(slide_path).stem
    candidates = [
        mask_dir / f"{stem}.npy",
        mask_dir / f"{stem}.png",
        mask_dir / f"{stem}.tif",
        mask_dir / f"{stem}.tiff",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No mask found for slide '{stem}' in {mask_dir}")


def _load_mask_summary(mask_dir: Path) -> dict[str, dict]:
    summary = mask_dir / "mask_summary.csv"
    if not summary.exists():
        return {}
    df = pd.read_csv(summary, dtype={"slide_id": "string"})
    if "slide_id" not in df.columns:
        return {}
    out = {}
    for _, row in df.iterrows():
        sid = str(row["slide_id"]).strip()
        out[sid] = dict(row)
    return out


def _to_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    if isinstance(x, (int, float)):
        if isinstance(x, float) and np.isnan(x):
            return False
        return bool(int(x))
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def _to_float(x, default=np.nan) -> float:
    if x is None:
        return float(default)
    if isinstance(x, (int, float)):
        if isinstance(x, float) and np.isnan(x):
            return float(default)
        return float(x)
    try:
        return float(str(x).strip())
    except Exception:
        return float(default)


def _resolve_slide_diag(slide_diag: dict) -> dict:
    slide_mask_status_global = str(slide_diag.get("mask_status", "unknown"))
    status_eff_raw = slide_diag.get("mask_status_effective", np.nan)
    status_eff_str = str(status_eff_raw).strip()
    if status_eff_str and status_eff_str.lower() not in {"nan", "none"}:
        slide_mask_status_effective = status_eff_str
    else:
        slide_mask_status_effective = slide_mask_status_global
    return {
        "mask_status": slide_mask_status_effective,
        "mask_status_global": slide_mask_status_global,
        "mask_status_effective": slide_mask_status_effective,
        "status_basis": str(slide_diag.get("status_basis", "global")),
        "fallback_used": _to_bool(slide_diag.get("fallback_used", False)),
        "mask_frac": _to_float(slide_diag.get("mask_frac", np.nan)),
        "mask_frac_effective": _to_float(slide_diag.get("mask_frac_effective", np.nan)),
        "sparse_canvas_mode": _to_bool(slide_diag.get("sparse_canvas_mode", False)),
        "profile_used": str(slide_diag.get("profile_used", "")),
    }


def _infer_mask_level(mask: np.ndarray, wsi) -> int:
    mh, mw = mask.shape[:2]
    best_lvl = 0
    best_err = float("inf")
    for lvl, (w, h) in enumerate(wsi.level_dimensions):
        err = abs(h - mh) + abs(w - mw)
        if err < best_err:
            best_err = err
            best_lvl = lvl
    return best_lvl


def _mask_xy_to_level0_xy(xy_mask: np.ndarray, wsi, mask_level: int) -> np.ndarray:
    ds = float(wsi.level_downsamples[mask_level])
    xy0 = np.round(xy_mask.astype(np.float32) * ds).astype(np.int32)
    return xy0


def _clip_level0_coords(xy0: np.ndarray, wsi, read_size: int) -> np.ndarray:
    W0, H0 = wsi.level_dimensions[0]
    x = np.clip(xy0[:, 0], 0, max(0, W0 - read_size))
    y = np.clip(xy0[:, 1], 0, max(0, H0 - read_size))
    return np.stack([x, y], axis=1).astype(np.int32)


def run_on_slides(slide_paths: list[str], cfg: dict) -> None:
    from src.io.wsi import open_wsi
    from src.preprocess.mask import load_mask
    from src.qc.metrics import (
        adipose_score,
        artifact_fraction,
        brightness_mean,
        focus_score,
        rbc_fraction,
        tissue_fraction,
    )
    from src.select.sampling import apply_spatial_cap, quota_select
    from src.select.scoring import compute_scores_and_types
    from src.select.viz import save_montage

    out_root = Path(cfg["paths"]["out_dir"])
    mask_dir = Path(cfg["paths"]["mask_dir"])
    _ensure_dir(out_root)

    coords_root = out_root / "coords"
    qc_root = out_root / "qc"
    qc_pool_root = out_root / "qc_pool"
    patches_lvl0_root = out_root / "patches_lvl0"
    _ensure_dir(coords_root)
    _ensure_dir(qc_root)
    _ensure_dir(qc_pool_root)
    _ensure_dir(patches_lvl0_root)
    run_summary_rows: list[dict] = []
    mask_diag = _load_mask_summary(mask_dir)

    seed = int(cfg["run"].get("seed", 1337))
    rng = np.random.default_rng(seed)

    out_size = int(cfg["wsi"]["out_patch_size"])
    scale_factor = int(cfg["wsi"]["scale_factor"])
    read_size = out_size * scale_factor  # 448 for 224 & factor=2

    stride0 = int(cfg["candidate_grid"].get("stride_level0", 512))
    max_candidates = int(cfg["candidate_grid"]["max_candidates"])

    # QC thresholds
    min_tissue = float(cfg["qc"]["min_tissue_frac"])
    min_focus = float(cfg["qc"]["min_focus"])
    min_brightness = float(cfg["qc"].get("min_brightness", 60.0))
    max_brightness = cfg["qc"].get("max_brightness", None)
    max_artifact_frac = float(cfg["qc"].get("max_artifact_frac", 0.30))
    max_rbc_frac = float(cfg["qc"].get("max_rbc_frac", 0.45))

    for sp in tqdm(slide_paths, desc="[qc] slides", unit="slide", dynamic_ncols=True):
        sid = _stem(sp)
        print(f"\n=== {sid} ===")
        wsi = open_wsi(sp)
        slide_diag = mask_diag.get(sid, {})
        resolved_diag = _resolve_slide_diag(slide_diag)
        slide_mask_status = str(resolved_diag["mask_status"])
        slide_mask_status_global = str(resolved_diag["mask_status_global"])
        slide_mask_status_effective = str(resolved_diag["mask_status_effective"])
        slide_status_basis = str(resolved_diag["status_basis"])
        slide_fallback_used = bool(resolved_diag["fallback_used"])
        slide_mask_frac = float(resolved_diag["mask_frac"])
        slide_mask_frac_effective = float(resolved_diag["mask_frac_effective"])
        slide_sparse_canvas_mode = bool(resolved_diag["sparse_canvas_mode"])
        slide_profile_used = str(resolved_diag["profile_used"])

        def _base_summary_row() -> dict:
            return {
                "slide_id": sid,
                "mask_status": slide_mask_status,
                "mask_status_global": slide_mask_status_global,
                "mask_status_effective": slide_mask_status_effective,
                "status_basis": slide_status_basis,
                "fallback_used": slide_fallback_used,
                "mask_frac": slide_mask_frac,
                "mask_frac_effective": slide_mask_frac_effective,
                "sparse_canvas_mode": slide_sparse_canvas_mode,
                "profile_used": slide_profile_used,
            }

        # load mask
        mask_path = _find_mask(mask_dir, sp)
        mask = load_mask(mask_path)

        mask_level = _infer_mask_level(mask, wsi)
        ds = float(wsi.level_downsamples[mask_level])

        # adaptive stride in mask pixels
        stride = max(4, int(round(stride0 / ds)))

        # denser sampling for relatively fine mask levels (helps C3L-like cases)
        if ds <= 16:
            stride = max(4, stride // 2)

        print(
            f"Mask: {mask_path.name} shape={mask.shape} inferred_level={mask_level} "
            f"ds={ds:.2f} -> stride_mask={stride} (stride_level0={stride0})"
        )

        # candidate grid in mask space
        mh, mw = mask.shape[:2]
        ys = np.arange(0, mh, stride, dtype=np.int32)
        xs = np.arange(0, mw, stride, dtype=np.int32)
        grid = np.array([(x, y) for y in ys for x in xs], dtype=np.int32)

        keep = mask[grid[:, 1], grid[:, 0]] > 0
        grid = grid[keep]

        if len(grid) == 0:
            print("No tissue candidates from mask. Skipping.")
            row = _base_summary_row()
            row.update(
                {
                    "candidates_after_mask": 0,
                    "qc_pass": 0,
                    "selected_count": 0,
                    "good_count": 0,
                    "typeA": 0,
                    "typeB": 0,
                    "typeC": 0,
                    "typeD": 0,
                    "mean_blood_score": np.nan,
                    "mean_rbc_frac": np.nan,
                    "median_cell_rich": np.nan,
                }
            )
            run_summary_rows.append(row)
            wsi.close()
            continue

        # subsample for speed
        if len(grid) > max_candidates:
            idx = rng.choice(len(grid), size=max_candidates, replace=False)
            grid = grid[idx]

        # map to level0 coords
        xy0 = _mask_xy_to_level0_xy(grid, wsi, mask_level)
        xy0 = _clip_level0_coords(xy0, wsi, read_size=read_size)

        # read half-mag patches (448@0 -> 224)
        tiles = np.zeros((len(xy0), out_size, out_size, 3), dtype=np.uint8)
        for i, (x0, y0) in enumerate(
            tqdm(xy0, desc=f"[qc] {sid} read", unit="patch", dynamic_ncols=True, leave=False)
        ):
            tiles[i] = wsi.read_half_mag_patch(
                int(x0), int(y0), out_size=out_size, scale_factor=scale_factor
            )

        # QC
        tf = tissue_fraction(tiles, cfg=cfg.get("qc", {}))
        fs = focus_score(tiles)
        bm = brightness_mean(tiles)
        af = artifact_fraction(tiles)
        rbcf = rbc_fraction(tiles, cfg=cfg.get("qc", {}))

        qc_keep = (
            (tf >= min_tissue)
            & (fs >= min_focus)
            & (bm >= min_brightness)
            & (af <= max_artifact_frac)
            & (rbcf <= max_rbc_frac)
        )
        if max_brightness is not None:
            qc_keep &= (bm <= float(max_brightness))

        if bool(cfg["qc"]["reject_adipose"]):
            ad = adipose_score(tiles)
            qc_keep &= (ad < float(cfg["qc"]["adipose_whiteness"]))
        else:
            ad = np.full(len(tiles), np.nan, dtype=np.float32)

        tiles_qc = tiles[qc_keep]
        xy0_qc = xy0[qc_keep]
        grid_qc = grid[qc_keep]
        tf_qc = tf[qc_keep]
        fs_qc = fs[qc_keep]
        bm_qc = bm[qc_keep]
        af_qc = af[qc_keep]
        rbcf_qc = rbcf[qc_keep]
        ad_qc = ad[qc_keep] if np.ndim(ad) else ad

        print(f"Candidates(after mask): {len(grid)}  QC-pass: {len(tiles_qc)}")
        if len(tiles_qc) == 0:
            print("No QC-passing candidates. Skipping.")
            row = _base_summary_row()
            row.update(
                {
                    "candidates_after_mask": int(len(grid)),
                    "qc_pass": 0,
                    "selected_count": 0,
                    "good_count": 0,
                    "typeA": 0,
                    "typeB": 0,
                    "typeC": 0,
                    "typeD": 0,
                    "mean_blood_score": np.nan,
                    "mean_rbc_frac": np.nan,
                    "median_cell_rich": np.nan,
                }
            )
            run_summary_rows.append(row)
            wsi.close()
            continue

        if bool(cfg["outputs"].get("write_qc_pool", False)):
            slide_out_pool = qc_pool_root / sid
            _ensure_dir(slide_out_pool)
            qc_meta = pd.DataFrame(
                {
                    "x_mask": grid_qc[:, 0],
                    "y_mask": grid_qc[:, 1],
                    "x0": xy0_qc[:, 0],
                    "y0": xy0_qc[:, 1],
                    "tissue_frac": tf_qc,
                    "focus": fs_qc,
                    "brightness": bm_qc,
                    "qc_artifact_frac": af_qc,
                    "qc_rbc_frac": rbcf_qc,
                    "adipose_proxy": ad_qc,
                    "mask_status": slide_mask_status,
                    "mask_status_global": slide_mask_status_global,
                    "mask_status_effective": slide_mask_status_effective,
                    "status_basis": slide_status_basis,
                    "fallback_used": slide_fallback_used,
                    "mask_frac": slide_mask_frac,
                    "mask_frac_effective": slide_mask_frac_effective,
                    "sparse_canvas_mode": slide_sparse_canvas_mode,
                    "profile_used": slide_profile_used,
                }
            )
            qc_meta.to_csv(slide_out_pool / "qc_meta.csv", index=False)
            np.save(slide_out_pool / "qc_coords_level0.npy", xy0_qc.astype(np.int32))
            np.save(slide_out_pool / "qc_coords_mask.npy", grid_qc.astype(np.int32))
            if bool(cfg["outputs"].get("write_qc_tiles_npy", False)):
                np.save(slide_out_pool / "qc_tiles_uint8.npy", tiles_qc.astype(np.uint8))

        # scoring + types
        if bool(cfg.get("scoring", {}).get("extract_qc_pool_only", False)):
            print("QC-pool-only mode: skipping classical scoring/quota selection.")
            row = _base_summary_row()
            row.update(
                {
                    "candidates_after_mask": int(len(grid)),
                    "qc_pass": int(len(tiles_qc)),
                    "selected_count": int(len(tiles_qc)),
                    "good_count": np.nan,
                    "typeA": np.nan,
                    "typeB": np.nan,
                    "typeC": np.nan,
                    "typeD": np.nan,
                    "mean_blood_score": np.nan,
                    "mean_rbc_frac": float(np.mean(rbcf_qc)) if len(rbcf_qc) else np.nan,
                    "median_cell_rich": np.nan,
                }
            )
            run_summary_rows.append(row)
            wsi.close()
            continue

        # scoring + types
        scores_df = compute_scores_and_types(tiles_qc, tf_qc, fs_qc, cfg)
        scores_df["brightness"] = bm_qc
        scores_df["qc_artifact_frac"] = af_qc
        scores_df["qc_rbc_frac"] = rbcf_qc
        scores_df["adipose_proxy"] = ad_qc
        scores_df["x_mask"] = grid_qc[:, 0]
        scores_df["y_mask"] = grid_qc[:, 1]
        scores_df["x0"] = xy0_qc[:, 0]
        scores_df["y0"] = xy0_qc[:, 1]
        scores_df["mask_status"] = slide_mask_status
        scores_df["mask_status_global"] = slide_mask_status_global
        scores_df["mask_status_effective"] = slide_mask_status_effective
        scores_df["status_basis"] = slide_status_basis
        scores_df["fallback_used"] = slide_fallback_used
        scores_df["mask_frac"] = slide_mask_frac
        scores_df["mask_frac_effective"] = slide_mask_frac_effective
        scores_df["sparse_canvas_mode"] = slide_sparse_canvas_mode
        scores_df["profile_used"] = slide_profile_used

        # spatial cap (in mask space)
        if bool(cfg["spatial"]["enable_spatial_cap"]):
            sel_idx = apply_spatial_cap(
                xy_mask=grid_qc,
                scores_df=scores_df,
                cell_size=int(cfg["spatial"]["cell_size"]),
                max_per_cell=int(cfg["spatial"]["max_per_cell"]),
                sort_col="cell_rich_p",
            )
            tiles_cap = tiles_qc[sel_idx]
            scores_cap = scores_df.iloc[sel_idx].reset_index(drop=True)
        else:
            tiles_cap = tiles_qc
            scores_cap = scores_df

        # quota selection
        target = int(cfg["scoring"]["target_patches"])
        sel_idx2 = quota_select(scores_cap, target, cfg)
        tiles_sel = tiles_cap[sel_idx2]
        meta_sel = scores_cap.iloc[sel_idx2].reset_index(drop=True)
        selected_count = int(len(meta_sel))

        # reject/flag based on good patches (A+B)
        good = int(meta_sel["type"].isin(["A", "B"]).sum())
        if good < int(cfg["scoring"]["min_good_patches_reject"]):
            print(f"REJECT slide: only {good} good patches (A+B).")
            row = _base_summary_row()
            row.update(
                {
                    "candidates_after_mask": int(len(grid)),
                    "qc_pass": int(len(tiles_qc)),
                    "selected_count": selected_count,
                    "good_count": good,
                    "typeA": int((meta_sel["type"] == "A").sum()),
                    "typeB": int((meta_sel["type"] == "B").sum()),
                    "typeC": int((meta_sel["type"] == "C").sum()),
                    "typeD": int((meta_sel["type"] == "D").sum()),
                    "mean_blood_score": float(meta_sel["blood_score"].mean()) if selected_count else np.nan,
                    "mean_rbc_frac": float(meta_sel["rbc_frac"].mean()) if selected_count else np.nan,
                    "median_cell_rich": float(meta_sel["cell_rich_score"].median()) if selected_count else np.nan,
                }
            )
            run_summary_rows.append(row)
            wsi.close()
            continue

        if good < int(cfg["scoring"]["min_good_patches_flag"]):
            print(f"FLAG slide: only {good} good patches (A+B). Keeping outputs.")

        # save outputs
        slide_out_coords = coords_root / sid
        slide_out_qc = qc_root / sid
        slide_out_patches = patches_lvl0_root / sid
        _ensure_dir(slide_out_coords)
        _ensure_dir(slide_out_qc)
        _ensure_dir(slide_out_patches)

        coords_np = meta_sel[["x0", "y0"]].to_numpy(dtype=np.int32)
        meta_sel["selected_count"] = selected_count
        meta_sel["good_count"] = good
        np.save(slide_out_coords / "selected_coords_level0.npy", coords_np)
        meta_sel.to_csv(slide_out_coords / "selected_meta.csv", index=False)

        slide_summary = pd.DataFrame(
            [
                {
                    **_base_summary_row(),
                    "candidates_after_mask": int(len(grid)),
                    "qc_pass": int(len(tiles_qc)),
                    "selected_count": selected_count,
                    "good_count": good,
                    "typeA": int((meta_sel["type"] == "A").sum()),
                    "typeB": int((meta_sel["type"] == "B").sum()),
                    "typeC": int((meta_sel["type"] == "C").sum()),
                    "typeD": int((meta_sel["type"] == "D").sum()),
                    "mean_blood_score": float(meta_sel["blood_score"].mean()) if selected_count else np.nan,
                    "mean_rbc_frac": float(meta_sel["rbc_frac"].mean()) if selected_count else np.nan,
                    "median_cell_rich": float(meta_sel["cell_rich_score"].median()) if selected_count else np.nan,
                },
            ]
        )
        slide_summary.to_csv(slide_out_qc / "summary.csv", index=False)
        run_summary_rows.extend(slide_summary.to_dict(orient="records"))

        if bool(cfg["outputs"]["write_montage"]):
            save_montage(
                tiles=tiles_sel,
                out_path=slide_out_qc / "montage.png",
                n=int(cfg["outputs"]["montage_n"]),
                seed=seed,
            )

        if bool(cfg["outputs"]["write_patches"]):
            from PIL import Image
            for i in range(len(tiles_sel)):
                Image.fromarray(tiles_sel[i]).save(slide_out_patches / f"{i:05d}.png")

        print(f"Saved -> {slide_out_coords}")
        wsi.close()

    if run_summary_rows:
        run_summary_path = qc_root / "run_summary.csv"
        run_summary = pd.DataFrame(run_summary_rows).sort_values("slide_id").reset_index(drop=True)
        if run_summary_path.exists():
            prev = pd.read_csv(run_summary_path, dtype={"slide_id": "string"})
            prev["slide_id"] = prev["slide_id"].astype(str).str.strip()
            run_summary["slide_id"] = run_summary["slide_id"].astype(str).str.strip()
            run_summary = pd.concat([prev, run_summary], ignore_index=True, sort=False)
            run_summary = (
                run_summary.drop_duplicates(subset=["slide_id"], keep="last")
                .sort_values("slide_id")
                .reset_index(drop=True)
            )
        run_summary.to_csv(run_summary_path, index=False)
        print(f"Run summary -> {run_summary_path}")

# scripts/run_pilot.py
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.select.pipeline import run_on_slides


WSI_EXTS = (".svs", ".ndpi", ".mrxs", ".tif", ".tiff")
MASK_EXTS = (".npy", ".png", ".tif", ".tiff")


def _resolve_path(project_root: Path, p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def _as_list(v: Any) -> list:
    if isinstance(v, list):
        return v
    if v is None:
        return []
    return [v]


def list_wsi_files(wsi_dirs: list[Path], recursive: bool = True) -> list[tuple[str, str]]:
    files: list[tuple[Path, str]] = []
    for wsi_dir in wsi_dirs:
        center = wsi_dir.name
        if recursive:
            files.extend([(p, center) for p in sorted(wsi_dir.rglob("*")) if p.is_file() and p.suffix.lower() in WSI_EXTS])
        else:
            files.extend([(p, center) for p in sorted(wsi_dir.iterdir()) if p.is_file() and p.suffix.lower() in WSI_EXTS])
    # de-dup + stable order
    uniq: list[tuple[str, str]] = []
    seen = set()
    for p, center in files:
        if p.stem not in seen:
            uniq.append((str(p), str(center)))
            seen.add(p.stem)
    return uniq


def find_mask_for_slide(mask_dir: Path, slide_path: str) -> Path | None:
    stem = Path(slide_path).stem
    for ext in MASK_EXTS:
        p = mask_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def run_precheck(slides: list[str], cfg: dict) -> bool:
    mask_dir = Path(cfg["paths"]["mask_dir"])
    ok = True

    print("\n=== PRECHECK ===")
    print(f"Mask dir: {mask_dir}")
    if not mask_dir.exists():
        print("FAIL: mask directory does not exist.")
        return False

    missing_masks: list[str] = []
    for sp in slides:
        if find_mask_for_slide(mask_dir, sp) is None:
            missing_masks.append(Path(sp).name)

    found = len(slides) - len(missing_masks)
    print(f"Mask files found: {found}/{len(slides)}")
    if missing_masks:
        ok = False
        print("Missing masks:")
        for name in missing_masks:
            print(f"  - {name}")

    summary_path = mask_dir / "mask_summary.csv"
    if summary_path.exists():
        with summary_path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        by_id = {str(r.get("slide_id", "")): r for r in rows}
        in_run = [by_id.get(Path(s).stem) for s in slides]
        in_run = [r for r in in_run if r is not None]
        print(f"Mask summary rows matched to run slides: {len(in_run)}/{len(slides)}")
        status_key = "mask_status_effective"
        if in_run and status_key in in_run[0]:
            counts: dict[str, int] = {}
            for r in in_run:
                st = str(r.get(status_key, "")).strip() or "unknown"
                counts[st] = counts.get(st, 0) + 1
            print("Effective status counts:")
            for k in sorted(counts):
                print(f"  {k}: {counts[k]}")
    else:
        print("Mask summary not found (data/masks/mask_summary.csv).")

    print("PRECHECK: PASS" if ok else "PRECHECK: FAIL")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Run patch-selection pilot pipeline.")
    parser.add_argument("--config", default="configs/pilot.yaml", help="Path to config yaml.")
    parser.add_argument("--n-slides", type=int, default=None, help="Override number of slides. <=0 means all.")
    parser.add_argument(
        "--multi-worker-mode",
        action="store_true",
        help="Enable parallel center execution when possible.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="CPU workers for center-parallel execution.",
    )
    parser.add_argument(
        "--precheck-only",
        action="store_true",
        help="Validate mask coverage/readiness for selected slides and exit without extraction.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg_path = _resolve_path(project_root, args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)
    run_cfg = cfg.get("run", {})
    cfg_multi = bool(run_cfg.get("multi_worker_mode", False))
    cfg_workers = int(run_cfg.get("cpu_workers", 1))
    multi_worker_mode = bool(args.multi_worker_mode or cfg_multi)
    workers = int(args.workers) if args.workers is not None else int(cfg_workers)
    workers = max(1, workers)

    raw_wsi_dirs = _as_list(cfg["paths"].get("wsi_dir", "data/raw_wsi"))
    wsi_recursive = bool(cfg["paths"].get("wsi_recursive", True))
    wsi_dirs = [_resolve_path(project_root, p) for p in raw_wsi_dirs]
    for wsi_dir in wsi_dirs:
        if not wsi_dir.exists():
            raise FileNotFoundError(f"WSI dir not found: {wsi_dir}")
    mask_dir = _resolve_path(project_root, cfg["paths"].get("mask_dir", "data/masks"))
    out_dir = _resolve_path(project_root, cfg["paths"].get("out_dir", "data/out"))
    cfg["paths"]["wsi_dir"] = [str(p) for p in wsi_dirs]
    cfg["paths"]["mask_dir"] = str(mask_dir)
    cfg["paths"]["out_dir"] = str(out_dir)

    slide_items = list_wsi_files(wsi_dirs, recursive=wsi_recursive)
    if len(slide_items) == 0:
        raise RuntimeError(f"No WSIs found in {wsi_dirs} with extensions {WSI_EXTS}")

    n = int(args.n_slides) if args.n_slides is not None else int(cfg["run"].get("n_slides", 0))
    if n > 0:
        slide_items = slide_items[:n]

    print("\n=== PILOT RUN ===")
    print(f"Config: {cfg_path}")
    print(f"WSI dirs: {[str(p) for p in wsi_dirs]}")
    print(f"WSI recursive scan: {wsi_recursive}")
    print(f"Multi-worker mode: {multi_worker_mode} (workers={workers})")
    print(f"Mask dir: {mask_dir}")
    print(f"Out dir:  {out_dir}")
    print(f"Slides ({len(slide_items)}):")
    for i, (s, center) in enumerate(slide_items, 1):
        print(f"  {i:02d}. [{center}] {Path(s).name}")

    by_center: dict[str, list[str]] = {}
    for sp, center in slide_items:
        by_center.setdefault(center, []).append(sp)

    all_ok = True
    center_jobs: list[tuple[str, list[str], dict]] = []
    for center, center_slides in sorted(by_center.items()):
        center_mask_dir = mask_dir / center / "mask"
        center_out_dir = out_dir / center
        center_cfg = dict(cfg)
        center_cfg["paths"] = dict(cfg["paths"])
        center_cfg["paths"]["mask_dir"] = str(center_mask_dir)
        center_cfg["paths"]["out_dir"] = str(center_out_dir)
        print(f"\n--- Center: {center} ({len(center_slides)} slides) ---")
        precheck_ok = run_precheck(center_slides, center_cfg)
        if not precheck_ok:
            all_ok = False
            continue
        center_jobs.append((center, center_slides, center_cfg))

    if not all_ok:
        raise SystemExit(1)
    if args.precheck_only:
        return

    if multi_worker_mode and workers > 1 and len(center_jobs) > 1:
        n_pool = min(workers, len(center_jobs))
        print(f"Running QC extraction in parallel across centers (workers={n_pool})")
        with ThreadPoolExecutor(max_workers=n_pool) as ex:
            fut_map = {
                ex.submit(run_on_slides, center_slides, center_cfg): center
                for center, center_slides, center_cfg in center_jobs
            }
            for fut in as_completed(fut_map):
                center = fut_map[fut]
                try:
                    fut.result()
                    print(f"Center completed: {center}")
                except Exception as e:
                    print(f"Center failed: {center} ({type(e).__name__}: {e})")
                    raise
    else:
        for center, center_slides, center_cfg in center_jobs:
            run_on_slides(center_slides, center_cfg)

    # Aggregate per-center QC summaries into a global summary for gate checks.
    agg_rows: list[pd.DataFrame] = []
    for center in sorted(by_center.keys()):
        center_summary = out_dir / center / "qc" / "run_summary.csv"
        if center_summary.exists():
            df = pd.read_csv(center_summary, dtype={"slide_id": "string"})
            df["slide_id"] = df["slide_id"].astype(str).str.strip()
            df["center"] = str(center)
            agg_rows.append(df)
    if agg_rows:
        qc_global_dir = out_dir / "qc"
        qc_global_dir.mkdir(parents=True, exist_ok=True)
        agg = pd.concat(agg_rows, ignore_index=True, sort=False)
        agg = agg.drop_duplicates(subset=["slide_id"], keep="last").sort_values("slide_id").reset_index(drop=True)
        agg_path = qc_global_dir / "run_summary.csv"
        agg.to_csv(agg_path, index=False)
        print(f"Aggregated QC summary -> {agg_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

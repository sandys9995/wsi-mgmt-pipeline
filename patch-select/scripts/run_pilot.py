# scripts/run_pilot.py
from __future__ import annotations

import argparse
import csv
import gc
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.select.pipeline import run_on_slides
from src.utils.runlog import progress, stage_logger
from src.utils.slides import list_slide_records, slide_key_from_row, slide_match


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


def list_wsi_files(wsi_dirs: list[Path], recursive: bool = True) -> list[dict[str, str]]:
    return list_slide_records(wsi_dirs, recursive=recursive, exts=WSI_EXTS)


def find_mask_for_slide(mask_dir: Path, slide_rec: dict[str, str]) -> Path | None:
    stem = str(slide_rec["slide_uid"]).strip()
    for ext in MASK_EXTS:
        p = mask_dir / f"{stem}{ext}"
        if p.exists():
            return p
    stem_fallback = str(slide_rec["slide_id"]).strip()
    for ext in MASK_EXTS:
        p = mask_dir / f"{stem_fallback}{ext}"
        if p.exists():
            return p
    return None


def run_precheck(slides: list[dict[str, str]], cfg: dict, logger) -> bool:
    mask_dir = Path(cfg["paths"]["mask_dir"])
    ok = True
    known_mask_failure_statuses = {"read_error", "open_error", "mask_error"}

    logger.info("=== PRECHECK ===")
    logger.info(f"Mask dir: {mask_dir}")
    if not mask_dir.exists():
        logger.error("FAIL: mask directory does not exist.")
        return False

    by_id: dict[str, dict] = {}
    summary_path = mask_dir / "mask_summary.csv"
    if summary_path.exists():
        with summary_path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        by_id = {slide_key_from_row(r): r for r in rows}

    missing_masks: list[str] = []
    known_failures: list[tuple[str, str, str]] = []
    for rec in slides:
        if find_mask_for_slide(mask_dir, rec) is None:
            row = by_id.get(str(rec["slide_uid"]).strip(), {})
            status = str(row.get("mask_status_effective") or row.get("mask_status") or "").strip()
            if status in known_mask_failure_statuses:
                known_failures.append((Path(str(rec["path"])).name, status, str(row.get("error_message", "")).strip()))
                continue
            missing_masks.append(Path(str(rec["path"])).name)

    found = len(slides) - len(missing_masks)
    logger.info(f"Mask files found: {found}/{len(slides)}")
    if missing_masks:
        ok = False
        logger.error("Missing masks:")
        for name in missing_masks:
            logger.error(f"  - {name}")
    if known_failures:
        logger.warning(f"Known mask failures (will be skipped downstream): {len(known_failures)}")
        for name, status, err in known_failures[:10]:
            suffix = f" | {err}" if err else ""
            logger.warning(f"  - {name}: {status}{suffix}")
        if len(known_failures) > 10:
            logger.warning(f"  ... and {len(known_failures) - 10} more")

    if summary_path.exists():
        in_run = [by_id.get(str(rec["slide_uid"]).strip()) for rec in slides]
        in_run = [r for r in in_run if r is not None]
        logger.info(f"Mask summary rows matched to run slides: {len(in_run)}/{len(slides)}")
        status_key = "mask_status_effective"
        if in_run and status_key in in_run[0]:
            counts: dict[str, int] = {}
            for r in in_run:
                st = str(r.get(status_key, "")).strip() or "unknown"
                counts[st] = counts.get(st, 0) + 1
            logger.info("Effective status counts:")
            for k in sorted(counts):
                logger.info(f"  {k}: {counts[k]}")
    else:
        logger.warning("Mask summary not found (data/masks/mask_summary.csv).")

    logger.info("PRECHECK: PASS" if ok else "PRECHECK: FAIL")
    return ok


def _is_slide_done_for_qc(slide_rec: dict[str, str], cfg: dict) -> bool:
    sid = str(slide_rec["slide_uid"])
    out_dir = Path(cfg["paths"]["out_dir"])
    extract_qc_pool_only = bool(cfg.get("scoring", {}).get("extract_qc_pool_only", False))
    if extract_qc_pool_only:
        return (out_dir / "qc_pool" / sid / "qc_meta.csv").exists()
    return (out_dir / "coords" / sid / "selected_meta.csv").exists()


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
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip slides that already have stage outputs (default from config run.resume, fallback true).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable detailed console logs.")
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
    cfg_resume = bool(run_cfg.get("resume", True))
    multi_worker_mode = bool(args.multi_worker_mode or cfg_multi)
    workers = int(args.workers) if args.workers is not None else int(cfg_workers)
    workers = max(1, workers)
    resume = cfg_resume if args.resume is None else bool(args.resume)

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

    logger, interactive = stage_logger("qc_driver", out_dir, verbose=bool(args.verbose))
    n = int(args.n_slides) if args.n_slides is not None else int(cfg["run"].get("n_slides", 0))
    if n > 0:
        slide_items = slide_items[:n]

    logger.info("=== PILOT RUN ===")
    logger.info(f"Config: {cfg_path}")
    logger.info(f"WSI recursive scan: {wsi_recursive}")
    logger.info(f"Multi-worker mode: {multi_worker_mode} (workers={workers})")
    logger.info(f"Resume mode: {resume}")
    logger.info(f"Mask dir: {mask_dir}")
    logger.info(f"Out dir: {out_dir}")
    logger.info(f"Slides: {len(slide_items)} across {len({rec['center'] for rec in slide_items})} centers")

    by_center: dict[str, list[dict[str, str]]] = {}
    for rec in slide_items:
        by_center.setdefault(str(rec["center"]), []).append(rec)

    all_ok = True
    center_jobs: list[tuple[str, list[dict[str, str]], dict]] = []
    for center, center_slides in progress(
        sorted(by_center.items()), interactive=interactive, desc="[qc] centers", unit="center"
    ):
        center_mask_dir = mask_dir / center / "mask"
        center_out_dir = out_dir / center
        center_cfg = dict(cfg)
        center_cfg["paths"] = dict(cfg["paths"])
        center_cfg["paths"]["mask_dir"] = str(center_mask_dir)
        center_cfg["paths"]["out_dir"] = str(center_out_dir)
        logger.info(f"--- Center: {center} ({len(center_slides)} slides) ---")
        precheck_ok = run_precheck(center_slides, center_cfg, logger)
        if not precheck_ok:
            all_ok = False
            continue
        if resume:
            pending_slides = [rec for rec in center_slides if not _is_slide_done_for_qc(rec, center_cfg)]
            skipped_done = len(center_slides) - len(pending_slides)
            logger.info(f"Resume scan: pending={len(pending_slides)} skipped_done={skipped_done}")
        else:
            pending_slides = list(center_slides)
        if pending_slides:
            center_jobs.append((center, pending_slides, center_cfg))
        else:
            logger.info(f"No pending slides for center={center}; skipping extraction.")

    if not all_ok:
        raise SystemExit(1)
    if args.precheck_only:
        return
    if not center_jobs:
        logger.info("No pending slides across all centers. Nothing to run.")
        return

    worker_interactive = interactive and not (multi_worker_mode and workers > 1 and len(center_jobs) > 1)

    if multi_worker_mode and workers > 1 and len(center_jobs) > 1:
        n_pool = min(workers, len(center_jobs))
        logger.info(f"Running QC extraction in parallel across centers (workers={n_pool})")
        with ThreadPoolExecutor(max_workers=n_pool) as ex:
            fut_map = {
                ex.submit(run_on_slides, center_slides, center_cfg, logger, worker_interactive): center
                for center, center_slides, center_cfg in center_jobs
            }
            for fut in as_completed(fut_map):
                center = fut_map[fut]
                try:
                    fut.result()
                    logger.info(f"Center completed: {center}")
                except Exception as e:
                    logger.error(f"Center failed: {center} ({type(e).__name__}: {e})")
                    raise
    else:
        for center, center_slides, center_cfg in progress(
            center_jobs, interactive=interactive, desc="[qc] center runs", unit="center"
        ):
            run_on_slides(center_slides, center_cfg, logger, worker_interactive)
            gc.collect()

    # Aggregate per-center QC summaries into a global summary for gate checks.
    agg_rows: list[pd.DataFrame] = []
    for center in progress(sorted(by_center.keys()), interactive=interactive, desc="[qc] aggregate", unit="center"):
        center_summary = out_dir / center / "qc" / "run_summary.csv"
        if center_summary.exists():
            df = pd.read_csv(center_summary, dtype={"slide_id": "string", "slide_uid": "string"})
            if "slide_uid" in df.columns:
                df["slide_uid"] = df["slide_uid"].astype(str).str.strip()
            df["center"] = str(center)
            agg_rows.append(df)
    if agg_rows:
        qc_global_dir = out_dir / "qc"
        qc_global_dir.mkdir(parents=True, exist_ok=True)
        agg = pd.concat(agg_rows, ignore_index=True, sort=False)
        key_col = "slide_uid" if "slide_uid" in agg.columns else "slide_id"
        agg = agg.drop_duplicates(subset=[key_col], keep="last").sort_values(["center", "slide_id"]).reset_index(drop=True)
        agg_path = qc_global_dir / "run_summary.csv"
        agg.to_csv(agg_path, index=False)
        logger.info(f"Aggregated QC summary -> {agg_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()

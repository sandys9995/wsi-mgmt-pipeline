from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import shutil
import sys

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.runlog import stage_logger
from src.utils.slides import list_slide_records

WSI_EXTS = {".svs", ".ndpi", ".mrxs", ".tif", ".tiff"}


def _resolve_path(project_root: Path, p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def _as_list(v):
    if isinstance(v, list):
        return v
    if v is None:
        return []
    return [v]


def _backup_csv(path: Path) -> Path:
    backup = path.with_suffix(path.suffix + ".legacy.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def _legacy_summary_key(row: pd.Series) -> tuple[str, str]:
    center = str(row.get("center", "")).strip()
    slide_id = str(row.get("slide_id", "")).strip()
    return center, slide_id


def _row_has_new_uid(row: pd.Series) -> bool:
    slide_uid = str(row.get("slide_uid", "")).strip()
    slide_id = str(row.get("slide_id", "")).strip()
    return bool(slide_uid and slide_uid != slide_id)


def _transform_summary_df(
    df: pd.DataFrame,
    *,
    record_map: dict[tuple[str, str], dict[str, str]],
    duplicate_keys: set[tuple[str, str]],
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    rows: list[dict] = []
    dropped: list[dict[str, str]] = []
    for _, row in df.iterrows():
        key = _legacy_summary_key(row)
        rec = record_map.get(key)
        if _row_has_new_uid(row):
            row_dict = row.to_dict()
            row_dict["slide_uid"] = str(row_dict.get("slide_uid", "")).strip()
            row_dict["slide_relpath"] = str(row_dict.get("slide_relpath", "")).strip()
            rows.append(row_dict)
            continue
        if key in duplicate_keys:
            dropped.append(
                {
                    "center": key[0],
                    "slide_id": key[1],
                    "reason": "ambiguous_duplicate_basename",
                }
            )
            continue
        if rec is None:
            dropped.append(
                {
                    "center": key[0],
                    "slide_id": key[1],
                    "reason": "no_matching_slide_record",
                }
            )
            continue
        row_dict = row.to_dict()
        row_dict["slide_uid"] = str(rec["slide_uid"])
        row_dict["slide_relpath"] = str(rec["slide_relpath"])
        rows.append(row_dict)

    if not rows:
        return pd.DataFrame(columns=list(df.columns) + ["slide_uid", "slide_relpath"]), dropped

    out = pd.DataFrame(rows)
    if "slide_uid" not in out.columns:
        out["slide_uid"] = ""
    if "slide_relpath" not in out.columns:
        out["slide_relpath"] = ""
    out["slide_uid"] = out["slide_uid"].astype(str).str.strip()
    out["slide_relpath"] = out["slide_relpath"].astype(str).str.strip()
    out = out.drop_duplicates(subset=["slide_uid"], keep="last").sort_values(["center", "slide_id"]).reset_index(drop=True)
    return out, dropped


def _move_or_copy(src: Path, dst: Path, *, copy_only: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_only:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate legacy mask outputs from slide_id names to slide_uid names.")
    parser.add_argument("--config", default="configs/pilot.yaml", help="Path to config yaml.")
    parser.add_argument("--apply", action="store_true", help="Apply changes. Default is dry-run.")
    parser.add_argument("--copy", action="store_true", help="Copy legacy mask files instead of moving them.")
    parser.add_argument(
        "--delete-ambiguous",
        action="store_true",
        help="Delete ambiguous legacy basename masks instead of quarantining them.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths_cfg = cfg.get("paths", {})
    wsi_dirs = [_resolve_path(project_root, p) for p in _as_list(paths_cfg.get("wsi_dir"))]
    recursive = bool(paths_cfg.get("wsi_recursive", True))
    mask_dir = _resolve_path(project_root, paths_cfg.get("mask_dir", "data/masks"))
    logger, _ = stage_logger("mask_migrate", mask_dir, verbose=False)

    records = list_slide_records(wsi_dirs, recursive=recursive, exts=WSI_EXTS)
    basename_counts = Counter((str(r["center"]), str(r["slide_id"])) for r in records)
    duplicate_keys = {k for k, n in basename_counts.items() if n > 1}
    record_map = {(str(r["center"]), str(r["slide_id"])): r for r in records if basename_counts[(str(r["center"]), str(r["slide_id"]))] == 1}

    logger.info(
        f"Mask migration scan: slides={len(records)} unique_keys={len(record_map)} duplicate_keys={len(duplicate_keys)} "
        f"apply={args.apply} copy={args.copy} delete_ambiguous={args.delete_ambiguous}"
    )

    migration_dir = mask_dir / "migration"
    quarantine_dir = migration_dir / "ambiguous_legacy"
    migration_dir.mkdir(parents=True, exist_ok=True)

    file_rows: list[dict[str, str]] = []
    counts = {
        "migrated": 0,
        "already_new": 0,
        "missing_legacy": 0,
        "ambiguous": 0,
        "deleted_ambiguous": 0,
        "quarantined_ambiguous": 0,
    }
    handled_ambiguous_keys: set[tuple[str, str]] = set()

    for rec in records:
        center = str(rec["center"])
        slide_id = str(rec["slide_id"])
        slide_uid = str(rec["slide_uid"])
        key = (center, slide_id)
        center_mask_dir = mask_dir / center / "mask"
        legacy_npy = center_mask_dir / f"{slide_id}.npy"
        legacy_png = center_mask_dir / f"{slide_id}.png"
        new_npy = center_mask_dir / f"{slide_uid}.npy"
        new_png = center_mask_dir / f"{slide_uid}.png"
        is_duplicate = key in duplicate_keys

        if new_npy.exists() or new_png.exists():
            counts["already_new"] += 1
            file_rows.append(
                {
                    "center": center,
                    "slide_id": slide_id,
                    "slide_uid": slide_uid,
                    "legacy_npy": str(legacy_npy),
                    "legacy_png": str(legacy_png),
                    "new_npy": str(new_npy),
                    "new_png": str(new_png),
                    "action": "already_new",
                    "status": "ok",
                }
            )
            continue

        if is_duplicate:
            counts["ambiguous"] += 1
            action = "ambiguous_needs_rerun"
            if key not in handled_ambiguous_keys and (legacy_npy.exists() or legacy_png.exists()):
                if args.apply:
                    if args.delete_ambiguous:
                        if legacy_npy.exists():
                            legacy_npy.unlink()
                        if legacy_png.exists():
                            legacy_png.unlink()
                        counts["deleted_ambiguous"] += 1
                        action = "ambiguous_deleted"
                    else:
                        q_dir = quarantine_dir / center
                        if legacy_npy.exists():
                            _move_or_copy(legacy_npy, q_dir / legacy_npy.name, copy_only=False)
                        if legacy_png.exists():
                            _move_or_copy(legacy_png, q_dir / legacy_png.name, copy_only=False)
                        counts["quarantined_ambiguous"] += 1
                        action = "ambiguous_quarantined"
                handled_ambiguous_keys.add(key)
            file_rows.append(
                {
                    "center": center,
                    "slide_id": slide_id,
                    "slide_uid": slide_uid,
                    "legacy_npy": str(legacy_npy),
                    "legacy_png": str(legacy_png),
                    "new_npy": str(new_npy),
                    "new_png": str(new_png),
                    "action": action,
                    "status": "rerun_required",
                }
            )
            continue

        if not legacy_npy.exists() and not legacy_png.exists():
            counts["missing_legacy"] += 1
            file_rows.append(
                {
                    "center": center,
                    "slide_id": slide_id,
                    "slide_uid": slide_uid,
                    "legacy_npy": str(legacy_npy),
                    "legacy_png": str(legacy_png),
                    "new_npy": str(new_npy),
                    "new_png": str(new_png),
                    "action": "missing_legacy",
                    "status": "rerun_required",
                }
            )
            continue

        if args.apply:
            if legacy_npy.exists():
                _move_or_copy(legacy_npy, new_npy, copy_only=args.copy)
            if legacy_png.exists():
                _move_or_copy(legacy_png, new_png, copy_only=args.copy)
        counts["migrated"] += 1
        file_rows.append(
            {
                "center": center,
                "slide_id": slide_id,
                "slide_uid": slide_uid,
                "legacy_npy": str(legacy_npy),
                "legacy_png": str(legacy_png),
                "new_npy": str(new_npy),
                "new_png": str(new_png),
                "action": "migrated" if args.apply else "would_migrate",
                "status": "ok",
            }
        )

    file_report = pd.DataFrame(file_rows).sort_values(["center", "slide_id", "slide_uid"]).reset_index(drop=True)
    file_report_path = migration_dir / "mask_file_migration_report.csv"
    file_report.to_csv(file_report_path, index=False)

    dropped_rows: list[dict[str, str]] = []
    root_summary_path = mask_dir / "mask_summary.csv"
    if root_summary_path.exists():
        root_df = pd.read_csv(root_summary_path, dtype={"slide_uid": "string", "slide_id": "string"})
        new_root_df, dropped = _transform_summary_df(root_df, record_map=record_map, duplicate_keys=duplicate_keys)
        dropped_rows.extend(dropped)
        if args.apply:
            _backup_csv(root_summary_path)
            new_root_df.to_csv(root_summary_path, index=False)

    for center in sorted({str(r["center"]) for r in records}):
        center_summary_path = mask_dir / center / "mask" / "mask_summary.csv"
        if not center_summary_path.exists():
            continue
        center_df = pd.read_csv(center_summary_path, dtype={"slide_uid": "string", "slide_id": "string"})
        new_center_df, dropped = _transform_summary_df(center_df, record_map=record_map, duplicate_keys=duplicate_keys)
        dropped_rows.extend(dropped)
        if args.apply:
            _backup_csv(center_summary_path)
            new_center_df.to_csv(center_summary_path, index=False)

    dropped_report = pd.DataFrame(dropped_rows).drop_duplicates().sort_values(["center", "slide_id", "reason"]) if dropped_rows else pd.DataFrame(columns=["center", "slide_id", "reason"])
    dropped_report_path = migration_dir / "mask_summary_dropped_rows.csv"
    dropped_report.to_csv(dropped_report_path, index=False)

    logger.info(
        f"Mask migration complete. migrated={counts['migrated']} already_new={counts['already_new']} "
        f"ambiguous={counts['ambiguous']} missing_legacy={counts['missing_legacy']}"
    )
    logger.info(
        f"Ambiguous actions: quarantined={counts['quarantined_ambiguous']} deleted={counts['deleted_ambiguous']} "
        f"reports={file_report_path} dropped_rows={dropped_report_path}"
    )
    if not args.apply:
        logger.info("Dry-run only. Re-run with --apply to perform migration.")


if __name__ == "__main__":
    main()

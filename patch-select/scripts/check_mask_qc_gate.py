from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import pandas as pd


def _pick_status(row: pd.Series) -> str:
    v = row.get("mask_status_effective", None)
    if v is None:
        return str(row.get("mask_status", "unknown"))
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none"}:
        return str(row.get("mask_status", "unknown"))
    return s


def _parse_statuses(raw: str) -> set[str]:
    out: set[str] = set()
    for s in str(raw).split(","):
        v = s.strip().lower()
        if v:
            out.add(v)
    if not out:
        out = {"ok"}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Check mask+QC acceptance gate.")
    ap.add_argument("--mask-summary", default="data/masks/mask_summary.csv")
    ap.add_argument("--run-summary", default="data/out/qc/run_summary.csv")
    ap.add_argument(
        "--mode",
        choices=["adaptive", "strict"],
        default="adaptive",
        help="adaptive: thresholds scale with run size; strict: fixed absolute thresholds.",
    )
    ap.add_argument(
        "--ok-statuses",
        default="ok,high_tissue",
        help="Comma-separated mask statuses accepted by gate.",
    )
    ap.add_argument("--min-candidates", type=int, default=200)
    ap.add_argument("--min-qc-pass", type=int, default=80)
    ap.add_argument("--dataset-pass-min", type=int, default=16)
    ap.add_argument("--digital-pass-min", type=int, default=3)
    ap.add_argument(
        "--dataset-pass-rate",
        type=float,
        default=16.0 / 18.0,
        help="Adaptive mode: required passed fraction of all slides.",
    )
    ap.add_argument(
        "--digital-pass-rate",
        type=float,
        default=3.0 / 4.0,
        help="Adaptive mode: required passed fraction of DigitalSlide* subset.",
    )
    args = ap.parse_args()

    mask_path = Path(args.mask_summary)
    run_path = Path(args.run_summary)
    if not mask_path.exists():
        print(f"Missing file: {mask_path}")
        return 2
    if not run_path.exists():
        print(f"Missing file: {run_path}")
        return 2

    mask_df = pd.read_csv(mask_path, dtype={"slide_id": "string"})
    run_df = pd.read_csv(run_path, dtype={"slide_id": "string"})
    if "slide_id" not in mask_df.columns or "slide_id" not in run_df.columns:
        print("Both CSVs must contain a 'slide_id' column.")
        return 2
    mask_df["slide_id"] = mask_df["slide_id"].astype(str).str.strip()
    run_df["slide_id"] = run_df["slide_id"].astype(str).str.strip()

    merged = mask_df.merge(
        run_df[["slide_id", "candidates_after_mask", "qc_pass"]],
        on="slide_id",
        how="left",
        suffixes=("", "_run"),
    )

    ok_statuses = _parse_statuses(args.ok_statuses)
    merged["status_for_gate"] = merged.apply(_pick_status, axis=1)
    merged["status_for_gate_norm"] = merged["status_for_gate"].astype(str).str.strip().str.lower()
    merged["candidates_after_mask"] = merged["candidates_after_mask"].fillna(0).astype(int)
    merged["qc_pass"] = merged["qc_pass"].fillna(0).astype(int)
    merged["slide_pass"] = (
        (merged["status_for_gate_norm"].isin(ok_statuses))
        & (merged["candidates_after_mask"] >= int(args.min_candidates))
        & (merged["qc_pass"] >= int(args.min_qc_pass))
    )

    n_total = int(len(merged))
    n_pass = int(merged["slide_pass"].sum())
    is_digital = merged["slide_id"].astype(str).str.startswith("DigitalSlide")
    n_digital = int(is_digital.sum())
    n_digital_pass = int((merged["slide_pass"] & is_digital).sum())

    if args.mode == "strict":
        req_dataset = int(args.dataset_pass_min)
        req_digital = int(args.digital_pass_min)
    else:
        req_dataset = int(math.ceil(float(args.dataset_pass_rate) * max(0, n_total))) if n_total > 0 else 0
        req_digital = int(math.ceil(float(args.digital_pass_rate) * max(0, n_digital))) if n_digital > 0 else 0

    print("Mask+QC Gate")
    print(f"- Mode: {args.mode}")
    print(f"- Accepted statuses: {sorted(ok_statuses)}")
    print(f"- Slides: {n_pass}/{n_total} passed (required >= {req_dataset})")
    print(f"- DigitalSlide: {n_digital_pass}/{n_digital} passed (required >= {req_digital})")

    failed = merged.loc[~merged["slide_pass"], ["slide_id", "status_for_gate", "candidates_after_mask", "qc_pass"]]
    if not failed.empty:
        print("\nFailed slides:")
        for _, r in failed.sort_values("slide_id").iterrows():
            print(
                f"- {r['slide_id']}: status={r['status_for_gate']} "
                f"candidates={int(r['candidates_after_mask'])} qc_pass={int(r['qc_pass'])}"
            )

    dataset_ok = n_pass >= int(req_dataset)
    digital_ok = n_digital_pass >= int(req_digital)
    if dataset_ok and digital_ok:
        print("\nRESULT: PASS")
        return 0
    print("\nRESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())

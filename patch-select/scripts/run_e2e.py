from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


VALID_STAGES = ("mask", "qc", "gate", "tumor", "uni")


def _resolve_path(project_root: Path, p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def _run_stage(name: str, cmd: list[str], cwd: Path) -> float:
    print(f"\n=== STAGE: {name} ===")
    print(" ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd))
    dt = float(time.time() - t0)
    print(f"[stage:{name}] exit={proc.returncode} elapsed={dt:.1f}s")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return dt


def _parse_stages(raw: str) -> list[str]:
    out: list[str] = []
    for s in raw.split(","):
        k = s.strip().lower()
        if not k:
            continue
        if k not in VALID_STAGES:
            raise ValueError(f"Invalid stage '{k}'. Valid: {VALID_STAGES}")
        out.append(k)
    if not out:
        raise ValueError("No stages selected.")
    return out


def _load_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run end-to-end patch pipeline: mask -> qc -> gate -> tumor -> uni."
    )
    ap.add_argument("--config", default="configs/pilot.yaml", help="Path to config yaml.")
    ap.add_argument(
        "--stages",
        default="mask,qc,gate,tumor,uni",
        help="Comma-separated stages from: mask,qc,gate,tumor,uni",
    )
    ap.add_argument("--n-slides", type=int, default=None, help="Override n_slides for qc/tumor/uni stages.")
    ap.add_argument(
        "--multi-worker-mode",
        action="store_true",
        help="Enable multi-worker mode across stages where supported.",
    )
    ap.add_argument(
        "--cpu-workers",
        type=int,
        default=None,
        help="Override CPU worker count for stages that support it.",
    )
    ap.add_argument(
        "--io-workers",
        type=int,
        default=None,
        help="Override IO worker count for patch-reading heavy stages (tumor/uni).",
    )
    ap.add_argument(
        "--smoke-gate",
        action="store_true",
        help="Relax gate thresholds for smoke runs (dataset>=1, digital>=0).",
    )
    ap.add_argument("--overwrite-uni", action="store_true", help="Pass --overwrite to uni stage.")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg_path = _resolve_path(project_root, args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = _load_cfg(cfg_path)

    paths_cfg = cfg.get("paths", {})
    mask_summary = _resolve_path(project_root, paths_cfg.get("mask_dir", "data/masks")) / "mask_summary.csv"
    qc_summary = _resolve_path(project_root, paths_cfg.get("out_dir", "data/out")) / "qc" / "run_summary.csv"

    stages = _parse_stages(args.stages)
    python = sys.executable
    timings: dict[str, float] = {}

    print("=== E2E PIPELINE ===")
    print(f"Config: {cfg_path}")
    print(f"Stages: {stages}")
    if args.n_slides is not None:
        print(f"n_slides override: {args.n_slides}")
    if args.multi_worker_mode:
        print("multi_worker_mode: enabled")
    if args.cpu_workers is not None:
        print(f"cpu_workers override: {args.cpu_workers}")
    if args.io_workers is not None:
        print(f"io_workers override: {args.io_workers}")
    if args.smoke_gate:
        print("smoke_gate: enabled (dataset-pass-min=1, digital-pass-min=0)")

    for stage in stages:
        if stage == "mask":
            cmd = [python, "-u", "scripts/make_masks.py", "--config", str(cfg_path)]
            if args.n_slides is not None:
                cmd += ["--n-slides", str(args.n_slides)]
            if args.multi_worker_mode:
                cmd += ["--multi-worker-mode"]
            if args.cpu_workers is not None:
                cmd += ["--workers", str(args.cpu_workers)]
        elif stage == "qc":
            cmd = [python, "-u", "scripts/run_pilot.py", "--config", str(cfg_path)]
            if args.n_slides is not None:
                cmd += ["--n-slides", str(args.n_slides)]
            if args.multi_worker_mode:
                cmd += ["--multi-worker-mode"]
            if args.cpu_workers is not None:
                cmd += ["--workers", str(args.cpu_workers)]
        elif stage == "gate":
            cmd = [
                python,
                "-u",
                "scripts/check_mask_qc_gate.py",
                "--mask-summary",
                str(mask_summary),
                "--run-summary",
                str(qc_summary),
            ]
            if args.smoke_gate:
                cmd += ["--dataset-pass-min", "1", "--digital-pass-min", "0"]
        elif stage == "tumor":
            cmd = [python, "-u", "scripts/run_tumor_gate_pilot.py", "--config", str(cfg_path)]
            if args.n_slides is not None:
                cmd += ["--n-slides", str(args.n_slides)]
            if args.multi_worker_mode:
                cmd += ["--multi-worker-mode"]
            if args.cpu_workers is not None:
                cmd += ["--workers", str(args.cpu_workers)]
            if args.io_workers is not None:
                cmd += ["--io-workers", str(args.io_workers)]
        elif stage == "uni":
            cmd = [python, "-u", "scripts/run_uni_features.py", "--config", str(cfg_path)]
            if args.n_slides is not None:
                cmd += ["--n-slides", str(args.n_slides)]
            if args.multi_worker_mode:
                cmd += ["--multi-worker-mode"]
            if args.cpu_workers is not None:
                cmd += ["--workers", str(args.cpu_workers)]
            if args.io_workers is not None:
                cmd += ["--io-workers", str(args.io_workers)]
            if args.overwrite_uni:
                cmd += ["--overwrite"]
        else:
            raise RuntimeError(f"Unhandled stage '{stage}'")

        timings[stage] = _run_stage(stage, cmd, cwd=project_root)

    total = float(sum(timings.values()))
    print("\n=== E2E DONE ===")
    for k in stages:
        print(f"{k}: {timings[k]:.1f}s")
    print(f"total: {total:.1f}s")


if __name__ == "__main__":
    main()

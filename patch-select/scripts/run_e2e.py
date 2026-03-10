from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.runlog import progress, stage_logger


VALID_STAGES = ("mask", "qc", "gate", "tumor", "uni")


def _resolve_path(project_root: Path, p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def _run_stage(name: str, cmd: list[str], cwd: Path, logger) -> tuple[float, int]:
    logger.info(f"=== STAGE: {name} ===")
    logger.debug(" ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd))
    dt = float(time.time() - t0)
    logger.info(f"[stage:{name}] exit={proc.returncode} elapsed={dt:.1f}s")
    return dt, int(proc.returncode)


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
    ap.add_argument(
        "--strict-gate",
        action="store_true",
        help="Treat gate stage failure as fatal. Default is non-blocking gate.",
    )
    ap.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="Continue remaining stages even if a non-gate stage fails.",
    )
    ap.add_argument("--overwrite-uni", action="store_true", help="Pass --overwrite to uni stage.")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg_path = _resolve_path(project_root, args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = _load_cfg(cfg_path)
    run_cfg = cfg.get("run", {})
    strict_gate_cfg = bool(run_cfg.get("strict_gate", False))
    continue_on_fail_cfg = bool(run_cfg.get("continue_on_fail", True))
    strict_gate = bool(args.strict_gate or strict_gate_cfg)
    continue_on_fail = bool(args.continue_on_fail or continue_on_fail_cfg)

    paths_cfg = cfg.get("paths", {})
    log_root = _resolve_path(project_root, paths_cfg.get("out_dir", "data/out"))
    logger, interactive = stage_logger("e2e", log_root, verbose=False)
    mask_summary = _resolve_path(project_root, paths_cfg.get("mask_dir", "data/masks")) / "mask_summary.csv"
    qc_summary = _resolve_path(project_root, paths_cfg.get("out_dir", "data/out")) / "qc" / "run_summary.csv"

    stages = _parse_stages(args.stages)
    python = sys.executable
    timings: dict[str, float] = {}

    logger.info("=== E2E PIPELINE ===")
    logger.info(f"Config: {cfg_path}")
    logger.info(f"Stages: {stages}")
    if args.n_slides is not None:
        logger.info(f"n_slides override: {args.n_slides}")
    if args.multi_worker_mode:
        logger.info("multi_worker_mode: enabled")
    if args.cpu_workers is not None:
        logger.info(f"cpu_workers override: {args.cpu_workers}")
    if args.io_workers is not None:
        logger.info(f"io_workers override: {args.io_workers}")
    if args.smoke_gate:
        logger.info("smoke_gate: enabled (dataset-pass-min=1, digital-pass-min=0)")
    logger.info(f"strict_gate: {strict_gate}")
    logger.info(f"continue_on_fail: {continue_on_fail}")

    stage_codes: dict[str, int] = {}
    hard_fail_code = 0

    for stage in progress(stages, interactive=interactive, desc="[e2e] stages", unit="stage"):
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
                "--ok-statuses",
                "ok,high_tissue",
            ]
            if strict_gate:
                cmd += ["--mode", "strict"]
                if args.smoke_gate:
                    cmd += ["--dataset-pass-min", "1", "--digital-pass-min", "0"]
            else:
                cmd += ["--mode", "adaptive"]
                if args.smoke_gate:
                    cmd += ["--dataset-pass-rate", "0.20", "--digital-pass-rate", "0.0"]
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

        dt, code = _run_stage(stage, cmd, cwd=project_root, logger=logger)
        timings[stage] = dt
        stage_codes[stage] = int(code)
        if code == 0:
            continue

        if stage == "gate" and (not strict_gate):
            logger.warning(
                "[warn] gate failed but pipeline will continue "
                "(non-blocking gate mode)."
            )
            logger.warning(
                "[warn] If this was a pilot run, use --smoke-gate for relaxed "
                "gate thresholds, or run --stages mask,qc,tumor,uni."
            )
            continue

        if continue_on_fail:
            if hard_fail_code == 0:
                hard_fail_code = int(code)
            logger.warning(
                f"[warn] stage '{stage}' failed (exit={code}) "
                "but continuing due to --continue-on-fail."
            )
            continue

        raise SystemExit(code)

    total = float(sum(timings.values()))
    logger.info("=== E2E DONE ===")
    for k in stages:
        code = int(stage_codes.get(k, 0))
        logger.info(f"{k}: {timings[k]:.1f}s exit={code}")
    logger.info(f"total: {total:.1f}s")
    if hard_fail_code != 0:
        raise SystemExit(hard_fail_code)


if __name__ == "__main__":
    main()

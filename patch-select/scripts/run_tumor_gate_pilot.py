from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torchvision import models
from torchstain.torch.normalizers.macenko import TorchMacenkoNormalizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io.wsi import open_wsi
from src.utils.runlog import PeriodicProgress, progress, stage_logger
from src.utils.slides import list_slide_records, slide_key_from_row, slide_match


WSI_EXTS = (".svs", ".ndpi", ".mrxs", ".tif", ".tiff")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


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


def _to_int(x, default: int = 0) -> int:
    if x is None:
        return int(default)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return int(default)
        return int(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return int(default)
    try:
        return int(float(s))
    except Exception:
        return int(default)


def _root_summary_path(stage_root: Path, main_out_root: Path, stage_name: str) -> Path:
    if stage_root.resolve() == main_out_root.resolve():
        return stage_root / f"{stage_name}_run_summary.csv"
    return stage_root / "run_summary.csv"


def list_wsi_files(wsi_dirs: list[Path], recursive: bool = True) -> list[dict[str, str]]:
    return list_slide_records(wsi_dirs, recursive=recursive, exts=WSI_EXTS)


def pil_to_tensor255_chw(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1)


def tensor_to_pil_safe(x: torch.Tensor) -> Image.Image:
    if isinstance(x, (tuple, list)):
        x = x[0]
    x = x.detach().cpu()

    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got {tuple(x.shape)}")
    if x.shape[0] != 3 and x.shape[-1] == 3:
        x = x.permute(2, 0, 1)
    if x.shape[0] != 3:
        raise ValueError(f"Expected 3 channels, got {tuple(x.shape)}")

    x = x.float()
    mx = float(x.max().item()) if x.numel() else 1.0
    if mx <= 5.0:
        x = (x.clamp(0, 1) * 255.0).round()
    else:
        x = x.clamp(0, 255).round()

    arr = x.to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(arr, mode="RGB")


def to_model_tensor(x: torch.Tensor) -> torch.Tensor:
    if isinstance(x, (tuple, list)):
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got {tuple(x.shape)}")
    if x.shape[0] != 3 and x.shape[-1] == 3:
        x = x.permute(2, 0, 1)
    if x.shape[0] != 3:
        raise ValueError(f"Expected 3 channels, got {tuple(x.shape)}")

    x = x.float()
    if float(x.max().item()) > 5.0:
        x = x / 255.0
    else:
        x = x.clamp(0.0, 1.0)

    mean = IMAGENET_MEAN.to(x.device)
    std = IMAGENET_STD.to(x.device)
    return (x - mean) / std


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_path: Path, device: str):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    try:
        state = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


def build_normalizer(reference_tile_path: Path, norm_device: str):
    ref_img = Image.open(reference_tile_path).convert("RGB")
    ref_tensor = pil_to_tensor255_chw(ref_img).to(norm_device)

    normalizer = TorchMacenkoNormalizer()
    if hasattr(normalizer, "to"):
        normalizer = normalizer.to(norm_device)
    normalizer.fit(ref_tensor)
    return normalizer


def load_qc_run_summary(out_dir: Path) -> dict[str, dict]:
    run_summary = out_dir / "qc" / "run_summary.csv"
    if not run_summary.exists():
        return {}
    df = pd.read_csv(run_summary, dtype={"slide_id": "string", "slide_uid": "string"})
    if "slide_id" not in df.columns and "slide_uid" not in df.columns:
        return {}
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        out[slide_key_from_row(dict(row))] = dict(row)
    return out


def _apply_uni_gate_fields(row: dict, cfg: dict) -> dict:
    min_high = int(cfg["min_high_tumor_patches_for_uni"])
    min_scored = int(cfg["min_scored_patches_for_uni"])
    min_ratio = float(cfg["min_high_ratio_for_uni"])
    high_only = int(row.get("high_tumor_only_count", 0))
    scored_ok = int(row.get("scored_ok", 0))
    high_ratio = float(row.get("high_ratio", 0.0))

    if high_only < min_high:
        ready = False
        reason = f"high_tumor_only<{min_high}"
    elif scored_ok < min_scored:
        ready = False
        reason = f"scored_ok<{min_scored}"
    elif high_ratio < min_ratio:
        ready = False
        reason = f"high_ratio<{min_ratio:.2f}"
    else:
        ready = True
        reason = ""

    row["uni_ready"] = bool(ready)
    row["uni_discard_reason"] = reason
    row["uni_min_high_tumor_patches"] = min_high
    row["uni_min_scored_patches"] = min_scored
    row["uni_min_high_ratio"] = min_ratio
    return row


def _write_scored_outputs(out_slide_dir: Path, scored: pd.DataFrame, cfg: dict) -> dict:
    topk = int(cfg["topk"])
    tumor_thr = float(cfg["tumor_thr"])
    export_high_only = bool(cfg["export_high_only"])
    high_only_source = str(cfg["high_only_source"])
    if high_only_source not in {"all_scored", "topk_scored"}:
        high_only_source = "all_scored"

    scored = scored.sort_values("tumor_prob", ascending=False).reset_index(drop=True)
    kept_topk = min(topk, len(scored))
    topk_df = scored.head(kept_topk).copy()

    scored.to_csv(out_slide_dir / "all_scored.csv", index=False)
    topk_df.to_csv(out_slide_dir / "topk_scored.csv", index=False)
    topk_coords = (
        topk_df[["x0", "y0"]].to_numpy(dtype=np.int32)
        if kept_topk > 0
        else np.zeros((0, 2), dtype=np.int32)
    )
    np.save(out_slide_dir / "topk_coords_level0.npy", topk_coords)

    if high_only_source == "topk_scored":
        high_base = topk_df
    else:
        high_base = scored
    high_df = high_base[high_base["tumor_prob"] >= tumor_thr].copy().reset_index(drop=True)
    high_coords = (
        high_df[["x0", "y0"]].to_numpy(dtype=np.int32)
        if len(high_df) > 0
        else np.zeros((0, 2), dtype=np.int32)
    )
    if export_high_only:
        high_df.to_csv(out_slide_dir / "high_tumor_only.csv", index=False)
        np.save(out_slide_dir / "high_tumor_coords_level0.npy", high_coords)

    return {
        "scored_df": scored,
        "topk_df": topk_df,
        "kept_topk": int(kept_topk),
        "high_tumor_only_count": int(len(high_df)),
        "high_only_source": high_only_source,
    }


def _read_filter_source(out_slide_dir: Path) -> pd.DataFrame | None:
    all_scored = out_slide_dir / "all_scored.csv"
    topk_scored = out_slide_dir / "topk_scored.csv"
    if all_scored.exists():
        return pd.read_csv(all_scored)
    if topk_scored.exists():
        return pd.read_csv(topk_scored)
    return None


def score_from_tiles(
    sid: str,
    slide_uid: str,
    meta: pd.DataFrame,
    tiles: np.ndarray,
    out_slide_dir: Path,
    cfg: dict,
    model,
    normalizer,
    model_device: str,
    norm_device: str,
    interactive: bool = False,
) -> dict:
    out_slide_dir.mkdir(parents=True, exist_ok=True)

    topk = int(cfg["topk"])
    batch_size = int(cfg["batch_size"])
    tumor_thr = float(cfg["tumor_thr"])
    mid_thr = float(cfg["mid_thr"])
    save_preview = bool(cfg["save_preview_patches"])
    save_preview_limit = int(cfg["save_preview_limit"])
    save_preview_normalized = bool(cfg["save_preview_normalized"])
    preview_dir = out_slide_dir / "preview_top"
    if save_preview and save_preview_limit > 0:
        preview_dir.mkdir(parents=True, exist_ok=True)

    n_input = int(len(meta))
    if n_input == 0:
        empty = meta.copy()
        empty["tumor_prob"] = []
        empty["tumor_band"] = []
        empty.to_csv(out_slide_dir / "all_scored.csv", index=False)
        np.save(out_slide_dir / "topk_coords_level0.npy", np.zeros((0, 2), dtype=np.int32))
        empty.to_csv(out_slide_dir / "topk_scored.csv", index=False)
        return {
            "slide_id": sid,
            "slide_uid": slide_uid,
            "selected_input": 0,
            "scored_ok": 0,
            "norm_or_read_fail": 0,
            "tumor_prob_ge_thr": 0,
            "tumor_prob_mid": 0,
            "tumor_prob_low": 0,
            "kept_topk": 0,
            "high_ratio": 0.0,
            "tumor_thr": tumor_thr,
            "mid_thr": mid_thr,
        }

    batch_imgs: list[torch.Tensor] = []
    batch_idx: list[int] = []
    scored_rows: list[int] = []
    probs_out: list[float] = []
    fail_count = 0

    for i in range(len(meta)):
        try:
            patch = tiles[i]
            pil = Image.fromarray(patch, mode="RGB")
            tensor = pil_to_tensor255_chw(pil).to(norm_device)
            out = normalizer.normalize(tensor)
            norm_t = out[0] if isinstance(out, (tuple, list)) else out
            model_t = to_model_tensor(norm_t).to(model_device)
        except Exception:
            fail_count += 1
            continue

        batch_imgs.append(model_t)
        batch_idx.append(i)
        if len(batch_imgs) == batch_size:
            batch_tensor = torch.stack(batch_imgs)
            with torch.no_grad():
                probs = torch.softmax(model(batch_tensor), dim=1)[:, 1].detach().cpu().numpy()
            probs_out.extend([float(p) for p in probs])
            scored_rows.extend(batch_idx)
            batch_imgs.clear()
            batch_idx.clear()

    if batch_imgs:
        batch_tensor = torch.stack(batch_imgs)
        with torch.no_grad():
            probs = torch.softmax(model(batch_tensor), dim=1)[:, 1].detach().cpu().numpy()
        probs_out.extend([float(p) for p in probs])
        scored_rows.extend(batch_idx)

    scored = meta.iloc[scored_rows].copy().reset_index(drop=True)
    scored["_src_idx"] = np.array(scored_rows, dtype=np.int32)
    scored["tumor_prob"] = np.array(probs_out, dtype=np.float32)
    scored["tumor_band"] = np.where(
        scored["tumor_prob"] >= tumor_thr,
        "high_tumor",
        np.where(scored["tumor_prob"] >= mid_thr, "mid_tumor", "low_tumor"),
    )
    scored = scored.sort_values("tumor_prob", ascending=False).reset_index(drop=True)

    write_out = _write_scored_outputs(out_slide_dir=out_slide_dir, scored=scored, cfg=cfg)
    topk_df = write_out["topk_df"]
    kept_topk = int(write_out["kept_topk"])
    high_tumor_only_count = int(write_out["high_tumor_only_count"])
    high_only_source = str(write_out["high_only_source"])

    if save_preview and save_preview_limit > 0 and kept_topk > 0:
        n_save = min(save_preview_limit, kept_topk)
        for rank, row in topk_df.head(n_save).iterrows():
            src_idx = int(row["_src_idx"])
            prob = float(row["tumor_prob"])
            x0 = int(row["x0"])
            y0 = int(row["y0"])
            img = Image.fromarray(tiles[src_idx], mode="RGB")
            if save_preview_normalized:
                try:
                    t = pil_to_tensor255_chw(img).to(norm_device)
                    out = normalizer.normalize(t)
                    t_norm = out[0] if isinstance(out, (tuple, list)) else out
                    img = tensor_to_pil_safe(t_norm)
                except Exception:
                    pass
            img.save(preview_dir / f"rank{rank:05d}__x{x0}__y{y0}__p{prob:.4f}.png")

    n_high = int((scored["tumor_prob"] >= tumor_thr).sum())
    n_mid = int(((scored["tumor_prob"] >= mid_thr) & (scored["tumor_prob"] < tumor_thr)).sum())
    n_low = int((scored["tumor_prob"] < mid_thr).sum())
    scored_ok = int(len(scored))
    high_ratio = float(n_high / max(scored_ok, 1))

    return {
        "slide_id": sid,
        "slide_uid": slide_uid,
        "selected_input": n_input,
        "scored_ok": scored_ok,
        "norm_or_read_fail": int(fail_count),
        "tumor_prob_ge_thr": n_high,
        "tumor_prob_mid": n_mid,
        "tumor_prob_low": n_low,
        "kept_topk": int(kept_topk),
        "high_tumor_only_count": high_tumor_only_count,
        "high_only_source": high_only_source,
        "high_ratio": high_ratio,
        "tumor_thr": tumor_thr,
        "mid_thr": mid_thr,
    }


def score_slide(
    slide_rec: dict[str, str],
    selected_meta_path: Path,
    out_slide_dir: Path,
    cfg: dict,
    model,
    normalizer,
    model_device: str,
    norm_device: str,
    interactive: bool = False,
) -> dict:
    slide_path = Path(str(slide_rec["path"]))
    sid = str(slide_rec["slide_id"])
    slide_uid = str(slide_rec["slide_uid"])
    out_slide_dir.mkdir(parents=True, exist_ok=True)

    topk = int(cfg["topk"])
    batch_size = int(cfg["batch_size"])
    tumor_thr = float(cfg["tumor_thr"])
    mid_thr = float(cfg["mid_thr"])

    meta = pd.read_csv(selected_meta_path)
    n_input = int(len(meta))
    if n_input == 0:
        empty = meta.copy()
        empty["tumor_prob"] = []
        empty["tumor_band"] = []
        empty.to_csv(out_slide_dir / "all_scored.csv", index=False)
        np.save(out_slide_dir / "topk_coords_level0.npy", np.zeros((0, 2), dtype=np.int32))
        empty.to_csv(out_slide_dir / "topk_scored.csv", index=False)
        return {
            "slide_id": sid,
            "slide_uid": slide_uid,
            "selected_input": 0,
            "scored_ok": 0,
            "norm_or_read_fail": 0,
            "tumor_prob_ge_thr": 0,
            "tumor_prob_mid": 0,
            "tumor_prob_low": 0,
            "kept_topk": 0,
            "high_ratio": 0.0,
            "tumor_thr": tumor_thr,
            "mid_thr": mid_thr,
        }

    wsi = open_wsi(str(slide_path))
    out_size = int(cfg["out_patch_size"])
    scale_factor = int(cfg["scale_factor"])
    save_preview = bool(cfg["save_preview_patches"])
    save_preview_limit = int(cfg["save_preview_limit"])
    save_preview_normalized = bool(cfg["save_preview_normalized"])
    preview_dir = out_slide_dir / "preview_top"
    if save_preview and save_preview_limit > 0:
        preview_dir.mkdir(parents=True, exist_ok=True)

    batch_imgs: list[torch.Tensor] = []
    batch_idx: list[int] = []
    scored_rows: list[int] = []
    probs_out: list[float] = []
    fail_count = 0

    try:
        coords = meta[["x0", "y0"]].to_numpy(dtype=np.int32)
        for i, (x0, y0) in enumerate(coords):
            try:
                patch = wsi.read_half_mag_patch(
                    int(x0), int(y0), out_size=out_size, scale_factor=scale_factor
                )
                pil = Image.fromarray(patch, mode="RGB")
                tensor = pil_to_tensor255_chw(pil).to(norm_device)
                out = normalizer.normalize(tensor)
                norm_t = out[0] if isinstance(out, (tuple, list)) else out
                model_t = to_model_tensor(norm_t).to(model_device)
            except Exception:
                fail_count += 1
                continue

            batch_imgs.append(model_t)
            batch_idx.append(i)

            if len(batch_imgs) == batch_size:
                batch_tensor = torch.stack(batch_imgs)
                with torch.no_grad():
                    probs = torch.softmax(model(batch_tensor), dim=1)[:, 1].detach().cpu().numpy()
                probs_out.extend([float(p) for p in probs])
                scored_rows.extend(batch_idx)
                batch_imgs.clear()
                batch_idx.clear()

        if batch_imgs:
            batch_tensor = torch.stack(batch_imgs)
            with torch.no_grad():
                probs = torch.softmax(model(batch_tensor), dim=1)[:, 1].detach().cpu().numpy()
            probs_out.extend([float(p) for p in probs])
            scored_rows.extend(batch_idx)
    finally:
        wsi.close()

    scored = meta.iloc[scored_rows].copy().reset_index(drop=True)
    scored["tumor_prob"] = np.array(probs_out, dtype=np.float32)
    scored["tumor_band"] = np.where(
        scored["tumor_prob"] >= tumor_thr,
        "high_tumor",
        np.where(scored["tumor_prob"] >= mid_thr, "mid_tumor", "low_tumor"),
    )
    scored = scored.sort_values("tumor_prob", ascending=False).reset_index(drop=True)

    write_out = _write_scored_outputs(out_slide_dir=out_slide_dir, scored=scored, cfg=cfg)
    topk_df = write_out["topk_df"]
    kept_topk = int(write_out["kept_topk"])
    high_tumor_only_count = int(write_out["high_tumor_only_count"])
    high_only_source = str(write_out["high_only_source"])

    if save_preview and save_preview_limit > 0 and kept_topk > 0:
        wsi2 = open_wsi(str(slide_path))
        try:
            n_save = min(save_preview_limit, kept_topk)
            for rank, row in topk_df.head(n_save).iterrows():
                x0 = int(row["x0"])
                y0 = int(row["y0"])
                prob = float(row["tumor_prob"])
                patch = wsi2.read_half_mag_patch(
                    x0, y0, out_size=out_size, scale_factor=scale_factor
                )
                img = Image.fromarray(patch, mode="RGB")
                if save_preview_normalized:
                    try:
                        t = pil_to_tensor255_chw(img).to(norm_device)
                        out = normalizer.normalize(t)
                        t_norm = out[0] if isinstance(out, (tuple, list)) else out
                        img = tensor_to_pil_safe(t_norm)
                    except Exception:
                        pass
                img.save(preview_dir / f"rank{rank:05d}__x{x0}__y{y0}__p{prob:.4f}.png")
        finally:
            wsi2.close()

    n_high = int((scored["tumor_prob"] >= tumor_thr).sum())
    n_mid = int(((scored["tumor_prob"] >= mid_thr) & (scored["tumor_prob"] < tumor_thr)).sum())
    n_low = int((scored["tumor_prob"] < mid_thr).sum())
    scored_ok = int(len(scored))
    high_ratio = float(n_high / max(scored_ok, 1))

    return {
        "slide_id": sid,
        "slide_uid": slide_uid,
        "selected_input": n_input,
        "scored_ok": scored_ok,
        "norm_or_read_fail": int(fail_count),
        "tumor_prob_ge_thr": n_high,
        "tumor_prob_mid": n_mid,
        "tumor_prob_low": n_low,
        "kept_topk": int(kept_topk),
        "high_tumor_only_count": high_tumor_only_count,
        "high_only_source": high_only_source,
        "high_ratio": high_ratio,
        "tumor_thr": tumor_thr,
        "mid_thr": mid_thr,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pilot-2: score patch-select outputs with Macenko+ResNet and keep top-k tumor-rich patches."
    )
    parser.add_argument("--config", default="configs/pilot.yaml", help="Path to pilot config.")
    parser.add_argument(
        "--n-slides",
        type=int,
        default=None,
        help="Override number of slides. <=0 means all slides.",
    )
    parser.add_argument(
        "--filter-only",
        action="store_true",
        help="Do not rescore. Read existing all_scored/topk_scored and export high_tumor_only + summary gate fields.",
    )
    parser.add_argument(
        "--multi-worker-mode",
        action="store_true",
        help="Enable worker overrides from run config for stage execution.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="CPU workers hint from run config (reserved for stage-level parallelism).",
    )
    parser.add_argument(
        "--io-workers",
        type=int,
        default=None,
        help="IO worker hint from run config (reserved for patch-loading paths).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing tumor-gate slide outputs; otherwise reuse existing scored csv when present.",
    )
    parser.add_argument("--slide-id", default=None, help="Optional single slide_id or slide_uid to run.")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed console logs.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg_path = _resolve_path(project_root, args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_cfg = cfg.get("run", {})
    cfg_multi = bool(run_cfg.get("multi_worker_mode", False))
    cfg_cpu_workers = int(run_cfg.get("cpu_workers", 1))
    cfg_io_workers = int(run_cfg.get("io_workers", 0))
    multi_worker_mode = bool(args.multi_worker_mode or cfg_multi)
    cpu_workers = int(args.workers) if args.workers is not None else cfg_cpu_workers
    io_workers = int(args.io_workers) if args.io_workers is not None else int(cfg_io_workers)
    cpu_workers = max(1, cpu_workers)
    io_workers = max(0, io_workers)

    raw_wsi_dirs = _as_list(cfg["paths"].get("wsi_dir", "data/raw_wsi"))
    wsi_recursive = bool(cfg["paths"].get("wsi_recursive", True))
    wsi_dirs = [_resolve_path(project_root, p) for p in raw_wsi_dirs]
    out_dir = _resolve_path(project_root, cfg["paths"]["out_dir"])
    for wsi_dir in wsi_dirs:
        if not wsi_dir.exists():
            raise FileNotFoundError(f"WSI dir not found: {wsi_dir}")

    tg_raw = cfg.get("tumor_gate", {})
    tg_cfg = {
        "topk": int(tg_raw.get("topk", 5000)),
        "tumor_thr": float(tg_raw.get("tumor_thr", 0.738)),
        "mid_thr": float(tg_raw.get("mid_thr", 0.50)),
        "batch_size": int(tg_raw.get("batch_size", 256)),
        "out_patch_size": int(cfg["wsi"]["out_patch_size"]),
        "scale_factor": int(cfg["wsi"]["scale_factor"]),
        "save_preview_patches": bool(tg_raw.get("save_preview_patches", True)),
        "save_preview_limit": int(tg_raw.get("save_preview_limit", 200)),
        "save_preview_normalized": bool(tg_raw.get("save_preview_normalized", True)),
        "io_workers": int(tg_raw.get("io_workers", io_workers)),
        "overwrite": bool(tg_raw.get("overwrite", False) or args.overwrite),
        "prefer_qc_pool": bool(tg_raw.get("prefer_qc_pool", True)),
        "export_high_only": bool(tg_raw.get("export_high_only", True)),
        "high_only_source": str(tg_raw.get("high_only_source", "all_scored")),
        "min_high_tumor_patches_for_uni": int(tg_raw.get("min_high_tumor_patches_for_uni", 200)),
        "min_scored_patches_for_uni": int(tg_raw.get("min_scored_patches_for_uni", 300)),
        "min_high_ratio_for_uni": float(tg_raw.get("min_high_ratio_for_uni", 0.20)),
    }

    tumor_out_root = _resolve_path(project_root, tg_raw.get("out_dir", "data/out/tumor_gate"))
    tumor_out_root.mkdir(parents=True, exist_ok=True)
    logger, interactive = stage_logger("tumor_gate", tumor_out_root, verbose=bool(args.verbose))

    slide_items = list_wsi_files(wsi_dirs, recursive=wsi_recursive)
    if len(slide_items) == 0:
        raise RuntimeError(f"No WSIs found in {wsi_dirs} with extensions {WSI_EXTS}")
    if args.slide_id:
        want = str(args.slide_id).strip()
        slide_items = [rec for rec in slide_items if slide_match(rec, want)]
        if not slide_items:
            raise RuntimeError(f"--slide-id '{want}' not found in WSI dirs.")

    n_from_cfg = int(cfg.get("run", {}).get("n_slides", 0))
    n_slides = n_from_cfg if args.n_slides is None else int(args.n_slides)
    if n_slides > 0:
        slide_items = slide_items[:n_slides]
    logger.info("=== TUMOR GATE ===")
    logger.info(f"WSI recursive scan: {wsi_recursive}")
    logger.info(f"Slides: {len(slide_items)} across {len({rec['center'] for rec in slide_items})} centers")

    if args.filter_only:
        model = None
        normalizer = None
        model_device = "na"
        norm_device = "na"
        logger.info(
            f"filter-only mode | topk={tg_cfg['topk']} "
            f"tumor_thr={tg_cfg['tumor_thr']:.3f} source={tg_cfg['high_only_source']}"
        )
    else:
        model_path = _resolve_path(
            project_root,
            tg_raw.get(
                "model_path",
                "../Tumor patch extraction/src/results/model_patch/best.pt",
            ),
        )
        reference_tile_path = _resolve_path(
            project_root,
            tg_raw.get(
                "reference_tile_path",
                "../Tumor patch extraction/src/ref_image/target_reference_clean.png",
            ),
        )
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not reference_tile_path.exists():
            raise FileNotFoundError(f"Reference tile not found: {reference_tile_path}")
        model_device = get_device()
        norm_device = "cpu" if model_device == "mps" else model_device
        logger.info(f"model_device={model_device} norm_device={norm_device}")
        logger.info(f"topk={tg_cfg['topk']} tumor_thr={tg_cfg['tumor_thr']:.3f}")
        logger.info(
            f"multi_worker_mode={multi_worker_mode} "
            f"cpu_workers={cpu_workers} io_workers={int(tg_cfg['io_workers'])} "
            f"overwrite={bool(tg_cfg['overwrite'])}"
        )
        model = load_model(model_path, model_device)
        normalizer = build_normalizer(reference_tile_path, norm_device)
    by_center: dict[str, list[dict[str, str]]] = {}
    for rec in slide_items:
        by_center.setdefault(str(rec["center"]), []).append(rec)

    rows: list[dict] = []
    reporter = PeriodicProgress(logger, "tumor-gate", total=len(slide_items), every=25)
    processed = 0
    for center, center_slides in progress(
        sorted(by_center.items()), interactive=interactive, desc="[tumor-gate] centers", unit="center"
    ):
        center_out_dir = out_dir / center
        coords_root = center_out_dir / "coords"
        qc_pool_root = center_out_dir / "qc_pool"
        qc_lookup = load_qc_run_summary(center_out_dir)
        center_tumor_out_root = tumor_out_root / center / "tumor_gate"
        center_tumor_out_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Center: {center} slides={len(center_slides)}")

        for rec in progress(center_slides, interactive=interactive, desc=f"[tumor-gate] {center} slides", unit="slide", leave=False):
            slide_path = Path(str(rec["path"]))
            sid = str(rec["slide_id"])
            slide_uid = str(rec["slide_uid"])
            slide_relpath = str(rec["slide_relpath"])
            selected_meta_path = coords_root / slide_uid / "selected_meta.csv"
            qc_pool_meta_path = qc_pool_root / slide_uid / "qc_meta.csv"
            qc_pool_tiles_path = qc_pool_root / slide_uid / "qc_tiles_uint8.npy"
            out_slide_dir = center_tumor_out_root / slide_uid
            try:
                used_input_source = "missing"
                if (not args.filter_only) and (not bool(tg_cfg["overwrite"])):
                    in_df = _read_filter_source(out_slide_dir)
                    if in_df is not None and "tumor_prob" in in_df.columns:
                        in_df = in_df.copy()
                        in_df["tumor_band"] = np.where(
                            in_df["tumor_prob"] >= tg_cfg["tumor_thr"],
                            "high_tumor",
                            np.where(in_df["tumor_prob"] >= tg_cfg["mid_thr"], "mid_tumor", "low_tumor"),
                        )
                        write_out = _write_scored_outputs(out_slide_dir=out_slide_dir, scored=in_df, cfg=tg_cfg)
                        scored_ok = int(len(in_df))
                        n_high = int((in_df["tumor_prob"] >= tg_cfg["tumor_thr"]).sum())
                        n_mid = int(
                            ((in_df["tumor_prob"] >= tg_cfg["mid_thr"]) & (in_df["tumor_prob"] < tg_cfg["tumor_thr"])).sum()
                        )
                        n_low = int((in_df["tumor_prob"] < tg_cfg["mid_thr"]).sum())
                        row = {
                            "slide_id": sid,
                            "slide_uid": slide_uid,
                            "slide_relpath": slide_relpath,
                            "selected_input": scored_ok,
                            "scored_ok": scored_ok,
                            "norm_or_read_fail": 0,
                            "tumor_prob_ge_thr": n_high,
                            "tumor_prob_mid": n_mid,
                            "tumor_prob_low": n_low,
                            "kept_topk": int(write_out["kept_topk"]),
                            "high_tumor_only_count": int(write_out["high_tumor_only_count"]),
                            "high_only_source": str(write_out["high_only_source"]),
                            "high_ratio": float(n_high / max(scored_ok, 1)),
                            "tumor_thr": tg_cfg["tumor_thr"],
                            "mid_thr": tg_cfg["mid_thr"],
                            "selected_meta_missing": False,
                        }
                        row = _apply_uni_gate_fields(row, tg_cfg)
                        used_input_source = "existing_scores_skip"
                    else:
                        row = None
                else:
                    row = None

                if row is not None:
                    pass
                elif args.filter_only:
                    in_df = _read_filter_source(out_slide_dir)
                    if in_df is None or "tumor_prob" not in in_df.columns:
                        row = {
                            "slide_id": sid,
                            "slide_uid": slide_uid,
                            "slide_relpath": slide_relpath,
                            "selected_input": 0,
                            "scored_ok": 0,
                            "norm_or_read_fail": 0,
                            "tumor_prob_ge_thr": 0,
                            "tumor_prob_mid": 0,
                            "tumor_prob_low": 0,
                            "kept_topk": 0,
                            "high_tumor_only_count": 0,
                            "high_only_source": tg_cfg["high_only_source"],
                            "high_ratio": 0.0,
                            "tumor_thr": tg_cfg["tumor_thr"],
                            "mid_thr": tg_cfg["mid_thr"],
                            "selected_meta_missing": True,
                        }
                    else:
                        in_df = in_df.copy()
                        in_df["tumor_band"] = np.where(
                            in_df["tumor_prob"] >= tg_cfg["tumor_thr"],
                            "high_tumor",
                            np.where(in_df["tumor_prob"] >= tg_cfg["mid_thr"], "mid_tumor", "low_tumor"),
                        )
                        write_out = _write_scored_outputs(out_slide_dir=out_slide_dir, scored=in_df, cfg=tg_cfg)
                        scored_ok = int(len(in_df))
                        n_high = int((in_df["tumor_prob"] >= tg_cfg["tumor_thr"]).sum())
                        n_mid = int(
                            ((in_df["tumor_prob"] >= tg_cfg["mid_thr"]) & (in_df["tumor_prob"] < tg_cfg["tumor_thr"])).sum()
                        )
                        n_low = int((in_df["tumor_prob"] < tg_cfg["mid_thr"]).sum())
                        row = {
                            "slide_id": sid,
                            "slide_uid": slide_uid,
                            "slide_relpath": slide_relpath,
                            "selected_input": scored_ok,
                            "scored_ok": scored_ok,
                            "norm_or_read_fail": 0,
                            "tumor_prob_ge_thr": n_high,
                            "tumor_prob_mid": n_mid,
                            "tumor_prob_low": n_low,
                            "kept_topk": int(write_out["kept_topk"]),
                            "high_tumor_only_count": int(write_out["high_tumor_only_count"]),
                            "high_only_source": str(write_out["high_only_source"]),
                            "high_ratio": float(n_high / max(scored_ok, 1)),
                            "tumor_thr": tg_cfg["tumor_thr"],
                            "mid_thr": tg_cfg["mid_thr"],
                            "selected_meta_missing": False,
                        }
                        row = _apply_uni_gate_fields(row, tg_cfg)
                    used_input_source = "filter_only_existing_scores"
                elif tg_cfg["prefer_qc_pool"] and qc_pool_meta_path.exists():
                    if qc_pool_tiles_path.exists():
                        qc_meta = pd.read_csv(qc_pool_meta_path)
                        qc_tiles = np.load(qc_pool_tiles_path, mmap_mode=None)
                        row = score_from_tiles(
                            sid=sid,
                            slide_uid=slide_uid,
                            meta=qc_meta,
                            tiles=qc_tiles,
                            out_slide_dir=out_slide_dir,
                            cfg=tg_cfg,
                            model=model,
                            normalizer=normalizer,
                            model_device=model_device,
                            norm_device=norm_device,
                            interactive=interactive,
                        )
                        used_input_source = "qc_pool_tiles"
                        del qc_tiles, qc_meta
                    else:
                        row = score_slide(
                            slide_rec=rec,
                            selected_meta_path=qc_pool_meta_path,
                            out_slide_dir=out_slide_dir,
                            cfg=tg_cfg,
                            model=model,
                            normalizer=normalizer,
                            model_device=model_device,
                            norm_device=norm_device,
                            interactive=interactive,
                        )
                        used_input_source = "qc_pool_coords"
                    row["selected_meta_missing"] = False
                    row = _apply_uni_gate_fields(row, tg_cfg)
                elif not selected_meta_path.exists():
                    row = {
                        "slide_id": sid,
                        "slide_uid": slide_uid,
                        "slide_relpath": slide_relpath,
                        "selected_input": 0,
                        "scored_ok": 0,
                        "norm_or_read_fail": 0,
                        "tumor_prob_ge_thr": 0,
                        "tumor_prob_mid": 0,
                        "tumor_prob_low": 0,
                        "kept_topk": 0,
                        "high_tumor_only_count": 0,
                        "high_only_source": tg_cfg["high_only_source"],
                        "high_ratio": 0.0,
                        "tumor_thr": tg_cfg["tumor_thr"],
                        "mid_thr": tg_cfg["mid_thr"],
                        "selected_meta_missing": True,
                    }
                    row = _apply_uni_gate_fields(row, tg_cfg)
                else:
                    row = score_slide(
                        slide_rec=rec,
                        selected_meta_path=selected_meta_path,
                        out_slide_dir=out_slide_dir,
                        cfg=tg_cfg,
                        model=model,
                        normalizer=normalizer,
                        model_device=model_device,
                        norm_device=norm_device,
                        interactive=interactive,
                    )
                    row["selected_meta_missing"] = False
                    used_input_source = "selected_meta_wsi_read"
                    row = _apply_uni_gate_fields(row, tg_cfg)

                qc_row = qc_lookup.get(slide_uid) or qc_lookup.get(sid, {})
                row.setdefault("slide_uid", slide_uid)
                row.setdefault("slide_relpath", slide_relpath)
                row["center"] = str(center)
                row["all_candidates_total"] = _to_int(qc_row.get("candidates_after_mask", 0), 0)
                row["after_mask_qc"] = _to_int(qc_row.get("qc_pass", 0), 0)
                row["selected_count_stage1"] = _to_int(qc_row.get("selected_count", 0), 0)
                row["input_source"] = used_input_source
                rows.append(row)
                processed += 1

                logger.debug(
                    f"[tumor-gate] {slide_uid}: total={row['all_candidates_total']} "
                    f"qc={row['after_mask_qc']} selected={row['selected_input']} "
                    f"high={row['tumor_prob_ge_thr']} kept={row['kept_topk']}"
                )
            except Exception as e:
                processed += 1
                rows.append(
                    {
                        "slide_id": sid,
                        "slide_uid": slide_uid,
                        "slide_relpath": slide_relpath,
                        "center": str(center),
                        "selected_input": 0,
                        "scored_ok": 0,
                        "norm_or_read_fail": 0,
                        "tumor_prob_ge_thr": 0,
                        "tumor_prob_mid": 0,
                        "tumor_prob_low": 0,
                        "kept_topk": 0,
                        "high_tumor_only_count": 0,
                        "high_ratio": 0.0,
                        "tumor_thr": tg_cfg["tumor_thr"],
                        "mid_thr": tg_cfg["mid_thr"],
                        "selected_meta_missing": False,
                        "input_source": "failed_exception",
                        "uni_ready": False,
                        "uni_discard_reason": f"{type(e).__name__}: {e}",
                        "all_candidates_total": 0,
                        "after_mask_qc": 0,
                        "selected_count_stage1": 0,
                    }
                )
                logger.error(f"[tumor-gate] {slide_uid}: unexpected failure {type(e).__name__}: {e}")
            finally:
                gc.collect()
                if model_device == "cuda":
                    torch.cuda.empty_cache()
                reporter.update(processed, ready=sum(1 for r in rows if bool(r.get("uni_ready", False))), total=len(rows))

        center_rows = [r for r in rows if str(r.get("center", "")) == str(center)]
        if center_rows:
            center_df = pd.DataFrame(center_rows).sort_values(["center", "slide_id"]).reset_index(drop=True)
            center_csv = center_tumor_out_root / "run_summary.csv"
            center_df.to_csv(center_csv, index=False)

    if rows:
        df = pd.DataFrame(rows).sort_values(["center", "slide_id"]).reset_index(drop=True)
        out_csv = _root_summary_path(tumor_out_root, out_dir, "tumor_gate")
        if out_csv.exists():
            prev = pd.read_csv(out_csv, dtype={"slide_id": "string", "slide_uid": "string"})
            prev["slide_id"] = prev["slide_id"].astype(str).str.strip()
            if "center" in prev.columns:
                prev["center"] = prev["center"].astype(str).str.strip()
            if "slide_uid" in prev.columns:
                prev["slide_uid"] = prev["slide_uid"].astype(str).str.strip()
            df["slide_id"] = df["slide_id"].astype(str).str.strip()
            if "center" in df.columns:
                df["center"] = df["center"].astype(str).str.strip()
            if "slide_uid" in df.columns:
                df["slide_uid"] = df["slide_uid"].astype(str).str.strip()
            subset = ["slide_uid"] if "slide_uid" in df.columns else (["center", "slide_id"] if "center" in df.columns else ["slide_id"])
            df = pd.concat([prev, df], ignore_index=True, sort=False)
            df = df.drop_duplicates(subset=subset, keep="last").sort_values(subset).reset_index(drop=True)
        df.to_csv(out_csv, index=False)
        logger.info(f"Run summary -> {out_csv}")


if __name__ == "__main__":
    main()

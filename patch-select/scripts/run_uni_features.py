from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torchstain.torch.normalizers.macenko import TorchMacenkoNormalizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io.wsi import open_wsi

WSI_EXTS = (".svs", ".ndpi", ".mrxs", ".tif", ".tiff")


class _WSICoordDataset(Dataset):
    def __init__(self, slide_path: Path, coords: np.ndarray, out_size: int, scale_factor: int):
        self.slide_path = str(slide_path)
        self.coords = np.asarray(coords, dtype=np.int32)
        self.out_size = int(out_size)
        self.scale_factor = int(scale_factor)
        self._wsi = None

    def _get_wsi(self):
        if self._wsi is None:
            self._wsi = open_wsi(self.slide_path)
        return self._wsi

    def __len__(self) -> int:
        return int(len(self.coords))

    def __getitem__(self, idx: int):
        x0, y0 = self.coords[int(idx)]
        try:
            patch = self._get_wsi().read_half_mag_patch(
                int(x0), int(y0), out_size=self.out_size, scale_factor=self.scale_factor
            )
            return int(idx), patch
        except Exception:
            return int(idx), None

    def __del__(self) -> None:
        try:
            if self._wsi is not None:
                self._wsi.close()
        except Exception:
            pass


def _collate_wsi_patch_batch(batch):
    good_idx: list[int] = []
    good_patches: list[np.ndarray] = []
    failed = 0
    for idx, patch in batch:
        if patch is None:
            failed += 1
            continue
        good_idx.append(int(idx))
        good_patches.append(np.asarray(patch, dtype=np.uint8))
    if not good_patches:
        empty = np.zeros((0, 0, 0, 3), dtype=np.uint8)
        return np.asarray(good_idx, dtype=np.int32), empty, int(failed)
    return np.asarray(good_idx, dtype=np.int32), np.stack(good_patches, axis=0), int(failed)


def _resolve_path(project_root: Path, p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def _to_int(x: Any, default: int = 0) -> int:
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


def _to_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return float(default)
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return float(default)
        return float(x)
    if isinstance(x, (int, np.integer)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return float(default)
    try:
        return float(s)
    except Exception:
        return float(default)


def _to_bool(x: Any, default: bool = False) -> bool:
    if x is None:
        return bool(default)
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        return bool(int(x))
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return bool(default)
        return bool(int(x))
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n", "", "none", "nan"}:
        return False
    return bool(default)


def _as_list(v: Any) -> list:
    if isinstance(v, list):
        return v
    if v is None:
        return []
    return [v]


def list_wsi_files(wsi_dirs: list[Path], recursive: bool = True) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for wsi_dir in wsi_dirs:
        center = wsi_dir.name
        if recursive:
            files.extend([(p, center) for p in sorted(wsi_dir.rglob("*")) if p.is_file() and p.suffix.lower() in WSI_EXTS])
        else:
            files.extend([(p, center) for p in sorted(wsi_dir.iterdir()) if p.is_file() and p.suffix.lower() in WSI_EXTS])

    uniq: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for p, center in files:
        if p.stem not in seen:
            uniq.append((p, center))
            seen.add(p.stem)
    return uniq


def pil_to_tensor255_chw(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1)


def get_device(prefer_device: str = "auto") -> str:
    d = str(prefer_device).strip().lower()
    if d in {"cuda", "mps", "cpu"}:
        if d == "cuda" and torch.cuda.is_available():
            return "cuda"
        if d == "mps" and torch.backends.mps.is_available():
            return "mps"
        if d == "cpu":
            return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _read_coords_csv(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if source == "high_tumor_only":
        needed = {"x0", "y0", "tumor_prob"}
    elif source in {"all_scored", "topk_scored"}:
        needed = {"x0", "y0", "tumor_prob"}
    else:
        needed = {"x0", "y0"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing} in {path}")
    keep = [c for c in ["x0", "y0", "tumor_prob", "tumor_band"] if c in df.columns]
    out = df[keep].copy()
    out["x0"] = out["x0"].map(_to_int)
    out["y0"] = out["y0"].map(_to_int)
    if "tumor_prob" in out.columns:
        out["tumor_prob"] = out["tumor_prob"].map(_to_float)
    return out


def _load_tumor_summary(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype={"slide_id": "string"})
    if "slide_id" not in df.columns:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        sid = str(row["slide_id"]).strip()
        out[sid] = dict(row)
    return out


def _load_qc_summary(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype={"slide_id": "string"})
    if "slide_id" not in df.columns:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        sid = str(row["slide_id"]).strip()
        out[sid] = dict(row)
    return out


def _build_macenko_normalizer(reference_tile_path: Path, norm_device: str):
    ref_img = Image.open(reference_tile_path).convert("RGB")
    ref_tensor = pil_to_tensor255_chw(ref_img).to(norm_device)
    normalizer = TorchMacenkoNormalizer()
    if hasattr(normalizer, "to"):
        normalizer = normalizer.to(norm_device)
    normalizer.fit(ref_tensor)
    return normalizer


def _precheck(
    cfg_path: Path,
    wsi_dirs: list[Path],
    tumor_summary_path: Path,
    uni_repo_dir: Path,
    use_macenko: bool,
    reference_tile_path: Path,
) -> None:
    print("\n=== UNI PRECHECK ===")
    print(f"Config: {cfg_path}")
    for d in wsi_dirs:
        print(f"WSI dir: {d}")
        if not d.exists():
            raise FileNotFoundError(f"WSI dir not found: {d}")
    print(f"Tumor-gate summary: {tumor_summary_path}")
    if not tumor_summary_path.exists():
        raise FileNotFoundError(f"Tumor-gate run summary not found: {tumor_summary_path}")
    print(f"UNI repo dir: {uni_repo_dir}")
    if not uni_repo_dir.exists():
        raise FileNotFoundError(f"UNI repo not found: {uni_repo_dir}")
    if use_macenko:
        print(f"Macenko reference: {reference_tile_path}")
        if not reference_tile_path.exists():
            raise FileNotFoundError(f"Macenko reference tile not found: {reference_tile_path}")
    print("PRECHECK: PASS")


def main():
    ap = argparse.ArgumentParser(
        description="Extract UNI features from tumor-gated patch coordinates with optional Macenko normalization."
    )
    ap.add_argument("--config", default="configs/pilot.yaml", help="Path to config yaml.")
    ap.add_argument("--n-slides", type=int, default=None, help="Override number of slides. <=0 means all.")
    ap.add_argument("--slide-id", default=None, help="Optional single slide_id to run.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing done slides.")
    ap.add_argument(
        "--multi-worker-mode",
        action="store_true",
        help="Enable worker overrides for patch IO prefetch.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="CPU workers hint used when multi-worker mode is enabled.",
    )
    ap.add_argument(
        "--io-workers",
        type=int,
        default=None,
        help="Number of DataLoader workers for patch extraction from WSI.",
    )
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg_path = _resolve_path(project_root, args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    uni_raw = cfg.get("uni", {})
    if not uni_raw:
        raise KeyError("Missing `uni` section in config.")
    run_cfg = cfg.get("run", {})
    cfg_multi = bool(run_cfg.get("multi_worker_mode", False))
    cfg_cpu_workers = int(run_cfg.get("cpu_workers", 1))
    cfg_io_workers = int(run_cfg.get("io_workers", 0))
    cli_cpu_workers = int(args.workers) if args.workers is not None else cfg_cpu_workers
    multi_worker_mode = bool(args.multi_worker_mode or cfg_multi)

    wsi_dirs = [_resolve_path(project_root, p) for p in _as_list(cfg["paths"]["wsi_dir"])]
    wsi_recursive = bool(cfg.get("paths", {}).get("wsi_recursive", True))
    out_root = _resolve_path(project_root, cfg["paths"]["out_dir"])
    tumor_out_root = _resolve_path(project_root, cfg.get("tumor_gate", {}).get("out_dir", "data/out/tumor_gate"))
    uni_out_root = _resolve_path(project_root, uni_raw.get("out_dir", "data/out/uni"))
    uni_out_root.mkdir(parents=True, exist_ok=True)

    tumor_summary_path = tumor_out_root / "run_summary.csv"
    qc_summary_path = out_root / "qc" / "run_summary.csv"
    tumor_lookup = _load_tumor_summary(tumor_summary_path)
    qc_lookup = _load_qc_summary(qc_summary_path)

    uni_repo_dir = _resolve_path(project_root, uni_raw.get("uni_repo_dir", "../UNI"))
    use_macenko = bool(uni_raw.get("use_macenko", True))
    reference_tile_path = _resolve_path(
        project_root,
        uni_raw.get("reference_tile_path", "../Tumor patch extraction/src/ref_image/target_reference_clean.png"),
    )
    _precheck(
        cfg_path=cfg_path,
        wsi_dirs=wsi_dirs,
        tumor_summary_path=tumor_summary_path,
        uni_repo_dir=uni_repo_dir,
        use_macenko=use_macenko,
        reference_tile_path=reference_tile_path,
    )

    # Make UNI package importable from local checkout.
    sys.path.insert(0, str(uni_repo_dir))
    from uni.get_encoder import get_encoder  # type: ignore

    encoder_name = str(uni_raw.get("encoder_name", "uni")).strip()
    encoder_ckpt = str(uni_raw.get("checkpoint", "pytorch_model.bin")).strip()
    encoder_assets_dir = _resolve_path(
        project_root,
        uni_raw.get("assets_dir", str((uni_repo_dir / "assets" / "ckpts").resolve())),
    )

    out_patch_size = int(cfg["wsi"].get("out_patch_size", 224))
    scale_factor = int(cfg["wsi"].get("scale_factor", 2))
    batch_size = int(uni_raw.get("batch_size", 128))
    io_workers_cfg = int(uni_raw.get("io_workers", cfg_io_workers))
    io_workers = int(args.io_workers) if args.io_workers is not None else io_workers_cfg
    prefer_device = str(uni_raw.get("device", "auto")).strip()
    model_device = get_device(prefer_device)
    torch_device = torch.device(model_device)
    norm_device = "cpu" if model_device == "mps" else model_device
    if multi_worker_mode and io_workers <= 0:
        io_workers = max(1, min(8, int(cli_cpu_workers)))
    io_workers = max(0, int(io_workers))

    print("\n=== UNI SETUP ===")
    print(f"Encoder: {encoder_name} checkpoint={encoder_ckpt}")
    print(f"Assets dir: {encoder_assets_dir}")
    print(f"Model device: {model_device} | Norm device: {norm_device}")
    print(f"Patch size: {out_patch_size} | Scale factor: {scale_factor} | Batch size: {batch_size}")
    print(f"Multi-worker mode: {multi_worker_mode} | io_workers={io_workers}")
    print(f"WSI recursive scan: {wsi_recursive}")

    model, eval_transform = get_encoder(
        enc_name=encoder_name,
        checkpoint=encoder_ckpt,
        img_resize=out_patch_size,
        center_crop=False,
        device=torch_device,
        assets_dir=str(encoder_assets_dir),
        test_batch=1,
    )
    if model is None or eval_transform is None:
        raise RuntimeError(f"Failed to initialize UNI encoder={encoder_name}")

    macenko = None
    if use_macenko:
        macenko = _build_macenko_normalizer(reference_tile_path=reference_tile_path, norm_device=norm_device)
        print("Macenko normalizer: enabled")
    else:
        print("Macenko normalizer: disabled")

    slide_items = list_wsi_files(wsi_dirs, recursive=wsi_recursive)
    if len(slide_items) == 0:
        raise RuntimeError(f"No WSIs found in {wsi_dirs} with extensions {WSI_EXTS}")
    if args.slide_id:
        slide_id = str(args.slide_id).strip()
        slide_items = [(p, center) for p, center in slide_items if p.stem == slide_id]
        if not slide_items:
            raise RuntimeError(f"--slide-id '{slide_id}' not found in WSI dirs.")

    n_cfg = int(cfg.get("run", {}).get("n_slides", 0))
    n_slides = int(args.n_slides) if args.n_slides is not None else n_cfg
    if n_slides > 0:
        slide_items = slide_items[:n_slides]

    coord_source = str(uni_raw.get("coord_source", "high_tumor_only")).strip()
    use_uni_ready_only = bool(uni_raw.get("use_uni_ready_only", True))
    max_patches = int(uni_raw.get("max_patches_per_slide", 5000))
    min_ok_patches = int(uni_raw.get("min_ok_patches_per_slide", 200))
    output_dtype = str(uni_raw.get("output_dtype", "float16")).strip().lower()
    overwrite = bool(uni_raw.get("overwrite", False) or args.overwrite)
    merge_run_summary = bool(uni_raw.get("merge_run_summary", True))

    print("\n=== UNI RUN ===")
    print(f"Slides to consider: {len(slide_items)}")
    print(f"Coord source: {coord_source} | use_uni_ready_only={use_uni_ready_only}")
    print(f"max_patches_per_slide={max_patches} min_ok_patches_per_slide={min_ok_patches}")
    print(f"output_dtype={output_dtype} overwrite={overwrite} merge_run_summary={merge_run_summary}")

    rows: list[dict[str, Any]] = []
    for slide_path, center in tqdm(slide_items, desc="[uni] slides", unit="slide", dynamic_ncols=True):
        sid = slide_path.stem
        t0 = time.time()
        row: dict[str, Any] = {
            "slide_id": sid,
            "center": str(center),
            "status": "unknown",
            "reason": "",
            "elapsed_sec": np.nan,
            "input_coords": 0,
            "processed_ok": 0,
            "failed": 0,
            "feature_dim": 0,
            "uni_feature_ready": False,
            "multi_worker_mode": bool(multi_worker_mode),
            "io_workers": int(io_workers),
        }

        tumor_row = tumor_lookup.get(sid, {})
        uni_ready_in = _to_bool(tumor_row.get("uni_ready", False), False)
        row["uni_ready_input"] = uni_ready_in
        row["high_tumor_only_count"] = _to_int(tumor_row.get("high_tumor_only_count", 0), 0)
        row["selected_input"] = _to_int(tumor_row.get("selected_input", 0), 0)
        row["after_mask_qc"] = _to_int(tumor_row.get("after_mask_qc", 0), 0)
        row["all_candidates_total"] = _to_int(tumor_row.get("all_candidates_total", 0), 0)
        row["mask_status_effective"] = str(qc_lookup.get(sid, {}).get("mask_status_effective", ""))

        out_slide_dir = uni_out_root / str(center) / "uni" / sid
        out_slide_dir.mkdir(parents=True, exist_ok=True)
        done_path = out_slide_dir / "done.json"
        feat_path = out_slide_dir / "patch_features.npy"
        meta_path = out_slide_dir / "patch_meta.csv"
        coords_csv_path = out_slide_dir / "coords_level0.csv"
        coords_npy_path = out_slide_dir / "coords_level0.npy"
        slide_feat_path = out_slide_dir / "slide_feature.npy"

        if done_path.exists() and feat_path.exists() and (not overwrite):
            try:
                with done_path.open("r", encoding="utf-8") as f:
                    done = json.load(f)
                row.update(
                    {
                        "status": "skipped_done",
                        "reason": "done_json_exists",
                        "input_coords": _to_int(done.get("input_coords", 0)),
                        "processed_ok": _to_int(done.get("processed_ok", 0)),
                        "failed": _to_int(done.get("failed", 0)),
                        "feature_dim": _to_int(done.get("feature_dim", 0)),
                        "uni_feature_ready": bool(done.get("uni_feature_ready", False)),
                    }
                )
                row["elapsed_sec"] = float(time.time() - t0)
                rows.append(row)
                print(f"[uni] {sid}: skip(done) processed={row['processed_ok']} dim={row['feature_dim']}")
                continue
            except Exception:
                pass

        if use_uni_ready_only and (not uni_ready_in):
            row["status"] = "skipped_not_ready"
            row["reason"] = "uni_ready_input_false"
            row["elapsed_sec"] = float(time.time() - t0)
            rows.append(row)
            print(f"[uni] {sid}: skip(uni_ready=false)")
            continue

        coord_center_root = tumor_out_root / str(center) / "tumor_gate" / sid
        if coord_source == "high_tumor_only":
            coord_path = coord_center_root / "high_tumor_only.csv"
            coord_path_fallback = tumor_out_root / sid / "high_tumor_only.csv"
        elif coord_source == "topk_scored":
            coord_path = coord_center_root / "topk_scored.csv"
            coord_path_fallback = tumor_out_root / sid / "topk_scored.csv"
        elif coord_source == "all_scored":
            coord_path = coord_center_root / "all_scored.csv"
            coord_path_fallback = tumor_out_root / sid / "all_scored.csv"
        else:
            raise ValueError(f"Unsupported uni.coord_source={coord_source}")

        if (not coord_path.exists()) and coord_path_fallback.exists():
            coord_path = coord_path_fallback

        if not coord_path.exists():
            row["status"] = "skipped_missing_coords"
            row["reason"] = str(coord_path)
            row["elapsed_sec"] = float(time.time() - t0)
            rows.append(row)
            print(f"[uni] {sid}: skip(missing coords)")
            continue

        try:
            coords_df = _read_coords_csv(coord_path, source=coord_source)
        except Exception as e:
            row["status"] = "failed_bad_coords"
            row["reason"] = f"{type(e).__name__}: {e}"
            row["elapsed_sec"] = float(time.time() - t0)
            rows.append(row)
            print(f"[uni] {sid}: fail(read coords) {e}")
            continue

        if len(coords_df) == 0:
            row["status"] = "skipped_empty_coords"
            row["reason"] = "0_coords"
            row["elapsed_sec"] = float(time.time() - t0)
            rows.append(row)
            print(f"[uni] {sid}: skip(empty coords)")
            continue

        if max_patches > 0 and len(coords_df) > max_patches:
            coords_df = coords_df.head(max_patches).copy()
        row["input_coords"] = int(len(coords_df))

        features_batches: list[np.ndarray] = []
        keep_idx: list[int] = []
        fail_count = 0
        tensors: list[torch.Tensor] = []
        idxs: list[int] = []

        def flush_batch() -> None:
            nonlocal tensors, idxs, features_batches
            if not tensors:
                return
            batch = torch.stack(tensors).to(torch_device)
            with torch.inference_mode():
                out = model(batch).detach().cpu().numpy()
            features_batches.append(out.astype(np.float32))
            keep_idx.extend(idxs)
            tensors = []
            idxs = []

        if io_workers > 0:
            coords_arr = coords_df[["x0", "y0"]].to_numpy(dtype=np.int32)
            dataset = _WSICoordDataset(
                slide_path=slide_path,
                coords=coords_arr,
                out_size=out_patch_size,
                scale_factor=scale_factor,
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=int(io_workers),
                pin_memory=bool(model_device == "cuda"),
                collate_fn=_collate_wsi_patch_batch,
                persistent_workers=bool(io_workers > 0),
            )
            for batch_pack in tqdm(loader, desc=f"[uni] {sid} batches", unit="batch", dynamic_ncols=True, leave=False):
                if batch_pack is None:
                    continue
                batch_idxs, patches, batch_fail = batch_pack
                fail_count += int(batch_fail)
                for local_i, patch in enumerate(patches):
                    src_idx = int(batch_idxs[local_i])
                    try:
                        pil = Image.fromarray(np.asarray(patch, dtype=np.uint8), mode="RGB")
                        if macenko is not None:
                            tensor255 = pil_to_tensor255_chw(pil).to(norm_device)
                            norm_out = macenko.normalize(tensor255)
                            norm_t = norm_out[0] if isinstance(norm_out, (tuple, list)) else norm_out
                            norm_t = norm_t.detach().cpu()
                            if norm_t.ndim == 3 and norm_t.shape[0] == 3:
                                arr = norm_t.permute(1, 2, 0).numpy()
                            else:
                                arr = norm_t.numpy()
                            if arr.max() <= 5.0:
                                arr = np.clip(arr, 0.0, 1.0) * 255.0
                            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
                            pil = Image.fromarray(arr, mode="RGB")
                        t = eval_transform(pil)
                        tensors.append(t)
                        idxs.append(src_idx)
                    except Exception:
                        fail_count += 1
                        continue
                flush_batch()
        else:
            try:
                wsi = open_wsi(str(slide_path))
            except Exception as e:
                row["status"] = "failed_open_wsi"
                row["reason"] = f"{type(e).__name__}: {e}"
                row["elapsed_sec"] = float(time.time() - t0)
                rows.append(row)
                print(f"[uni] {sid}: fail(open wsi) {e}")
                continue

            try:
                for i, r in tqdm(
                    coords_df.iterrows(),
                    total=len(coords_df),
                    desc=f"[uni] {sid} patches",
                    unit="patch",
                    dynamic_ncols=True,
                    leave=False,
                ):
                    x0 = _to_int(r.get("x0", 0), 0)
                    y0 = _to_int(r.get("y0", 0), 0)
                    try:
                        patch = wsi.read_half_mag_patch(x0, y0, out_size=out_patch_size, scale_factor=scale_factor)
                        pil = Image.fromarray(patch, mode="RGB")
                        if macenko is not None:
                            tensor255 = pil_to_tensor255_chw(pil).to(norm_device)
                            norm_out = macenko.normalize(tensor255)
                            norm_t = norm_out[0] if isinstance(norm_out, (tuple, list)) else norm_out
                            norm_t = norm_t.detach().cpu()
                            if norm_t.ndim == 3 and norm_t.shape[0] == 3:
                                arr = norm_t.permute(1, 2, 0).numpy()
                            else:
                                arr = norm_t.numpy()
                            if arr.max() <= 5.0:
                                arr = np.clip(arr, 0.0, 1.0) * 255.0
                            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
                            pil = Image.fromarray(arr, mode="RGB")

                        t = eval_transform(pil)
                        tensors.append(t)
                        idxs.append(i)
                        if len(tensors) >= batch_size:
                            flush_batch()
                    except Exception:
                        fail_count += 1
                        continue
                flush_batch()
            finally:
                wsi.close()

        if features_batches:
            feats = np.vstack(features_batches)
            kept = coords_df.iloc[keep_idx].copy().reset_index(drop=True)
        else:
            feats = np.zeros((0, 0), dtype=np.float32)
            kept = coords_df.iloc[:0].copy()

        dim = int(feats.shape[1]) if feats.ndim == 2 and feats.size > 0 else 0
        ok_count = int(feats.shape[0]) if feats.ndim == 2 else 0
        row["processed_ok"] = ok_count
        row["failed"] = int(fail_count)
        row["feature_dim"] = dim

        if ok_count == 0:
            row["status"] = "failed_no_features"
            row["reason"] = "0_features"
            row["uni_feature_ready"] = False
            row["elapsed_sec"] = float(time.time() - t0)
            rows.append(row)
            print(f"[uni] {sid}: fail(no features)")
            continue

        if output_dtype == "float16":
            feats_to_save = feats.astype(np.float16, copy=False)
        elif output_dtype == "float32":
            feats_to_save = feats.astype(np.float32, copy=False)
        else:
            raise ValueError(f"Unsupported uni.output_dtype={output_dtype}")
        np.save(feat_path, feats_to_save)

        kept.to_csv(meta_path, index=False)
        kept[["x0", "y0"]].to_csv(coords_csv_path, index=False)
        np.save(coords_npy_path, kept[["x0", "y0"]].to_numpy(dtype=np.int32))
        slide_feat = feats.astype(np.float32).mean(axis=0, keepdims=False)
        np.save(slide_feat_path, slide_feat.astype(np.float32))

        uni_feature_ready = bool(ok_count >= min_ok_patches)
        row["uni_feature_ready"] = uni_feature_ready
        row["status"] = "ok" if uni_feature_ready else "ok_low_count"
        if not uni_feature_ready:
            row["reason"] = f"processed_ok<{min_ok_patches}"

        done_payload = {
            "slide_id": sid,
            "status": row["status"],
            "reason": row["reason"],
            "input_coords": row["input_coords"],
            "processed_ok": ok_count,
            "failed": int(fail_count),
            "feature_dim": dim,
            "uni_feature_ready": uni_feature_ready,
            "feature_path": str(feat_path),
            "meta_path": str(meta_path),
            "coords_csv_path": str(coords_csv_path),
            "coords_npy_path": str(coords_npy_path),
            "slide_feature_path": str(slide_feat_path),
            "encoder_name": encoder_name,
            "coord_source": coord_source,
            "use_macenko": use_macenko,
            "multi_worker_mode": multi_worker_mode,
            "io_workers": int(io_workers),
            "output_dtype": output_dtype,
            "timestamp_unix": int(time.time()),
        }
        with done_path.open("w", encoding="utf-8") as f:
            json.dump(done_payload, f, indent=2)

        row["elapsed_sec"] = float(time.time() - t0)
        rows.append(row)
        print(
            f"[uni] {sid}: input={row['input_coords']} ok={ok_count} fail={fail_count} "
            f"dim={dim} ready={uni_feature_ready}"
        )

    run_df = pd.DataFrame(rows).sort_values("slide_id").reset_index(drop=True)
    run_summary_path = uni_out_root / "run_summary.csv"
    if merge_run_summary and run_summary_path.exists() and len(run_df) > 0:
        prev = pd.read_csv(run_summary_path, dtype={"slide_id": "string"})
        prev["slide_id"] = prev["slide_id"].astype(str).str.strip()
        if "center" in prev.columns:
            prev["center"] = prev["center"].astype(str).str.strip()
        run_df["slide_id"] = run_df["slide_id"].astype(str).str.strip()
        if "center" in run_df.columns:
            run_df["center"] = run_df["center"].astype(str).str.strip()
        run_df = pd.concat([prev, run_df], ignore_index=True, sort=False)
        subset = ["center", "slide_id"] if "center" in run_df.columns else ["slide_id"]
        run_df = run_df.drop_duplicates(subset=subset, keep="last").sort_values(subset).reset_index(drop=True)
    run_df.to_csv(run_summary_path, index=False)
    if len(run_df) and "center" in run_df.columns:
        for center, g in run_df.groupby("center", dropna=False):
            center_dir = uni_out_root / str(center) / "uni"
            center_dir.mkdir(parents=True, exist_ok=True)
            g.sort_values("slide_id").reset_index(drop=True).to_csv(center_dir / "run_summary.csv", index=False)
    print("\n=== UNI SUMMARY ===")
    print(f"Run summary -> {run_summary_path}")
    if len(run_df):
        counts = run_df["status"].value_counts(dropna=False).to_dict()
        print(f"Status counts: {counts}")
        ready = int(run_df["uni_feature_ready"].map(lambda x: _to_bool(x, False)).sum())
        print(f"UNI feature ready: {ready}/{len(run_df)}")


if __name__ == "__main__":
    main()

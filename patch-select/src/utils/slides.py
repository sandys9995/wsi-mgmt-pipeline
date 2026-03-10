from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable


def _sanitize_part(text: str, max_len: int = 80) -> str:
    s = str(text).strip().replace("\\", "/")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._")
    if not s:
        s = "item"
    return s[:max_len]


def slide_relative_stem(root_dir: Path, slide_path: Path) -> str:
    try:
        rel = slide_path.relative_to(root_dir)
    except ValueError:
        rel = Path(slide_path.name)
    rel_no_suffix = rel.with_suffix("")
    return rel_no_suffix.as_posix()


def build_slide_uid(center: str, rel_stem: str) -> str:
    rel_norm = str(rel_stem).replace("\\", "/").strip("/")
    stem = Path(rel_norm).name
    digest = hashlib.sha1(f"{center}::{rel_norm}".encode("utf-8")).hexdigest()[:10]
    return f"{_sanitize_part(center, 40)}__{_sanitize_part(stem, 80)}__{digest}"


def build_slide_record(path: Path, center: str, root_dir: Path) -> dict[str, str]:
    rel_stem = slide_relative_stem(root_dir, path)
    return {
        "path": str(path),
        "center": str(center),
        "center_root": str(root_dir),
        "slide_id": path.stem,
        "slide_uid": build_slide_uid(str(center), rel_stem),
        "slide_relpath": rel_stem,
    }


def list_slide_records(
    wsi_dirs: Iterable[Path],
    *,
    recursive: bool,
    exts: Iterable[str],
) -> list[dict[str, str]]:
    exts_norm = {str(x).lower() for x in exts}
    files: list[dict[str, str]] = []
    for wsi_dir in wsi_dirs:
        center = wsi_dir.name
        if recursive:
            paths = [p for p in sorted(wsi_dir.rglob("*")) if p.is_file() and p.suffix.lower() in exts_norm]
        else:
            paths = [p for p in sorted(wsi_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts_norm]
        for path in paths:
            files.append(build_slide_record(path=path, center=str(center), root_dir=wsi_dir))

    uniq: list[dict[str, str]] = []
    seen: set[str] = set()
    for rec in files:
        uid = str(rec["slide_uid"])
        if uid in seen:
            continue
        uniq.append(rec)
        seen.add(uid)
    return uniq


def slide_match(rec: dict[str, str], query: str) -> bool:
    q = str(query).strip()
    return q in {
        str(rec.get("slide_id", "")).strip(),
        str(rec.get("slide_uid", "")).strip(),
        Path(str(rec.get("path", ""))).stem,
    }


def slide_key_from_row(row: dict) -> str:
    slide_uid = str(row.get("slide_uid", "")).strip()
    if slide_uid:
        return slide_uid
    return str(row.get("slide_id", "")).strip()

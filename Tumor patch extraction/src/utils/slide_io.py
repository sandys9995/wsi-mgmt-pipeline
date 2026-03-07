import os
import glob
from typing import Optional, Dict, Tuple

import openslide

EXTS = (".svs", ".ndpi", ".mrxs", ".tif", ".tiff")


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def build_slide_index(slide_dir: str, recursive: bool = True) -> Dict[str, str]:
    index: Dict[str, str] = {}
    pattern_root = os.path.join(slide_dir, "**") if recursive else slide_dir

    for ext in EXTS:
        pattern = os.path.join(pattern_root, f"*{ext}")
        for fp in glob.glob(pattern, recursive=recursive):
            sid = _stem(fp)
            if sid not in index:
                index[sid] = fp
    return index


def open_slide_by_id(slide_index: Dict[str, str], slide_id: str) -> openslide.OpenSlide:
    sid = slide_id
    for ext in EXTS:
        if sid.lower().endswith(ext):
            sid = sid[: -len(ext)]
            break

    if sid not in slide_index:
        matches = [k for k in slide_index.keys() if k == sid or k.startswith(sid) or sid.startswith(k)]
        if matches:
            sid = matches[0]
        else:
            raise FileNotFoundError(
                f"Slide '{slide_id}' not found. Example IDs: {list(slide_index.keys())[:10]}"
            )
    return openslide.OpenSlide(slide_index[sid])


def get_mpp(slide: openslide.OpenSlide) -> Optional[float]:
    props = slide.properties
    if "openslide.mpp-x" in props and "openslide.mpp-y" in props:
        try:
            mx = float(props["openslide.mpp-x"])
            my = float(props["openslide.mpp-y"])
            if mx > 0 and my > 0:
                return (mx + my) / 2.0
        except Exception:
            pass

    if "aperio.MPP" in props:
        try:
            v = float(props["aperio.MPP"])
            if v > 0:
                return v
        except Exception:
            pass

    obj = props.get("openslide.objective-power", props.get("aperio.AppMag", None))
    if obj is not None:
        try:
            obj = float(obj)
            if obj >= 40:
                return 0.25
            if obj >= 20:
                return 0.50
        except Exception:
            pass
    return None


def get_bounds(slide: openslide.OpenSlide) -> Tuple[int, int, int, int]:
    """
    Returns (bounds_x, bounds_y, bounds_w, bounds_h).
    If slide has bounds, use them; else return full canvas.
    """
    props = slide.properties
    if (
        "openslide.bounds-x" in props and
        "openslide.bounds-y" in props and
        "openslide.bounds-width" in props and
        "openslide.bounds-height" in props
    ):
        bx = int(float(props["openslide.bounds-x"]))
        by = int(float(props["openslide.bounds-y"]))
        bw = int(float(props["openslide.bounds-width"]))
        bh = int(float(props["openslide.bounds-height"]))
        return bx, by, bw, bh

    W, H = slide.dimensions
    return 0, 0, W, H
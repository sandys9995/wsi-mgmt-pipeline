# src/select/sampling.py
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_spatial_cap(
    xy_mask: np.ndarray,
    scores_df: pd.DataFrame,
    cell_size: int = 256,
    max_per_cell: int = 40,
    sort_col: str = "cell_rich_p",
) -> np.ndarray:
    """
    Cap number of candidates per spatial cell (in mask coordinate space).
    Keeps highest scoring (sort_col) within each cell.
    Returns indices to keep.
    """
    x = xy_mask[:, 0]
    y = xy_mask[:, 1]
    cx = (x // cell_size).astype(np.int32)
    cy = (y // cell_size).astype(np.int32)

    # sort indices by score descending
    order = np.argsort(-scores_df[sort_col].to_numpy())

    kept = []
    counts = {}
    for idx in order:
        key = (int(cx[idx]), int(cy[idx]))
        c = counts.get(key, 0)
        if c < max_per_cell:
            kept.append(idx)
            counts[key] = c + 1

    kept = np.array(sorted(kept), dtype=np.int32)
    return kept


def quota_select(scores_df: pd.DataFrame, target: int, cfg: dict) -> np.ndarray:
    """
    Select up to `target` items with type quotas.
    Types: A, B, C, D as produced by scoring.py
    Returns indices in scores_df to keep.
    """
    quotas = cfg["scoring"]["quotas"]
    fracA = float(quotas.get("typeA_frac", 0.70))
    fracB = float(quotas.get("typeB_frac", 0.25))
    fracC = float(quotas.get("typeC_frac", 0.05))
    fracD = float(quotas.get("typeD_frac", 0.00))

    nA = int(round(target * fracA))
    nB = int(round(target * fracB))
    nC = int(round(target * fracC))
    nD = max(0, target - (nA + nB + nC))  # remainder

    score_col = "cell_rich_p" if "cell_rich_p" in scores_df.columns else "tex_p"
    score = scores_df[score_col].to_numpy()
    order_all = np.argsort(-score)

    def take_type(t, n):
        idx = np.where(scores_df["type"].to_numpy() == t)[0]
        if len(idx) == 0 or n <= 0:
            return np.array([], dtype=np.int32)
        idx_sorted = idx[np.argsort(-score[idx])]
        return idx_sorted[: min(n, len(idx_sorted))]

    selA = take_type("A", nA)
    selB = take_type("B", nB)

    # cap blood explicitly using blood_cap_frac
    cap_frac = float(cfg["scoring"].get("blood_cap_frac", 0.05))
    nC_cap = min(int(round(target * cap_frac)), nC)
    selC = take_type("C", nC_cap)

    sel = np.unique(np.concatenate([selA, selB, selC]).astype(np.int32))
    allow_under_target = bool(cfg["scoring"].get("allow_under_target", True))
    max_type_d_frac = float(cfg["scoring"].get("max_typeD_frac", 0.10))
    max_d = max(0, int(round(target * max_type_d_frac)))

    # fill remainder with best overall (excluding already selected)
    if len(sel) < target:
        need = target - len(sel)
        used = set(int(i) for i in sel.tolist())
        fill = []
        for idx in order_all:
            if int(idx) in used:
                continue
            # avoid Type D in first fill pass
            if scores_df.iloc[idx]["type"] == "D":
                continue
            fill.append(int(idx))
            if len(fill) >= need:
                break

        sel = np.concatenate([sel, np.array(fill, dtype=np.int32)])

    # if still short, consider Type D under cap.
    if len(sel) < target:
        need = target - len(sel)
        used = set(int(i) for i in sel.tolist())
        curr_d = int((scores_df.iloc[sel]["type"] == "D").sum()) if len(sel) > 0 else 0
        d_room = max(0, max_d - curr_d)
        fill = []
        for idx in order_all:
            if int(idx) in used:
                continue
            if scores_df.iloc[idx]["type"] != "D":
                continue
            fill.append(int(idx))
            if len(fill) >= min(need, d_room):
                break
        sel = np.concatenate([sel, np.array(fill, dtype=np.int32)])

    if len(sel) > target:
        sel = sel[:target]

    if not allow_under_target and len(sel) < target:
        need = target - len(sel)
        used = set(int(i) for i in sel.tolist())
        fill = []
        for idx in order_all:
            if int(idx) in used:
                continue
            fill.append(int(idx))
            if len(fill) >= need:
                break
        sel = np.concatenate([sel, np.array(fill, dtype=np.int32)])

    return sel.astype(np.int32)

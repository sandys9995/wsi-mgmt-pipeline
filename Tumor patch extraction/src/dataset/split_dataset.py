# dataset/split_dataset.py
# - Builds patch DF from results/patches_norm
# - Creates HARD validation split (force hard negatives if available)
# - Robust to single-class slides (pos-only / neg-only)
# - Ensures VAL has both labels globally whenever possible
# - Saves slide-level summary + split summary
# - Uses tqdm if installed

import os
import json
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# -------------------------- helpers --------------------------

def build_patch_df(patches_norm_dir: str) -> pd.DataFrame:
    """
    Build a patch-level dataframe from:
      patches_norm_dir/0/*.png
      patches_norm_dir/1/*.png

    Expects filenames like:
      slideid__x123__y456__mpp0.243__r462.png

    Returns columns:
      slide_id, x, y, label, png_path
    """
    rows = []
    labels = ["0", "1"]

    all_paths = []
    for lab in labels:
        all_paths.extend(sorted(glob.glob(os.path.join(patches_norm_dir, lab, "*.png"))))

    iterator = all_paths
    if tqdm is not None:
        iterator = tqdm(all_paths, desc="[split] indexing patches", unit="patch")

    for p in iterator:
        lab = os.path.basename(os.path.dirname(p))
        fname = os.path.basename(p)

        slide_id = fname.split("__x")[0]

        x = None
        y = None
        try:
            x = int(fname.split("__x")[1].split("__y")[0])
            y = int(fname.split("__y")[1].split("__")[0])
        except Exception:
            pass

        rows.append(
            {
                "slide_id": slide_id,
                "x": x,
                "y": y,
                "label": int(lab),
                "png_path": p,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No PNGs found under: {patches_norm_dir}/0 and /1")

    df = df.dropna(subset=["slide_id", "label", "png_path"]).reset_index(drop=True)
    return df


def save_split(train_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    train_csv = os.path.join(out_dir, "train.csv")
    val_csv = os.path.join(out_dir, "val.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    return train_csv, val_csv


def summarize(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    def slide_stats(df):
        g = df.groupby("slide_id")["label"]
        tumor_frac = g.mean()
        return {
            "n_slides": int(tumor_frac.shape[0]),
            "n_patches": int(len(df)),
            "mean_tumor_frac_across_slides": float(tumor_frac.mean()),
            "tumor_frac_min": float(tumor_frac.min()),
            "tumor_frac_max": float(tumor_frac.max()),
        }

    def label_counts(df):
        vc = df["label"].value_counts()
        return {str(int(k)): int(v) for k, v in vc.items()}

    return {
        "train": slide_stats(train_df),
        "val": slide_stats(val_df),
        "train_label_counts": label_counts(train_df),
        "val_label_counts": label_counts(val_df),
    }


def save_slide_summary(df: pd.DataFrame, out_dir: str) -> str:
    """
    Per-slide patch counts + class composition.
    """
    os.makedirs(out_dir, exist_ok=True)
    g = df.groupby("slide_id")["label"]
    s = g.agg(n_patches="size", n_pos="sum", tumor_frac="mean").reset_index()
    s["n_neg"] = s["n_patches"] - s["n_pos"]
    s["pos_only"] = s["n_neg"] == 0
    s["neg_only"] = s["n_pos"] == 0
    s["mixed"] = (~s["pos_only"]) & (~s["neg_only"])
    s["single_class"] = s["pos_only"] | s["neg_only"]

    out_path = os.path.join(out_dir, "slide_label_summary.csv")
    s.to_csv(out_path, index=False)

    print("[split] slides total:", len(s))
    print("[split] single-class slides:", int(s["single_class"].sum()))
    print("[split] pos-only slides:", int(s["pos_only"].sum()))
    print("[split] neg-only slides:", int(s["neg_only"].sum()))
    return out_path


# -------------------------- main split logic --------------------------

def make_slide_level_split(
    df: pd.DataFrame,
    val_frac: float = 0.2,
    seed: int = 42,
    low_thr: float = 0.20,     # define "hard negative" slides (tumor_frac <= low_thr)
    high_thr: float = 0.80,    # very tumor-heavy slides
    min_low_in_val: int = 2,   # force at least this many low slides in val (if available)
    prefer_mid: bool = True,   # include some mid slides too, if possible
    ensure_val_has_both_labels: bool = True,  # global label coverage in VAL
):
    """
    Slide-level split WITHOUT leakage, but with a HARD validation set.
    Robust to slides that are single-class (pos-only / neg-only).

    Key behavior:
    - Preferentially puts low-tumor-fraction slides into validation (hard negatives).
    - Optionally enforces that validation contains both labels globally when possible.
    - Falls back to GroupShuffleSplit if low slides do not exist.

    Returns: train_df, val_df, train_slides, val_slides
    """
    rng = np.random.RandomState(seed)

    # --- per-slide stats ---
    g = df.groupby("slide_id")["label"]
    slides_df = pd.DataFrame({
        "slide_id": g.size().index,
        "tumor_frac": g.mean().values,
        "n_patches": g.size().values,
    })

    n_slides = len(slides_df)
    if n_slides < 2:
        raise ValueError("Need at least 2 slides to split.")

    n_val = int(round(n_slides * val_frac))
    n_val = max(1, min(n_val, n_slides - 1))

    # Identify slide types
    # pos-only => tumor_frac == 1
    # neg-only => tumor_frac == 0
    pos_only = slides_df[slides_df["tumor_frac"] >= 1.0 - 1e-12]["slide_id"].tolist()
    neg_only = slides_df[slides_df["tumor_frac"] <= 0.0 + 1e-12]["slide_id"].tolist()
    mixed    = slides_df[(slides_df["tumor_frac"] > 0.0 + 1e-12) & (slides_df["tumor_frac"] < 1.0 - 1e-12)]["slide_id"].tolist()

    # strata by tumor_frac
    low_df  = slides_df[slides_df["tumor_frac"] <= low_thr].copy()
    mid_df  = slides_df[(slides_df["tumor_frac"] > low_thr) & (slides_df["tumor_frac"] < high_thr)].copy()
    high_df = slides_df[slides_df["tumor_frac"] >= high_thr].copy()

    low_ids  = low_df["slide_id"].tolist()
    mid_ids  = mid_df["slide_id"].tolist()
    high_ids = high_df["slide_id"].tolist()

    rng.shuffle(low_ids)
    rng.shuffle(mid_ids)
    rng.shuffle(high_ids)

    # Helper to evaluate global label coverage in a candidate val set
    def _val_has_both_labels(val_slide_set: set) -> bool:
        vdf = df[df["slide_id"].isin(val_slide_set)]
        if vdf.empty:
            return False
        labs = set(vdf["label"].unique().tolist())
        return (0 in labs) and (1 in labs)

    # ---- fallback if no low slides exist ----
    if len(low_ids) == 0:
        print(f"[split] WARNING: no low-tumor slides found (tumor_frac <= {low_thr}). "
              f"Falling back to GroupShuffleSplit.")
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        groups = slides_df["slide_id"].values
        X_dummy = np.zeros(len(groups))
        train_idx, val_idx = next(splitter.split(X_dummy, groups=groups))
        train_slides = set(slides_df.iloc[train_idx]["slide_id"].tolist())
        val_slides   = set(slides_df.iloc[val_idx]["slide_id"].tolist())

        # Optional post-fix: ensure VAL has both labels if possible
        if ensure_val_has_both_labels and not _val_has_both_labels(val_slides):
            # Try to swap one slide from train into val that adds missing label
            # We do it deterministically by scanning.
            missing = set([0, 1]) - set(df[df["slide_id"].isin(val_slides)]["label"].unique().tolist())
            if missing:
                missing_lab = list(missing)[0]
                # Find a train slide that contains missing label
                for sid in list(train_slides):
                    labs = set(df[df["slide_id"] == sid]["label"].unique().tolist())
                    if missing_lab in labs:
                        # swap: move sid to val, move one val slide back to train (pick the "easiest" high slide if possible)
                        val_to_move = None
                        # prefer moving back a high slide if we added a neg-rich slide, etc.
                        for vsid in list(val_slides):
                            val_to_move = vsid
                            break
                        if val_to_move is not None:
                            train_slides.add(val_to_move)
                            val_slides.remove(val_to_move)
                            train_slides.remove(sid)
                            val_slides.add(sid)
                        break

        train_df = df[df["slide_id"].isin(train_slides)].reset_index(drop=True)
        val_df   = df[df["slide_id"].isin(val_slides)].reset_index(drop=True)
        return train_df, val_df, train_slides, val_slides

    # ---- HARD VAL construction ----
    val_list = []

    # (A) If enforcing global label coverage, seed VAL with one pos-only and one neg-only if available
    if ensure_val_has_both_labels:
        rng.shuffle(pos_only)
        rng.shuffle(neg_only)
        if len(pos_only) > 0 and len(neg_only) > 0 and n_val >= 2:
            val_list.append(neg_only[0])
            val_list.append(pos_only[0])

    # (B) Force low slides into validation (hard negatives)
    low_pool = [sid for sid in low_ids if sid not in val_list]
    n_low = min(len(low_pool), min_low_in_val, n_val - len(val_list))
    val_list.extend(low_pool[:n_low])

    remaining = n_val - len(val_list)

    # (C) Prefer some mid slides
    if remaining > 0 and prefer_mid and len(mid_ids) > 0:
        mid_pool = [sid for sid in mid_ids if sid not in val_list]
        n_mid = min(len(mid_pool), max(1, remaining // 2))
        val_list.extend(mid_pool[:n_mid])
        remaining = n_val - len(val_list)

    # (D) Fill rest from high, then mid, then low leftovers
    if remaining > 0:
        high_pool = [sid for sid in high_ids if sid not in val_list]
        take = min(len(high_pool), remaining)
        val_list.extend(high_pool[:take])
        remaining = n_val - len(val_list)

    if remaining > 0:
        mid_left = [sid for sid in mid_ids if sid not in val_list]
        low_left = [sid for sid in low_ids if sid not in val_list]
        rng.shuffle(mid_left)
        rng.shuffle(low_left)

        take = min(len(mid_left), remaining)
        val_list.extend(mid_left[:take])
        remaining = n_val - len(val_list)

        if remaining > 0:
            take = min(len(low_left), remaining)
            val_list.extend(low_left[:take])
            remaining = n_val - len(val_list)

    val_slides = set(val_list)
    train_slides = set(slides_df["slide_id"].tolist()) - val_slides

    if len(train_slides) == 0 or len(val_slides) == 0:
        raise RuntimeError("Split failed: empty train or val set.")

    train_df = df[df["slide_id"].isin(train_slides)].reset_index(drop=True)
    val_df   = df[df["slide_id"].isin(val_slides)].reset_index(drop=True)

    # Final safety: if we asked for both labels but val ended up single-label, do a cheap fix if possible
    if ensure_val_has_both_labels and not _val_has_both_labels(val_slides):
        print("[split] WARNING: VAL ended up with a single label. Attempting a swap fix...")
        val_labs = set(val_df["label"].unique().tolist())
        missing = set([0, 1]) - val_labs
        if missing:
            missing_lab = list(missing)[0]
            # find a train slide that contains missing label
            candidate_train = None
            for sid in list(train_slides):
                labs = set(df[df["slide_id"] == sid]["label"].unique().tolist())
                if missing_lab in labs:
                    candidate_train = sid
                    break
            if candidate_train is not None:
                # move one "least useful" val slide back to train (pick a high slide first)
                val_df2 = slides_df[slides_df["slide_id"].isin(val_slides)].copy()
                # prefer swapping out a very tumor-heavy slide to keep val hard
                val_df2 = val_df2.sort_values("tumor_frac", ascending=False)
                swap_out = val_df2.iloc[0]["slide_id"]

                train_slides.add(swap_out)
                val_slides.remove(swap_out)

                train_slides.remove(candidate_train)
                val_slides.add(candidate_train)

                train_df = df[df["slide_id"].isin(train_slides)].reset_index(drop=True)
                val_df   = df[df["slide_id"].isin(val_slides)].reset_index(drop=True)

    # Debug print
    def _summ(slides_set):
        sdf = slides_df[slides_df["slide_id"].isin(slides_set)]
        return {
            "n_slides": int(len(sdf)),
            "tumor_frac_min": float(sdf["tumor_frac"].min()),
            "tumor_frac_max": float(sdf["tumor_frac"].max()),
            "tumor_frac_mean": float(sdf["tumor_frac"].mean()),
            "n_low": int((sdf["tumor_frac"] <= low_thr).sum()),
            "n_mid": int(((sdf["tumor_frac"] > low_thr) & (sdf["tumor_frac"] < high_thr)).sum()),
            "n_high": int((sdf["tumor_frac"] >= high_thr).sum()),
            "n_pos_only": int((sdf["tumor_frac"] >= 1.0 - 1e-12).sum()),
            "n_neg_only": int((sdf["tumor_frac"] <= 0.0 + 1e-12).sum()),
        }

    print("[split] HARD VAL enabled")
    print("[split] train:", _summ(train_slides))
    print("[split] val:  ", _summ(val_slides))

    return train_df, val_df, train_slides, val_slides


# -------------------------- CLI entry --------------------------

if __name__ == "__main__":
    PATCHES_NORM_DIR = "results/patches_norm"
    OUT_DIR = "results/splits"

    df = build_patch_df(PATCHES_NORM_DIR)

    # Always save slide composition report (so we can diagnose “single-class slide” issues later)
    slide_summary_csv = save_slide_summary(df, OUT_DIR)
    print("[split] slide summary:", slide_summary_csv)

    train_df, val_df, train_slides, val_slides = make_slide_level_split(
        df,
        val_frac=0.2,
        seed=42,
        low_thr=0.20,
        high_thr=0.80,
        min_low_in_val=2,
        prefer_mid=True,
        ensure_val_has_both_labels=True,
    )

    train_csv, val_csv = save_split(train_df, val_df, OUT_DIR)

    summary = summarize(train_df, val_df)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    print(" ", train_csv)
    print(" ", val_csv)
    print("Summary:", json.dumps(summary, indent=2))
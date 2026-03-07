# train_patch_classifier.py
import os
import json
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from dataset.patch_dataset import PatchCSVDataset

try:
    import torchvision.models as models
except Exception as e:
    raise RuntimeError("torchvision is required. pip install torchvision") from e

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

patience = 3          # stop after 3 bad epochs
min_delta = 1e-4      # minimum AUC improvement
bad_epochs = 0
best_auc = -1.0

# --------------------------- metrics + saving helpers ---------------------------

def compute_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= thr).astype(int)

    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    sens = tp / (tp + fn + 1e-9)  # recall for tumor
    spec = tn / (tn + fp + 1e-9)  # recall for non-tumor

    # extra, useful in practice
    prec = tp / (tp + fp + 1e-9)
    npv = tn / (tn + fn + 1e-9)
    fpr = fp / (fp + tn + 1e-9)
    fnr = fn / (fn + tp + 1e-9)

    return {
        "auc": float(auc),
        "ap": float(ap),
        "acc": float(acc),
        "sens": float(sens),
        "spec": float(spec),
        "precision": float(prec),
        "npv": float(npv),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def find_best_threshold(y_true, y_prob, mode="youden"):
    """
    mode:
      - 'youden': maximize (TPR - FPR)
      - 'f1': maximize F1
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    thr_grid = np.linspace(0.0, 1.0, 501)
    best_thr = 0.5
    best_score = -1e9
    best_metrics = None

    for thr in thr_grid:
        m = compute_metrics(y_true, y_prob, thr=thr)
        # youden J = sens + spec - 1
        if mode == "youden":
            score = m["sens"] + m["spec"] - 1.0
        elif mode == "f1":
            # f1 = 2*prec*rec/(prec+rec)
            score = (2 * m["precision"] * m["sens"]) / (m["precision"] + m["sens"] + 1e-9)
        else:
            raise ValueError("mode must be 'youden' or 'f1'")

        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_metrics = m

    return best_thr, float(best_score), best_metrics


def save_curves(out_dir, y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    curves = {}

    # ROC
    if len(np.unique(y_true)) > 1:
        fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
        curves["roc"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thr": roc_thr.tolist(),
        }

        # PR
        prec, rec, pr_thr = precision_recall_curve(y_true, y_prob)
        curves["pr"] = {
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            # sklearn returns thr with len = len(precision)-1
            "thr": pr_thr.tolist(),
        }

    with open(os.path.join(out_dir, "curves.json"), "w") as f:
        json.dump(curves, f, indent=2)


def save_val_predictions(out_dir, val_ds, y_true, y_prob, epoch, tag):
    """
    Save:
      - NPY: val_y.npy, val_prob.npy (last run for that tag)
      - CSV: val_predictions_{tag}.csv (overwritten each time)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    np.save(os.path.join(out_dir, f"val_y_{tag}.npy"), y_true)
    np.save(os.path.join(out_dir, f"val_prob_{tag}.npy"), y_prob)

    # Keep it simple: align by order of val_ds.df (since DataLoader shuffle=False)
    dfp = val_ds.df.copy().reset_index(drop=True)
    if len(dfp) == len(y_true):
        dfp["y_true"] = y_true
        dfp["y_prob"] = y_prob
        dfp["y_pred_0p5"] = (y_prob >= 0.5).astype(int)
        dfp["epoch"] = int(epoch)
        dfp.to_csv(os.path.join(out_dir, f"val_predictions_{tag}.csv"), index=False)
    else:
        # if mismatch happens, still save arrays; CSV alignment can't be trusted
        with open(os.path.join(out_dir, f"val_predictions_{tag}.txt"), "w") as f:
            f.write(f"WARNING: cannot align df (n={len(dfp)}) with y_true (n={len(y_true)})\n")


def save_hard_fp_fn_examples(out_dir, val_ds, y_true, y_prob, epoch, thr=0.5, topk=50):
    """
    Save “hard mistakes” for quick inspection:
      - top FP: y_true=0 but y_prob high
      - top FN: y_true=1 but y_prob low
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    df = val_ds.df.copy().reset_index(drop=True)

    if len(df) != len(y_true):
        return  # can't align reliably

    df["y_true"] = y_true
    df["y_prob"] = y_prob
    df["y_pred"] = (y_prob >= thr).astype(int)
    df["epoch"] = int(epoch)

    fp = df[(df["y_true"] == 0) & (df["y_pred"] == 1)].copy().sort_values("y_prob", ascending=False).head(topk)
    fn = df[(df["y_true"] == 1) & (df["y_pred"] == 0)].copy().sort_values("y_prob", ascending=True).head(topk)

    fp.to_csv(os.path.join(out_dir, f"hard_fp_epoch{epoch}_thr{thr:.2f}.csv"), index=False)
    fn.to_csv(os.path.join(out_dir, f"hard_fn_epoch{epoch}_thr{thr:.2f}.csv"), index=False)


# --------------------------- training script ---------------------------

def run():
    TRAIN_CSV = "results/splits/train.csv"
    VAL_CSV   = "results/splits/val.csv"
    OUT_DIR   = "results/model_patch"
    os.makedirs(OUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("[train] device:", device)

    train_ds = PatchCSVDataset(TRAIN_CSV, train=True)
    val_ds   = PatchCSVDataset(VAL_CSV, train=False)

    # class weights (inverse frequency)
    y_train = train_ds.df["label"].values
    n0 = int((y_train == 0).sum())
    n1 = int((y_train == 1).sum())
    w0 = (n0 + n1) / (2.0 * max(n0, 1))
    w1 = (n0 + n1) / (2.0 * max(n1, 1))
    class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)
    print(f"[train] class weights: w0={w0:.3f} w1={w1:.3f}")

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=8, pin_memory=(device=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=8, pin_memory=(device=="cuda"))

    # model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    crit = nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

    best_auc = -1.0
    best_epoch = -1
    history = []

    for epoch in range(1, 11):
        # ---- train ----
        model.train()
        t0 = time.time()
        running = 0.0

        it = train_loader
        if tqdm is not None:
            it = tqdm(train_loader, desc=f"[train] epoch {epoch}", unit="batch")

        for xb, yb in it:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

            running += float(loss.item())

        train_loss = running / max(len(train_loader), 1)

        # ---- val ----
        model.eval()
        val_loss = 0.0
        y_true, y_prob = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                logits = model(xb)
                loss = crit(logits, yb)
                val_loss += float(loss.item())

                prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                y_prob.extend(prob.tolist())
                y_true.extend(yb.detach().cpu().numpy().tolist())

        val_loss /= max(len(val_loader), 1)

        # metrics at default threshold 0.5
        metrics_05 = compute_metrics(y_true, y_prob, thr=0.5)

        # also compute a “best threshold” summary (useful later)
        best_thr_youden, bestJ, m_best_thr = find_best_threshold(y_true, y_prob, mode="youden")

        scheduler.step()

        rec = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),

            # default-threshold metrics (thr=0.5)
            **{f"thr0p5_{k}": v for k, v in metrics_05.items()},

            # best threshold (Youden) summary
            "best_thr_youden": float(best_thr_youden),
            "best_youden_J": float(bestJ),
            **{f"bestthr_{k}": v for k, v in (m_best_thr or {}).items()},

            "lr": float(opt.param_groups[0]["lr"]),
            "sec": float(time.time() - t0),
        }
        history.append(rec)

        print(
            f"[val] epoch={epoch} loss={val_loss:.4f} "
            f"auc={metrics_05['auc']:.4f} ap={metrics_05['ap']:.4f} "
            f"acc={metrics_05['acc']:.4f} sens={metrics_05['sens']:.4f} spec={metrics_05['spec']:.4f} "
            f"(thr=0.5) | best_thr={best_thr_youden:.3f}"
        )

        # Always save "last" preds/curves for plotting (overwrites each epoch)
        save_val_predictions(OUT_DIR, val_ds, y_true, y_prob, epoch=epoch, tag="last")
        save_curves(OUT_DIR, y_true, y_prob)
        save_hard_fp_fn_examples(OUT_DIR, val_ds, y_true, y_prob, epoch=epoch, thr=0.5, topk=50)

        # # save best by AUC
        # if not np.isnan(metrics_05["auc"]) and metrics_05["auc"] > best_auc:
        #     best_auc = float(metrics_05["auc"])
        #     best_epoch = int(epoch)

        #     torch.save(model.state_dict(), os.path.join(OUT_DIR, "best.pt"))
        #     with open(os.path.join(OUT_DIR, "best_metrics.json"), "w") as f:
        #         json.dump(rec, f, indent=2)

        #     # Also freeze best-epoch predictions for ROC/PR plots
        #     save_val_predictions(OUT_DIR, val_ds, y_true, y_prob, epoch=epoch, tag="best")
        #     save_curves(OUT_DIR, y_true, y_prob)
        #     save_hard_fp_fn_examples(OUT_DIR, val_ds, y_true, y_prob, epoch=epoch, thr=0.5, topk=200)
        if not np.isnan(metrics_05["auc"]) and metrics_05["auc"] > best_auc + min_delta:
            best_auc = float(metrics_05["auc"])
            best_epoch = int(epoch)
            bad_epochs = 0

            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best.pt"))
            with open(os.path.join(OUT_DIR, "best_metrics.json"), "w") as f:
                json.dump(rec, f, indent=2)

            save_val_predictions(OUT_DIR, val_ds, y_true, y_prob, epoch=epoch, tag="best")
            save_curves(OUT_DIR, y_true, y_prob)
            save_hard_fp_fn_examples(
                OUT_DIR, val_ds, y_true, y_prob,
                epoch=epoch, thr=0.5, topk=200
            )

        else:
            bad_epochs += 1

        with open(os.path.join(OUT_DIR, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
            
        if bad_epochs >= patience:
            print(
                f"[early stop] no AUC improvement for {patience} epochs. "
                f"Stopping at epoch {epoch}."
            )
            break

    # final summary
    with open(os.path.join(OUT_DIR, "train_summary.json"), "w") as f:
        json.dump({"best_auc": best_auc, "best_epoch": best_epoch}, f, indent=2)

    print("[train] done. best_auc=", best_auc, "best_epoch=", best_epoch)
    print("[train] saved to:", OUT_DIR)
    print("[train] for plotting, use:")
    print("  results/model_patch/val_y_best.npy, val_prob_best.npy")
    print("  results/model_patch/curves.json")
    print("  results/model_patch/val_predictions_best.csv")
    print("  results/model_patch/hard_fp_epochX_thr0.50.csv + hard_fn_epochX_thr0.50.csv")


if __name__ == "__main__":
    run()
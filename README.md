# WSI MGMT Pipeline: Important Run Commands

Run all commands from:

```bash
cd /Users/sandeepsharma/Downloads/WSI_PATCH_selection/patch-select
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate projects
mkdir -p logs
```

## 1) Dependency Sanity

```bash
python -c "import cv2, openslide, torch, pandas; print('cv2', cv2.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'mps', torch.backends.mps.is_available())"
```

## 2) Quick Unit Tests

```bash
pytest -q tests/test_strategy.py tests/test_masking.py tests/test_pipeline_diag.py
```

## 3) Mask Smoke Test (n=1)

```bash
python -u scripts/make_masks.py --config configs/pilot.yaml --n-slides 1
```

## 4) QC Precheck Only (n=1)

```bash
python -u scripts/run_pilot.py --config configs/pilot.yaml --n-slides 1 --precheck-only
```

## 5) End-to-End Smoke (n=1)

```bash
python -u scripts/run_e2e.py --config configs/pilot.yaml --n-slides 1 --smoke-gate
```

## 6) Medium Pilot (n=10)

```bash
python -u scripts/run_e2e.py --config configs/pilot.yaml --n-slides 10
```

## 7) Full Validation / Full Dataset Mode

`configs/pilot.yaml` already uses `run.n_slides: 0` (meaning all slides).

```bash
python -u scripts/run_e2e.py --config configs/pilot.yaml
```

## 8) Explicit Mask+QC Gate Check

```bash
python -u scripts/check_mask_qc_gate.py \
  --mask-summary data/masks/mask_summary.csv \
  --run-summary data/out/qc/run_summary.csv
```

## 9) Long Run in Background (Cluster Style)

```bash
nohup python -u scripts/run_e2e.py --config configs/pilot.yaml > logs/e2e_full_$(date +%F_%H%M).log 2>&1 &
tail -f logs/e2e_full_*.log
```

## 10) Multi-Worker End-to-End Run (Cluster)

Use this to activate end-to-end worker mode from one command.

```bash
nohup python -u scripts/run_e2e.py \
  --config configs/pilot.yaml \
  --multi-worker-mode \
  --cpu-workers 16 \
  --io-workers 8 \
  > logs/e2e_mw_$(date +%F_%H%M).log 2>&1 &
tail -f logs/e2e_mw_*.log
```

Notes:
- Gate is non-blocking by default in `run_e2e.py` (pipeline continues to tumor/uni even if gate fails).
- Use `--strict-gate` if you want gate failure to stop the run.
- Use `--continue-on-fail` to continue even if non-gate stages fail.
- Gate uses adaptive thresholds by default (small pilots no longer fail only due sample size).
- Resume is enabled by default (`run.resume: true`) so reruns skip slides that already have QC outputs.
- Long loops now show `tqdm` progress bars in `run_e2e.py`, `make_masks.py`, `run_pilot.py`, `run_tumor_gate_pilot.py`, `run_uni_features.py`, and QC patch reads in `src/select/pipeline.py`.

## Result Layout

Use one center-first `results/` root on cluster:

```text
results/
├── mask_summary.csv
├── qc/
│   └── run_summary.csv
├── tumor_gate_run_summary.csv
├── uni_run_summary.csv
├── <center>/
│   ├── mask/
│   │   ├── <slide_id>.npy
│   │   ├── <slide_id>.png
│   │   └── mask_summary.csv
│   ├── qc/
│   │   ├── run_summary.csv
│   │   └── <slide_id>/
│   ├── qc_pool/
│   │   └── <slide_id>/
│   ├── coords/
│   │   └── <slide_id>/
│   ├── tumor_gate/
│   │   ├── run_summary.csv
│   │   └── <slide_id>/
│   └── uni/
│       ├── run_summary.csv
│       └── <slide_id>/
```

Root merged summaries:
- `mask_summary.csv`
- `qc/run_summary.csv`
- `tumor_gate_run_summary.csv`
- `uni_run_summary.csv`

## Optional Single-Slide Commands

Single slide UNI:

```bash
python -u scripts/run_uni_features.py --config configs/pilot.yaml --slide-id 1039403
```

Single slide tumor-gate:

```bash
python -u scripts/run_tumor_gate_pilot.py --config configs/pilot.yaml --n-slides 1
```

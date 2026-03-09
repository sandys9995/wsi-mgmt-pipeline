## Result Layout

Recommended cluster result layout:

```text
results/
├── mask_summary.csv
├── qc/
│   └── run_summary.csv
├── tumor_gate_run_summary.csv
├── uni_run_summary.csv
├── <center>/
│   ├── mask/
│   ├── qc/
│   ├── qc_pool/
│   ├── coords/
│   ├── tumor_gate/
│   └── uni/
```

This keeps each center together while still preserving one merged summary per stage at the root.

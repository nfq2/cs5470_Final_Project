# cs5470_Final_Project

Model implementations live in separate folders:

- [ResNet-18](resnet/README.md): Saesha's baseline.
- [RepViT-M0.9](repvit_m0_9/README.md): Nicole's model workflow.
- [MobileNetV3-Small](mobilenet_v3_small/README.md): Amelia's baseline.
- [DINOv2](dino/README.md): Akhil's frozen-backbone experiments and diagnostics.
- [ResNet + RepViT ensemble](ensemble/README.md): five validation-selected fusion strategies.

Midpoint metrics are collected in [results/summary.csv](results/summary.csv).
The committed ResNet result is the basic baseline only; large checkpoints remain
outside Git.

Keep the sensitive dataset on Perlmutter, outside this repository. See
[DATASET.md](DATASET.md) for the dataset and
[common/DATA_SETUP.md](common/DATA_SETUP.md) for getting a copy.

## Shared baseline protocol (`common/`)

All four architectures should follow
[the shared baseline protocol](common/BASELINE_PROTOCOL.md). It fixes training
defaults, uses all five folds, pools all 416 held-out predictions, and produces
one accuracy-versus-latency comparison. Each teammate adapts their own model
folder to the documented interface.

| File | Use |
|---|---|
| [common/data.py](common/data.py) | `load_split(fold, root)`, `load_all_folds(root)` |
| [common/metrics.py](common/metrics.py) | `report(labels, probs)` |
| [common/latency.py](common/latency.py) | `forward_latency(fn)` |
| [common/check.py](common/check.py) | Checks the real dataset (prints counts only) |
| [common/protocol.py](common/protocol.py) | Shared training and latency constants |
| [common/run_five_folds.py](common/run_five_folds.py) | Trains/evaluates one model on folds 0–4 |
| [common/aggregate_folds.py](common/aggregate_folds.py) | Pools the 416 held-out predictions |
| [common/compare_results.py](common/compare_results.py) | Makes the final CSV and plot |

### Split

The five episode-based folds supplied in `splits.json`. No new splits are made.

| Fold | Train | Validation | Test |
|---|---|---|---|
| 0 | 267 | 80 | 69 |
| 1 | 271 | 73 | 72 |
| 2 | 270 | 75 | 71 |
| 3 | 263 | 72 | 81 |
| 4 | 241 | 52 | 123 |

- Train: fit the model and any input normalization.
- Validation: choose epochs, hyperparameters and models.
- Test: report only.
- The five test groups cover all 416 cases once. To compare models, train each fold
  and pool the five test groups (one fold alone is 69 to 123 cases).

### Usage

```python
sys.path.append(str(Path(__file__).resolve().parents[1]))  # repository root
from common.data import load_split
from common.metrics import report
from common.latency import forward_latency

cases, split = load_split(fold=0, root=data_root)   # split["train"] etc. index into cases
results = report(labels, probs)                     # 1 = advance, 0 = retract; probs = P(advance)
latency = forward_latency(lambda: model(images, history))
```

- `fold` is the stored fold ID. `root` defaults to `$SHORTHORIZON_ROOT`, then
  `$PSCRATCH/finalproject/shorthorizon2`.
- `load_split` raises an error if groups overlap or share an episode.
- `report` returns overall, advance, retract and balanced accuracy (fractions, 0.5
  threshold), AUROC, and `always_advance_accuracy` on the same cases, which is the
  floor to beat.
- `forward_latency` times one decision at batch 1 with inputs already on the GPU: 20
  warm-up calls, 100 timed. Use FP32 with TF32 disabled.

Check on Perlmutter: `python common/check.py` and `python -m unittest discover common/tests`.

Older result files may use different settings or only fold 0. Label those as
preliminary; do not mix them into the standardized four-architecture graph.

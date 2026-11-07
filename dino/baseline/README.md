# DINOv2-S standardized baseline

Implements [`common/BASELINE_PROTOCOL.md`](../../common/BASELINE_PROTOCOL.md) for
DINOv2-S (`vit_small_patch14_dinov2.lvd142m`, timm). Same interface and classifier as
`resnet/`; only the backbone differs.

| File | What it does |
| --- | --- |
| data.py | Seven images resized to 448, ImageNet normalization, 12 history values (depth, then time). |
| model.py | Frozen DINOv2-S; the CLS token (384-d) is the per-image feature; protocol classifier. |
| train.py | Trains the classifier for one fold, saves `best.pt` at the lowest validation loss. |
| evaluate.py | Predictions, shared metrics and the standardized latency, with the backbone running live. |

The backbone is frozen with no augmentation, so `train.py` encodes each image once per
fold and trains the classifier on those vectors; the numbers are identical to re-encoding
every epoch. `evaluate.py` runs the backbone live for every prediction and for latency.

## Run (GPU allocation, from the repository root)

```bash
module load pytorch/2.8.0 && export HF_HOME=$PSCRATCH/huggingface   # timm installed with pip --user
python -m common.run_five_folds \
  --model-dir dino/baseline \
  --data-root "$PSCRATCH/finalproject/shorthorizon2" \
  --output-root "$PSCRATCH/finalproject/standardized_results/dinov2_small"
```

The experiments in the parent folder (`dino/`) are separate from this baseline: different
head, fixed epoch count, cached features, probes, tiles and fine-tuning. See
[`../PLAN.md`](../PLAN.md).

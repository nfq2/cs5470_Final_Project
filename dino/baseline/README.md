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

## Checkpoints and predictions (ensemble handoff)

The finished five-fold run is on Perlmutter, readable by the m4999 group:

```
/global/cfs/cdirs/m4999/waldo_shared/standardized_results/dinov2_small/
  fold0/ .. fold4/
    best.pt          lowest-val-loss checkpoint for that fold (~88 MB each)
    validation.json  predictions on the fold's validation split
    test.json        predictions on the fold's test group
    config.json      training settings (same as results/dino/standardized_baseline/foldN/)
    training.csv     per-epoch train/val loss and val accuracies
  pooled_test.json   the five test.json files pooled over all 416 cases
  summary.csv        one row, same columns as results/summary.csv
```

Original copy: `$PSCRATCH/finalproject/standardized_results/dinov2_small` (ak2682).
Nothing here goes into Git; the aggregate files are in `results/dino/standardized_baseline/`.

`best.pt` is a dict with the same keys as the ResNet checkpoint: `weights` (full
`state_dict`, frozen encoder included, so no download is needed), `fold`, `image_size`
(448), `epoch`, `config`. Load it exactly like `collect_resnet` in `ensemble/evaluate.py`:

```python
from dino.baseline.data import WireDataset, load_data
from dino.baseline.model import WireModel

saved = torch.load(path, map_location="cpu", weights_only=True)
model = WireModel(saved["image_size"], pretrained=False,
                  dropout=saved["config"].get("dropout", 0.3)).to(device)
model.load_state_dict(saved["weights"])
model.eval()
cases, split = load_data(data_root, saved["fold"])
dataset = WireDataset(data_root, cases, split["validation"], saved["image_size"])
# dataset[i] -> (images [7,3,448,448], history [12], label); model(images, history) -> logits [B,2]
# P(advance) = logits.softmax(1)[:, 1]
```

`WireDataset` walks `split[...]` in order, so case order matches `resnet/` for the same
fold. Needs `timm` (`pip install --user timm`); the timm model is built with
`pretrained=False`, so nothing is fetched from the hub.

`validation.json` and `test.json` already contain `case_indices`, `labels` and
`advance_probabilities` (P(advance), 0.5 threshold) in that same order, plus the
protocol metrics, so a late-fusion ensemble can use them without running the model.

The experiments in the parent folder (`dino/`) are separate from this baseline: different
head, fixed epoch count, cached features, probes, tiles and fine-tuning. See
[`../PLAN.md`](../PLAN.md).

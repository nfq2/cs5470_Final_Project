# RepViT-M0.9 — Nicole

## Short-horizon data loader

This folder contains Nicole's RepViT-M0.9 implementation. Run the commands below
from this folder (`cd repvit_m0_9` from the repository root). Model construction,
training, and evaluation scripts are included.

Requires Python 3.10+, PyTorch, torchvision, timm, and Pillow (`requirements.txt`). Install the
appropriate GPU-enabled PyTorch build in your Perlmutter environment for training.
Keep the real dataset, case outputs, and images on Perlmutter, outside this repo.
Local tests use synthetic images only.

`data_loader.py` loads the supplied `manifest.json` and `splits.json`. Split
entries index the manifest's cases array. `fold` selects the stored fold ID;
everyone should use the same fold for comparisons. No new splits are generated.

```python
from data_loader import ImagePreprocess, make_dataloader

loader = make_dataloader(
    "/pscratch/sd/n/nicoleq/shorthorizon2",
    split="train", fold=0, batch_size=8,
    transform=ImagePreprocess(size=(224, 224)),
)
batch = next(iter(loader))
```

| Batch field | Shape | Meaning |
|---|---|---|
| `goal` | B × 3 × H × W | Target-marked map; separate from history |
| `frames` | B × 6 × 3 × H × W | Chronological f1 through f6 |
| `history_offsets_s` | B × 6 | Original relative times in seconds |
| `depth_relative_mm` | B × 6 | Original relative insertion depths in mm |
| `label` | B | Integer RETRACT=0 / ADVANCE=1 |

Training is shuffled reproducibly; validation and test preserve supplied order.
Partial batches are retained. Images and numeric inputs are float32 CPU tensors;
labels are int64 (suitable for two-logit cross entropy). For a one-logit binary
loss, convert labels to float in the training code. Move tensors to the GPU in
the training/evaluation loop. Numeric inputs are left in their original units;
if standardizing, fit statistics on training data only.

Configure image size, mean, and standard deviation for the selected RepViT
pretrained weights, or supply a deterministic PIL-to-
CHW-tensor transform. Defaults use ImageNet normalization and a provisional
224×224 direct resize; inspect guidewire visibility on Perlmutter before settling
on resolution. The loader does not crop or randomly augment images. Independent
random transforms would misalign frames; any future spatial augmentation must
be synchronized at case level and preserve the target.

On Perlmutter, check one batch without printing case data (no GPU required):

```bash
python3 data_loader.py --data-root "$PSCRATCH/shorthorizon2" --fold 0
```

Run synthetic tests:

```bash
python3 -m unittest discover -s tests -v
```

## Model and training

`model.py` uses timm's `repvit_m0_9.dist_450e_in1k` pretrained backbone.
[Upstream implementation and checkpoint definitions](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/repvit.py).
Each image is encoded with shared weights. Goal features and six ordered history
features are concatenated with 12 timing/depth values, then passed through a
128-unit ReLU/dropout classifier with two output logits. Numeric normalization is
fit on training cases only and stored in the checkpoint. No case IDs are inputs.
The initial full-image resize deliberately avoids cropping away the target; it
is not the original ImageNet center-crop evaluation recipe.

Install dependencies in your chosen Perlmutter Python environment:

```bash
python3 -m pip install -r requirements.txt
```

Download public pretrained weights once on a login node before allocating a GPU.
Use the same HF_HOME for training so the cached weights are found:

```bash
export HF_HOME="$PSCRATCH/huggingface"
python3 -c 'from model import WireModel; WireModel(pretrained=True); print("Weights cached")'
```

Inside an allocated GPU session, train the classifier with the backbone frozen:

```bash
export HF_HOME="$PSCRATCH/huggingface"
export HF_HUB_OFFLINE=1
python3 train.py \
  --data-root "$PSCRATCH/shorthorizon2" \
  --output-dir "$PSCRATCH/repvit_runs/fold0_head" \
  --fold 0 --epochs 20 --batch-size 8
```

Training defaults to CUDA and FP32, ordinary unweighted cross entropy, AdamW,
head learning rate 0.001, and seed 42. Frozen backbone parameters and BatchNorm
statistics stay fixed. The best checkpoint is chosen by **validation loss**;
training never loads the test split. Each output directory must be new to prevent
accidentally overwriting a run. Epoch metrics are saved in `history.jsonl`.

Optional fine-tuning from that checkpoint (all backbone weights become trainable):

```bash
python3 train.py \
  --data-root "$PSCRATCH/shorthorizon2" \
  --output-dir "$PSCRATCH/repvit_runs/fold0_finetune" \
  --init-checkpoint "$PSCRATCH/repvit_runs/fold0_head/best.pt" \
  --finetune --lr 1e-4 --backbone-lr 1e-5 --epochs 10
```

Compare candidates using validation results before selecting one for test
assessment. Do not use repeated test results to tune settings. Fold and image
preprocessing must match when initializing from a prior checkpoint.

## Evaluation

On an allocated GPU, evaluate the chosen checkpoint:

```bash
python3 evaluate.py \
  --data-root "$PSCRATCH/shorthorizon2" \
  --checkpoint "$PSCRATCH/repvit_runs/fold0_head/best.pt" \
  --output "$PSCRATCH/repvit_runs/fold0_head/test_metrics.json" \
  --latency
```

Checkpoint loading does not download pretrained weights. The saved fold and
preprocessing are reused automatically. Use `--split validation` for validation
assessment. Output files must be new. Accuracy values are fractions, not percent.
Reports include overall accuracy, per-class recall (called advance/retract
accuracy), balanced accuracy, the same-split always-advance baseline, and a
confusion matrix with rows=true and columns=predicted, ordered [retract, advance].
A missing class produces null for its recall rather than a misleading zero.

Latency uses FP32, batch size one, ten warmup forwards, and three timed forwards
per case, with CUDA synchronization. It covers all seven image encodings and the
classifier, but excludes disk loading, preprocessing, and CPU-to-GPU transfer.
Mean/median/p95 and device name are recorded. Compare only with results measured
using the same scope, GPU, precision, and input size. These are operator-action
proxy labels; episode splits do not establish anatomy generalization.

For synthetic CPU smoke tests only, `train.py` accepts `--device cpu
--random-init --image-size 64 --epochs 1`. Random-initialization results are marked
in checkpoints/reports and are not a pretrained-model baseline. GPU memory,
pretrained weight download, real-data accuracy, and CUDA timing require verification
on Perlmutter.

## Diagnose all-advance predictions

Inside a GPU allocation, run this training-only check of actual weight updates,
feature variation, and memorization of eight balanced training cases:

```bash
srun --ntasks=1 --gpus=1 python diagnose.py \
  --data-root "$PSCRATCH/shorthorizon2" \
  --checkpoint "$PSCRATCH/repvit_runs/fold0_head/best.pt"
```

It prints aggregate statistics only and does not change saved checkpoints.
A positive classifier weight change proves updates occurred. Tiny-batch fitting
uses cached frozen features with dropout disabled; it checks fitting capacity,
not generalization. Omit the checkpoint argument to check a fresh pretrained
backbone and newly initialized head. A failed memorization check calls for further
inspection; it does not by itself identify a cause.

## Retract-weighted training and fine-tuning

`--retract-weight 1.5` multiplies each true retract case's cross-entropy by 1.5;
advance cases retain weight 1.0. The weighted per-case losses are averaged over
the batch. This increases the relative penalty for missed retracts; it does not
reward incorrect retract predictions. Default 1.0 preserves the original run.

`loss` remains ordinary unweighted cross-entropy for consistent reporting.
`weighted_training_loss` records the optimization objective when weighting is on.
Validation and test are unweighted, and checkpoint selection still uses minimum
unweighted validation loss. The weight and fine-tuning settings are saved in the
run config and checkpoint. Improved retract recall can trade off advance recall;
compare both and balanced accuracy on validation before test evaluation.

For a controlled comparison, start both runs from the pretrained backbone and the
same seed, not from a previous trained classifier. First run weighted head-only:

```bash
srun --ntasks=1 --gpus=1 python train.py \
  --data-root "$PSCRATCH/shorthorizon2" \
  --output-dir "$PSCRATCH/repvit_runs/fold0_weighted_head" \
  --fold 0 --epochs 20 --batch-size 8 --num-workers 2 \
  --lr 1e-4 --retract-weight 1.5
```

Then the same configuration with the backbone unfrozen:

```bash
srun --ntasks=1 --gpus=1 python train.py \
  --data-root "$PSCRATCH/shorthorizon2" \
  --output-dir "$PSCRATCH/repvit_runs/fold0_weighted_finetune" \
  --fold 0 --epochs 20 --batch-size 8 --num-workers 2 \
  --lr 1e-4 --retract-weight 1.5 --finetune --backbone-lr 1e-5
```

In fine-tuning mode all encoder weights and BatchNorm statistics can change.
The smaller backbone learning rate limits the size of changes to pretrained
features. Fine-tuning uses more GPU memory and can overfit the small training set;
its benefit must be measured rather than assumed.

## Input and BatchNorm audit

Use `audit.py --data-root ... --checkpoint ... --output-dir ...` inside a GPU
allocation. Choose a new output directory under `$PSCRATCH`, outside the repo.
It independently checks eight sampled training cases against decoded image files,
reports duplicate-image counts, evaluates train and validation in evaluation mode,
and probes BatchNorm sensitivity on training cases with dropout disabled. It never
reads test cases. The checkpoint epoch is printed: this audits `best.pt`, not the
last epoch unless those coincide. Model buffers and checkpoint files are preserved.

The audit directory includes `audit.json`, native goal/current-frame previews, and
resized seven-view contact sheets for two training cases. These PNGs contain
sensitive dataset imagery: keep the entire directory on Perlmutter. View using an
approved Perlmutter-hosted notebook/desktop; do not commit or download previews.
Inspect whether the wire and goal marking survive the actual resize, comparing the
native images at full resolution. Zero tensor mismatches verifies sampled file
loading/order, not image or label correctness. Duplicate frames may be legitimate.
A large BatchNorm prediction shift suggests sensitivity to running statistics,
not proof of a bug or permission to use validation batches to recalibrate them.

## Fixed eight-case fine-tuning check

To check fitting capacity before another full experiment, run on a GPU:

```bash
srun --ntasks=1 --gpus=1 python overfit.py \
  --data-root "$PSCRATCH/shorthorizon2" \
  --output-dir "$PSCRATCH/repvit_overfit_$(date +%Y%m%d_%H%M%S)"
```

This starts a fresh pretrained backbone and classifier, selecting four cases of
each class from the chosen training fold using seed 42. The same eight cases are
loaded once and reused for 200 optimizer steps, with classifier LR 1e-4 and
backbone LR 1e-5. Both parts train; BatchNorm running statistics update during
training. Dropout, weight decay, and class weighting are disabled to make fitting
easier. Numeric normalization is fit on training data only.

Every 25 steps it reports scores on these same eight cases using running
statistics (`eval_mode`) and temporary batch statistics (`batch_statistics_mode`),
with dropout off in both. Scoring does not modify buffers. First-update gradient
norms and parameter changes verify both backbone and classifier learn. Config,
progress, and summary JSON are saved, but no model checkpoint or case identifiers.
Existing training runs are untouched. No validation or test cases are accessed.

Near-perfect tiny-set accuracy and low loss show the pipeline can fit a small
set, not that it generalizes. A gap between the two modes suggests BatchNorm
sensitivity in this deliberately tiny experiment. Failure to fit after 200 steps
calls for further diagnosis, not a definitive conclusion about the model.

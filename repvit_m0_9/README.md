# RepViT-M0.9 — Nicole

## Short-horizon data loader

This folder contains Nicole's RepViT-M0.9 implementation. Run the commands below
from this folder (`cd repvit_m0_9` from the repository root). Model construction,
training, and evaluation scripts are included.

Requires Python 3.10+, PyTorch, torchvision, timm, and Pillow (`requirements.txt`). Install the
appropriate GPU-enabled PyTorch build in your Perlmutter environment for training.
Keep the real dataset, case outputs, and images on Perlmutter, outside this repo.
Local tests use synthetic images only.

`data_loader.py` gets cases and splits from the team's shared loader
(`common/data.py`), which reads `cases.jsonl` and `splits.json`. Split
entries index the cases in file order. `fold` selects the stored fold ID;
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
statistics stay fixed. `best.pt` is chosen by **validation loss**;
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
Validation and test are unweighted. `best.pt` uses minimum unweighted validation
loss; `best_balanced.pt` separately tracks balanced accuracy. The weight and fine-tuning settings are saved in the
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

## Controlled dropout comparison

`train.py --dropout 0` disables classifier dropout. The default remains 0.3;
this setting is saved in config/checkpoints and printed at startup. Evaluation
always disables dropout, and older checkpoints remain loadable.

Every epoch now includes `train_eval`: the entire training set scored after the
updates, with dropout disabled and BatchNorm using running statistics, exactly
like validation. This extra pass performs no updates and uses a separate loader
RNG so training order and dropout randomness are not affected. `train` still
reports metrics collected during updates. Compare `train_eval` with `validation`
to distinguish poor training-set fitting from poor generalization. The extra pass
adds time; `best.pt` selection remains unweighted validation loss.

For the comparison with the previous unweighted fine-tuning run, keep weight
decay (1e-4), seed, fold, learning rates, batch size, and resolution unchanged:

```bash
srun --ntasks=1 --gpus=1 python train.py \
  --data-root "$PSCRATCH/shorthorizon2" \
  --output-dir "$PSCRATCH/repvit_runs/fold0_finetune_no_dropout_$(date +%Y%m%d_%H%M%S)" \
  --fold 0 --epochs 20 --batch-size 8 --num-workers 2 \
  --lr 1e-4 --retract-weight 1.0 --finetune --backbone-lr 1e-5 --dropout 0
```

This is full-data training, not the eight-case diagnostic. It starts from fresh
pretrained weights and a fresh classifier. Improved tiny-case fitting does not
guarantee improvement on validation; assess the recorded curves before testing.

## Fine-tune only the last backbone stage

Use `--finetune-last-stage` instead of `--finetune`. Only `encoder.stages[-1]`
and the classifier receive updates. Earlier backbone parameters and BatchNorm
running statistics stay frozen. The last stage uses `--backbone-lr`; the
classifier uses `--lr`. With neither flag the entire backbone stays frozen.
The two flags are mutually exclusive.

```bash
srun --ntasks=1 --gpus=1 python train.py \
  --data-root "$PSCRATCH/shorthorizon2" \
  --output-dir "$PSCRATCH/repvit_runs/fold0_last_stage_$(date +%Y%m%d_%H%M%S)" \
  --fold 0 --epochs 20 --batch-size 8 --num-workers 2 \
  --lr 1e-4 --retract-weight 1.0 --dropout 0 \
  --finetune-last-stage --backbone-lr 1e-5
```

This keeps the no-dropout run's settings and changes only the trainable backbone
scope. A fresh pretrained model is used unless `--init-checkpoint` is supplied.

New runs save three independent files after completed epochs:

| File | Selection |
|---|---|
| `best.pt` | Lowest unweighted validation loss (same meaning as before) |
| `best_balanced.pt` | Highest validation balanced accuracy; ties prefer lower validation loss |
| `last.pt` | Latest completed epoch; final epoch if the run completes |

All contain model weights, normalization, configuration, epoch, validation metrics,
and selection criterion. Writes replace the old file only after the new file is
complete. These are model checkpoints, not full optimizer/RNG resume snapshots.
An interrupted partial epoch is not saved. If validation lacks a class, balanced
accuracy is undefined and no balanced checkpoint is selected. Earlier runs cannot
recover discarded epochs using this change.

`evaluate.py --checkpoint ...` accepts any of the three files and reports its
selection criterion. Decide the criterion using validation and the project goals
before evaluating test performance; do not pick among them based on test results.
Compare methods under the same declared selection rule.

### Validation alignment and fold comparison

The audit now checks both train and validation samples, saving their previews in
separate `train/` and `validation/` subdirectories. `--samples 100000` checks every
case in those two splits (more disk reads). It compares loaded labels, time/depth
tensors, and all seven images to their manifest record. When supplied, label names,
source timestamps versus relative history offsets, and chronological source frame
indices are checked too. The goal is excluded from chronology checks. It reports
case/episode overlap and split class counts without emitting identifiers. Missing
optional source metadata is reflected by zero checks; it is not a pass.

These checks establish consistency with the provided records, not whether the
original action-label generation used the correct future time window. That requires
the benchmark construction details or advisor confirmation. Anatomy IDs are absent.

After the audit, repeat the existing last-stage, weight-1.2, dropout-0.1 configuration
on fold 1 without changing other settings:

```bash
srun --ntasks=1 --gpus=1 python train.py \
  --data-root "$PSCRATCH/shorthorizon2" \
  --output-dir "$PSCRATCH/repvit_runs/fold1_last_stage_w12_d01_$(date +%Y%m%d_%H%M%S)" \
  --fold 1 --epochs 20 --batch-size 8 --num-workers 4 \
  --lr 1e-4 --retract-weight 1.2 --dropout 0.1 \
  --finetune-last-stage --backbone-lr 1e-5
```

Compare validation results under the same checkpoint criterion across folds. This
is a cross-fold validation check, not test evaluation; no fold's test data is used
by training. Supplied folds may overlap each other, so scores are not necessarily
independent and pooled test use would need a declared evaluation protocol.

## Numerical-history-only baseline

`history_baseline.py` uses only six relative timestamps and six relative insertion
depths. It reads the supplied manifest and fold indices, but never opens images or
constructs RepViT. The model is 12 inputs → 32-unit ReLU → dropout → two logits
(482 parameters). Standardization is fitted on training cases only and saved in
model buffers. No identifiers, absolute timestamps, labels, or image-derived fields
are included as inputs.

Run the fixed comparison on fold 0 inside your existing allocation:

```bash
srun --ntasks=1 --gpus=1 python history_baseline.py \
  --data-root "$PSCRATCH/shorthorizon2" \
  --output-dir "$PSCRATCH/repvit_runs/fold0_history_w12_d01_$(date +%Y%m%d_%H%M%S)" \
  --fold 0 --epochs 20 --batch-size 8 --lr 1e-4 \
  --retract-weight 1.2 --dropout 0.1
```

CPU is also supported with `--device cpu` on an appropriate allocated node. No
pretrained download is needed. This keeps the latest visual run's fold, seed,
batch size, learning rate, class weights, dropout, epochs and AdamW weight decay.
The smaller classifier is a simple baseline, not an exact matched-capacity ablation.
It reports train, train_eval, and validation, writes history/config, and uses the
same three checkpoint names. For this comparison, use `best_balanced.pt` for BOTH
methods; its selected epoch and validation metrics are printed at completion.
Poor performance alone does not prove numerical inputs have no signal: this is one
model and optimization budget, so also inspect its training-set fitting.

`evaluate.py` detects numerical checkpoints and uses the image-free loader. To
recheck validation (not test) with an exact run path:

```bash
python evaluate.py --data-root "$PSCRATCH/shorthorizon2" \
  --checkpoint /path/to/run/best_balanced.pt \
  --split validation --device cpu --output /path/to/run/validation_metrics.json
```

Do not evaluate test while selecting configurations. Any later latency comparison
must use the same device; the numerical forward naturally excludes visual encoding.

### Shared team loader and evaluation

RepViT uses `common.data.load_split` and the shared image dataset, with a small
adapter that separates `images[0]` into `goal` and `images[1:]` into `frames`.
RepViT retains its deterministic resize/normalization and input validation.
The numerical-history baseline uses the same split loader without opening images.

`evaluate.py` computes the team's `common.metrics.report` (including AUROC) from
probabilities collected during the same forward pass as loss and confusion matrix.
Predictions now follow the shared threshold: P(advance) >= 0.5 means advance.
Only exact ties differ from older RepViT argmax reports, which chose retract.
Existing checkpoints remain compatible; no retraining is required to evaluate them.
RepViT's optional latency remains its existing all-case measurement, rather than
the shared helper's repeated single-input benchmark; do not compare those timing
protocols directly. Use `shared_report` for classification comparisons.

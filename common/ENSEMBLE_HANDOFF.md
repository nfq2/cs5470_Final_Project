# Ensemble artifact handoff

This checklist applies to ResNet-18, MobileNetV3-Small, RepViT-M0.9, and DINOv2.
Aggregate CSV files are for plots; ensemble training instead uses case-level
prediction JSON files and trained checkpoints. Those artifacts stay on
Perlmutter because they contain derived case-level data and large model weights.

Each model's repository results folder should contain an
`ensemble_artifacts.csv` manifest with counts, Perlmutter paths, checkpoint
epochs, and SHA-256 hashes. ResNet's completed example is
`results/resnet_baseline/ensemble_artifacts.csv`.

## What every model owner provides

For every fold, provide:

- `best.pt`: that fold's minimum-validation-loss checkpoint;
- `validation.json`: aligned validation labels and advance probabilities;
- `test.json`: aligned held-out test labels and advance probabilities; and
- the exact preprocessing/configuration stored with the checkpoint.

Use one scratch artifact root with this layout:

```text
MODEL_ROOT/
  fold0/best.pt
  fold0/validation.json
  fold0/test.json
  ...
  fold4/best.pt
  fold4/validation.json
  fold4/test.json
```

Every prediction JSON must contain:

```text
protocol: midpoint_baseline_v1
model, fold, split, epoch
case_indices
labels
advance_probabilities
image_size, precision, gpu
```

The three arrays must have equal length. Labels use retract=0 and advance=1.
The manifest records expected counts and hashes so copied files can be checked.

For a given fold and split, the two models must have identical `case_indices`
and `labels` in the same order. Do not email or commit checkpoints, images, or
case-level prediction files. Share them through an approved Perlmutter project
directory with group-readable permissions, or give the owner a readable scratch
path.

## Pair selection and stacker training

There are six possible pairs among ResNet, MobileNet, RepViT, and DINO. For each
pair and each fold:

1. Use the two models' validation probabilities as the two stacker inputs.
2. Fit a class-balanced logistic regression using validation labels. Use
   out-of-fold predictions inside validation when comparing pairs or thresholds.
3. Rank pairs using validation balanced accuracy and AUROC only.
4. Select one pair before inspecting its pooled test result.
5. Apply each fold-specific stacker once to that fold's test probabilities.
6. Pool the five test outputs and compute metrics over all 416 cases.

Training-history CSVs and aggregate metrics are not stacker inputs. Test labels
are used only for the final report, never to fit weights, choose a pair, or tune
the decision threshold.

## Final latency

Precomputed probabilities cannot measure ensemble latency. After selecting the
pair, load both real checkpoints and time this complete prediction path:

```text
seven images + history
  -> model A forward
  -> model B forward
  -> probability/log-odds features
  -> logistic stacker
  -> advance/retract
```

Use one GPU, batch size one, FP32 with TF32 disabled, 20 warmups, and 100
synchronized repetitions per fold. Inputs are already preprocessed and on the
GPU, matching `common.latency.forward_latency`. Report the pooled median over
500 timings and state that I/O, preprocessing, transfer, and startup are
excluded. The stacker is negligible; the reported number mainly measures both
backbones. Measure the selected pair directly on the same GPU rather than adding
latencies previously collected on different GPU models.

If the ensemble becomes the deployable final model, also train its two base
models on all data permitted by the final protocol and report an end-to-end
latency separately if required.

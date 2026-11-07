# Shared architecture baseline protocol

Use this protocol for the midpoint comparison of ResNet-18,
MobileNetV3-Small, RepViT-M0.9, and DINO. `protocol.py` is the source of truth
for the exact values. Each model owner adapts their own folder to this interface;
the common code does not implement or alter their architecture.

## Fixed training setup

- Use the five supplied episode-based folds; never create a random split.
- Start each fold from the same architecture's public pretrained weights.
- Train only the new classifier; keep backbone weights and BatchNorm statistics
  frozen.
- Use all seven ordered images and the same 12 numerical history values.
- Directly resize the full image to 448 x 448 and use ImageNet normalization.
  Do not crop or augment the baseline.
- Use a 128-unit ReLU classifier, dropout 0.3, unweighted cross-entropy, Adam
  with learning rate 0.001 and zero weight decay, batch size 8, 20 epochs, and
  seed 42.
- Use FP32 with TF32 disabled. After every epoch, compute ordinary unweighted
  cross-entropy on that fold's validation set. Save `best.pt` whenever this
  validation loss is strictly lower than every earlier epoch; therefore a tie
  keeps the earlier checkpoint.
- Do not tune settings using any test fold.

Only the architecture should change in this baseline. Architecture-specific
optimization experiments belong in separately named runs.

The standard baseline settings are deliberately simple rather than necessarily
optimal for every architecture. If a model normally prefers a different input
size, optimizer, fine-tuning strategy, or augmentation, report that later as a
separate optimized variant—not as the standardized baseline.

## Fixed evaluation setup

Train five independent models. The model trained for fold 0 is evaluated only
on fold 0's held-out test group, and likewise for folds 1 through 4. Save the
case index, true label, and advance probability for every test prediction.

Checkpoint selection is performed separately inside each fold. This produces
five `best.pt` files, potentially from five different epochs. Do not select the
best-performing fold, compare validation losses between folds, or choose a
checkpoint using test accuracy.

Pool the five test prediction files and compute metrics once over all 416 cases.
Do not choose the best fold and do not take an unweighted mean of five fold
percentages. The aggregator rejects missing, duplicated, or misplaced cases.

Measure latency with batch size one and inputs already on the GPU: 20 warmups
and 100 synchronized forwards per fold. It includes all seven image encodings,
history fusion, and classification, while excluding file I/O, preprocessing,
transfer, and startup. All architectures must use the same GPU model.

## Run one architecture

From the repository root, using the Python environment required by that model:

```bash
python -m common.run_five_folds \
  --model-dir resnet \
  --data-root "$PSCRATCH/finalproject/shorthorizon2" \
  --output-root "$PSCRATCH/finalproject/standardized_results/resnet18"
```

Replace `resnet` and the final output name for another architecture. A compatible
model folder accepts the common `train.py` and `evaluate.py` arguments used by
the runner. The output contains five fold directories, `pooled_test.json`, and a
one-row `summary.csv`.

## Required model-folder interface

Each teammate keeps ownership of their model code. To use the common runner,
their `train.py` must accept:

```text
--data-root PATH --output-dir NEW_DIRECTORY --fold 0..4
```

It must use the values in `common.protocol` and save its minimum-validation-loss
checkpoint as `best.pt`. Its `evaluate.py` must accept:

```text
--data-root PATH --checkpoint CHECKPOINT --split validation|test --output NEW_JSON
```

Every evaluation JSON must contain these fields so the common aggregator can
verify and pool it:

```text
protocol               midpoint_baseline_v1
model                  stable architecture name
fold, split            evaluated fold and split
case_indices           original indices in exact split order
labels                 true 0=retract / 1=advance labels
advance_probabilities  model P(advance) values in the same order
latency_samples_ms     100 standardized batch-one timings
gpu, image_size, precision
```

Use `common.metrics.report` for per-fold metrics and `common.latency.forward_latency`
for timing. The aggregator recomputes the final metrics from raw held-out
predictions; it does not trust pre-averaged percentages.

After all architectures finish, create the comparison table and plot:

```bash
python -m common.compare_results \
  --inputs "$PSCRATCH"/finalproject/standardized_results/*/pooled_test.json \
  --csv "$PSCRATCH/finalproject/standardized_results/comparison.csv" \
  --plot "$PSCRATCH/finalproject/standardized_results/accuracy_latency.png"
```

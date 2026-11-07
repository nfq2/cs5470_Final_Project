# MobileNetV3-Small — Initial Baseline Results

## Method

ImageNet-pretrained MobileNetV3-Small with a frozen backbone. Seven image feature vectors are concatenated with insertion-depth and timing history and passed through a 128-unit classifier.

- **Images:** target-marked goal map and six chronological frames, resized to 448×448.
- **Training:** Adam, learning rate 0.001, batch size 8, 20 epochs, seed 42.
- **Checkpoint selection:** lowest validation loss.
- **Evaluation:** five supplied episode-based folds, with a fresh model trained for each fold.

## Standardized pooled test results

The five existing checkpoints were re-evaluated under `midpoint_baseline_v1`
without retraining. Each checkpoint was evaluated on its own held-out test
fold, and the metrics were computed from all 416 predictions together.

| Metric | Result |
|---|---:|
| Test cases | 416 |
| Overall accuracy | 58.2% (242/416) |
| Always-advance baseline | 62.3% (259/416) |
| Advance recall | 78.4% (203/259) |
| Retract recall | 24.8% (39/157) |
| Balanced accuracy | 51.6% |
| Pooled AUROC | 0.4962 |
| Median forward latency | 8.13 ms |
| Mean forward latency | 8.05 ms |
| Timed forward passes | 500 |
| GPU | NVIDIA A100-PCIE-40GB |

Use the pooled median for the standardized comparison when the other models
were timed on the same GPU model and with the same timing scope.

## Per-fold test results

| Fold | Test cases | Accuracy | Always-advance baseline | Advance recall | Retract recall | Earlier mean forward latency (A100-SXM4-80GB) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 69 | 58.0% | 62.3% | 86.0% | 11.5% | 3.786 ms |
| 1 | 72 | 63.9% | 56.9% | 100.0% | 16.1% | 3.724 ms |
| 2 | 71 | 62.0% | 62.0% | 100.0% | 0.0% | 3.783 ms |
| 3 | 81 | 46.9% | 61.7% | 56.0% | 32.3% | 3.787 ms |
| 4 | 123 | 60.2% | 65.9% | 65.4% | 50.0% | 3.698 ms |

## Latency setup

The standardized run used one **NVIDIA A100-PCIE-40GB**, FP32 with TF32
disabled, batch size one decision, and 20 warm-ups and 100 synchronized
timed forwards per fold. The median and mean above are computed over the
500 timings together.

Timing includes all seven image encodings and the classifier. It excludes loading, preprocessing, CPU-to-GPU transfers, and startup.

The earlier **3.70–3.79 ms** range is from per-fold means measured on an
**A100-SXM4-80GB**. Those measurements are separate from the standardized
A100-PCIE timing and should not be used interchangeably.

## Findings

The baseline predicts quickly but falls **4.1 percentage points below always-advance accuracy**. It correctly identifies only **39 of 157 retract cases**, indicating weak retract recognition. These results describe the initial frozen-backbone configuration; the cause of its limited accuracy has not yet been isolated.

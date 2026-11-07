# MobileNetV3-Small — Initial Baseline Results

## Method

ImageNet-pretrained MobileNetV3-Small with a frozen backbone. Seven image feature vectors are concatenated with insertion-depth and timing history and passed through a 128-unit classifier.

- **Images:** target-marked goal map and six chronological frames, resized to 448×448.
- **Training:** Adam, learning rate 0.001, batch size 8, 20 epochs, seed 42.
- **Checkpoint selection:** lowest validation loss.
- **Evaluation:** five supplied episode-based folds, with a fresh model trained for each fold.

## Test results

| Fold | Test cases | Accuracy | Always-advance baseline | Advance recall | Retract recall | Mean forward latency |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 69 | 58.0% | 62.3% | 86.0% | 11.5% | 3.786 ms |
| 1 | 72 | 63.9% | 56.9% | 100.0% | 16.1% | 3.724 ms |
| 2 | 71 | 62.0% | 62.0% | 100.0% | 0.0% | 3.783 ms |
| 3 | 81 | 46.9% | 61.7% | 56.0% | 32.3% | 3.787 ms |
| 4 | 123 | 60.2% | 65.9% | 65.4% | 50.0% | 3.698 ms |
| **Combined** | **416** | **58.2%** | **62.3%** | **78.4%** | **24.8%** | **3.70–3.79 ms range** |

Combined balanced accuracy: **51.6%**. Combined metrics use pooled correct/class counts rather than averaging fold percentages.

## Latency setup

Measured on one **NVIDIA A100-SXM4-80GB**, using FP32 with TF32 disabled, batch size one decision, 20 warm-ups, and 100 timed repetitions with CUDA synchronization.

Timing includes all seven image encodings and the classifier. It excludes loading, preprocessing, CPU-to-GPU transfers, and startup.

## Findings

The baseline predicts quickly but falls **4.1 percentage points below always-advance accuracy**. It correctly identifies only **39 of 157 retract cases**, indicating weak retract recognition. These results describe the initial frozen-backbone configuration; the cause of its limited accuracy has not yet been isolated.

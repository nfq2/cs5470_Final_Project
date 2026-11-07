# MobileNetV3-Small baseline

Amelia's five saved frozen-backbone MobileNet checkpoints were re-evaluated
under `midpoint_baseline_v1`. Each checkpoint was selected by minimum
validation loss in its own fold. No retraining was performed.

| File | Contents |
|---|---|
| [`pooled_test.json`](pooled_test.json) | Standardized metrics across all 416 held-out cases and 500 forward timings |
| [`summary.csv`](summary.csv) | One graph-ready row for the pooled run |
| [`config.json`](config.json) | Training and evaluation settings |
| [`folds.csv`](folds.csv) | Earlier per-fold test summaries, timed on the A100-SXM4-80GB |
| [`training_history.csv`](training_history.csv) | All 100 epoch summaries at the precision printed in the original training logs |

The pooled run achieved **58.2% accuracy**, **78.4% advance recall**,
**24.8% retract recall**, **51.6% balanced accuracy**, and **0.4962 AUROC**.
Always predicting advance achieved **62.3%** on the same cases.

The pooled median was **8.13 ms** on an **NVIDIA A100-PCIE-40GB**, using
batch-one FP32 forward timing. Inputs were already on the GPU. Use the pooled
metrics for comparison; the earlier fold timings are from different hardware.
ResNet and DINOv2's standardized timings in the shared table use an
A100-SXM4-40GB, so they are not a matched hardware latency comparison.

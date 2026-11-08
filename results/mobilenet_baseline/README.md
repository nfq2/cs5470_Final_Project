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

The matched rerun measured a pooled median of **4.34 ms** (mean **4.26 ms**)
on an **NVIDIA A100-SXM4-40GB**, the same GPU model as the other standardized
baselines. It used the same five checkpoints and produced the same test
metrics. Timing is batch-one FP32 forward latency with inputs already on the
GPU. The original **8.13 ms** median on the A100-PCIE-40GB and the earlier
A100-SXM4-80GB fold timings are historical measurements.

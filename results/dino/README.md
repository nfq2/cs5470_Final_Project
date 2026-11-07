# DINOv2 results (Akhil)

Aggregate files only; per-case predictions stay on Perlmutter. Everything is pooled over
the five folds unless marked fold 0.

## Standardized baseline (`standardized_baseline/`)

`midpoint_baseline_v1` via `common/run_five_folds.py` on `dino/baseline`: frozen DINOv2-S
(ViT-S/14, CLS token per image), protocol classifier, lowest validation loss per fold.
`pooled_test.json` and `summary.csv` are the files for the four-architecture plot.

| Split | n | Accuracy | Advance | Retract | Balanced | AUROC | Always-advance | Latency (median) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pooled test | 416 | 61.5% | 98.1% | 1.3% | 49.7% | 0.490 | 62.3% | 29.0 ms (A100-40GB) |

## Experiments (`experiments/`)

Not protocol runs: attention-pooling head over patch tokens, derived history features,
AdamW, 40 fixed epochs (last epoch kept), seed 0. Details and the diagnostics that do not
produce result files (probes, MIL, wire probe, human check, clicked tips) are in
[`dino/PLAN.md`](../../dino/PLAN.md).

| Run | Split | n | Accuracy | Retract | AUROC | Model / end-to-end latency |
|---|---|---:|---:|---:|---:|---:|
| S @ 448 head | val | 352 | 54.0% | 22.1% | 0.483 | 28.8 / 30.5 ms |
| S @ 448 head | test | 416 | 61.1% | 12.1% | 0.470 | |
| History only | val | 352 | 56.0% | 4.5% | 0.525 | |
| History only | test | 416 | 60.6% | 6.4% | 0.492 | |
| S @ 448, 2x2 tiles | val | 352 | 52.3% | 31.2% | 0.508 | 105.7 / 108.1 ms |
| S @ 448, 2x2 tiles | test | 416 | 53.8% | 23.6% | 0.462 | |
| Fine-tuned last 4 blocks (tiles, goal+f1+f6) | val | 352 | 56.2% | 36.4% | 0.563 | |
| Fine-tuned last 4 blocks (tiles, goal+f1+f6) | test | 416 | 53.4% | 33.8% | 0.502 | |
| S @ 224 head (fold 0 only) | test | 69 | 52.2% | 30.8% | 0.372 | 7.5 / 9.1 ms |
| S @ 672 head (fold 0 only) | test | 69 | 42.0% | 19.2% | 0.338 | 80.0 / 82.0 ms |
| B @ 448 head (fold 0 only) | test | 69 | 56.5% | 15.4% | 0.354 | 88.6 / 90.3 ms |

Always-advance: 56.2% on pooled validation, 62.3% on pooled test and on fold 0 test.
Model latency = preprocessed tensors on the GPU to logits; end-to-end = decoded frames in
host memory to a decision on the CPU; batch 1, FP32, TF32 off, 20 warm-up / 100 timed.

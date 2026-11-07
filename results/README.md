# Midpoint results

`summary.csv` is the shared, graph-ready table. Add one pooled five-fold row per
architecture using the same columns. Accuracy values are fractions (for example,
`0.75` means 75%).

Each model may also have a subfolder containing its raw evaluation output and
training history. Do not add datasets or large model checkpoints to Git.

The completed standardized ResNet baseline is in
[`resnet_baseline/`](resnet_baseline/). Its `summary.csv` is ready for the
four-architecture comparison, and `training_history.csv` contains all epoch-level
values needed for learning-curve graphs.

## Current result

`resnet_baseline/` contains Saesha's official standardized five-fold baseline.
It uses a frozen ImageNet-pretrained backbone, dropout 0.3, no class weighting,
and per-fold checkpoint selection by minimum validation loss. It excludes all
later optimization and ensemble experiments.

The reported latency is a batch-one FP32 forward pass for seven images and the
insertion-depth/time history. Inputs were already on the GPU, so data loading,
preprocessing, CPU-to-GPU transfer, and startup are excluded.

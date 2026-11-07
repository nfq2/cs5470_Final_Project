# Midpoint results

`summary.csv` is the earlier fold-0 table. Add legacy results using the same
columns and report accuracy values as fractions (for example, `0.75` means 75%).
The final accuracy-versus-latency plot must instead use each model's pooled
five-fold `summary.csv` produced by `common/run_five_folds.py`.

Each model may also have a subfolder containing its raw evaluation output and
training history. Do not add datasets or large model checkpoints to Git.

## Current result

`resnet_baseline/` contains Saesha's unmodified fold-0 ResNet-18 baseline run.
It uses a frozen ImageNet-pretrained backbone, dropout 0.3, no class weighting,
and checkpoint selection by minimum validation loss. These files do not include
any later optimization or ensemble experiments.

This is a preliminary fold-0 result produced before the shared five-fold
`midpoint_baseline_v1` protocol. Do not place it on the final standardized
four-architecture graph. New baseline runs should use `common/run_five_folds.py`
and report the pooled 416-case output.

The reported latency is a batch-one FP32 forward pass for seven images and the
insertion-depth/time history. Inputs were already on the GPU, so data loading,
preprocessing, CPU-to-GPU transfer, and startup are excluded.

# Midpoint results

`summary.csv` is the shared, graph-ready table. Add one row per model and split
using the same columns and report accuracy values as fractions (for example,
`0.75` means 75%). Use the test row for the final accuracy-versus-latency plot.

Each model may also have a subfolder containing its raw evaluation output and
training history. Do not add datasets or large model checkpoints to Git.

## Current result

`resnet_baseline/` contains Saesha's unmodified fold-0 ResNet-18 baseline run.
It uses a frozen ImageNet-pretrained backbone, dropout 0.3, no class weighting,
and checkpoint selection by minimum validation loss. These files do not include
any later optimization or ensemble experiments.

The reported latency is a batch-one FP32 forward pass for seven images and the
insertion-depth/time history. Inputs were already on the GPU, so data loading,
preprocessing, CPU-to-GPU transfer, and startup are excluded.

# ResNet-18 standardized five-fold baseline

This is Saesha's `midpoint_baseline_v1` result. Five independent frozen-backbone
ResNet-18 models were trained, one for each supplied episode fold. Each fold's
lowest-validation-loss checkpoint was evaluated only on its matching test group.
The five test groups cover all 416 cases exactly once.

## Pooled result

| Metric | Result |
|---|---:|
| Cases | 416 |
| Overall accuracy | 62.26% |
| Advance accuracy | 100.00% |
| Retract accuracy | 0.00% |
| Balanced accuracy | 50.00% |
| AUROC | 48.62% |
| Always-advance accuracy | 62.26% |
| Median forward latency | 6.149 ms |

The model predicted advance for every held-out case, so its overall accuracy
exactly equals the always-advance baseline. This is an important negative result:
the frozen ResNet features and unweighted classifier did not learn a useful
retract decision at the fixed 0.5 threshold.

`summary.csv` is the graph-ready pooled row. `folds.csv` gives the checkpoint
epoch and metrics for each held-out fold. `pooled_test.json` is the same pooled
result in JSON, and `config.json` records the standardized settings.
`training_history.csv` contains every epoch's training loss, validation loss,
and validation accuracies for all five folds; use its `fold` column to plot each
curve separately or aggregate across folds.

Latency comprises 500 synchronized measurements: 100 from each fold on an
NVIDIA A100-SXM4-40GB. It is a batch-one forward over seven images plus numeric
history with inputs already on the GPU; I/O, preprocessing, transfer, and startup
are excluded.

The five checkpoints, full prediction files, validation files, and training
curves remain outside Git under
`$PSCRATCH/finalproject/standardized_results/resnet18`.

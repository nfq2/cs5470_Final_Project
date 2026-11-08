# Pairwise logistic ensembles

These are interim, validation-only results for the three pairs currently having
complete shared artifacts: ResNet-18, MobileNetV3-Small, and DINOv2-S. Each pair
uses two clipped P(advance) log-odds as inputs to a class-balanced logistic
regression. Predictions are five-way out-of-fold within each outer fold's
validation set, then pooled across the five outer folds (352 predictions).

`comparison.csv` has the same accuracy and latency fields for all available
pairs and includes separate validation-OOF and pooled-test rows. Each pair
directory contains a full `results.json` and `summary.csv`. These files
deliberately contain no case-level probabilities.

The first run did not read test predictions. At the project owner's request, a
second exploratory run evaluates all currently available pairs on the pooled
416-case test set and adds those rows to `comparison.csv` and each `summary.csv`.
Because this happened before RepViT's three pairs were available, these test
metrics must not be used to select the final pair. The recorded validation OOF
metrics remain the proper pair-comparison values.

`benchmark_latency.py` fills the latency columns by timing both real models plus
the fitted logistic stacker with GPU-resident batch-one inputs, 20 warmups and
100 repetitions per fold (500 measurements per pair). It does not add latency
numbers previously collected on different GPU types.

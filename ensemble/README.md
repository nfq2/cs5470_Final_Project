# Pairwise ensembles

`stack_pairs.py` is the standardized five-fold pairwise logistic-stacking
workflow. It consumes the saved `validation.json` files without loading images
or checkpoints. By default it never reads test predictions: it creates one
directory per available pair with pooled out-of-fold validation metrics. After
all six pairs are available, compare these validation results, explicitly select
one pair, and rerun with `--selected-pair NAME1+NAME2` to evaluate only that
pair on the pooled 416-case test set.

The stacker receives two inputs: the clipped log-odds of each base model's
P(advance). A separate class-balanced logistic regression is trained for every
outer fold. Its validation prediction is itself five-way stratified
out-of-fold, and one threshold is chosen from the pooled OOF validation scores.
No case-level predictions are written into the repository result files.

`evaluate.py` below is retained as the earlier exploratory ResNet + RepViT
experiment; it is not the standardized five-fold result.

## Earlier exploratory experiment

`evaluate.py` compares five fusion rules on fold validation predictions, selects
the highest balanced-accuracy rule, and reports test metrics only for that rule.
Both input checkpoints remain unchanged. Derived results stay on Perlmutter.

The five rules are a plain probability average, tuned weighted average, retract
veto using the lower score, rank-calibrated average, and a balanced two-score
logistic stacker with five-fold out-of-fold validation predictions.

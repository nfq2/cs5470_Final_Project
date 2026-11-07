# ResNet + RepViT ensemble

`evaluate.py` compares five fusion rules on fold validation predictions, selects
the highest balanced-accuracy rule, and reports test metrics only for that rule.
Both input checkpoints remain unchanged. Derived results stay on Perlmutter.

The five rules are a plain probability average, tuned weighted average, retract
veto using the lower score, rank-calibrated average, and a balanced two-score
logistic stacker with five-fold out-of-fold validation predictions.

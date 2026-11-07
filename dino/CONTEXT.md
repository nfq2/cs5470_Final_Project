# DINOv2 track

Akhil's frozen-DINOv2 model for the short-horizon advance/retract decision, and the language used to evaluate it against teammates' models and the VLM baselines.

## Inputs

**Decision point**:
One moment in a recorded procedure where the model must choose the next move; one case in `shorthorizon2`.
_Avoid_: Sample, example, case (when talking about the task rather than the file)

**Goal map**:
The target-marked artery image supplied with a decision point; a separate input, not an earlier time step.
_Avoid_: Goal frame, first frame, f0

**History frames**:
The six chronological X-ray frames f1..f6, ending at the decision moment.
_Avoid_: Recent frames, clip, video

**Depth history**:
The six insertion-depth values relative to now, with their six time offsets.
_Avoid_: Insertion history, metadata

## Evaluation

**Overall accuracy**:
Fraction of decision points where the prediction matches the expert's move; the metric Astra's 82% and always-advance's 62% are quoted in.
_Avoid_: Accuracy (unqualified), agreement

**Advance accuracy**:
Fraction of true-advance decision points predicted as advance (recall on advance).
_Avoid_: Advance precision, TPR

**Retract accuracy**:
Fraction of true-retract decision points predicted as retract (recall on retract).
_Avoid_: Retract precision, specificity

**Balanced accuracy**:
Mean of advance accuracy and retract accuracy; always-advance scores 50%.
_Avoid_: Average accuracy, mean accuracy

**Selection metric**:
Validation AUROC averaged over folds; the only number used to choose between variants and settings.
_Avoid_: Validation accuracy, best epoch

**Model latency**:
Time for one decision point from preprocessed GPU tensors to logits; the number compared across teammates' models.
_Avoid_: Latency (unqualified), inference time, forward time

**End-to-end latency**:
Time for one decision point from decoded frames in host memory to a decision on the CPU; the number compared with the VLM baselines and the real-time ceiling.
_Avoid_: Wall-clock latency, system latency

**Operating point**:
A model's (1 − retract accuracy, advance accuracy) pair; how hard-decision baselines such as the VLMs and always-advance appear on an ROC plot.

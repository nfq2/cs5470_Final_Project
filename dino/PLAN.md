# DINOv2 track — Akhil

Status: draft, being grilled. Midpoint is Mon Nov 9; final is Dec 7.

## Goal

Add the team's one self-supervised transformer: a frozen DINOv2 backbone with a small
trained head that takes a decision point and outputs ADVANCE (1) or RETRACT (0).
The teammates' models are all ImageNet-supervised CNNs (ResNet-18, MobileNetV3-Small,
RepViT-M0.9), so this covers the "accurate but larger" end of the accuracy-latency plot.

**Priority: maximize accuracy, subject to a latency ceiling of ~100 ms per decision
(end to end, one A100, batch 1).** A cardiologist decides several times per second, so
anything under the ceiling counts as real-time; the gap between 10 ms and 50 ms matters
far less than accuracy when the alternative (Astra) takes 3-4 s. Latency is still measured
for every variant. Astra's 82% is a reference until we confirm it was measured on this
416-case set; the comparison we can compute is always-advance on the same test cases.

## Hypothesis

A 2-3 px guidewire in a 1065x1050 frame is a fine, local signal. DINOv2's self-supervised
patch features keep local detail that classification-pretrained features discard, and a
frozen backbone avoids overfitting 267 training cases. Expect DINOv2 > CNNs on accuracy,
with higher latency, and expect 448 px >> 224 px.

## Model

1. Each of the 7 images (goal map, f1..f6) goes through frozen DINOv2 and gives patch tokens.
2. Attention pooling over each image's patches gives one vector per image (not the CLS token,
   which may average away the wire).
3. Each image vector is projected 384 -> 64. The 7 vectors stay in order (goal kept in
   its own slot, not treated as a time step).
4. Depth history becomes ~7 derived numbers: speed in each of the 5 intervals
   (change in depth / change in time), total depth change, and time span. Raw values are
   not used: the last depth and offset are always 0, and spacing varies (0.65-4 s), so
   raw depth hides speed. These go through a small MLP to a 32-dim embedding, so 12
   numbers are not drowned out by 2,688 image numbers. Statistics come from training data only.
5. A small MLP on [7 x 64 image, 32 history] outputs 2 logits.

The same derived history numbers feed the history-only baseline (logistic regression).

## Midpoint scope

| Priority | What |
|---|---|
| Must | DINOv2-S @ 448, frozen, model above; fold 0 at minimum, 5 folds ideally; overall / advance / retract accuracy and latency |
| Nice | History-only baseline; DINOv2-S @ 224 |
| After midpoint | Patch-difference variant, DINOv2-B, 672 px, DeiT-Tiny |

## Variants (each one is a point on the accuracy-latency plot)

| Variant | Question it answers |
|---|---|
| DINOv2-S/14 @ 224 | Low-resolution reference |
| DINOv2-S/14 @ 448 | Main model; does resolution matter for the wire? |
| DINOv2-B/14 @ 448 (if time) | Does a larger backbone help, and at what latency cost? |
| + f6 - f5 difference input (if time) | Does explicit motion help? |

Input is always the full frame, resized (no crop). Resolution is the experimental
variable: 224, 448, and 672 if time allows. If accuracy keeps rising with resolution,
fine wire detail is the bottleneck, which justifies a wire-tip crop (needs a tip
locator) after the midpoint. A crop around `goal_xy` is rejected: the wire tip is often
far from the target, and wrong-branch retracts happen away from it.

Motion, in build order (checked Nov 3: history spans ~4 s, wire depth moves in 415/416
cases, but ~2% of pixels change between frames, so heartbeat/noise move too):
1. Pool each frame, then concatenate (the model above).
2. History-only model (depth history alone) to measure how much images add.
3. Patch-difference variant (subtract same-position patch features before pooling);
   an experiment, not the expected winner, since non-wire motion muddies the differences.

DeiT-Tiny is deferred; add it only if the results raise "is it the architecture or the pretraining?"

## Pipeline

The backbone is frozen, so features are extracted once per variant and cached. Head
training then runs on the cached features in seconds, which makes 5 folds and later
hyperparameter sweeps cheap.

| File | Job |
|---|---|
| `extract.py` | Runs all 416 x 7 images through the backbone once and saves patch tokens (fp16, ~2.3 GB for S @ 448) |
| `head.py` | Attention pooling, history input, classifier |
| `train.py` | Trains the head on cached features for a given fold |
| `evaluate.py` | Accuracy (overall, advance, retract, balanced) and two latencies (below) |

Latency is reported two ways, both batch 1 (one decision point = 7 images + history),
FP32 with TF32 off, 20 warmups and 100 timed runs with CUDA sync, as in `resnet/evaluate.py`.
Both run the backbone live on images, never the cached features.

| Number | Timer covers | Compare with |
|---|---|---|
| Model latency | Preprocessed tensors already on GPU -> backbone -> head -> logits | Teammates' models (same boundary as `resnet/evaluate.py`) |
| End-to-end latency | Decoded uint8 frames in host memory -> GPU copy -> resize/normalize on GPU -> backbone -> head -> decision on CPU | Astra / VLMs and the ~100 ms ceiling |

PNG decoding is excluded from both: in deployment frames arrive as raw pixels from the
video feed. Suggest the team add the end-to-end column too.

Reuse Saesha's `resnet/data.py` loading and split logic rather than writing a third loader.

## Training defaults (midpoint)

Deliberately plain, so the post-midpoint experiments have a clean "before": Nicole's
retract weighting and Amelia's augmentation are measured against this model, and Akhil's
tuning starts from it.

| Setting | Value |
|---|---|
| Loss | Plain cross-entropy, no class weights |
| Optimizer | AdamW, lr 1e-3, weight decay 0.05 |
| Dropout | 0.3 |
| Epochs | One fixed count for all folds, from fold-averaged validation AUROC (expect 30-50) |
| Augmentation | None |
| Seeds | 1 for midpoint; 3 during tuning |

Cached features rule out per-epoch augmentation; see `docs/adr/0001-cache-frozen-features.md`.

## Perlmutter layout

Everything below is in personal space; nothing touches teammates' files.

```
~/cs5470_Final_Project/finalproject/dino/      code (git)
$PSCRATCH/finalproject/shorthorizon2/           dataset copy (read only)
$PSCRATCH/huggingface/                          model weights (HF_HOME)
$PSCRATCH/finalproject/dino/features/<variant>/ cached features
$PSCRATCH/finalproject/dino/runs/<variant>/fold<k>/  checkpoints, results JSON
```

Only aggregate numbers and plots leave Perlmutter. No images, features or per-case outputs.

## Evaluation

- Fold 0 for first results; all 5 folds for the midpoint table if time allows.
- Checked on Perlmutter (Nov 3): the 5 test sets partition the 416 cases exactly (no overlap),
  so report pooled test accuracy over all 416 (about ±2%). Test sizes are uneven:
  69 / 72 / 71 / 81 / 123.
- Choose variants and settings by the selection metric: validation AUROC averaged over folds.
  It is threshold-free, unaffected by the 62/38 imbalance, and smoother than accuracy on
  ~30 validation retract cases. Balanced accuracy is not used for selection because it
  conflates the model with its 0.5 threshold.
- Use one fixed epoch count for all folds, chosen from fold-averaged validation curves;
  no per-fold best-epoch picking.
- Report on test at threshold 0.5: overall accuracy (headline, comparable to Astra and
  always-advance), advance accuracy, retract accuracy, balanced accuracy.
- ROC plot with baselines as operating points, if per-class baseline numbers are available.
- Run test only on final settings.
- Raise with the team: everyone should pick by the same rule (Saesha's script uses val loss).

## Timeline

| When | Task |
|---|---|
| Tue Nov 3 – Wed Nov 4 | Tell the team; clone repo; install timm; download weights on login node. (Done Nov 3: data copied, fold check, motion check.) |
| Thu Nov 5 | Write the four files; extract S @ 448; train fold 0; beat always-advance on validation |
| Fri Nov 6 – Sun Nov 8 | Remaining variants; 5 folds; results table; look at ~10 errors on Perlmutter |
| Mon Nov 9 | Send numbers for the shared plot; 1-2 slides |
| Nov 9 – Dec 1 | Hyperparameter tuning on the team's selected model, always 5-fold |

## Results log

Nov 3, first run: validation pooled over 5 folds (352 cases, always-advance = 56.2%), default
training settings, 40 epochs, threshold 0.5.

| Run | AUROC | Overall | Advance | Retract | Model / end-to-end latency |
|---|---|---|---|---|---|
| DINOv2-S @ 448 | 0.483 | 54.0% | 78.8% | 22.1% | 28.8 / 30.5 ms (A100, FP32) |
| History-only | 0.525 | 56.0% | 96.0% | 4.5% | — |

Both are at chance. The DINO head overfits: on fold 0 its train loss falls 0.74 -> 0.40 while
val loss rises 0.72 -> 1.01 and val AUROC never clearly exceeds 0.55. Latency is well under
the 100 ms ceiling.

Nov 3, diagnostics (`probe.py`):

- **Recent motion says nothing about the next move.** The wire was advancing in the last
  interval in 377/416 cases; P(next = advance) is 0.62 whether it was advancing (0.62) or
  still (0.68), i.e. the base rate. Decision points are "keep pushing or pull back" moments.
- **Heavily regularized head** (S @ 448, 15 epochs, wd 0.5, dropout 0.5): val AUROC 0.523,
  collapses to always-advance.
- **Logistic-regression probes**, 42 combinations (S @ 224/448/672, B @ 448; CLS goal+f6,
  CLS all 7, mean patch, CLS differences, top-k patch differences; with/without history):
  best pooled val AUROC 0.582 (B @ 448, CLS goal+f6 + history). Mean-patch and
  patch-difference probes are at or below chance.
- **That best probe fails on test**, chosen in advance, pooled over all 416 cases:
  AUROC 0.507 (B @ 448), 0.501 (S @ 672), 0.507 (S @ 448). The validation edge was
  selection noise.
- **Latency** (model / end-to-end, A100 FP32): S @ 224 7.5 / 9.1 ms; S @ 448 28.8 / 30.5 ms;
  S @ 672 80.0 / 82.0 ms; B @ 448 88.6 / 90.3 ms.

Conclusion: frozen DINOv2 features, pooled per image, carry no usable signal for this
decision at any tested resolution or size. The decision likely depends on where the wire tip
is relative to the path to the target, a small geometric relation that generic frozen
features, pooled, do not keep.

Nov 3, human check (Akhil, non-expert, `$PSCRATCH/finalproject/dino/human_check.ipynb`,
23 of 30 random cases answered, 1 unsure): 77% (17/22) vs. always-advance 64% (14/22) on the
same cases. Advance 14/14, retract 3/8. Every retract call (3/3) was right; all 5 errors were
missed retracts. Picking 3 true retracts out of 8 by chance with 3 guesses has p ~ 0.04
(56/1540), so some retracts are visibly identifiable, but n is small.

Nov 3, local multiple-instance model (`local.py`): a shared per-patch MLP over the DINO patch
grid (goal / f6 / f6 - f1 at the same position), aggregated by max, top-8 or mean, so the
model can fire on one small place. 36 configs across S @ 448, S @ 672, B @ 448: val AUROC
0.47-0.55, all at chance; best (S @ 672, mean, f6+diff, val 0.546) scored **0.513 on pooled
test**. So DINO's 14-px patch tokens do not encode the 1-3 px wire tip even when the model
is allowed to look locally. The tip finder must come from somewhere else (thin-line filter +
frame differencing, or a small model trained on clicked tips at full resolution).

Nov 4, geometric pipeline, first full run (`geo.py`, `tip.py`, `rule.py`; outputs in
`$PSCRATCH/finalproject/dino/geo/`):

- **Vessel map**: Frangi (sigmas 3-13) on `goal_raw.png`, hysteresis threshold, closing,
  skeleton; along-vessel distance to target as a min-cost path (cost 1 on vessels, 25 off).
  Vessel mask covers ~6.7% of the image. Target lands within 15 px of a detected vessel in
  **342/416 (82%)**; median 5 px. Skeleton is fragmented (median 8 pieces), hence the cost-path
  distance instead of a hard graph.
- **Automatic tips** (label-free, unverified): candidates A (moving thin line nearest target)
  and B (thin-line endpoint with most motion) disagree almost always (median 440 px apart).
- **Rule on automatic tips**, no fitting, all 416 cases, AUROC for retract:
  A: 0.54 (progress), and A's progress is 0 for both classes, i.e. A sits on a static
  structure. B: 0.56 (progress), 0.55 (moved away); retract cases "moved away from target"
  53% vs 43% for advance. Weak but in the right direction.
- Blocking question: is this the tip finder or the rule? Only the 30 clicked tips
  (`tip_clicks.ipynb`) can tell: the rule on clicked tips isolates the rule; candidate error vs
  clicks isolates the tip finder.

Nov 6, clicked tips (Akhil, `tip_clicks.ipynb`, same 30 cases as the human check, tip visible
in f6 and f1 in 30/30):

- **Both automatic tip finders fail.** Median error vs. click: A 170 px (f6) / 335 px (f1),
  B 239 / 219 px; within 50 px only 7-20% of the time. The AUROC 0.56 above was noise.
- **With correct tips the progress rule is weak**: AUROC 0.627 (-progress), 0.597 (moved
  away), 0.51 (d6), n = 29-30, so roughly +-0.1.
- **Retract is anticipatory.** Retract cases still made progress toward the target (median
  34 px along the vessel vs. 68 px for advance) and only 42% had moved away. The operator
  retracts when the tip is about to enter the wrong branch, before it has gone the wrong way.
  The cue is the tip's heading at the bifurcation, not its past trajectory.

Nov 7, resolution and fine-tuning (`extract.py --tiles 2`, `wire_probe.py`, `finetune.py`;
jobs `ab.sh`, `c.sh`; ~1.5 GPU-hours in total):

- **Frozen DINOv2-S patch tokens see the wire.** Logistic probe per patch, trained on the train
  group and scored on the pooled test groups: wire vs. other *vessel* patches AUROC **0.951**
  (448 full frame, 33 px/patch) and 0.950 (2x2 tiles, 17 px/patch); wire vs. all patches
  0.979 / 0.985; target-circle control 0.991 / 0.986. The clicked tip's patch ranks in the
  top 3.4% (median) of its image by wire score, in the top 5% for 53% of cases, with no tip
  supervision. The Nov 3 conclusion "patch tokens do not encode the wire" was wrong; what
  they do not give a pooling head is the decision.
- **Tiles do not help the decision**: tiled head val AUROC 0.508; tiled MIL test 0.44;
  latency 106 / 108 ms (over budget).
- **Fine-tuning memorizes**: last 4 blocks + head on 2x2 tiles of goal, f1, f6, 20 epochs,
  5 folds in parallel (15 min each): pooled **val AUROC 0.563, test 0.502**. Train loss
  reaches 0.002; val AUROC peaks ~0.59 around epochs 5-13 then falls.

Conclusion as of Nov 7: perception is not the bottleneck. The features locate the wire; no
model learns the decision from 416 labels because the decision is relational (tip heading vs.
the branch that leads to the target). A human gets 77% from the same images. Everything
tested in the "bigger / sharper / trained backbone" direction is now closed.

Literature check (ChatGPT deep research, Nov 6-7; verify each before citing): closest
precedents are Cao et al. (same Mentice simulator, device segmentation U-Net from 102
images, tip + heading from the skeleton, 33 ms), FM-OSD (frozen DINO + small decoders, one
annotated template, ~1.5 mm on X-rays), Scarponi et al. (state = tip curvature vs. the
wrong branch, anticipatory by design), MSLNet / Ravigopal (endpoint tangent by PCA over the
distal 10-40 px). No paper shows a 30 -> 100 -> 150 tip-label learning curve on frozen
DINOv2; that would be ours.

## Midpoint story (poster, Nov 9)

1. No model beats always-advance (62.3%) yet: DINO heads, probes, MIL, tiles, fine-tuning all
   AUROC 0.50-0.56; speed is fine (9-29 ms).
2. The images do contain the answer: human 77%, all retract calls right.
3. DINO sees the wire (patch AUROC 0.95) but cannot decide; fine-tuning memorizes.
4. Retract is anticipatory (clicks): heading at the bifurcation, not progress.
5. Plan: tip locator on the frozen features (~100 clicks), heading-vs-branch rule, then a
   local-crop classifier where the team's augmentation / weighting / tuning experiments fit.

## After the midpoint (Nov 9 - Dec 1)

One line of work, in order. Each step names what would make us stop.

1. **More clicks, new protocol: tip + 2 shaft points** (about 20 and 50 px back along the
   wire) in f6 and f1, ~70 more cases to reach ~100. The shaft points give a directed heading
   label. ~1 min per case.
2. **Heading-vs-branch rule on clicked tips** (no training): signed PCA tangent over the
   distal 10-30 px, compared with the direction of the on-path branch and of the competing
   branch at the nearest bifurcation of the vessel graph. Stop if it is at chance on 100
   clicked cases; then the labels, not the geometry, are the limit.
3. **Tip locator** on cached DINOv2-S @ 448 tokens: small conv heatmap head on the 32x32 grid,
   local soft-argmax, optional native-resolution refinement crop (FM-OSD); baseline =
   template matching against clicked tips with no training. Report error at 30 / 60 / 100
   labels on held-out episodes. Target: median < 20 px. Keep heading and tip velocity as
   separate features (retracting reverses motion, not heading).
4. **Decision model on [geometric scalars + native-resolution crop around the located tip +
   goal-map crop + history]**. This is where Amelia's augmentation, Nicole's class weighting
   and Akhil's hyperparameter sweep plug in, always 5-fold pooled, selection by val AUROC.
5. **Error analysis** with clicked tips: tip not found / graph wrong / both right but label
   disagrees. Share of each.
6. **Pipeline latency**, model and end-to-end; the vessel graph is per-episode and does not
   count per decision; say so.
7. Stretch: fluoroscopy-specific pretraining (CathAction) for the locator.

Closed, with the evidence: fine-tuning DINO (Nov 7, test 0.50); higher resolution / tiles
(Nov 7); DeiT-Tiny (weaker backbone, same approach); V-JEPA / video (motion at chance);
small VLM (seconds of latency, 56% best open-weight on the clinical benchmark).

## Open questions

- Astra's 82% is on 512 *clinical* procedures (projectwaldo.ai), not this simulator set.
  Confirm with Jamal and state it on the poster.
- Can error-analysis images go on the poster, or only descriptions? (Ask Jamal.)
- Is the student set simulated, phantom, or clinical? DATASET.md does not say; the poster
  should use whichever word Jamal confirms.
- Teammates' pooled 5-fold numbers (Nov 8, results/summary.csv, all A100-SXM4-40GB): ResNet-18
  62.3% / AUROC 0.486 / 6.15 ms; MobileNetV3-S 58.2% / 0.496 / 4.34 ms; RepViT-M0.9 62.3% /
  0.515 / 10.75 ms; DINOv2-S 61.5% / 0.490 / 28.98 ms. All six pairwise logistic ensembles:
  validation AUROC 0.55-0.60, test AUROC 0.49-0.52 (ensemble/results/comparison.csv). Answered.

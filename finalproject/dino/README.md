# Frozen DINOv2 — Akhil

A frozen DINOv2 backbone with a small trained head. Design and decisions are in
[PLAN.md](PLAN.md); terms are defined in [CONTEXT.md](CONTEXT.md).

| File | What it does |
| --- | --- |
| extract.py | Runs the frozen backbone once over all 416 x 7 images and saves patch features to `$PSCRATCH`; `--tiles 2` encodes 2x2 tiles per image. |
| head.py | Attention pooling per image, derived depth history (speeds), classifier; also the history-only baseline. |
| train.py | Trains the head on saved features for one fold, for a fixed number of epochs. |
| evaluate.py | Accuracy pooled over every trained fold, and model / end-to-end latency with the backbone running live. |
| probe.py | Logistic-regression probes over CLS / mean-patch / patch-difference features. |
| local.py | Per-patch multiple-instance head (max / top-k / mean over patches). |
| wire_probe.py | Patch-level probes: do the frozen features detect the wire, and where does the clicked tip rank? |
| finetune.py | End-to-end fine-tuning of the last blocks on 2x2 tiles of goal, f1, f6; `--summarize` pools folds. |
| geo.py | Vessel skeleton from each goal map and along-vessel distance to the target (min-cost path). |
| tip.py | Two label-free wire-tip candidates per frame (both fail against clicked tips). |
| rule.py | Progress-along-vessel rule, scored by AUROC on automatic or clicked tips. |

Results and decisions, in date order, are in [PLAN.md](PLAN.md). The click notebooks
(`human_check.ipynb`, `tip_clicks.ipynb`) live only in `$PSCRATCH/finalproject/dino/` because
they render dataset images.

Cases, splits and metrics come from the shared `common/` package at the repository root.

## Setup (login node, once)

```bash
module load pytorch/2.8.0
pip install --user timm
export HF_HOME=$PSCRATCH/huggingface
python -c "import timm; timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True)"
```

## Run (GPU node)

```bash
salloc --nodes 1 --qos shared_interactive --time 01:00:00 --constraint gpu --gpus 1 -c 32 --account m4999
cd ~/cs5470_Final_Project/finalproject/dino
module load pytorch/2.8.0 && export HF_HOME=$PSCRATCH/huggingface

python extract.py --size 448                    # once per variant, a few minutes
for fold in 0 1 2 3 4; do python train.py --features vit_small_dinov2_448 --fold $fold; done
for fold in 0 1 2 3 4; do python train.py --history-only --fold $fold; done
python evaluate.py --run vit_small_dinov2_448 --split validation
python evaluate.py --run history_only --split validation
python evaluate.py --run vit_small_dinov2_448 --latency
```

Outputs go to `$PSCRATCH/finalproject/dino/` (`features/<variant>/`, `runs/<run>/fold<k>/`).
Training a fold again overwrites it; pass `--name` to keep a separate run.
Only aggregate numbers leave Perlmutter. `*_predictions.json` holds per-case outputs for
error analysis and stays there.

## What is measured

- Accuracy uses threshold 0.5 on P(advance). Variants are compared by validation AUROC
  pooled over folds; test is run only on final settings.
- Model latency starts from preprocessed tensors on the GPU (same boundary as
  `resnet/evaluate.py`). End-to-end latency starts from decoded frames in host memory and
  includes the GPU copy and resize. Both are batch 1, FP32 with TF32 off, 20 warmups and
  100 timed runs, with the backbone running live, never on cached features.

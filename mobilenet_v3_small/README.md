# MobileNetV3-Small — Amelia

This first baseline adapts `finalproject/resnet/`. It keeps fold 0, full-image
448x448 resizing and ImageNet normalization, seven ordered image features plus
12 history values (depths then times), a 128-unit ReLU/dropout-0.3 classifier,
20 epochs, batch size 8, Adam at 0.001, seed 42, and unweighted cross entropy.
Only the classifier trains. The backbone weights and BatchNorm statistics stay
frozen. History normalization is fitted on training cases only and saved.
`best.pt` is selected by minimum validation loss. No test images are loaded by
training. No augmentation, class weighting or mixed precision is used.

The architecture change is the torchvision pretrained MobileNetV3-Small encoder
in place of ResNet-18. Its ImageNet classifier is removed, yielding 576 pooled
features per image rather than ResNet's 512; the new classifier input is adjusted.
The goal has a separate position before the six chronological history frames.
MobileNet's standard ImageNet crop recipe is deliberately replaced by the same
full-image resize as the team's ResNet baseline. This does not establish that
448 pixels preserves enough guidewire detail; inspect visibility on Perlmutter.

Code is in this folder; the dataset and derived outputs must stay on Perlmutter
outside the repository. Scripts require PyTorch, torchvision and Pillow, e.g.
the existing team's `module load pytorch/2.8.0` environment. GPU training and
accuracy/latency on the real dataset must be verified on Perlmutter.

## 1. Prepare on a Perlmutter login node

From `~/cs5470_Final_Project/mobilenet_v3_small`:

```bash
module load pytorch/2.8.0
export TORCH_HOME="$PSCRATCH/torch_cache"
python -c 'from torchvision.models import MobileNet_V3_Small_Weights; MobileNet_V3_Small_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)'
python data.py --data-root "$PSCRATCH/shorthorizon2" --fold 0
```

The data check reports split counts and tensor shapes, not sensitive records.
It should show 267 train, 80 validation, 69 test, images `(7, 3, 448, 448)` and
history `(12,)`. Adjust `--data-root` only if your dataset is elsewhere.

## 2. Allocate one GPU

Use the GPU allocation command authorized by your course, then enter its compute
node with `srun --pty bash` if needed. Check `nvidia-smi` there. Do not train on a
login node. This repository does not establish your GPU allocation account.

## 3. Train inside the allocated GPU session

```bash
cd ~/cs5470_Final_Project/mobilenet_v3_small
module load pytorch/2.8.0
export TORCH_HOME="$PSCRATCH/torch_cache"
python train.py --data-root "$PSCRATCH/shorthorizon2" \
  --output-dir "$PSCRATCH/mobilenet_runs/fold0_baseline" --fold 0
```

The output directory must be new to prevent overwriting experiments. Outputs are
`config.json`, `best.pt` and `training.csv`. The CSV is written when training
finishes; the best checkpoint is written whenever validation loss improves.
Checkpoints contain weights/settings, not complete optimizer/RNG resume state.
Use `--epochs 1` and a distinct output directory for a short pipeline check.

## 4. Evaluate validation, then test after fixing settings

On the same kind of allocated GPU:

```bash
python evaluate.py --data-root "$PSCRATCH/shorthorizon2" \
  --checkpoint "$PSCRATCH/mobilenet_runs/fold0_baseline/best.pt" \
  --split validation --output "$PSCRATCH/mobilenet_runs/fold0_baseline/validation.json"
```

Once the configuration is fixed, change `--split validation` to `--split test`
and use a new filename `test.json`. Never tune on test results. Metrics are
fractions: 0.75 means 75%. Advance/retract accuracy is recall within that class;
the always-advance baseline is computed on the evaluated subset.

Latency matches the ResNet script: batch one, all seven encodings and classifier,
FP32 with TF32 disabled, 20 warmups and 100 synchronized timed repetitions of the
first example. Inputs are already on the GPU; excludes disk loading, resize,
normalization, transfer, startup and returning the action to CPU. This is forward
latency, not end-to-end latency. RepViT uses a different timing protocol and
resolution by default; coordinate comparisons. Record GPU type.

For other folds, use `--fold 1` through `--fold 4` with new output directories,
starting fresh each time. Agree the cross-fold tuning/reporting protocol with
teammates before using test results; validation across folds can reuse episodes
that another fold holds out for testing.

## Predict one case

```bash
python predict.py --data-root "$PSCRATCH/shorthorizon2" \
  --checkpoint "$PSCRATCH/mobilenet_runs/fold0_baseline/best.pt" --case-index 13
```

This prints an action only. Choose case indices according to the agreed split
protocol. No pretrained download occurs when loading a trained checkpoint.

Scratch is temporary. Preserve final artifacts in approved persistent storage
on Perlmutter, within project permissions and quota.

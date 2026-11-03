# Simple ResNet-18 baseline

Four Python files. You allocate the GPU and run each step yourself.
This replaces the previous larger starter package; it is a separate, simpler version.

## What each file does

| File | What it does |
| --- | --- |
| data.py | Opens seven images, reads depth/time history, and uses the provided split. |
| model.py | Uses ResNet-18 to extract image features and a small network to predict the action. |
| train.py | Trains the small network, checks validation loss, and saves the best model. |
| evaluate.py | Loads the saved model and reports overall/advance/retract accuracy and latency. |

## How the model works

1. Resize each entire image to 448 x 448 and apply ImageNet normalization.
2. Pass the goal and six history frames through the same pretrained ResNet-18.
3. Keep their seven feature vectors in order and concatenate them. The goal has
   its own position, separate from the six chronological frames.
4. Append six depth values and six actual time offsets. Normalize these using
   means and standard deviations computed only from training examples.
5. Train a small classifier to predict RETRACT=0 or ADVANCE=1.

For this first baseline, ResNet's weights and BatchNorm statistics stay frozen;
only the added classifier learns. This is a starting experiment, not evidence
that frozen features or 448-pixel resizing are optimal for thin guidewires.

## Put these files on Perlmutter

Upload the four Python files into ~/finalproject. Keep the data at:
$PSCRATCH/finalproject/shorthorizon2

Use a terminal opened inside ~/finalproject. You do not need the old submit.sh
or batch scripts for this version. No dataset or derived outputs should leave
Perlmutter without your advisor's permission.

## 1. Load PyTorch and cache public weights on the login node

```bash
module load pytorch/2.8.0
export TORCH_HOME="$PSCRATCH/finalproject/torch_cache"
python -c 'from torchvision.models import ResNet18_Weights; ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)'
```

This only downloads public pretrained weights; it does not read or upload data.
Cache weights before allocating a GPU because GPU-node downloads may be restricted.

## 2. Allocate a GPU manually

Use the allocation command from your course, for example:

```bash
salloc --nodes=1 --ntasks=1 --cpus-per-task=8 --qos=interactive --time=01:00:00 --constraint=gpu --gpus=1 --account=YOUR_GPU_ACCOUNT
srun --pty bash
```

Replace YOUR_GPU_ACCOUNT with your actual allocation. The data's project is
m4999; its GPU account may be m4999_g, but that has not been verified for you.
If already inside an allocated GPU terminal, skip this allocation step.

## 3. Train on that GPU

```bash
cd ~/finalproject
module load pytorch/2.8.0
export TORCH_HOME="$PSCRATCH/finalproject/torch_cache"
python train.py
```

Settings are ordinary constants at the top of train.py: first provided fold
(index 0), 448-pixel images, 20 epochs, batch size 8, learning rate 0.001.
Coordinate split and preprocessing with your teammates. No random image split,
augmentation, class weighting or mixed precision is used in this baseline.

Data loaders use four CPU workers to prepare images. Each epoch reports overall,
advance, retract and balanced validation accuracy. Balanced accuracy is the mean
of advance and retract accuracy. Checkpoint selection still uses validation loss.

Training writes resnet18.pt and training.csv in the current directory. Running
training again overwrites these files; copy them to a separate run folder on
Perlmutter first if you want to preserve an experiment.

## 4. Evaluate validation performance

```bash
python evaluate.py validation
```

It prints metrics (including balanced accuracy) and writes validation_results.json. Accuracies are fractions:
0.75 means 75%. Advance/retract accuracy means recall on that class. The
always-advance baseline is computed on this subset, rather than assuming 62%.

Latency is a batch-one forward pass over all seven images plus history, including
the classifier. It uses 20 warmups, 100 timed repetitions and CUDA synchronization.
Inputs are already on the GPU: image loading, resizing, transfer, model startup,
and returning the action to the CPU are excluded. Record this timing boundary
when comparing with teammates and VLM baselines. All computation uses FP32 with
TF32 disabled. This script does not claim end-to-end system latency.

## 5. Evaluate test performance after fixing the model settings

```bash
python evaluate.py test
```

This writes test_results.json. Use validation to choose settings; do not tune
based on repeated test results. This version runs one of the five provided splits.

To predict a batch in your own code, use model(images, history).argmax(1) with
model.eval() and torch.no_grad(), as shown in evaluate.py. Images have shape
[batch, 7, 3, 448, 448]; history is [batch, 12], depths followed by time offsets.
The saved state includes the history normalization.

The submitted code has no results on the sensitive dataset yet. GPU accuracy
and latency must be measured on Perlmutter.

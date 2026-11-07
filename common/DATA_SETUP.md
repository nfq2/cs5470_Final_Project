# Setting up the short-horizon dataset on Perlmutter

These steps reproduce our local dataset setup for the Project Waldo advance/retract task. Run the commands in a terminal on Perlmutter using an account with access to project `m4999`.

**The dataset is sensitive and must stay on Perlmutter. Do not download it to your computer.**

## 1. Open the team repository

```bash
cd ~/cs5470_Final_Project
```

The model folders and shared utilities are already in this repository.

## 2. Copy the dataset into your scratch space

Jamal provided the dataset at:

```text
/global/cfs/cdirs/m4999/studentdata/shorthorizon2
```

Copy it into your own `$PSCRATCH`:

```bash
mkdir -p "$PSCRATCH/finalproject"
cp -r /global/cfs/cdirs/m4999/studentdata/shorthorizon2 "$PSCRATCH/finalproject/"
```

Wait for the copy to finish. Your dataset will be at:

```text
$PSCRATCH/finalproject/shorthorizon2
```

`$PSCRATCH` is specific to your account, so these commands work without replacing another student's username.

## 3. Check the files and read the dataset README

```bash
ls "$PSCRATCH/finalproject/shorthorizon2"
cat "$PSCRATCH/finalproject/shorthorizon2/README.md"
```

The directory should contain:

| Name | Contents |
| --- | --- |
| `cases/` | Goal image, six history frames, and metadata for each decision point |
| `frames/` | Unique unannotated frames |
| `cases.jsonl` | One decision-point record per line |
| `manifest.json` | Case records and five provided episode-based splits |
| `splits.json` | Training, validation, and test indices for each split |
| `validation.json` | Checks performed when preparing the dataset |
| `README.md` | Dataset description |

The supplied student dataset contains 416 decision points from 120 episodes: 259 advance and 157 retract cases.

## 4. Optionally add a data shortcut to the repository

The repository already contains `DATASET.md`. To make the Perlmutter data visible
in the repository file browser, create an ignored shortcut:

```bash
ln -s "$PSCRATCH/finalproject/shorthorizon2" ~/cs5470_Final_Project/data
```

Create this shortcut once. If `data` already exists, skip that command. Refresh the file browser after creating it. The shortcut points to the scratch copy; it does not move or duplicate the dataset.

## 5. Check the split format

Run this on Perlmutter to inspect the structure without printing individual records:

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["PSCRATCH"]) / "finalproject/shorthorizon2"
with open(root / "splits.json") as f:
    splits = json.load(f)

print("Number of provided splits:", len(splits))
for name in ["train", "validation", "test"]:
    indices = splits[0][name]
    print(name, "examples:", len(indices), "entry type:", type(indices[0]).__name__)
PY
```

For the supplied first split, the expected counts are 267 training, 80 validation, and 69 test examples. Entries are integers, used by our loader as zero-based row indices into `cases.jsonl`. Use the provided episode-based splits rather than randomly splitting individual images.

## Dataset path used by the model code

The Python loader reads from:

```python
DATA_DIR = Path(os.environ["PSCRATCH"]) / "finalproject/shorthorizon2"
```

Each case's image paths are relative to that directory. `paths` contains the target-marked goal image first, followed by six chronological frames. The goal is separate from the frame history. The labels are `ADVANCE = 1` and `RETRACT = 0`.

This completes the dataset setup. A GPU is not needed for these copying and inspection steps.

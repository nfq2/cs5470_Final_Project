# RepViT-M0.9 — Nicole

## Short-horizon data loader

This folder contains Nicole's RepViT-M0.9 implementation. Run the commands below
from this folder (`cd repvit_m0_9` from the repository root). Model construction
and training scripts have not been implemented yet.

Requires Python 3.10+, PyTorch, and Pillow (`requirements.txt`). Install the
appropriate GPU-enabled PyTorch build in your Perlmutter environment for training.
Keep the real dataset, case outputs, and images on Perlmutter, outside this repo.
Local tests use synthetic images only.

`data_loader.py` loads the supplied `manifest.json` and `splits.json`. Split
entries index the manifest's cases array. `fold` selects the stored fold ID;
everyone should use the same fold for comparisons. No new splits are generated.

```python
from data_loader import ImagePreprocess, make_dataloader

loader = make_dataloader(
    "/pscratch/sd/n/nicoleq/shorthorizon2",
    split="train", fold=0, batch_size=8,
    transform=ImagePreprocess(size=(224, 224)),
)
batch = next(iter(loader))
```

| Batch field | Shape | Meaning |
|---|---|---|
| `goal` | B × 3 × H × W | Target-marked map; separate from history |
| `frames` | B × 6 × 3 × H × W | Chronological f1 through f6 |
| `history_offsets_s` | B × 6 | Original relative times in seconds |
| `depth_relative_mm` | B × 6 | Original relative insertion depths in mm |
| `label` | B | Integer RETRACT=0 / ADVANCE=1 |

Training is shuffled reproducibly; validation and test preserve supplied order.
Partial batches are retained. Images and numeric inputs are float32 CPU tensors;
labels are int64 (suitable for two-logit cross entropy). For a one-logit binary
loss, convert labels to float in the training code. Move tensors to the GPU in
the training/evaluation loop. Numeric inputs are left in their original units;
if standardizing, fit statistics on training data only.

Configure image size, mean, and standard deviation for the selected RepViT
pretrained weights, or supply a deterministic PIL-to-
CHW-tensor transform. Defaults use ImageNet normalization and a provisional
224×224 direct resize; inspect guidewire visibility on Perlmutter before settling
on resolution. The loader does not crop or randomly augment images. Independent
random transforms would misalign frames; any future spatial augmentation must
be synchronized at case level and preserve the target.

On Perlmutter, check one batch without printing case data (no GPU required):

```bash
python3 data_loader.py --data-root "$PSCRATCH/shorthorizon2" --fold 0
```

Run synthetic tests:

```bash
python3 -m unittest discover -s tests -v
```

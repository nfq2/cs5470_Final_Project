# Cache frozen DINOv2 features instead of running the backbone during training

The backbone is frozen, so each image's patch features are extracted once per variant and saved to `$PSCRATCH`; the head trains on the saved features. This makes 5-fold runs, variant sweeps and hyperparameter tuning take seconds instead of minutes. The cost is that per-epoch augmentation is impossible, because the backbone only ever sees each image once, unaltered.

## Consequences

- Latency is always measured with the backbone running live on images, never on cached features.
- If augmentation (Amelia's post-midpoint experiment) joins the final model, either cache several augmented copies per decision point, with the same transform applied to all 7 images, or run the backbone live during training.
- Patch-token caches are large (~2.3 GB for ViT-S at 448, ~5 GB at 672) and live in `$PSCRATCH`, which purges unused files; they can always be regenerated with `extract.py`.

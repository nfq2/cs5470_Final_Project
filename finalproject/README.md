# shorthorizon2 — student dataset

Predict the next operator-action proxy: **ADVANCE = 1**, **RETRACT = 0**.
This package contains 416 decision points from 120 episodes: 259 advance and
157 retract cases. It includes all images needed for these cases.

## Files

- `manifest.json`: all case records and five episode-based splits.
- `cases.jsonl`: the same case records in the same order, one JSON object per line.
- `splits.json`: the same five splits, for convenience.
- `cases/<case>/`: target-marked `goal.png`, unannotated `goal_raw.png`,
  six chronological frames `f1.png` through `f6.png`, and `metadata.json`.
- `frames/<episode>/<frame>.png`: the unique unannotated frames associated with
  the supplied cases. Case images that duplicate these frames are hard links
  within this package; they do not depend on the original dataset.
- `validation.json`: checks performed when preparing this handoff.

All image paths are relative to this directory. `paths` lists the seven task
images: target-marked goal map first, then the six history frames ending at now.
`clean_paths` lists the corresponding seven unannotated images. The goal map is
separate from the chronological history; do not treat it as an earlier time step.

## Case fields

- `case_id`, `episode`: sample identifier and procedure/episode grouping.
- `label`, `label_name`: binary action label and its name.
- `paths`, `clean_paths`, `render_dir`: local image paths and case directory.
- `image_roles`: roles matching the seven image paths.
- `image_size_wh`: image width and height, 1065 by 1050 pixels.
- `goal_xy`: target center in image pixel coordinates (x, y), measured from the
  top-left corner. `goal.png` already has the target circled.
- `source_frames`, `source_times_s`: original frame indices and timestamps in
  seconds, aligned with the seven images. These are numeric identifiers, not
  dependencies on the original videos.
- `history_offsets_s`: six history times relative to the current frame (0 seconds).
  Use these values rather than assuming constant spacing across cases.
- `depth_relative_mm`: six insertion-depth values aligned with the six history
  frames, relative to the current depth (last value = 0 mm).

Original video paths, original dataset paths, external build provenance, source
row references, and auxiliary clarity scores have been omitted. No model-generated
annotations or predictions are included.

## Interpretation and limitations

Labels are inferred operator-action proxies, not independently verified optimal
actions. Existing cohort and target limitations are inherited from the benchmark.
It is possible you find incorrect cases, mislabeled cases, or incorrect goal markings. If so, feel free to fix them.

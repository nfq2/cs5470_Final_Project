"""Single source of truth for the architecture-comparison baseline."""

NAME = "midpoint_baseline_v1"
FOLDS = (0, 1, 2, 3, 4)

EPOCHS = 20
BATCH_SIZE = 8
IMAGE_SIZE = 448
LEARNING_RATE = 1e-3
OPTIMIZER = "Adam"
WEIGHT_DECAY = 0.0
DROPOUT = 0.3
HIDDEN_DIM = 128
SEED = 42
NUM_WORKERS = 4

PRETRAINED = True
FREEZE_BACKBONE = True
LOSS = "unweighted cross entropy"
CHECKPOINT_SELECTION = "minimum validation loss"
PRECISION = "FP32; TF32 disabled"
HISTORY_ORDER = "depth_then_time"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LATENCY_WARMUP = 20
LATENCY_REPEATS = 100
LATENCY_SCOPE = (
    "Batch-one seven-image + history forward; excludes I/O, preprocessing, "
    "host-to-GPU transfer and startup"
)


def settings():
    """JSON-serializable record saved with every compliant run."""
    return {
        "name": NAME,
        "folds": list(FOLDS),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": OPTIMIZER,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "hidden_dim": HIDDEN_DIM,
        "seed": SEED,
        "num_workers": NUM_WORKERS,
        "pretrained": PRETRAINED,
        "freeze_backbone": FREEZE_BACKBONE,
        "loss": LOSS,
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "precision": PRECISION,
        "history_order": HISTORY_ORDER,
        "latency_warmup": LATENCY_WARMUP,
        "latency_repeats": LATENCY_REPEATS,
        "latency_scope": LATENCY_SCOPE,
    }

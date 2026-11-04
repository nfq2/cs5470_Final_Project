"""Training/evaluation helpers. Metrics use 0=retract, 1=advance."""
import torch
from torch.nn import functional as F


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def metrics_from_confusion(cm):
    cm = cm.cpu()
    total = int(cm.sum())
    counts = cm.sum(1)
    recalls = [float(cm[i, i] / counts[i]) if counts[i] else None for i in range(2)]
    return {
        "cases": total,
        "accuracy": float(cm.diag().sum() / total) if total else None,
        "retract_accuracy": recalls[0], "advance_accuracy": recalls[1],
        "balanced_accuracy": sum(recalls) / 2 if all(x is not None for x in recalls) else None,
        "always_advance_accuracy": float(counts[1] / total) if total else None,
        "confusion_matrix": cm.tolist(),
    }


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    cm = torch.zeros(2, 2, dtype=torch.long)
    total_loss = 0.0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            batch = move_batch(batch, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = F.cross_entropy(logits, batch["label"])
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            labels = batch["label"].detach().cpu()
            predictions = logits.detach().argmax(1).cpu()
            cm += torch.bincount(2 * labels + predictions, minlength=4).reshape(2, 2)
            total_loss += loss.item() * len(labels)
    if not cm.sum():
        raise ValueError("Cannot run on an empty split")
    metrics = metrics_from_confusion(cm)
    metrics["loss"] = total_loss / metrics["cases"]
    return metrics

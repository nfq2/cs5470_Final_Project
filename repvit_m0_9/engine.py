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


def run_epoch(model, loader, device, optimizer=None, class_weights=None, prediction_sink=None):
    if class_weights is not None:
        if optimizer is None:
            raise ValueError("Class weights apply to training only")
        class_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
        if class_weights.shape != (2,) or not torch.isfinite(class_weights).all() or not (class_weights > 0).all():
            raise ValueError("Expected positive finite weights ordered [retract, advance]")
    model.train(optimizer is not None)
    cm = torch.zeros(2, 2, dtype=torch.long)
    total_loss = 0.0
    total_objective = 0.0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            batch = move_batch(batch, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            per_case_loss = F.cross_entropy(logits, batch["label"], reduction="none")
            loss = per_case_loss.mean()
            # Divide by case count, not sum of weights: a retract example's
            # contribution is literally multiplied by its specified weight.
            objective = (per_case_loss * class_weights[batch["label"]]).mean() if class_weights is not None else loss
            if optimizer is not None:
                objective.backward()
                optimizer.step()
            labels = batch["label"].detach().cpu()
            probabilities = logits.detach().softmax(1)[:, 1].cpu()
            # Match common.metrics: an exact 0.5 tie predicts advance.
            predictions = (probabilities >= 0.5).long()
            if prediction_sink is not None:
                prediction_sink["labels"].extend(labels.tolist())
                prediction_sink["probs"].extend(probabilities.tolist())
            cm += torch.bincount(2 * labels + predictions, minlength=4).reshape(2, 2)
            total_loss += loss.item() * len(labels)
            total_objective += objective.item() * len(labels)
    if not cm.sum():
        raise ValueError("Cannot run on an empty split")
    metrics = metrics_from_confusion(cm)
    metrics["loss"] = total_loss / metrics["cases"]
    if class_weights is not None:
        metrics["weighted_training_loss"] = total_objective / metrics["cases"]
    return metrics

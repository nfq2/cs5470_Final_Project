"""Run on Perlmutter: aggregate-only checks of updates and tiny-batch learning.

Uses training cases only. Does not save or modify checkpoints or training runs.
"""
import argparse
import json

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from data_loader import ImagePreprocess, ShortHorizonDataset
from engine import move_batch, run_epoch
from model import WireModel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--checkpoint', help='Diagnose a saved model, or omit for a fresh pretrained model')
    p.add_argument('--device', default='cuda')
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-3)
    args = p.parse_args()
    if args.steps < 1:
        p.error('steps must be positive')
    torch.manual_seed(42)
    device = torch.device(args.device)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True) if args.checkpoint else None
    model = WireModel(pretrained=saved is None,
                      history_order=saved["config"].get("history_order", "time_then_depth")
                      if saved else "time_then_depth").to(device)
    if saved:
        model.load_state_dict(saved['model'])
        preprocessing = saved['preprocessing']
        fold = saved['config']['fold']
    else:
        preprocessing = dict(size=(224, 224), mean=model.encoder.pretrained_cfg['mean'],
                             std=model.encoder.pretrained_cfg['std'])
        fold = 0
    dataset = ShortHorizonDataset(args.data_root, split='train', fold=fold,
                                 transform=ImagePreprocess(**preprocessing))
    if saved is None:
        model.fit_history_normalization(dataset.cases)
    # Equal small numbers of each class eliminate the majority-class shortcut.
    selected = []
    for label in (0, 1):
        indices = [i for i, c in enumerate(dataset.cases) if c['label'] == label][:4]
        if len(indices) < 4:
            raise ValueError('Diagnostic requires at least four training cases of each class')
        selected.extend(indices)
    loader = DataLoader(Subset(dataset, selected), batch_size=8)
    batch = move_batch(next(iter(loader)), device)
    model.eval()
    with torch.no_grad():
        images = torch.cat([batch['goal'].unsqueeze(1), batch['frames']], dim=1)
        features = model.encoder(images.flatten(0, 1)).reshape(8, -1)
        history = torch.cat([batch[key] for key in model.history_keys()], dim=1)
        history = (history - model.history_mean) / model.history_std
        inputs = torch.cat([features, history], dim=1)
        probabilities = model(batch).softmax(1)[:, 1]
    report = {
        'training_cases_used': 8,
        'trainable_classifier_parameters': sum(p.numel() for p in model.classifier.parameters() if p.requires_grad),
        'trainable_backbone_parameters': sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),
        'features_finite': bool(torch.isfinite(features).all()),
        'feature_abs_mean': features.abs().mean().item(),
        'feature_variation_across_cases': features.std(0, unbiased=False).mean().item(),
        'normalized_history_finite': bool(torch.isfinite(history).all()),
        'normalized_history_abs_max': history.abs().max().item(),
        'initial_advance_probability_min': probabilities.min().item(),
        'initial_advance_probability_max': probabilities.max().item(),
    }
    before = {k: v.detach().clone() for k, v in model.classifier.named_parameters()}
    backbone_before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=1e-4)
    # Exercise the actual production training loop, not a separate imitation.
    run_epoch(model, loader, device, optimizer)
    report['classifier_gradient_l2'] = sum(
        p.grad.square().sum().item() for p in model.classifier.parameters() if p.grad is not None
    ) ** .5
    report['classifier_max_weight_change_after_one_batch'] = max(
        (p.detach() - before[k]).abs().max().item() for k, p in model.classifier.named_parameters()
    )
    report['frozen_backbone_unchanged'] = all(
        torch.equal(value, backbone_before[k]) for k, value in model.encoder.state_dict().items()
    )
    # Restore initial head and deliberately disable dropout for an easier
    # memorization check. Cached features are valid because the encoder is frozen.
    with torch.no_grad():
        for k, p in model.classifier.named_parameters():
            p.copy_(before[k])
    model.eval()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=0)
    def score():
        with torch.no_grad():
            logits = model.classifier(inputs)
            return {'loss': F.cross_entropy(logits, batch['label']).item(),
                    'accuracy': (logits.argmax(1) == batch['label']).float().mean().item()}
    report['tiny_batch_before'] = score()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model.classifier(inputs), batch['label'])
        loss.backward()
        optimizer.step()
    report['tiny_batch_after'] = score()
    report['tiny_batch_steps'] = args.steps
    report['tiny_batch_note'] = 'Training-only memorization with dropout off; not validation or test accuracy'
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

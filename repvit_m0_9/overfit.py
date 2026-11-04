"""Fine-tune on eight fixed training cases. Diagnostic, not benchmark accuracy."""
import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from data_loader import ImagePreprocess, ShortHorizonDataset
from engine import metrics_from_confusion, move_batch
from model import WireModel


def set_training_mode(model):
    model.train()
    # Make memorization easier while still adapting backbone weights and BN.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()


@torch.no_grad()
def score(model, batch, batch_statistics=False):
    model.eval()
    norms = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    original = [(m, m.training, m.track_running_stats) for m in norms]
    try:
        if batch_statistics:
            for module in norms:
                module.train()
                module.track_running_stats = False
        logits = model(batch)
        cm = torch.bincount(2 * batch['label'] + logits.argmax(1), minlength=4).reshape(2, 2)
        result = metrics_from_confusion(cm)
        result['loss'] = F.cross_entropy(logits, batch['label']).item()
        return result
    finally:
        for module, training, tracking in original:
            module.train(training)
            module.track_running_stats = tracking


def fit_tiny_batch(model, batch, steps=200, lr=1e-4, backbone_lr=1e-5,
                   log_every=25, emit=print):
    if model.freeze_backbone:
        raise ValueError('This diagnostic requires an unfrozen backbone')
    optimizer = torch.optim.AdamW([
        {'params': model.classifier.parameters(), 'lr': lr},
        {'params': model.encoder.parameters(), 'lr': backbone_lr},
    ], weight_decay=0)
    rows = []
    def record(step):
        row = {'step': step, 'eval_mode': score(model, batch),
               'batch_statistics_mode': score(model, batch, batch_statistics=True)}
        rows.append(row)
        emit(json.dumps(row))
    record(0)
    changes = {}
    for step in range(1, steps + 1):
        set_training_mode(model)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(batch), batch['label'])
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite tiny-batch training loss')
        loss.backward()
        if step == 1:
            before = {name: p.detach().clone() for name, p in model.named_parameters()}
            for group in ('encoder', 'classifier'):
                grads = [p.grad for p in getattr(model, group).parameters() if p.grad is not None]
                changes[group + '_gradient_l2'] = math.sqrt(sum(g.square().sum().item() for g in grads))
        optimizer.step()
        if step == 1:
            for group in ('encoder', 'classifier'):
                changes[group + '_max_weight_change'] = max(
                    (p.detach() - before[name]).abs().max().item()
                    for name, p in model.named_parameters() if name.startswith(group + '.')
                )
            del before
            emit(json.dumps({'first_update': changes}))
        if step % log_every == 0 or step == steps:
            record(step)
    return {'first_update': changes, 'history': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backbone-lr', type=float, default=1e-5)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--random-init', action='store_true', help='Synthetic test only')
    args = parser.parse_args()
    if args.steps < 1 or any(not math.isfinite(x) or x <= 0 for x in (args.lr, args.backbone_lr)):
        parser.error('steps and learning rates must be positive and finite')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; run inside a GPU allocation')
    torch.manual_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    model = WireModel(pretrained=not args.random_init, freeze_backbone=False)
    cfg = model.encoder.pretrained_cfg
    dataset = ShortHorizonDataset(args.data_root, 'train', args.fold,
                                  ImagePreprocess((args.image_size, args.image_size), cfg['mean'], cfg['std']))
    generator = torch.Generator().manual_seed(args.seed)
    selected = []
    for label in (0, 1):
        indices = [i for i, c in enumerate(dataset.cases) if c['label'] == label]
        if len(indices) < 4:
            raise ValueError('Need at least four training cases of each class')
        order = torch.randperm(len(indices), generator=generator).tolist()
        selected.extend(indices[i] for i in order[:4])
    model.fit_history_normalization(dataset.cases)
    model.to(device)
    batch = move_batch(next(iter(DataLoader(Subset(dataset, selected), batch_size=8))), device)
    config = dict(vars(args), cases=8, retract_cases=4, advance_cases=4,
                  dropout=False, weight_decay=0, class_weights=[1, 1], backbone_trainable=True)
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    print(json.dumps(config), flush=True)
    def emit(line):
        print(line, flush=True)
        with (output / 'history.jsonl').open('a') as stream:
            stream.write(line + '\n')
    result = fit_tiny_batch(model, batch, args.steps, args.lr, args.backbone_lr, emit=emit)
    result['note'] = 'Memorization of eight training cases only; no validation/test cases used, no checkpoint saved.'
    (output / 'summary.json').write_text(json.dumps(result, indent=2))
    print('Diagnostic complete. Summary:', output / 'summary.json', flush=True)


if __name__ == '__main__':
    main()

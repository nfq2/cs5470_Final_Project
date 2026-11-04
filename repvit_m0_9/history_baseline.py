"""Numerical-only baseline: six relative times plus six insertion depths."""
import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from checkpoints import CheckpointWriter
from data_loader import ShortHorizonDataset
from engine import run_epoch

MODEL_NAME = 'history_mlp'


class HistoryDataset(ShortHorizonDataset):
    """Reuse the supplied split mapping without opening any image files."""
    def __getitem__(self, index):
        case = self.cases[index]
        times = torch.tensor(case['history_offsets_s'], dtype=torch.float32)
        depths = torch.tensor(case['depth_relative_mm'], dtype=torch.float32)
        if times.shape != (6,) or depths.shape != (6,):
            raise ValueError('Expected six times and six depths')
        if not torch.isfinite(times).all() or not torch.isfinite(depths).all():
            raise ValueError('History must be finite')
        if not torch.all(times[1:] >= times[:-1]) or abs(times[-1].item()) > 1e-5 or abs(depths[-1].item()) > 1e-5:
            raise ValueError('History must be chronological with current time/depth zero')
        if case['label'] not in (0, 1):
            raise ValueError('Expected RETRACT=0 or ADVANCE=1')
        return {'history_offsets_s': times, 'depth_relative_mm': depths,
                'label': torch.tensor(case['label'], dtype=torch.long)}


class HistoryModel(nn.Module):
    def __init__(self, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.register_buffer('history_mean', torch.zeros(12))
        self.register_buffer('history_std', torch.ones(12))
        self.classifier = nn.Sequential(nn.Linear(12, hidden_dim), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    @torch.no_grad()
    def fit_history_normalization(self, cases):
        values = torch.tensor([c['history_offsets_s'] + c['depth_relative_mm'] for c in cases],
                              dtype=torch.float32, device=self.history_mean.device)
        if values.ndim != 2 or values.shape[1] != 12 or not len(values) or not torch.isfinite(values).all():
            raise ValueError('Training history must be nonempty and finite with 12 values per case')
        self.history_mean.copy_(values.mean(0))
        std = values.std(0, unbiased=False)
        self.history_std.copy_(torch.where(std < 1e-6, torch.ones_like(std), std))

    def forward(self, batch):
        history = torch.cat([batch['history_offsets_s'], batch['depth_relative_mm']], dim=1)
        return self.classifier((history - self.history_mean) / self.history_std)


def make_history_loader(root, split='train', fold=0, batch_size=8, seed=42,
                        num_workers=0, shuffle=None):
    return DataLoader(HistoryDataset(root, split=split, fold=fold), batch_size=batch_size,
                      shuffle=(split == 'train') if shuffle is None else shuffle,
                      num_workers=num_workers, generator=torch.Generator().manual_seed(seed))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--retract-weight', type=float, default=1.2)
    p.add_argument('--hidden-dim', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.hidden_dim < 1:
        p.error('epochs, batch-size, and hidden-dim must be positive')
    if not 0 <= args.dropout < 1 or any(not math.isfinite(x) or x <= 0 for x in (args.lr, args.retract_weight)):
        p.error('Invalid dropout, learning rate, or retract weight')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable; use an allocation or --device cpu')
    torch.manual_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    common = dict(root=args.data_root, fold=args.fold, batch_size=args.batch_size, seed=args.seed)
    train = make_history_loader(split='train', **common)
    train_eval = make_history_loader(split='train', shuffle=False, **common)
    validation = make_history_loader(split='validation', **common)
    model = HistoryModel(args.hidden_dim, args.dropout)
    model.fit_history_normalization(train.dataset.cases)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    weights = [args.retract_weight, 1.] if args.retract_weight != 1 else None
    checkpoints = CheckpointWriter(output)
    config = vars(args)
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    print(json.dumps({'model': MODEL_NAME, 'inputs': 'six times + six depths; no images',
                      'parameters': sum(p.numel() for p in model.parameters()),
                      'primary_comparison_checkpoint': 'best_balanced.pt', **config}), flush=True)
    for epoch in range(1, args.epochs + 1):
        training = run_epoch(model, train, device, optimizer, class_weights=weights)
        valid = run_epoch(model, validation, device)
        training_eval = run_epoch(model, train_eval, device)
        row = {'epoch': epoch, 'train': training, 'train_eval': training_eval, 'validation': valid}
        print(json.dumps(row), flush=True)
        with (output / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        checkpoints.save({'model': model.state_dict(), 'model_name': MODEL_NAME, 'config': config,
                          'epoch': epoch, 'validation': valid, 'pretrained': False,
                          'versions': {'torch': str(torch.__version__)}})
    if (output / 'best_balanced.pt').exists():
        best = torch.load(output / 'best_balanced.pt', map_location='cpu', weights_only=True)
        print(json.dumps({'best_balanced_epoch': best['epoch'], 'validation': best['validation']}), flush=True)
    else:
        print('Balanced accuracy unavailable: validation lacks one of the classes.', flush=True)
    print('Saved checkpoints:', output, flush=True)


if __name__ == '__main__':
    main()

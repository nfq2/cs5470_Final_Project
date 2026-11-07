"""Audit inputs and checkpoint behavior on Perlmutter; no test split is read.

Previews contain sensitive imagery and MUST stay on Perlmutter.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

from data_loader import ImagePreprocess, ShortHorizonDataset
from engine import metrics_from_confusion, move_batch, run_epoch
from model import WireModel


def image_digest(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


def audit_inputs(dataset, count, output):
    if not len(dataset):
        raise ValueError('Cannot audit an empty split')
    indices = np.linspace(0, len(dataset) - 1, min(count, len(dataset)), dtype=int).tolist()
    seen_native, seen_resized = set(), set()
    mismatches = 0
    alignment = {'label_mismatches': 0, 'time_tensor_mismatches': 0,
                 'depth_tensor_mismatches': 0, 'label_name_mismatches': 0,
                 'source_time_checks': 0, 'source_time_mismatches': 0,
                 'source_frame_checks': 0, 'source_frame_order_errors': 0}
    adjacent_native_duplicates = 0
    adjacent_resized_duplicates = 0
    total_images = 0
    shape_counts = {}
    transform = dataset.transform
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False)
    for sample, (index, batch) in enumerate(zip(indices, loader)):
        case = dataset.cases[index]
        alignment['label_mismatches'] += int(batch['label'][0].item() != case['label'])
        for field, counter in [('history_offsets_s', 'time_tensor_mismatches'),
                               ('depth_relative_mm', 'depth_tensor_mismatches')]:
            alignment[counter] += int(not torch.equal(batch[field][0], torch.tensor(case[field], dtype=torch.float32)))
        if 'label_name' in case:
            alignment['label_name_mismatches'] += int(case['label_name'].upper() != ('ADVANCE' if case['label'] == 1 else 'RETRACT'))
        if 'source_times_s' in case:
            alignment['source_time_checks'] += 1
            times = np.asarray(case['source_times_s'], dtype=float)
            valid = times.shape == (7,) and np.isfinite(times).all()
            valid = valid and np.all(np.diff(times[1:]) >= 0) and np.allclose(
                times[1:] - times[-1], case['history_offsets_s'], atol=1e-4, rtol=0)
            alignment['source_time_mismatches'] += int(not valid)
        if 'source_frames' in case:
            alignment['source_frame_checks'] += 1
            frames = np.asarray(case['source_frames'], dtype=float)
            valid = frames.shape == (7,) and np.isfinite(frames).all() and np.all(np.diff(frames[1:]) >= 0)
            alignment['source_frame_order_errors'] += int(not valid)
        loaded = torch.cat([batch['goal'][0].unsqueeze(0), batch['frames'][0]], dim=0)
        native_hashes, resized_hashes = [], []
        w, h = transform.size[1], transform.size[0]
        sheet = Image.new('RGB', (7 * w, h + 24), 'white') if sample < 2 else None
        for role, path in enumerate(case['paths']):
            # Independently decode and normalize the file named by this record,
            # then compare it with the actual DataLoader batch at this position.
            with Image.open(dataset.root / path) as opened:
                native = opened.convert('RGB')
            shape_counts[str(native.size)] = shape_counts.get(str(native.size), 0) + 1
            resized = native.resize((w, h), Image.Resampling.BILINEAR)
            expected = torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1).float() / 255
            expected = (expected - transform.mean) / transform.std
            mismatches += int(not torch.allclose(expected, loaded[role], rtol=0, atol=1e-6))
            native_hashes.append(image_digest(native))
            resized_hashes.append(image_digest(resized))
            if sheet is not None:
                sheet.paste(resized, (role * w, 24))
                ImageDraw.Draw(sheet).text((role * w + 4, 4), 'goal' if role == 0 else f'f{role}', fill='black')
                if role in (0, 6):
                    native.save(output / f'sample_{sample}_native_{"goal" if role == 0 else "f6"}.png')
        if sheet is not None:
            sheet.save(output / f'sample_{sample}_resized_views.png')
        # Exclude goal map from temporal adjacent-frame comparisons.
        adjacent_native_duplicates += sum(a == b for a, b in zip(native_hashes[1:-1], native_hashes[2:]))
        adjacent_resized_duplicates += sum(a == b for a, b in zip(resized_hashes[1:-1], resized_hashes[2:]))
        seen_native.update(native_hashes)
        seen_resized.update(resized_hashes)
        total_images += 7
    return {'sampled_cases': len(indices), 'images_checked': total_images,
            'record_alignment': alignment,
            'tensor_file_mismatches': mismatches,
            'native_image_sizes_wh': shape_counts,
            'unique_native_images': len(seen_native), 'unique_resized_images': len(seen_resized),
            'adjacent_history_pairs_checked': 5 * len(indices),
            'identical_native_adjacent_pairs': adjacent_native_duplicates,
            'identical_resized_adjacent_pairs': adjacent_resized_duplicates,
            'note': 'Duplicate images can be legitimate. Zero mismatches verifies sampled paths/order, not annotation correctness.'}


def audit_split_records(training, validation):
    result = {}
    for key in ('case_id', 'episode'):
        available = all(key in c for c in training + validation)
        result[key + '_overlap_count'] = (len({c[key] for c in training} & {c[key] for c in validation})
                                         if available else None)
    for name, cases in [('train', training), ('validation', validation)]:
        result[name] = {'cases': len(cases),
                        'advance': sum(c['label'] == 1 for c in cases),
                        'retract': sum(c['label'] == 0 for c in cases),
                        'episodes': len({c['episode'] for c in cases}) if all('episode' in c for c in cases) else None}
    result['note'] = 'Episode separation does not establish anatomy separation. Missing metadata is reported as null.'
    return result


@torch.no_grad()
def batchnorm_sensitivity(model, loader, device, max_batches=4):
    """Change only BN statistics source on TRAIN inputs, with dropout always off.

    Batch statistics are temporary; checkpoint buffers and parameters are preserved.
    This is a sensitivity diagnostic, not an alternative validation score.
    """
    model.eval()
    norms = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    before = {k: v.clone() for k, v in model.named_buffers()}
    original = [(m, m.training, m.track_running_stats) for m in norms]
    cm_running = torch.zeros(2, 2, dtype=torch.long)
    cm_batch = torch.zeros(2, 2, dtype=torch.long)
    shifts, flips, n = 0.0, 0, 0
    try:
        for number, raw in enumerate(loader):
            if number >= max_batches:
                break
            batch = move_batch(raw, device)
            for m in norms:
                m.eval()
            running_prob = model(batch).softmax(1)
            for m in norms:
                m.train()
                m.track_running_stats = False
            batch_prob = model(batch).softmax(1)
            labels = batch['label'].cpu()
            r, b = running_prob.argmax(1).cpu(), batch_prob.argmax(1).cpu()
            cm_running += torch.bincount(2 * labels + r, minlength=4).reshape(2, 2)
            cm_batch += torch.bincount(2 * labels + b, minlength=4).reshape(2, 2)
            shifts += (running_prob[:, 1] - batch_prob[:, 1]).abs().sum().item()
            flips += int((r != b).sum())
            n += len(labels)
    finally:
        for m, training, tracking in original:
            m.train(training)
            m.track_running_stats = tracking
    unchanged = all(torch.equal(v, before[k]) for k, v in model.named_buffers())
    if not unchanged:
        raise RuntimeError('Diagnostic unexpectedly changed model buffers')
    return {'training_cases_checked': n, 'batchnorm_layers': len(norms),
            'mean_absolute_advance_probability_shift': shifts / n,
            'prediction_flip_fraction': flips / n,
            'running_statistics': metrics_from_confusion(cm_running),
            'batch_statistics': metrics_from_confusion(cm_batch),
            'buffers_unchanged': unchanged,
            'note': 'Same training inputs, dropout off. Batch-statistic results depend on batch composition; not validation results.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output-dir', required=True, help='New directory on Perlmutter; includes sensitive previews')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--samples', type=int, default=8)
    args = p.parse_args()
    if args.samples < 1:
        p.error('samples must be positive')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    device = torch.device(args.device)
    model = WireModel(pretrained=False,
                      history_order=saved["config"].get("history_order", "time_then_depth")).to(device)
    model.load_state_dict(saved['model'])
    model.eval()
    datasets = {split: ShortHorizonDataset(args.data_root, split, saved['config']['fold'],
                                         ImagePreprocess(**saved['preprocessing']))
                for split in ('train', 'validation')}
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers) for split, dataset in datasets.items()}
    report = {'checkpoint_epoch': saved['epoch'], 'fold': saved['config']['fold'],
              'note': 'Audits the supplied checkpoint epoch, not necessarily the last logged epoch.'}
    print('Checking sampled image paths and saving Perlmutter-only previews...', flush=True)
    report['split_records'] = audit_split_records(datasets['train'].cases, datasets['validation'].cases)
    report['identical_preprocessing'] = (
        datasets['train'].transform.size == datasets['validation'].transform.size
        and torch.equal(datasets['train'].transform.mean, datasets['validation'].transform.mean)
        and torch.equal(datasets['train'].transform.std, datasets['validation'].transform.std))
    report['input_audit'] = {}
    for split, dataset in datasets.items():
        preview_dir = output / split
        preview_dir.mkdir()
        report['input_audit'][split] = audit_inputs(dataset, args.samples, preview_dir)
    print('Evaluating training and validation splits with dropout off...', flush=True)
    report['train_eval_mode'] = run_epoch(model, loaders['train'], device)
    report['validation_eval_mode'] = run_epoch(model, loaders['validation'], device)
    print('Checking BatchNorm sensitivity on training cases only...', flush=True)
    report['batchnorm_sensitivity'] = batchnorm_sensitivity(model, loaders['train'], device)
    (output / 'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print('Previews contain dataset imagery: keep this entire audit directory on Perlmutter.')


if __name__ == '__main__':
    main()

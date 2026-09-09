"""Robustness evaluation for CinC2017 ECG classifiers.

This script evaluates a trained checkpoint on the same labeled evaluation set
under controlled, reproducible perturbations of the raw ECG signal. It answers:

    ¿Qué tan sensible es el modelo a degradaciones de la señal ECG?

Default conditions:
  * original signal
  * additive Gaussian noise at SNR 20/10/5 dB
  * baseline wander (low-frequency sinusoidal drift)
  * amplitude scaling (0.5x and 1.5x)
  * central duration crops (10 s and 20 s)

Typical use from the repository root:

    python examples/cinc17/robustness.py \
      --data_json examples/cinc17/dev.json \
      --saved saved/cinc17_resnet \
      --out_dir results/cinc17/robustness_resnet

The output includes robustness_summary.csv, robustness_metrics.json,
robustness_report.md and, if matplotlib is installed, robustness_curves.png.
"""

from __future__ import absolute_import

import argparse
import csv
import json
import math
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

from ecg import load, network, util

try:
    from examples.cinc17 import evaluate as eval_mod
except ImportError:  # direct execution from examples/cinc17
    import evaluate as eval_mod


FS_DEFAULT = 300.0


def _load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _resolve_model(model_path, saved_dir):
    if model_path:
        return model_path
    best = util.find_best_model(saved_dir)
    if not best:
        raise SystemExit('No se encontró checkpoint. Usa --model_path o revisa --saved.')
    return best


def _load_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    preproc = ckpt.get('preproc')
    if preproc is None:
        preproc = util.load(os.path.dirname(model_path))
    config = ckpt.get('config', {})
    classes = ckpt.get('classes', preproc.classes)
    model = network.build_network(num_categories=len(classes), **config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    info = {
        'model_path': os.path.abspath(model_path),
        'model_type': ckpt.get('model_type') or ('CNN convencional' if config.get('is_regular_conv') else 'ResNet-34'),
        'is_regular_conv': bool(config.get('is_regular_conv', False)),
        'num_parameters': ckpt.get('num_parameters') or util.count_parameters(model),
        'classes': list(classes),
        'config': config,
    }
    return model, preproc, info


def _trim_to_step(x):
    n = load.STEP * int(len(x) // load.STEP)
    if n < load.STEP:
        return x[:0]
    return x[:n]


def _signal_power(x):
    x = np.asarray(x, dtype=np.float32)
    power = float(np.mean(np.square(x)))
    if power <= 1e-12:
        power = float(np.var(x))
    return max(power, 1e-12)


def add_noise_snr(x, snr_db, rng):
    """Add zero-mean Gaussian noise at a target SNR in dB."""
    power_signal = _signal_power(x)
    power_noise = power_signal / (10.0 ** (float(snr_db) / 10.0))
    noise = rng.normal(0.0, math.sqrt(power_noise), size=x.shape).astype(np.float32)
    return (x + noise).astype(np.float32)


def add_baseline_wander(x, fs, amplitude_std, freq_hz, rng):
    """Add a low-frequency sinusoidal baseline drift."""
    std = float(np.std(x))
    amp = float(amplitude_std) * max(std, 1e-6)
    t = np.arange(len(x), dtype=np.float32) / float(fs)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    drift = amp * np.sin(2.0 * np.pi * float(freq_hz) * t + phase)
    return (x + drift.astype(np.float32)).astype(np.float32)


def scale_amplitude(x, factor):
    return (x * float(factor)).astype(np.float32)


def crop_duration(x, seconds, fs):
    """Center-crop to the requested duration, preserving STEP multiples."""
    target = int(float(seconds) * float(fs))
    target = load.STEP * int(target // load.STEP)
    target = max(target, load.STEP)
    if len(x) <= target:
        return x.astype(np.float32)
    start = (len(x) - target) // 2
    return x[start:start + target].astype(np.float32)


def clip_amplitude(x, std_factor):
    """Clip extreme amplitudes to +/- std_factor * std around the mean."""
    mu = float(np.mean(x))
    sigma = max(float(np.std(x)), 1e-6)
    lo, hi = mu - float(std_factor) * sigma, mu + float(std_factor) * sigma
    return np.clip(x, lo, hi).astype(np.float32)


def make_conditions(args):
    conditions = [{
        'name': 'original',
        'description': 'ECG sin modificación',
        'kind': 'original',
        'params': {},
    }]
    for snr in args.snr_db:
        label = str(snr).replace('.', 'p').replace('-', 'm')
        conditions.append({
            'name': 'noise_snr_{}db'.format(label),
            'description': 'Ruido gaussiano aditivo con SNR={} dB'.format(snr),
            'kind': 'noise',
            'params': {'snr_db': float(snr)},
        })
    conditions.append({
        'name': 'baseline_wander',
        'description': 'Deriva de línea base senoidal: {} Hz, amplitud={}×STD'.format(
            args.baseline_freq_hz, args.baseline_amplitude_std),
        'kind': 'baseline',
        'params': {
            'freq_hz': float(args.baseline_freq_hz),
            'amplitude_std': float(args.baseline_amplitude_std),
        },
    })
    for factor in args.amplitude_factors:
        label = str(factor).replace('.', 'p').replace('-', 'm')
        conditions.append({
            'name': 'amplitude_x{}'.format(label),
            'description': 'Escalamiento de amplitud ×{}'.format(factor),
            'kind': 'scale',
            'params': {'factor': float(factor)},
        })
    for seconds in args.crop_seconds:
        label = str(seconds).replace('.', 'p').replace('-', 'm')
        conditions.append({
            'name': 'crop_{}s'.format(label),
            'description': 'Recorte central a {} segundos'.format(seconds),
            'kind': 'crop',
            'params': {'seconds': float(seconds)},
        })
    if args.include_clipping:
        conditions.append({
            'name': 'clip_2std',
            'description': 'Saturación/recorte de amplitud a ±2 STD',
            'kind': 'clip',
            'params': {'std_factor': 2.0},
        })
    return conditions


def apply_condition(ecg, condition, fs, rng):
    x = np.asarray(ecg, dtype=np.float32)
    kind = condition['kind']
    params = condition['params']
    if kind == 'original':
        out = x
    elif kind == 'noise':
        out = add_noise_snr(x, params['snr_db'], rng)
    elif kind == 'baseline':
        out = add_baseline_wander(x, fs, params['amplitude_std'], params['freq_hz'], rng)
    elif kind == 'scale':
        out = scale_amplitude(x, params['factor'])
    elif kind == 'crop':
        out = crop_duration(x, params['seconds'], fs)
    elif kind == 'clip':
        out = clip_amplitude(x, params['std_factor'])
    else:
        raise ValueError('Condición no soportada: {}'.format(kind))
    return _trim_to_step(out)


def predict_records_from_ecgs(ecgs, model, preproc, device):
    y_pred = []
    interval_counts = []
    with torch.no_grad():
        for ecg in ecgs:
            if len(ecg) < load.STEP:
                y_pred.append(preproc.pad_class)
                interval_counts.append(0)
                continue
            x = preproc.process_x([ecg])
            xb = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 2, 1))).to(device)
            probs = model(xb).cpu().numpy()[0]
            row = np.argmax(probs, axis=-1)
            vals, counts = np.unique(row, return_counts=True)
            y_pred.append(preproc.int_to_class[int(vals[np.argmax(counts)])])
            interval_counts.append(int(row.shape[0]))
    return y_pred, interval_counts


def evaluate_condition(raw_ecgs, y_true, labels, model, preproc, device, condition, fs, seed):
    rng = np.random.default_rng(seed)
    perturbed = [apply_condition(x, condition, fs, rng) for x in raw_ecgs]
    keep = [i for i, x in enumerate(perturbed) if len(x) >= load.STEP]
    if len(keep) < len(perturbed):
        print('Aviso: {} registros quedaron con longitud < STEP y se excluyeron en {}'.format(
            len(perturbed) - len(keep), condition['name']))
    ecgs_keep = [perturbed[i] for i in keep]
    y_true_keep = [y_true[i] for i in keep]
    y_pred, interval_counts = predict_records_from_ecgs(ecgs_keep, model, preproc, device)
    metrics = eval_mod.collect_metrics(y_true_keep, y_pred, labels)
    metrics['condition'] = condition['name']
    metrics['description'] = condition['description']
    metrics['params'] = condition['params']
    metrics['mean_intervals_per_record'] = round(float(np.mean(interval_counts)), 4) if interval_counts else 0.0
    metrics['min_intervals_per_record'] = int(np.min(interval_counts)) if interval_counts else 0
    metrics['max_intervals_per_record'] = int(np.max(interval_counts)) if interval_counts else 0
    return metrics


def _write_outputs(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, 'robustness_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(out_dir, 'robustness_summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'condition', 'description', 'n', 'accuracy', 'macro_f1',
            'weighted_f1', 'challenge_f1', 'mean_intervals_per_record'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results['conditions']:
            writer.writerow({k: row.get(k) for k in fieldnames})

    per_class_csv = os.path.join(out_dir, 'robustness_per_class.csv')
    with open(per_class_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['condition', 'class', 'precision', 'recall', 'f1', 'n_true', 'n_pred', 'tp']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cond in results['conditions']:
            for cls, row in cond['per_class'].items():
                writer.writerow({
                    'condition': cond['condition'],
                    'class': cls,
                    'precision': row['precision'],
                    'recall': row['recall'],
                    'f1': row['f1'],
                    'n_true': row['n_true'],
                    'n_pred': row['n_pred'],
                    'tp': row['tp'],
                })

    md_path = os.path.join(out_dir, 'robustness_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# Experimento: robustez frente a perturbaciones ECG\n\n')
        f.write('Modelo: **{}**  \n'.format(results['model']['model_type']))
        f.write('Checkpoint: `{}`  \n'.format(results['model']['model_path']))
        f.write('Dataset: `{}`  \n'.format(results['data_json']))
        f.write('Semilla de perturbaciones: `{}`\n\n'.format(results['seed']))
        f.write('| Condición | Descripción | Accuracy | Macro-F1 | Weighted-F1 | Challenge-F1 | n |\n')
        f.write('|---|---|---:|---:|---:|---:|---:|\n')
        for row in results['conditions']:
            f.write('| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {} |\n'.format(
                row['condition'], row['description'], row['accuracy'], row['macro_f1'],
                row['weighted_f1'], row['challenge_f1'], row['n']))
        f.write('\n> Challenge-F1 es el promedio oficial sobre N/A/O. Las perturbaciones se aplican al ECG crudo y luego se usa el mismo preprocesador del entrenamiento.\n')

    png_path = None
    try:
        png_path = _write_plot(results, out_dir)
    except Exception as exc:  # matplotlib is optional for this script
        print('No se pudo generar robustness_curves.png:', exc)

    paths = [json_path, csv_path, per_class_csv, md_path]
    if png_path:
        paths.append(png_path)
    return paths


def _write_plot(results, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = [c['condition'] for c in results['conditions']]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.9), 4.5))
    for key, label in [('accuracy', 'Accuracy'), ('macro_f1', 'Macro-F1'), ('challenge_f1', 'Challenge-F1')]:
        ax.plot(x, [c[key] for c in results['conditions']], marker='o', label=label)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel('Métrica')
    ax.set_title('Robustez ante perturbaciones ECG')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha='right')
    ax.grid(axis='y', alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, 'robustness_curves.png')
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_json', required=True,
                        help='dev/test JSONL con etiquetas verdaderas')
    parser.add_argument('--model_path', default=None,
                        help='checkpoint .pt exacto')
    parser.add_argument('--saved', default='saved/cinc17_resnet',
                        help='carpeta donde buscar automáticamente el mejor checkpoint')
    parser.add_argument('--out_dir', default='results/cinc17/robustness')
    parser.add_argument('--seed', type=int, default=1234,
                        help='semilla para ruido/fase de baseline wander')
    parser.add_argument('--fs', type=float, default=FS_DEFAULT,
                        help='frecuencia de muestreo esperada de CinC2017')
    parser.add_argument('--snr_db', nargs='*', type=float, default=[20.0, 10.0, 5.0],
                        help='niveles SNR para ruido gaussiano')
    parser.add_argument('--baseline_freq_hz', type=float, default=0.33)
    parser.add_argument('--baseline_amplitude_std', type=float, default=0.30,
                        help='amplitud de deriva como múltiplo de la desviación estándar de la señal')
    parser.add_argument('--amplitude_factors', nargs='*', type=float, default=[0.5, 1.5])
    parser.add_argument('--crop_seconds', nargs='*', type=float, default=[10.0, 20.0])
    parser.add_argument('--include_clipping', action='store_true',
                        help='incluye saturación/recorte de amplitud a ±2 STD')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = _resolve_model(args.model_path, args.saved)
    model, preproc, model_info = _load_model(model_path, device)

    data = _load_jsonl(args.data_json)
    y_true = [d['labels'][0] for d in data]
    raw_ecgs, _labels_repeated = load.load_dataset(args.data_json)
    labels = list(preproc.classes)

    conditions = make_conditions(args)
    results = {
        'data_json': os.path.abspath(args.data_json),
        'seed': int(args.seed),
        'fs': float(args.fs),
        'model': model_info,
        'notes': ('Perturbaciones controladas aplicadas a la señal cruda. '
                  'Predicción a nivel de registro por voto mayoritario sobre intervalos.'),
        'conditions': [],
    }

    for idx, cond in enumerate(conditions):
        print('Evaluando condición:', cond['name'])
        metrics = evaluate_condition(
            raw_ecgs, y_true, labels, model, preproc, device, cond, args.fs,
            seed=int(args.seed) + idx * 1009)
        results['conditions'].append(metrics)
        print('  acc={:.4f} macroF1={:.4f} challengeF1={:.4f}'.format(
            metrics['accuracy'], metrics['macro_f1'], metrics['challenge_f1']))

    paths = _write_outputs(results, args.out_dir)
    print('\nResultados guardados en:')
    for p in paths:
        print('  ', p)


if __name__ == '__main__':
    main()

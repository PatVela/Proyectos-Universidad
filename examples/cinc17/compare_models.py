"""Compare the ResNet-34 replica against a matched conventional CNN.

The script assumes both models were trained on the SAME train/dev split and then
computes record-level CinC2017 metrics on the SAME evaluation JSON. It writes a
CSV/JSON/Markdown report with:

  * Accuracy, Macro-F1, Weighted-F1 and official Challenge-F1 (N/A/O)
  * F1 per class
  * Number of parameters
  * Training time, when the checkpoint folder contains training_summary.json

Typical use from the repository root:

    python -m ecg.train examples/cinc17/config.json -e cinc17_resnet --seed 2018
    python -m ecg.train examples/cinc17/config_regular_cnn.json -e cinc17_cnn --seed 2018

    python examples/cinc17/compare_models.py \
      --data_json examples/cinc17/dev.json \
      --resnet_saved saved/cinc17_resnet \
      --cnn_saved saved/cinc17_cnn \
      --out_dir results/cinc17/resnet_vs_cnn
"""

from __future__ import absolute_import

import argparse
import csv
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from ecg import network, predict as pred_mod, util

try:
    from examples.cinc17 import evaluate as eval_mod
except ImportError:  # direct execution from examples/cinc17
    import evaluate as eval_mod


def _load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _resolve_model(model_path, saved_dir, model_name):
    if model_path:
        return model_path
    best = util.find_best_model(saved_dir)
    if not best:
        raise SystemExit(
            "No se encontró checkpoint para {}. Usa --{}_model o revisa --{}_saved.".format(
                model_name, model_name.lower(), model_name.lower()))
    return best


def _checkpoint_info(model_path):
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    config = ckpt.get('config', {})
    preproc = ckpt.get('preproc')
    classes = ckpt.get('classes', getattr(preproc, 'classes', []))
    if not classes:
        raise RuntimeError("El checkpoint no contiene clases/preproc: {}".format(model_path))

    model = network.build_network(num_categories=len(classes), **config)
    params = ckpt.get('num_parameters') or util.count_parameters(model)
    model_type = ckpt.get('model_type') or (
        'CNN convencional' if config.get('is_regular_conv') else 'ResNet-34')

    # Training metadata is produced by ecg.train in training_summary.json. Fall
    # back to per-checkpoint seconds for older runs.
    summary_path = os.path.join(os.path.dirname(model_path), 'training_summary.json')
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)

    training_seconds = summary.get('total_seconds', ckpt.get('training_seconds_so_far'))
    best_epoch = summary.get('best_epoch', ckpt.get('epoch'))
    return {
        'model_path': os.path.abspath(model_path),
        'model_type': model_type,
        'is_regular_conv': bool(config.get('is_regular_conv', False)),
        'num_parameters': params,
        'training_seconds': training_seconds,
        'training_minutes': None if training_seconds is None else float(training_seconds) / 60.0,
        'best_epoch': best_epoch,
        'config': config,
        'training_summary_path': summary_path if os.path.exists(summary_path) else None,
    }


def _record_metrics(data_json, model_path):
    records = _load_jsonl(data_json)
    y_true = [r['labels'][0] for r in records]
    probs, preproc, t_out = pred_mod.predict(data_json, model_path)
    y_pred, confidence = pred_mod.record_predictions(probs, preproc, t_out)
    metrics = eval_mod.collect_metrics(y_true, y_pred, list(preproc.classes))
    metrics['mean_majority_confidence'] = round(float(sum(confidence) / max(len(confidence), 1)), 4)
    return metrics


def _fmt_num(x):
    return '' if x is None else '{:,}'.format(int(x))


def _fmt_float(x, nd=4):
    return '' if x is None else ('{:.%df}' % nd).format(float(x))


def _write_outputs(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, 'comparison_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary_csv = os.path.join(out_dir, 'summary.csv')
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'model', 'architecture', 'model_path', 'parameters_total',
            'training_minutes', 'best_epoch', 'accuracy', 'macro_f1',
            'weighted_f1', 'challenge_f1', 'mean_majority_confidence'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, item in results['models'].items():
            info, m = item['info'], item['metrics']
            writer.writerow({
                'model': name,
                'architecture': info['model_type'],
                'model_path': info['model_path'],
                'parameters_total': info['num_parameters']['total'],
                'training_minutes': info['training_minutes'],
                'best_epoch': info['best_epoch'],
                'accuracy': m['accuracy'],
                'macro_f1': m['macro_f1'],
                'weighted_f1': m['weighted_f1'],
                'challenge_f1': m['challenge_f1'],
                'mean_majority_confidence': m['mean_majority_confidence'],
            })

    per_class_csv = os.path.join(out_dir, 'per_class_metrics.csv')
    with open(per_class_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['model', 'class', 'precision', 'recall', 'f1', 'n_true', 'n_pred', 'tp']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, item in results['models'].items():
            for cls, row in item['metrics']['per_class'].items():
                writer.writerow({
                    'model': name,
                    'class': cls,
                    'precision': row['precision'],
                    'recall': row['recall'],
                    'f1': row['f1'],
                    'n_true': row['n_true'],
                    'n_pred': row['n_pred'],
                    'tp': row['tp'],
                })

    md_path = os.path.join(out_dir, 'comparison_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# Experimento: ResNet-34 vs CNN convencional\n\n')
        f.write('Dataset de evaluación: `{}`\n\n'.format(results['data_json']))
        f.write('| Modelo | Arquitectura | Parámetros | Tiempo entrenamiento (min) | Accuracy | Macro-F1 | Weighted-F1 | Challenge-F1 |\n')
        f.write('|---|---:|---:|---:|---:|---:|---:|---:|\n')
        for name, item in results['models'].items():
            info, m = item['info'], item['metrics']
            f.write('| {} | {} | {} | {} | {} | {} | {} | {} |\n'.format(
                name,
                info['model_type'],
                _fmt_num(info['num_parameters']['total']),
                _fmt_float(info['training_minutes'], 2),
                _fmt_float(m['accuracy']),
                _fmt_float(m['macro_f1']),
                _fmt_float(m['weighted_f1']),
                _fmt_float(m['challenge_f1']),
            ))
        f.write('\n## F1 por clase\n\n')
        f.write('| Modelo | Clase | Precisión | Recall | F1 | n real | n pred |\n')
        f.write('|---|---:|---:|---:|---:|---:|---:|\n')
        for name, item in results['models'].items():
            for cls, row in item['metrics']['per_class'].items():
                f.write('| {} | {} | {:.4f} | {:.4f} | {:.4f} | {} | {} |\n'.format(
                    name, cls, row['precision'], row['recall'], row['f1'],
                    row['n_true'], row['n_pred']))
        f.write('\n> Challenge-F1 corresponde al promedio oficial de F1 en N/A/O; la clase `~` se excluye del promedio oficial.\n')

    return json_path, summary_csv, per_class_csv, md_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_json', required=True,
                        help='dev/test JSONL usado para comparar ambos modelos')
    parser.add_argument('--resnet_model', default=None,
                        help='checkpoint .pt exacto del ResNet-34')
    parser.add_argument('--resnet_saved', default='saved/cinc17_resnet',
                        help='carpeta donde buscar el mejor checkpoint ResNet')
    parser.add_argument('--cnn_model', default=None,
                        help='checkpoint .pt exacto de la CNN convencional')
    parser.add_argument('--cnn_saved', default='saved/cinc17_cnn',
                        help='carpeta donde buscar el mejor checkpoint CNN')
    parser.add_argument('--out_dir', default='results/cinc17/resnet_vs_cnn')
    args = parser.parse_args()

    resnet_path = _resolve_model(args.resnet_model, args.resnet_saved, 'resnet')
    cnn_path = _resolve_model(args.cnn_model, args.cnn_saved, 'cnn')

    results = {
        'data_json': os.path.abspath(args.data_json),
        'notes': ('Comparación a nivel de registro usando el mismo dataset, split, '
                  'preprocesamiento, entrenamiento y métricas. Challenge-F1 = promedio F1 en N/A/O.'),
        'models': {}
    }
    for name, path in [('ResNet-34', resnet_path), ('CNN convencional', cnn_path)]:
        print('Evaluando {} -> {}'.format(name, path))
        results['models'][name] = {
            'info': _checkpoint_info(path),
            'metrics': _record_metrics(args.data_json, path),
        }

    paths = _write_outputs(results, args.out_dir)
    print('\nResultados guardados en:')
    for p in paths:
        print('  ', p)

    print('\nResumen:')
    for name, item in results['models'].items():
        info, m = item['info'], item['metrics']
        print('  {} | params={} | tiempo={} min | acc={:.4f} | macroF1={:.4f} | challengeF1={:.4f}'.format(
            name,
            _fmt_num(info['num_parameters']['total']),
            _fmt_float(info['training_minutes'], 2) or 'NA',
            m['accuracy'], m['macro_f1'], m['challenge_f1']))


if __name__ == '__main__':
    main()

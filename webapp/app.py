"""Interactive ECG classification web app (Flask) for the PyTorch port of
awni/ecg (Hannun et al., Nature Medicine 2019), trained on PhysioNet CinC2017.

Run (from the project root, virtualenv active):
    python webapp/app.py --model saved/cinc17/<ts>/<best>.pt
    # or auto-select the best (lowest val_loss) checkpoint:
    python webapp/app.py --saved saved
    # custom port / host:
    python webapp/app.py --saved saved --port 5000 --host 0.0.0.0
"""

from __future__ import absolute_import

import argparse
import csv
import json
import logging
import os
import sys
import threading
import uuid

# Make the `ecg` package importable regardless of CWD
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# make sibling modules (prediction) importable when loaded as a package
_WEBAPP = os.path.dirname(os.path.abspath(__file__))
if _WEBAPP not in sys.path:
    sys.path.insert(0, _WEBAPP)

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

import prediction as pred_mod
import project_info as proj
import report_pdf

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger('ecg-webapp')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024   # 64 MB uploads
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')


@app.after_request
def add_security_headers(resp):
    """Security hardening. TLS itself is terminated by the reverse proxy or the
    waitress/gunicorn SSL options (see webapp/README.md 'Seguridad'); here we set
    browser security headers and enable HSTS only when serving over HTTPS."""
    resp.headers['X-Content-Type-Options'] = 'nosniff'
    resp.headers['X-Frame-Options'] = 'DENY'
    resp.headers['Referrer-Policy'] = 'no-referrer'
    resp.headers['X-XSS-Protection'] = '1; mode=block'
    resp.headers['Permissions-Policy'] = 'camera=(), microphone=(), geolocation=()'
    resp.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "img-src 'self' data: blob:; "
        "style-src 'self' 'unsafe-inline'; "
        "script-src 'self'; "
        "connect-src 'self'; "
        "font-src 'self'; "
        "frame-ancestors 'none'")
    resp.headers['Cache-Control'] = 'no-store'
    # enable HSTS only if we are actually serving HTTPS (proxy or direct TLS)
    if request.is_secure or request.headers.get('X-Forwarded-Proto') == 'https':
        resp.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return resp

SERVICE = None          # current PredictionService
SERVICE_LOCK = threading.Lock()
MODELS_DIR = 'saved'
REFERENCE_LABELS = {}   # {record_id -> true CinC2017 label}, loaded from REFERENCE-v3.csv
REFERENCE_PATH = None


def _jsonable(result):
    """Drop non-JSON-serialisable payloads (numpy arrays) before responding."""
    out = dict(result)
    out.pop('probs', None)
    return out


def _sanitize(err):
    """Return a short, client-safe error message (no internal paths/stack)."""
    text = str(err)
    # strip anything that looks like a path
    for frag in [os.path.abspath(os.curdir), _REPO_ROOT, '\\n', 'Traceback']:
        text = text.replace(frag, '...')
    return text[:300]


def _normalize_label(label):
    """Map human labels or CinC2017 labels to the canonical class code."""
    if label is None:
        return ''
    raw = str(label).strip()
    key = raw.lower()
    mapping = {
        'n': 'N', 'normal': 'N', 'ritmo normal': 'N',
        'a': 'A', 'af': 'A', 'fa': 'A', 'fibrilacion': 'A',
        'fibrilación': 'A', 'fibrilacion auricular': 'A',
        'fibrilación auricular': 'A',
        'o': 'O', 'otro': 'O', 'other': 'O', 'otro ritmo': 'O',
        '~': '~', 'ruido': '~', 'noise': '~', 'artefacto': '~',
        '|': '|', 'sin clasificar': '|', 'unclassified': '|',
    }
    return mapping.get(key, raw.upper())


def _record_id_from_filename(filename):
    """Extract a CinC2017 record id from an uploaded filename.

    Examples: A00001.mat -> A00001, A00001.dat -> A00001,
    A00001.csv -> A00001. If a user uploads a converted file with suffixes, the
    first token before '-'/'_' is also tried.
    """
    name = secure_filename(filename or '')
    stem = os.path.splitext(os.path.basename(name))[0].upper()
    if not stem:
        return ''
    # direct official format
    if stem.startswith('A') and len(stem) >= 6 and stem[1:6].isdigit():
        return stem[:6]
    # tolerate names such as A00001_export.csv or A00001-filtered.npy
    for sep in ('_', '-', ' '):
        token = stem.split(sep)[0]
        if token.startswith('A') and len(token) >= 6 and token[1:6].isdigit():
            return token[:6]
    return stem


def _candidate_reference_paths(explicit=None):
    """Candidate locations for REFERENCE-v3.csv, in priority order."""
    paths = []
    for p in (explicit, os.environ.get('ECG_REFERENCE')):
        if p:
            paths.append(p)
    paths.extend([
        os.path.join(_REPO_ROOT, 'dataset2017', 'REFERENCE-v3.csv'),
        os.path.join(_REPO_ROOT, 'training2017', 'REFERENCE-v3.csv'),
        os.path.join(_REPO_ROOT, 'data', 'REFERENCE-v3.csv'),
        os.path.join(_REPO_ROOT, 'examples', 'cinc17', 'REFERENCE-v3.csv'),
        os.path.join(_REPO_ROOT, 'REFERENCE-v3.csv'),
    ])
    # Arena/local attachment fallback; harmless outside this environment.
    paths.append(os.path.join(os.path.expanduser('~'), 'uploads', 'REFERENCE-v3.csv'))
    return paths


def _load_reference_csv(path):
    labels = {}
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            rec = row[0].strip().upper()
            lab = _normalize_label(row[1])
            if not rec or rec in ('RECORD', 'RECORDING') or not lab:
                continue
            labels[rec] = lab
    return labels


def _init_reference(reference_csv=None):
    """Load REFERENCE-v3.csv if available. The app still works without it."""
    global REFERENCE_LABELS, REFERENCE_PATH
    REFERENCE_LABELS = {}
    REFERENCE_PATH = None
    for path in _candidate_reference_paths(reference_csv):
        path = os.path.abspath(path)
        if not os.path.exists(path):
            continue
        try:
            labels = _load_reference_csv(path)
        except Exception as e:
            log.warning('No se pudo leer REFERENCE-v3.csv (%s): %s', path, e)
            continue
        if labels:
            REFERENCE_LABELS = labels
            REFERENCE_PATH = path
            log.info('Etiquetas reales cargadas: %d registros desde %s',
                     len(REFERENCE_LABELS), REFERENCE_PATH)
            return labels
    log.info('No se encontró REFERENCE-v3.csv; comparación automática desactivada.')
    return {}


def _lookup_reference_label(filename):
    rec = _record_id_from_filename(filename)
    if not rec:
        return None, None
    return rec, REFERENCE_LABELS.get(rec)


def _model_train_date(path):
    """Return the checkpoint file's modification date (ISO) as an estimate of
    the training finish time; empty string if not available."""
    if not path:
        return ''
    try:
        import datetime
        return datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime(
            '%Y-%m-%d %H:%M')
    except (OSError, ValueError):
        return ''


def _load_metrics_json():
    path = os.path.join(os.path.dirname(__file__), 'static', 'metrics', 'metrics.json')
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _load_json_if_exists(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _load_experiment_results():
    """Load optional experiment outputs generated by compare_models.py and
    robustness.py so the Flask app can display research results when present."""
    root = os.path.join(_REPO_ROOT, 'results', 'cinc17')
    comparison_path = os.path.join(root, 'resnet_vs_cnn', 'comparison_metrics.json')
    robustness_paths = {
        'ResNet-34': os.path.join(root, 'robustness_resnet', 'robustness_metrics.json'),
        'CNN convencional': os.path.join(root, 'robustness_cnn', 'robustness_metrics.json'),
    }
    out = {
        'comparison_path': comparison_path,
        'comparison': _load_json_if_exists(comparison_path),
        'robustness': {},
        'robustness_paths': robustness_paths,
    }
    # Backward-compatible fallback if the user keeps the default out_dir from
    # robustness.py instead of the named robustness_resnet folder.
    fallback = os.path.join(root, 'robustness', 'robustness_metrics.json')
    for name, path in robustness_paths.items():
        data = _load_json_if_exists(path)
        if data is None and name == 'ResNet-34':
            data = _load_json_if_exists(fallback)
            if data is not None:
                path = fallback
        if data is not None:
            out['robustness'][name] = data
    return out


@app.route('/metrics', methods=['GET'])
def metrics_status():
    path = os.path.join(os.path.dirname(__file__), 'static', 'metrics', 'metrics.json')
    return jsonify({
        'file_found': os.path.exists(path),
        'path': path,
        'metrics': _load_metrics_json(),
    })


@app.route('/reference', methods=['GET'])
def reference_status():
    """Return REFERENCE-v3.csv status and optionally the true label for ?record=."""
    record = (request.args.get('record') or '').strip()
    record = _record_id_from_filename(record) if record else ''
    return jsonify({
        'ok': True,
        'file_found': bool(REFERENCE_LABELS),
        'path': REFERENCE_PATH,
        'count': len(REFERENCE_LABELS),
        'record': record or None,
        'label': REFERENCE_LABELS.get(record) if record else None,
    })


@app.route('/experiments', methods=['GET'])
def experiments_status():
    exp = _load_experiment_results()
    return jsonify({
        'comparison_found': exp['comparison'] is not None,
        'comparison_path': exp['comparison_path'],
        'robustness_found': {k: True for k in exp['robustness']},
        'robustness_paths': exp['robustness_paths'],
        'experiments': exp,
    })


@app.route('/')
def index():
    info = SERVICE.info() if SERVICE is not None else None
    checkpoints = pred_mod.list_checkpoints(MODELS_DIR)
    # Number of trainable parameters in the loaded model (for the arch sheet).
    n_params = None
    if SERVICE is not None:
        try:
            n_params = sum(p.numel() for p in SERVICE.model.parameters())
        except Exception:
            n_params = None
    return render_template(
        'index.html', model_info=info,
        checkpoints=[os.path.relpath(p, MODELS_DIR) for p in checkpoints],
        models_dir=MODELS_DIR,
        project=proj.PROJECT_INFO, reference=proj.REFERENCE,
        model_train_date=(_model_train_date(SERVICE.model_path)
                          if SERVICE and SERVICE.model_path else ''),
        model_hash=pred_mod._short_model_id(SERVICE.model_path)
        if SERVICE and SERVICE.model_path else '',
        model_n_params=n_params,
        reference_path=REFERENCE_PATH,
        reference_count=len(REFERENCE_LABELS),
        metrics=_load_metrics_json(),
        experiments=_load_experiment_results())


@app.route('/models', methods=['GET'])
def list_models():
    rel = [os.path.relpath(p, MODELS_DIR) for p in pred_mod.list_checkpoints(MODELS_DIR)]
    return jsonify({'ok': True, 'models': rel, 'current': os.path.relpath(
        SERVICE.model_path, MODELS_DIR) if SERVICE else None})


@app.route('/use_model', methods=['POST'])
def use_model():
    """Switch the loaded checkpoint without restarting the server."""
    global SERVICE
    data = request.get_json(silent=True) or {}
    rel = data.get('model', '')
    path = os.path.join(MODELS_DIR, rel) if rel else pred_mod.find_best_model(MODELS_DIR)
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return jsonify({'ok': False, 'error': 'No existe el checkpoint: ' + rel}), 404
    try:
        svc = pred_mod.PredictionService(path)
    except Exception as e:
        return jsonify({'ok': False, 'error': _sanitize(e)}), 400
    with SERVICE_LOCK:
        SERVICE = svc
    log.info("Modelo cambiado a %s", path)
    return jsonify({'ok': True, 'info': svc.info()})


@app.route('/predict', methods=['POST'])
def predict():
    if SERVICE is None:
        return jsonify({'ok': False, 'error': 'No hay modelo cargado.'}), 500

    file = request.files.get('file')
    if file is None or file.filename == '':
        return jsonify({'ok': False, 'error': 'Selecciona un archivo.'}), 400

    # unique temporary name so concurrent uploads never collide; clean up after.
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ('.csv', '.mat', '.dat', '.npy', ''):
        return jsonify({'ok': False,
                        'error': 'Formato no soportado (usa CSV, .mat, .dat o .npy).'}), 400
    tmp_name = uuid.uuid4().hex + ext
    tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], tmp_name)
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    file.save(tmp_path)

    manual_label_raw = (request.form.get('label') or '').strip()
    manual_label = _normalize_label(manual_label_raw) if manual_label_raw else ''
    record_id, reference_label = _lookup_reference_label(file.filename)
    true_label = manual_label or reference_label
    label_source = 'manual' if manual_label else ('REFERENCE-v3.csv' if reference_label else None)
    try:
        info = pred_mod.load_signal(tmp_path, file.filename)
        mat, n_channels, fs = info['mat'], info['n_channels'], info['fs']

        # ---- multi-lead handling: require the user to choose a lead first ----
        selected_channel = None
        if n_channels > 1:
            channel = (request.form.get('channel') or '').strip()
            try:
                idx = int(channel)
            except (TypeError, ValueError):
                idx = -1
            if idx < 0 or idx >= n_channels:
                return jsonify({
                    'ok': True, 'requires_channel': True,
                    'n_channels': n_channels,
                    'channel_names': info['names'], 'fs': fs,
                })
            selected_channel = idx
            signal = pred_mod.select_channel(mat, idx)
        else:
            signal = mat[0]

        # NOTE: no start-trimming. The dark bar at the very start of the trace
        # is Plotly's native Range Slider control (a UI element), NOT signal
        # data, so the signal is analysed exactly as recorded.
        result = SERVICE.predict_signal(fs, signal)
        result['n_channels'] = n_channels
        result['channel'] = (selected_channel + 1) if selected_channel is not None else 1
        result['channel_name'] = (info['names'][selected_channel]
                                  if selected_channel is not None else info['names'][0])
        result['plot'] = 'data:image/png;base64,' + SERVICE.render_plot(fs, signal, result)
        result['plotly'] = SERVICE.render_plotly(fs, signal, result)
        result['models'] = [os.path.relpath(p, MODELS_DIR)
                            for p in pred_mod.list_checkpoints(MODELS_DIR)]
        result['info'] = SERVICE.info()
        result['record_id'] = record_id
        # Compare against a user-provided label or, if absent, the official
        # REFERENCE-v3.csv label matched by filename (A00001.mat -> A00001).
        if true_label:
            result['ground_truth'] = {
                'label': true_label,
                'correct': true_label == result['dominant'] if true_label in SERVICE.classes else None,
                'known_by_model': true_label in SERVICE.classes,
                'source': label_source,
                'record': record_id,
            }
    except Exception as e:
        log.warning("Predicción fallida: %s", e)
        return jsonify({'ok': False, 'error': _sanitize(e)}), 400
    finally:
        try:
            os.remove(tmp_path)             # never persist uploads
        except OSError:
            pass

    return jsonify({'ok': True, **_jsonable(result)})


def _synthetic_signal(kind='normal', n=256 * 30):
    """Generate a synthetic single-lead ECG-ish signal for quick demos.

    kind: 'normal' (sinusoidal-ish regular), 'af' (irregular irregularity,
    fibrillatory waves), 'noise' (predominantly noisy/artefact).
    """
    import numpy as np
    t = np.arange(n) / pred_mod.TRAIN_FS
    rng = np.random.default_rng(7)
    base = 0.05 * np.sin(2 * np.pi * 1.2 * t)
    base += 0.15 * np.sin(2 * np.pi * 1.3 * t).clip(min=0)
    if kind == 'af':
        # irregular RR + coarse fibrillatory waves (no clean P)
        rr = 0.6 + 0.25 * np.sin(2 * np.pi * 0.35 * t)
        base += 0.08 * np.sin(2 * np.pi * (5.0 + 1.5 * np.sin(2 * np.pi * 0.6 * t)) * t)
        base += 0.06 * rng.normal(0, 1, n)
        base *= 1.0 + 0.3 * rr
    elif kind == 'noise':
        base += 0.22 * rng.normal(0, 1, n)
        base += 0.05 * np.sin(2 * np.pi * 50 * t)
    else:
        base += 0.02 * rng.normal(0, 1, n)
    return (base + 0.0).astype(np.float32)


@app.route('/report.pdf', methods=['POST'])
def generate_report_pdf():
    """Build and return a real, well-formatted A4 clinical report as a PDF.

    The client posts the analysis (patient info + the base64 ECG PNG already
    rendered for the interactive view) as JSON; we return application/pdf so
    the browser downloads it automatically.
    """
    data = request.get_json(silent=True) or {}
    try:
        pdf = report_pdf.build_report(data, proj.PROJECT_INFO, proj.REFERENCE)
    except Exception as e:
        log.warning("Error generando el PDF del informe: %s", e)
        return jsonify({'ok': False, 'error': _sanitize(e)}), 400
    resp = app.response_class(pdf, mimetype='application/pdf')
    resp.headers['Content-Disposition'] = \
        "attachment; filename=informe_ecg.pdf"
    resp.headers['Content-Length'] = str(len(pdf))
    return resp


@app.route('/example', methods=['POST'])
def example():
    """Generate a synthetic single-lead signal server-side and classify it.
    `kind` selects the demo: normal | af | noise (default normal)."""
    import numpy as np

    if SERVICE is None:
        return jsonify({'ok': False, 'error': 'No hay modelo cargado.'}), 500
    kind = (request.get_json(silent=True) or {}).get('kind', 'normal')
    if kind not in ('normal', 'af', 'noise'):
        kind = 'normal'
    n = 256 * 30
    signal = _synthetic_signal(kind, n)
    try:
        result = SERVICE.predict_signal(pred_mod.TRAIN_FS, signal)
        result['n_channels'] = 1
        result['channel'] = 1
        result['channel_name'] = 'I'
        result['plot'] = 'data:image/png;base64,' + SERVICE.render_plot(
            pred_mod.TRAIN_FS, signal, result)
        result['plotly'] = SERVICE.render_plotly(pred_mod.TRAIN_FS, signal, result)
        result['models'] = [os.path.relpath(p, MODELS_DIR)
                            for p in pred_mod.list_checkpoints(MODELS_DIR)]
        result['info'] = SERVICE.info()
    except Exception as e:
        return jsonify({'ok': False, 'error': _sanitize(e)}), 400
    return jsonify({'ok': True, **_jsonable(result)})


def _init_service(saved_dir='saved', model_path=None):
    """Load (or reload) the PredictionService. Called by main() and by wsgi.py.

    Returns the PredictionService or exits if no model is found.
    """
    global SERVICE, MODELS_DIR
    MODELS_DIR = saved_dir
    path = model_path or pred_mod.find_best_model(MODELS_DIR)
    if not path:
        log.error("No se encontró modelo. Entrena primero con "
                  "`python ecg/train.py examples/cinc17/config.json -e cinc17` "
                  "o pasa --model <checkpoint.pt>.")
        sys.exit(1)
    log.info("Cargando modelo: %s", path)
    svc = pred_mod.PredictionService(path)
    with SERVICE_LOCK:
        SERVICE = svc
    log.info("Clases: %s | Device: %s", svc.classes, svc.device)
    return svc


def main():
    global SERVICE, MODELS_DIR
    parser = argparse.ArgumentParser(description="ECG classification web app")
    parser.add_argument("--model", help="path to a .pt checkpoint")
    parser.add_argument("--saved", default="saved",
                        help="directory with checkpoints (auto-selects the best)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--reference", default=None,
                        help="path to CinC2017 REFERENCE-v3.csv for automatic real-label comparison")
    args = parser.parse_args()

    _init_reference(args.reference)
    _init_service(args.saved, args.model)

    # Run Flask threaded so switching models / concurrent requests do not block.
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()

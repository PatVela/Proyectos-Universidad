"""Inference + plotting service for the ECG web app.

Wraps the trained PyTorch model (checkpoint produced by ecg/train.py) and the
saved `Preproc` (mean/std + class vocabulary) so the web app only needs to:

    service = PredictionService(model_path)
    result  = service.predict_signal(fs, signal)      # numpy floats
    b64     = service.render_plot(fs, signal, result)  # PNG data-URI
    traces ,= service.render_plotly(fs, signal, result, max_points)  # JSON traces
"""

from __future__ import absolute_import

import base64
import io
import os
import sys

import numpy as np

# Make the `ecg` package importable regardless of CWD (webapp/ lives in repo root)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from ecg import load, network, util

# The model was trained on CinC2017 single-lead data sampled at 300 Hz. Any
# uploaded signal is resampled to this rate before inference.
TRAIN_FS = 300
STEP = 256
MAX_PLOT_POINTS = 12000


# --------------------------------------------------------------------------
# Signal helpers
# --------------------------------------------------------------------------
def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def resample_to(signal, orig_fs, target_fs=TRAIN_FS):
    """Resample a 1-D signal to `target_fs` Hz (scipy.signal.resample).

    If the signal is already at the target rate it is returned unchanged.
    """
    if orig_fs is None or float(orig_fs) <= 0:
        return np.asarray(signal, dtype=np.float32), TRAIN_FS
    if float(orig_fs) == float(target_fs):
        return np.asarray(signal, dtype=np.float32), target_fs
    try:
        from scipy.signal import resample
    except ImportError:
        raise RuntimeError("Se requiere scipy para re-muestrear la señal.")
    n = len(signal)
    n_out = int(round(n * target_fs / float(orig_fs)))
    # scipy.signal.resample pads to the next FFT-friendly length internally;
    # use a length >= n so no trailing samples are discarded silently.
    new_len = max(_next_pow2(n), n)
    out = resample(np.asarray(signal, dtype=np.float32), new_len)
    return np.asarray(out[:n_out], dtype=np.float32), target_fs


# --------------------------------------------------------------------------
# Service
# --------------------------------------------------------------------------
class PredictionService:
    """Loads a trained model + preprocessor and runs single-lead inference."""

    def __init__(self, model_path, device=None):
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' not in ckpt:
            raise ValueError(
                "El archivo no es un checkpoint de train.py "
                "(falta 'model_state_dict').")
        self.preproc = ckpt.get('preproc')
        self.classes = ckpt.get('classes') or (self.preproc.classes if self.preproc
                                               else None)
        config = ckpt.get('config', {}) or {}
        self.config = config
        num_cat = len(self.classes) if self.classes else int(
            config.get('num_categories', 5))

        self.model = network.build_network(num_categories=num_cat, **config)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device).eval()
        self.model_type = ckpt.get('model_type') or (
            'CNN convencional' if config.get('is_regular_conv') else 'ResNet-34')
        self.num_parameters = ckpt.get('num_parameters') or util.count_parameters(self.model)

        # exposed training-time metrics (val_loss / val_acc were saved at each
        # checkpoint). Full dev metrics are available via examples/cinc17/evaluate.py
        self.meta = {
            'val_loss': ckpt.get('val_loss'),
            'val_acc': ckpt.get('val_acc'),
            'epoch': ckpt.get('epoch'),
            'train_fs': TRAIN_FS,
            'step': STEP,
        }
        self.class_to_desc = {
            'N': 'Normal',
            'A': 'Atrial fibrillation / flutter (AF)',
            'O': 'Other rhythm',
            '~': 'Noise / artifact',
            '|': 'Silence / unclassifiable',
        }

    def info(self):
        return {
            'model_path': self.model_path,
            'model_type': self.model_type,
            'is_regular_conv': bool(self.config.get('is_regular_conv', False)),
            'num_parameters': self.num_parameters,
            'classes': list(self.classes),
            'device': str(self.device),
            'train_fs': TRAIN_FS,
            'step': STEP,
            'meta': self.meta,
            'config': dict(self.config),
            'class_desc': {c: self.class_to_desc.get(c, c) for c in self.classes},
        }

    # ------------------------------------------------------------------ input
    def normalize(self, signal):
        """Normalise amplitude and truncate to a multiple of STEP.

        Returns (x_tensor (1,1,T), applied_fs). Amplitude normalisation uses the
        saved global mean/std (same as training).
        """
        if self.preproc is not None:
            mean = float(self.preproc.mean)
            std = float(self.preproc.std)
        else:
            mean = float(np.asarray(signal).mean())
            std = float(np.asarray(signal).std()) or 1.0
        x = (np.asarray(signal, dtype=np.float32) - mean) / std
        x = x[:STEP * (len(x) // STEP)]
        return torch.from_numpy(np.ascontiguousarray(x[None, None, :])).to(self.device)

    # -------------------------------------------------------------- inference
    def predict_signal(self, fs, signal):
        """fs -> (Hz) input sampling rate; signal -> 1-D array.

        Returns dict with probs per output interval + per-record summary.
        """
        fs = float(fs if fs and fs > 0 else TRAIN_FS)
        signal, applied_fs = resample_to(signal, fs, TRAIN_FS)
        x = self.normalize(signal)
        with torch.no_grad():
            probs = self.model(x).cpu().numpy()
        probs = probs[0]
        pred_idx = probs.argmax(axis=-1)
        labels = [self.classes[i] for i in pred_idx]

        counts = {c: int((pred_idx == self.classes.index(c)).sum()) for c in
                  self.classes if c in self.classes}
        total = len(labels) or 1
        summary = sorted(({'label': c, 'desc': self.class_to_desc.get(c, c),
                           'count': v, 'pct': round(100.0 * v / total, 1)}
                          for c, v in counts.items() if v),
                         key=lambda d: d['count'], reverse=True)
        per_interval = [{'idx': i, 'start_ms': round(i * STEP / applied_fs * 1000),
                         'label': labels[i],
                         'desc': self.class_to_desc.get(labels[i], labels[i]),
                         'prob': round(float(probs[i, pred_idx[i]]), 3)}
                        for i in range(len(labels))]

        return {'probs': probs, 'labels': labels, 'per_interval': per_interval,
                'summary': summary,
                'dominant': summary[0]['label'] if summary else None,
                'n_intervals': len(labels),
                'resampled': float(fs) != float(applied_fs),
                'orig_fs': float(fs), 'applied_fs': float(applied_fs),
                'n_samples_in': int(len(signal)),
                'n_samples_out': int(len(signal))}

    # ------------------------------------------------------------------ plot
    def render_plot(self, fs, signal, result):
        """Render ECG trace + color-coded prediction bands as a base64 PNG."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        signal = np.asarray(signal, dtype=np.float32)
        t_sec = np.arange(len(signal)) / (result['applied_fs'] or TRAIN_FS)

        palette = {'N': '#2ca02c', 'A': '#d62728', 'O': '#ff7f0e',
                   '~': '#7f7f7f', '|': '#1f77b4'}
        default = '#9467bd'

        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(13, 6.5), gridspec_kw={'height_ratios': [3, 1]},
            sharex=True)
        fig.subplots_adjust(hspace=0.15)
        ax.plot(t_sec, signal, color='#1a1a2e', linewidth=0.8)
        ax.set_ylabel('Amplitud (mV)')
        ax.set_title('ECG — predicción por intervalo de {} muestras (~{:.2f} s)'.format(
            STEP, STEP / (result['applied_fs'] or TRAIN_FS)))
        ax.grid(alpha=0.25)
        fs_eff = result['applied_fs'] or TRAIN_FS
        for i in range(result['n_intervals']):
            c = palette.get(result['labels'][i], default)
            start = i * STEP / fs_eff
            end = min((i + 1) * STEP / fs_eff, t_sec[-1])
            ax.axvspan(start, end, color=c, alpha=0.10, linewidth=0)
        handles = [plt.Rectangle((0, 0), 1, 1, color=palette.get(c, default))
                   for c in dict.fromkeys(result['labels'])]
        ax.legend(handles, list(dict.fromkeys(result['labels'])),
                  loc='upper right', fontsize=9, framealpha=0.9)

        s = result['summary']
        bar_labels = [d['label'] + ' (' + d['desc'][:16] + ')' for d in s]
        bar_vals = [d['pct'] for d in s]
        bar_cols = [palette.get(d['label'], default) for d in s]
        ax2.bar(range(len(s)), bar_vals, color=bar_cols, alpha=0.85)
        ax2.set_xticks(range(len(s)))
        ax2.set_xticklabels(bar_labels, rotation=20, ha='right', fontsize=8)
        ax2.set_ylabel('% del registro')
        ax2.set_ylim(0, 100)
        ax2.set_title('Distribución de clases en todo el registro', fontsize=10)
        ax2.grid(axis='y', alpha=0.25)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('ascii')

    # ------------------------------------------------- plotly (interactive)
    def render_plotly(self, fs, signal, result, max_points=MAX_PLOT_POINTS):
        """Return JSON-serialisable Plotly traces + layout.

        Decimates the raw signal for the browser and overlays per-interval
        colour bands, so the user gets zoom/pan/tooltips without a CDN.
        """
        signal = np.asarray(signal, dtype=np.float32)
        fs_eff = float(result['applied_fs'] or TRAIN_FS)
        t = (np.arange(len(signal)) / fs_eff).round(4)

        # optional light decimation for huge records
        step = max(1, int(np.ceil(len(signal) / max_points)))
        if step > 1:
            idx = np.arange(0, len(signal), step)
            t = t[idx]
            signal = signal[idx]

        # colour bands as filled rectangles (use trace_via vrect? no -> shapes)
        shapes = []
        for i in range(result['n_intervals']):
            c = _color(result['labels'][i])
            start = round(i * STEP / fs_eff, 4)
            end = round(min((i + 1) * STEP / fs_eff, t[-1] if len(t) else start), 4)
            shapes.append({'type': 'rect', 'xref': 'x', 'yref': 'paper',
                           'x0': start, 'x1': end, 'y0': 0, 'y1': 1,
                           'fillcolor': c, 'opacity': 0.10, 'line': {'width': 0}})

        trace = {'x': np.asarray(t).tolist(), 'y': np.asarray(signal).tolist(),
                 'mode': 'lines', 'name': 'ECG', 'type': 'scatter',
                 'line': {'color': '#132436', 'width': 1}}
        return {'traces': [trace], 'shapes': shapes,
                'interval_s': round(STEP / fs_eff, 4),
                'grid': {'fs': fs_eff}}


def _color(label):
    palette = {'N': '#2ca02c', 'A': '#d62728', 'O': '#ff7f0e',
               '~': '#7f7f7f', '|': '#1f77b4'}
    return palette.get(label, '#9467bd')


# --------------------------------------------------------------------------
# CSV / file parsing helpers (used by app.py)
# --------------------------------------------------------------------------
def parse_ecg_csv(text, default_fs=TRAIN_FS):
    """Parse an uploaded CSV of a single-lead ECG. Returns (fs, signal_1d)."""
    lines = [ln for ln in text.splitlines()
             if ln.strip() and not ln.strip().startswith(('#', '%', 'L'))]
    rows = []
    for ln in lines:
        toks = [t for t in ln.replace(',', ' ').split() if t]
        rows.append(toks)

    fs = None
    signal = []
    data_rows = []
    for toks in rows:
        floats, ok = [], True
        for t in toks:
            try:
                floats.append(float(t))
            except ValueError:
                ok = False
                break
        if not ok:
            try:
                candidate = float(toks[0])
                if fs is None and candidate > 0:
                    fs = candidate
            except (ValueError, IndexError):
                pass
            continue
        data_rows.append(floats)

    if not data_rows:
        raise ValueError("No se encontraron datos numéricos en el CSV.")

    if len(data_rows) == 1 or len(data_rows[0]) > 1:
        first = data_rows[0]
        if len(first) > 1 and fs is None and first[0] > 0:
            fs = first[0]
            signal = first[1:]
            for r in data_rows[1:]:
                signal.extend(r)
        else:
            signal = first
            for r in data_rows[1:]:
                signal.extend(r)
    else:
        signal = [r[0] for r in data_rows]

    signal = np.asarray(signal, dtype=np.float32)
    if fs is None:
        fs = float(default_fs)
    if len(signal) < STEP:
        raise ValueError("La señal es demasiado corta (< {} muestras).".format(STEP))
    return float(fs), signal


_LEAD_NAMES_12 = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                  'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def _to_2d(arr):
    """Normalise an array to a (channels, samples) matrix."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim == 2:
        r, c = arr.shape
        # standard multi-lead layout is (leads, samples); transpose if samples
        # is the smaller axis and looks like the 'leads' axis.
        if r <= 12 and r <= c and (r > 1 or c > 1):
            return arr
        if c <= 12 and c <= r:
            return arr.T
        return arr                                        # assume rows = leads
    # 3D or more: flatten trailing dims into samples, keep first as channels
    return arr.reshape(arr.shape[0], -1)


def _channel_names(n):
    if n <= 12:
        return list(_LEAD_NAMES_12[:n])
    return ['Derivación {}'.format(i + 1) for i in range(n)]


def _signal_info(mat, fs):
    """Build a reusable description of a loaded signal matrix."""
    mat = _to_2d(mat)
    n = mat.shape[0]
    return {'fs': float(fs), 'mat': mat, 'n_channels': n,
            'names': _channel_names(n)}


def select_channel(mat, index):
    """Return the 1-D signal for channel `index` (0-based)."""
    mat = _to_2d(mat)
    index = int(index)
    if index < 0 or index >= mat.shape[0]:
        raise ValueError("Canal fuera de rango (0..{})".format(mat.shape[0] - 1))
    return mat[index]


def load_signal(path, filename):
    """Load a CSV / .mat / .npy / .dat (format-212) upload.

    Returns a dict {fs, mat (channels x samples), n_channels, names}. It NEVER
    squeezes blindly, so multi-lead recordings are preserved for channel
    selection. For .mat/.dat the sampling rate may be unavailable, so it
    defaults to TRAIN_FS (the app announces this).
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext in ('.csv', ''):
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            fs, signal = parse_ecg_csv(f.read())
        return _signal_info(np.asarray(signal, dtype=np.float32)[None, :], fs)

    if ext == '.npy':
        arr = np.load(path).astype(np.float32)
        return _signal_info(arr, float(TRAIN_FS))

    # .mat (read the 'val' array if present) or .dat (PhysioNet format 212)
    if ext == '.mat':
        try:
            import scipy.io as sio
            d = sio.loadmat(path)
            key = 'val' if 'val' in d else next(
                (k for k, v in d.items() if not k.startswith('__')), None)
            if key is not None:
                return _signal_info(d[key], float(TRAIN_FS))
        except Exception:
            pass  # fall through to ecg.load_ecg
    sig = load.load_ecg(os.path.abspath(path))
    return _signal_info(sig, float(TRAIN_FS))


def find_best_model(models_dir):
    """Return the checkpoint with the lowest val_loss (first number in name)."""
    return util.find_best_model(models_dir)


def list_checkpoints(models_dir):
    """Return all .pt checkpoints under models_dir sorted by best (val_loss)."""
    return util.list_checkpoints(models_dir)


def _short_model_id(model_path):
    """A short, stable identifier for a checkpoint (for display in the UI).

    Uses a truncated SHA-256 of the file content, so the same model always
    shows the same id (a poor-man's 'version commit' for reproducibility).
    """
    if not model_path or not os.path.exists(model_path):
        return ''
    try:
        import hashlib
        h = hashlib.sha256()
        with open(model_path, 'rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                h.update(block)
        return h.hexdigest()[:10]
    except (OSError, ValueError):
        return ''

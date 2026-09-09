"""PyTorch port of awni/ecg util.py (save/load of the preprocessor).

Also hosts model-helper utilities shared by the CLI and the web app.
"""

import os
import pickle


def save(preproc, dirname):
    preproc_f = os.path.join(dirname, "preproc.bin")
    with open(preproc_f, 'wb') as f:
        pickle.dump(preproc, f)


def load(dirname):
    preproc_f = os.path.join(dirname, "preproc.bin")
    with open(preproc_f, 'rb') as f:
        return pickle.load(f)


def _best_key(p):
    """Sort key = checkpoint's val_loss (first number in filename)."""
    try:
        return float(os.path.basename(p).split('-')[0])
    except (ValueError, IndexError):
        return float('inf')


def find_best_model(models_dir):
    """Return the checkpoint with the lowest val_loss (first number in name)."""
    if not models_dir or not os.path.isdir(models_dir):
        return None
    ptfiles = [os.path.join(root, f) for root, _, fs in os.walk(models_dir)
               for f in fs if f.endswith('.pt')]
    if not ptfiles:
        return None
    return min(ptfiles, key=_best_key)


def list_checkpoints(models_dir):
    """Return all .pt checkpoints under models_dir sorted best-first."""
    if not models_dir or not os.path.isdir(models_dir):
        return []
    ptfiles = [os.path.join(root, f) for root, _, fs in os.walk(models_dir)
               for f in fs if f.endswith('.pt')]
    ptfiles.sort(key=_best_key)
    return ptfiles


def count_parameters(model):
    """Return total/trainable/non-trainable parameter counts for a torch model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': int(total),
        'trainable': int(trainable),
        'non_trainable': int(total - trainable),
    }

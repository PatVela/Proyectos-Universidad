"""PyTorch training entry point (port of awni/ecg train.py).

Usage:
    python ecg/train.py examples/cinc17/config.json -e cinc17

The relative paths in config.json ("train", "dev", "save_dir") are resolved
against the repository root (the directory containing this file's parent),
mirroring the original where you run from the repo root.
"""

from __future__ import print_function, division, absolute_import

import argparse
import csv
import json
import numpy as np
import os
import random
import sys
import time
import torch
import torch.nn as nn

MAX_EPOCHS = 100
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

try:
    from . import load, network, util
except ImportError:            # direct run as `python ecg/train.py`
    sys.path.insert(0, REPO_ROOT)
    from ecg import load, network, util


def make_save_dir(dirname, experiment_name):
    start_time = str(int(time.time())) + '-' + str(random.randrange(1000))
    save_dir = os.path.join(dirname, experiment_name, start_time)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def get_path_for_saving(save_dir, epoch, val_loss, val_acc, loss, acc, ext='pt'):
    name = "{val_loss:.3f}-{val_acc:.3f}-{epoch:03d}-{loss:.3f}-{acc:.3f}.{ext}".format(
        val_loss=val_loss, val_acc=val_acc, epoch=epoch, loss=loss, acc=acc, ext=ext)
    return os.path.join(save_dir, name)


def _get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _to_device(device, *tensors):
    return [t.to(device) for t in tensors]


def set_seed(seed):
    """Set Python/NumPy/PyTorch seeds for reproducible ablations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_type(params):
    return "CNN convencional" if params.get('is_regular_conv') else "ResNet-34"


def _save_training_artifacts(save_dir, history, params, model_info, best_epoch,
                             best_val_loss, total_seconds):
    """Persist per-epoch logs used by the comparison experiment."""
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'history.csv')
    fieldnames = [
        'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
        'epoch_seconds', 'learning_rate', 'checkpoint'
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k) for k in fieldnames})

    summary = {
        'model_type': model_info['model_type'],
        'is_regular_conv': bool(params.get('is_regular_conv', False)),
        'num_parameters': model_info['num_parameters'],
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'total_seconds': float(total_seconds),
        'total_minutes': float(total_seconds) / 60.0,
        'epochs_completed': len(history),
        'config': params,
        'history': history,
    }
    with open(os.path.join(save_dir, 'training_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def make_batch(device, preproc, xb, yb):
    """Build training tensors for a batch.

    The generator (load.data_generator) already returns *processed* arrays:
      xb -> (B, time, 1) float32, yb -> (B, time) int64 (padded)
    Here we only repack them into tensors.
    """
    x = torch.from_numpy(np.ascontiguousarray(xb.transpose(0, 2, 1)))  # (B,1,time)
    y = torch.from_numpy(yb)                                            # (B,time)
    x, y = _to_device(device, x, y)
    return x, y


def data_generator(batch_size, preproc, x, y, device):
    gen = load.data_generator(batch_size, preproc, x, y)
    while True:
        xb, yb = next(gen)
        yield make_batch(device, preproc, xb, yb)


def evaluate(model, loader, num_batches, device, criterion):
    model.eval()
    tot_loss, tot_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = next(loader)
            logits = model.logits(x)
            t = logits.shape[1]                      # network output time
            # targets are (B, time_in); the network outputs time_in/256 steps.
            # Downsample targets to the network output grid by taking every
            # 256-th step (they are constant per record, so it is exact).
            y_grid = y[:, :t]
            loss = criterion(logits.reshape(-1, logits.shape[-1]),
                             y_grid.reshape(-1))
            pred = logits.argmax(dim=-1)
            acc = (pred == y_grid).float().mean().item()
            tot_loss += loss.item() * x.shape[0]
            tot_acc += acc * x.shape[0]
            n += x.shape[0]
    model.train()
    return tot_loss / max(n, 1), tot_acc / max(n, 1)


def train(args, params):
    seed = args.seed if args.seed is not None else params.get('seed')
    if seed is not None:
        seed = int(seed)
        set_seed(seed)
        params['seed'] = seed
        print("Seed:", seed)

    device = _get_device()
    print("Device:", device)

    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)

    print("Loading training set...")
    train = load.load_dataset(resolve(params['train']))
    print("Loading dev set...")
    dev = load.load_dataset(resolve(params['dev']))
    print("Building preprocessor...")
    preproc = load.Preproc(*train)
    print("Training size: {}".format(len(train[0])))
    print("Dev size: {}".format(len(dev[0])))
    print("Classes ({}): {}".format(len(preproc.classes), preproc.classes))

    save_dir = make_save_dir(resolve(params['save_dir']), args.experiment)
    util.save(preproc, save_dir)

    num_categories = len(preproc.classes)
    model = network.build_network(num_categories=num_categories, **params)
    param_count = util.count_parameters(model)
    model_info = {
        'model_type': _model_type(params),
        'num_parameters': param_count,
    }
    print("Modelo:", model_info['model_type'])
    print("Parámetros: total={total:,} | entrenables={trainable:,}".format(**param_count))
    model.to(device)

    batch_size = int(params.get("batch_size", 32))
    lr = float(params.get("learning_rate", 1e-3))
    clipnorm = params.get("clipnorm", 1.0)
    max_epochs = args.epochs or params.get("max_epochs", MAX_EPOCHS)
    patience_es = int(params.get("early_stopping_patience", 8))
    patience_lr = int(params.get("reduce_lr_patience", 2))
    lr_factor = float(params.get("reduce_lr_factor", 0.1))
    min_lr = lr * 0.001

    train_steps = int(len(train[0]) // batch_size)
    dev_steps = int(len(dev[0]) // batch_size)
    print("Train steps/epoch:", train_steps, "| Dev steps/epoch:", dev_steps)

    # ----- optimiser (Adam with gradient clipping, as in add_compile) -----
    # Keras/TF Adam default epsilon is 1e-7; PyTorch defaults to 1e-8. Pin the
    # Keras value for numerical parity with the original training.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)

    # ----- loss: categorical cross-entropy over the flattened time grid -----
    criterion = nn.CrossEntropyLoss()

    print("Starting training.")
    best_val_loss = float('inf')
    best_epoch = None
    no_improve_es = 0
    no_improve_lr = 0
    history = []
    train_start_time = time.time()

    for epoch in range(max_epochs):
        epoch_start_time = time.time()
        train_gen = data_generator(batch_size, preproc, *train, device=device)
        dev_gen = data_generator(batch_size, preproc, *dev, device=device)

        model.train()
        epoch_loss, epoch_acc, n = 0.0, 0.0, 0
        for step in range(train_steps):
            x, y = next(train_gen)
            optimizer.zero_grad()
            logits = model.logits(x)
            t = logits.shape[1]
            y_grid = y[:, :t]
            loss = criterion(logits.reshape(-1, logits.shape[-1]),
                             y_grid.reshape(-1))
            loss.backward()
            # global norm clipping (tf clipnorm)
            nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
            optimizer.step()
            pred = logits.argmax(dim=-1)
            acc = (pred == y_grid).float().mean().item()
            bs = x.shape[0]
            epoch_loss += loss.item() * bs
            epoch_acc += acc * bs
            n += bs
            if step % 50 == 0:
                print("\tepoch{:03d}/{:03d} step{:05d}/{:05d} loss {:.4f} acc {:.4f}".format(
                    epoch, max_epochs, step, train_steps, loss.item(), acc))

        train_loss = epoch_loss / max(n, 1)
        train_acc = epoch_acc / max(n, 1)
        val_loss, val_acc = evaluate(model, dev_gen, dev_steps, device, criterion)
        epoch_seconds = time.time() - epoch_start_time
        total_seconds = time.time() - train_start_time
        print("Epoch {:03d} | train loss {:.4f} acc {:.4f} | val loss {:.4f} acc {:.4f} | {:.1f}s".format(
            epoch, train_loss, train_acc, val_loss, val_acc, epoch_seconds))

        # ----- checkpoint (saved every epoch, like ModelCheckpoint) -----
        cpath = get_path_for_saving(save_dir, epoch, val_loss, val_acc, train_loss, train_acc)
        torch.save({
            'model_state_dict': model.state_dict(),
            'preproc': preproc,
            'config': params,
            'classes': preproc.classes,
            'epoch': epoch,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'model_type': model_info['model_type'],
            'num_parameters': param_count,
            'epoch_seconds': epoch_seconds,
            'training_seconds_so_far': total_seconds,
        }, cpath)
        print("\tSaved:", os.path.basename(cpath))

        current_lr = optimizer.param_groups[0]['lr']
        history.append({
            'epoch': int(epoch),
            'train_loss': round(float(train_loss), 6),
            'train_acc': round(float(train_acc), 6),
            'val_loss': round(float(val_loss), 6),
            'val_acc': round(float(val_acc), 6),
            'epoch_seconds': round(float(epoch_seconds), 3),
            'learning_rate': float(current_lr),
            'checkpoint': os.path.basename(cpath),
        })

        # ----- LR scheduling on plateau (ReduceLROnPlateau) -----
        stop_early = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve_lr = 0
            no_improve_es = 0
        else:
            no_improve_lr += 1
            no_improve_es += 1
            if no_improve_lr >= patience_lr and optimizer.param_groups[0]['lr'] > min_lr:
                for g in optimizer.param_groups:
                    g['lr'] = max(g['lr'] * lr_factor, min_lr)
                print("\tReduceLR on plateau -> lr = {:.6f}".format(
                    optimizer.param_groups[0]['lr']))
                no_improve_lr = 0
            if no_improve_es >= patience_es:
                stop_early = True

        _save_training_artifacts(save_dir, history, params, model_info,
                                 best_epoch, float(best_val_loss), total_seconds)
        if stop_early:
            print("Early stopping at epoch", epoch)
            break

    print("Training finished. Best val loss {:.4f}.".format(best_val_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument("--experiment", "-e", help="experiment tag", default="default")
    parser.add_argument("--epochs", type=int, default=None, help="override max epochs")
    parser.add_argument("--seed", type=int, default=None,
                        help="override random seed from config for reproducible ablations")
    args = parser.parse_args()
    params = json.load(open(args.config_file, 'r'))
    train(args, params)

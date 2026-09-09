"""WSGI entry point for deploying the ECG web app with gunicorn / waitress.

Reads environment variables to decide which model/reference data to load at
import time:

    ECG_SAVED      -> directory of checkpoints (auto-selects the lowest val_loss)
    ECG_MODEL      -> explicit path to a .pt checkpoint (takes precedence)
    ECG_REFERENCE  -> optional path to CinC2017 REFERENCE-v3.csv

Defaults match the CLI (--saved saved, --model <given>). Example:

    export ECG_SAVED="saved"
    gunicorn -w 2 -b 0.0.0.0:5000 --timeout 120 "webapp.wsgi:app"

    # Windows native with waitress:
    waitress-serve --listen=*:5000 "webapp.wsgi:app"
"""

from __future__ import absolute_import

import os
import sys
import logging

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# make sibling modules (prediction) importable when loaded as "webapp.wsgi:app"
_WEBAPP = os.path.dirname(os.path.abspath(__file__))
if _WEBAPP not in sys.path:
    sys.path.insert(0, _WEBAPP)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger('ecg-wsgi')

import prediction as pred_mod
from app import app, _init_reference, _init_service

# Load the reference labels and model ONCE at import (shared per worker).
_SAVED = os.environ.get('ECG_SAVED', 'saved')
_MODEL = os.environ.get('ECG_MODEL')
_REFERENCE = os.environ.get('ECG_REFERENCE')
_init_reference(_REFERENCE)
_init_service(_SAVED, _MODEL)
log.info("WSGI app inicializada (saved=%s, model=%s, reference=%s)",
         _SAVED, _MODEL or '(auto)', _REFERENCE or '(auto)')

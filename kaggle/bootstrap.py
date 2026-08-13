"""
Kaggle bootstrap script — Cell 1 of every GPU notebook run.

Usage:
    %run /kaggle/input/<repo-slug>/kaggle/bootstrap.py

This script:
  1. Installs missing Python dependencies at pinned versions via pip.
  2. Sets environment variables that the rest of the codebase reads
     (REPO_ROOT, DATA_DIR, OUT_DIR).
  3. Prepends REPO_ROOT to sys.path so all repo modules are importable.

NOTE: The Time-Series-Library models are now vendored inside the repo
(models/tsl/ + layers/), so there is NO separate TSL dataset to attach
and no TSL_ROOT to configure. The repo is fully self-contained.

Run this cell before any import from the `experiments`, `data_pipeline`,
`models`, `baselines`, or `reproducibility` packages.
"""

import os
import sys
import subprocess

# ---------------------------------------------------------------------------
# 1. Install pinned dependencies not present in the Kaggle GPU base image.
#    NOTE: vmdpy is intentionally NOT installed here — VMD decomposition runs
#    only during LOCAL CPU preprocessing. Kaggle just reads the pre-computed
#    vmd_imfs.npz, so vmdpy is never imported on the training path.
# ---------------------------------------------------------------------------
subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '-q',
    'einops==0.8.0',
    'fightingcv-attention==1.0.0',
    'dill==0.3.8',
    'PyYAML==6.0.2',
    'tabulate==0.9.0',
])

# ---------------------------------------------------------------------------
# 2. Set environment variables (Kaggle mount convention).
#    Adjust the dataset slugs below to match how you named the uploads.
# ---------------------------------------------------------------------------
os.environ['REPO_ROOT'] = '/kaggle/input/itransformer-lstm-ca-kan-master'
os.environ['DATA_DIR']  = '/kaggle/input/sdwpf-turb1-preprocessed'
os.environ['OUT_DIR']   = '/kaggle/working'

# ---------------------------------------------------------------------------
# 3. Add repo root to sys.path (front, so repo modules take precedence).
# ---------------------------------------------------------------------------
sys.path.insert(0, os.environ['REPO_ROOT'])

print("bootstrap.py complete")
print(f"  REPO_ROOT : {os.environ['REPO_ROOT']}")
print(f"  DATA_DIR  : {os.environ['DATA_DIR']}")
print(f"  OUT_DIR   : {os.environ['OUT_DIR']}")
print(f"  sys.path[0]: {sys.path[0]}")

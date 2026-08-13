"""
experiments/runner.py
=====================
Unified train/eval interface for the wind-power forecasting experiment runner.

Provides:
- RunConfig  : dataclass describing a single experiment run
- run(cfg)   : end-to-end function that trains, evaluates, persists, and
               returns a RunRecord

Flow
----
1.  set_global_seed(cfg.seed)
2.  Load preprocessed CSV, apply scaler, assemble channel matrix
3.  (If vmd_on) load vmd_imfs.npz and prepend IMF channels
4.  Build WindowedSeriesDatasets + DataLoaders for train/valid/test
5.  Resolve model from MODEL_REGISTRY (or fallback builder)
6.  Train loop with EarlyStopping; record per-epoch losses
7.  Load best checkpoint; evaluate on test set
8.  inverse_transform_target on predictions
9.  compute_metrics on denormalised kW values
10. Persist:
      checkpoint  → model_save/wind/{run_id}.pt
      predictions → outputs/runs/{run_id}_preds.npz
      losses      → outputs/runs/{run_id}_losses.npz
      record      → outputs/runs/run_records.jsonl (append)
11. Return RunRecord

Custom exceptions
-----------------
InvalidHorizonError       – horizon not in [1, 24]
InsufficientWindowError   – lookback + horizon > n_partition_samples
UnknownModelError         – model_name not in MODEL_REGISTRY

Requirements: 3.1, 3.2, 3.4, 3.6, 7.1, 7.2, 7.7, 14.6
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import dill
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_pipeline.scaling import FeatureScaler
from data_pipeline.manifest import PartitionManifest, VMDManifest
from data_pipeline.windowing import WindowedSeriesDataset
from experiments.metrics import compute_metrics
from reproducibility.seeds import set_global_seed
from reproducibility.environment import capture_environment
from reproducibility.records import RunRecord, append_run_record
from utils.tools import EarlyStopping


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class InvalidHorizonError(ValueError):
    """Raised when horizon is not in the valid range [1, 24]."""


class InsufficientWindowError(ValueError):
    """Raised when lookback + horizon exceeds the number of samples in any partition."""


class UnknownModelError(KeyError):
    """Raised when model_name is not found in MODEL_REGISTRY."""


# ---------------------------------------------------------------------------
# RunConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """Complete specification for a single experiment run.

    Parameters
    ----------
    run_id : str
        Unique identifier for this run (e.g. uuid + slug).
    model_name : str
        Key into MODEL_REGISTRY. One of:
        'proposed' | 'lstm' | 'transformer' | 'informer' | 'fedformer' |
        'dlinear' | 'patchtst' | 'itransformer' | 'timesnet' |
        'ablation:itrans_off' | 'ablation:lstm_off' | 'ablation:fusion_concat' |
        'ablation:fusion_sum' | 'ablation:head_linear' | 'ablation:head_mlp' |
        'ablation:vmd_off' | 'ablation:outlier_off'
    seed : int
        Global random seed.
    lookback : int
        Input window length in time steps (default 144).
    horizon : int
        Forecast horizon length.  Must be in [1, 24].
    train : dict
        Training hyper-parameters:
            batch_size, learning_rate (or lr), epochs, early_stop_patience
            (or patience), early_stop_delta (or delta), optimizer ('adam'),
            scheduler ('reduce_on_plateau'), scheduler_factor,
            scheduler_patience, loss ('mae' | 'mse').
    model : dict
        Model-specific hyper-parameters passed to the registry factory.
    ablation : dict
        Ablation switches (use_itransformer, use_lstm, fusion_type, head_type).
    runtime : dict
        Execution context:
            execution_location ('local_cpu' | 'kaggle_gpu'),
            device ('auto' | 'cuda' | 'cpu'),
            deterministic (bool),
            tsl_root (str),
            out_dir (str).
    dataset : dict
        Dataset paths and config:
            csv_path, scaler_path, partition_path,
            vmd sub-dict (enabled bool, imf_path str, K int, …).
    """

    run_id: str
    model_name: str
    seed: int
    lookback: int
    horizon: int
    train: dict
    model: dict
    ablation: dict
    runtime: dict
    dataset: dict


# ---------------------------------------------------------------------------
# Helpers: device resolution
# ---------------------------------------------------------------------------

def _resolve_device(runtime: dict) -> torch.device:
    """Select cuda or cpu based on runtime.device and hardware availability."""
    device_str = runtime.get("device", "auto")
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Helpers: data loading
# ---------------------------------------------------------------------------

_FEATURE_COLS = ["Patv", "Wspd", "Wdir", "Etmp", "Itmp"]


def _load_data(cfg: RunConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and assemble the scaled feature matrix for all three partitions.

    Returns
    -------
    tuple of (train_data, valid_data, test_data), each shape (N_split, F_in)
    where F_in = 5 + K if vmd_on, else 5.
    """
    ds_cfg = cfg.dataset
    vmd_cfg = ds_cfg.get("vmd", {})
    vmd_enabled = vmd_cfg.get("enabled", False)
    cleaning_cfg = ds_cfg.get("cleaning", {})
    physical_rules_enabled = bool(cleaning_cfg.get("physical_rules", True))

    # ── Load partition boundaries ─────────────────────────────────────────
    partition_path = ds_cfg["partition_path"]
    indices = PartitionManifest.read(partition_path)

    # ── Load CSV + restrict to 5 features ────────────────────────────────
    csv_path = ds_cfg["csv_path"]
    df = pd.read_csv(csv_path)

    # Try to select our 5 feature columns (case-insensitive fallback)
    available = list(df.columns)
    cols_to_use = []
    for col in _FEATURE_COLS:
        if col in available:
            cols_to_use.append(col)
        else:
            # Case-insensitive match
            matches = [c for c in available if c.lower() == col.lower()]
            if matches:
                cols_to_use.append(matches[0])
            else:
                raise KeyError(
                    f"Expected feature column '{col}' not found in CSV. "
                    f"Available columns: {available}"
                )

    df = df[cols_to_use]

    # ── Apply (or skip) physical-rule outlier cleaning ─────────────────────
    # The CSV on disk (sdwpf_turb1_cleaned_final.csv) has already been
    # cleaned upstream, so physical_rule_clean typically finds zero
    # violations on this particular dataset regardless of this flag — but
    # the flag must still be honoured so that ablation:outlier_off is a
    # true no-op only because the *data* has nothing left to clean, not
    # because the *code* silently ignores the switch.
    from data_pipeline.cleaning import physical_rule_clean
    df, _clean_report = physical_rule_clean(df, enable=physical_rules_enabled)
    df = df.interpolate(method="linear", limit_direction="both")

    raw = df.values.astype(np.float64)

    # ── Load scaler ───────────────────────────────────────────────────────
    scaler_path = ds_cfg["scaler_path"]
    scaler = FeatureScaler.load(scaler_path)

    # ── Apply scaler to each partition ────────────────────────────────────
    tr_s, tr_e = indices.train
    va_s, va_e = indices.valid
    te_s, te_e = indices.test

    raw_train = raw[tr_s:tr_e]
    raw_valid = raw[va_s:va_e]
    raw_test  = raw[te_s:te_e]

    scaled_train = scaler.transform(raw_train).astype(np.float32)
    scaled_valid = scaler.transform(raw_valid).astype(np.float32)
    scaled_test  = scaler.transform(raw_test).astype(np.float32)

    if not vmd_enabled:
        # vmd_off: [Patv, Wspd, Wdir, Etmp, Itmp] — 5 channels
        return scaled_train, scaled_valid, scaled_test, scaler

    # ── Load pre-computed VMD IMFs ────────────────────────────────────────
    imf_path = vmd_cfg.get("imf_path")
    if imf_path is None or not Path(imf_path).exists():
        raise FileNotFoundError(
            f"VMD IMF file not found at '{imf_path}'. "
            "Run the preprocessing CLI first (scripts/preprocess_cli.py)."
        )

    imf_data = np.load(imf_path)
    # vmd_imfs.npz stores the IMF array under 'imfs' key; shape (N_total, K)
    if "imfs" in imf_data:
        all_imfs = imf_data["imfs"].astype(np.float32)
    else:
        # Fallback: take the first array
        key = list(imf_data.keys())[0]
        all_imfs = imf_data[key].astype(np.float32)

    imf_train = all_imfs[tr_s:tr_e]
    imf_valid = all_imfs[va_s:va_e]
    imf_test  = all_imfs[te_s:te_e]

    # ── Guard: sub-bands must additively reconstruct the target ───────────
    # The reported model trains on a partition-isolated db4 DWT, whose
    # defining property is y == A4 + sum(D1..D4) inside every partition.
    #
    # The pitfall this exists to catch: every run script hardcodes
    # imf_path="outputs/manifests/vmd_imfs.npz", and the DWT arrays only
    # reach that path via a `cp` inside scripts/run_dwt_experiments.sh. If
    # that copy has not been run on this machine, the file still holds VMD
    # modes, which are NOT an exact additive decomposition (observed max
    # error ~1.0, against ~1e-7 for the float32-stored DWT). Training would
    # then proceed silently on a different decomposition than the one
    # reported in the paper. Fail loudly instead of producing quiet garbage.
    if not vmd_cfg.get("allow_non_additive", False):
        worst = 0.0
        for imfs, scaled in ((imf_train, scaled_train),
                             (imf_valid, scaled_valid),
                             (imf_test,  scaled_test)):
            if imfs.shape[0] == 0:
                continue
            worst = max(worst, float(
                np.abs(imfs.sum(axis=1) - scaled[:, 0]).max()))
        if worst > 1e-4:
            raise ValueError(
                f"IMF file '{imf_path}' is not an additive decomposition of "
                f"the standardized target: max reconstruction error {worst:.3g}"
                f" (expected <1e-4). This is the signature of VMD modes "
                f"occupying the slot the DWT is expected to fill. Regenerate "
                f"with `python scripts/gen_dwt_imfs.py`, copy dwt_imfs.npz "
                f"over this path, or set vmd.allow_non_additive=true to train "
                f"on a non-additive decomposition deliberately."
            )

    # Channel layout: [Patv, IMF_1..IMF_K, Wspd, Wdir, Etmp, Itmp]
    # scaled_* columns: [0=Patv, 1=Wspd, 2=Wdir, 3=Etmp, 4=Itmp]
    # We insert IMFs after Patv (column 0) and before the other 4 features.
    def assemble(scaled: np.ndarray, imfs: np.ndarray) -> np.ndarray:
        patv   = scaled[:, 0:1]          # (N, 1)
        covars = scaled[:, 1:]           # (N, 4)
        return np.concatenate([patv, imfs, covars], axis=1)

    data_train = assemble(scaled_train, imf_train)
    data_valid = assemble(scaled_valid, imf_valid)
    data_test  = assemble(scaled_test,  imf_test)

    return data_train, data_valid, data_test, scaler


# ---------------------------------------------------------------------------
# Helpers: model building
# ---------------------------------------------------------------------------

def _build_model(cfg: RunConfig) -> nn.Module:
    """Instantiate model via MODEL_REGISTRY or a fallback builder.

    Raises
    ------
    UnknownModelError
        If the model_name is not registered.
    """
    # Lazy import to avoid circular deps and optional registry
    try:
        from baselines.registry import MODEL_REGISTRY
        registry = MODEL_REGISTRY
    except ImportError:
        # registry.py not yet implemented (task 5.2); use internal fallback
        registry = {}

    model_name = cfg.model_name

    if model_name not in registry:
        # Attempt built-in fallback for 'proposed' and 'ablation:*' variants
        builtin = _builtin_model_factory(cfg)
        if builtin is None:
            raise UnknownModelError(
                f"Model '{model_name}' is not in MODEL_REGISTRY and has no "
                f"built-in fallback.  Available: {list(registry.keys())}"
            )
        return builtin

    factory = registry[model_name]
    # Registry factories expect a plain dict (with 'model'/'dataset'/'ablation'
    # sub-dicts), but the runner works with a RunConfig dataclass.  Convert.
    return factory(_cfg_to_dict(cfg))


def _cfg_to_dict(cfg: RunConfig) -> dict:
    """Convert a RunConfig dataclass into the plain dict layout that the
    baselines registry factories expect.

    The registry factories read lookback/horizon from the ``dataset`` sub-dict,
    so we inject the top-level values there to guarantee they are present.
    """
    import copy
    dataset = copy.deepcopy(cfg.dataset)
    dataset.setdefault("lookback", cfg.lookback)
    dataset.setdefault("horizon", cfg.horizon)
    # Top-level lookback/horizon always win (they are the authoritative source).
    dataset["lookback"] = cfg.lookback
    dataset["horizon"] = cfg.horizon
    return {
        "run_id": cfg.run_id,
        "model_name": cfg.model_name,
        "seed": cfg.seed,
        "lookback": cfg.lookback,
        "horizon": cfg.horizon,
        "train": cfg.train,
        "model": cfg.model,
        "ablation": cfg.ablation,
        "runtime": cfg.runtime,
        "dataset": dataset,
    }


def _builtin_model_factory(cfg: RunConfig) -> Optional[nn.Module]:
    """Built-in factory for 'proposed' and 'ablation:*' models.

    Returns None if the model_name is not handled here.
    """
    model_name = cfg.model_name
    ds_cfg = cfg.dataset
    vmd_cfg = ds_cfg.get("vmd", {})
    vmd_enabled = vmd_cfg.get("enabled", False)
    K = int(vmd_cfg.get("K", 5)) if vmd_enabled else 0

    # Ablation switches.
    # NOTE: the default fusion for the full Proposed_Model is 'gated'
    # (adaptive soft-gating, SCGF-style). The earlier fixed 'cross_attention'
    # fusion is now only reachable as the 'ablation:fusion_cross_attention'
    # variant, since it was shown to be higher-variance and less accurate on
    # this low-channel-count wind dataset.
    ablation = cfg.ablation or {}
    use_itransformer = ablation.get("use_itransformer", True)
    use_lstm         = ablation.get("use_lstm", True)
    fusion_type      = ablation.get("fusion_type", "gated")
    head_type        = ablation.get("head_type", "kan")

    model_cfg = cfg.model or {}
    dim_embed    = int(model_cfg.get("dim_embed", 128))
    depth_itrans = int(model_cfg.get("depth_itrans", 4))
    heads_itrans = int(model_cfg.get("heads_itrans", 6))
    dim_lstm     = int(model_cfg.get("dim_lstm", 128))
    depth_lstm   = int(model_cfg.get("depth_lstm", 3))

    # Handle 'proposed' and all 'ablation:*' variants that use UnifiedProposedModel
    is_proposed = model_name == "proposed"
    is_ablation = model_name.startswith("ablation:")

    if is_proposed or is_ablation:
        # Apply ablation-specific overrides from the model_name suffix
        if is_ablation:
            variant = model_name.split(":", 1)[1]
            if variant == "itrans_off":
                use_itransformer = False
            elif variant == "lstm_off":
                use_lstm = False
            elif variant == "fusion_concat":
                fusion_type = "concat"
            elif variant == "fusion_sum":
                fusion_type = "sum"
            elif variant == "fusion_gated":
                fusion_type = "gated"
            elif variant == "fusion_cross_attention":
                fusion_type = "cross_attention"
            elif variant == "head_linear":
                head_type = "linear"
            elif variant == "head_mlp":
                head_type = "mlp"
            elif variant == "vmd_off":
                # VMD channels already excluded from data (vmd_enabled=False)
                pass
            elif variant == "outlier_off":
                # Outlier filtering handled during preprocessing; no model change
                pass

        from models.unified_proposed import UnifiedProposedModel

        n_target_channels   = 1 + K   # Patv + K IMFs (0 if vmd_off)
        n_covariate_channels = 4       # Wspd, Wdir, Etmp, Itmp

        return UnifiedProposedModel(
            lookback=cfg.lookback,
            horizon=cfg.horizon,
            n_target_channels=n_target_channels,
            n_covariate_channels=n_covariate_channels,
            use_itransformer=use_itransformer,
            use_lstm=use_lstm,
            fusion_type=fusion_type,
            head_type=head_type,
            dim_embed=dim_embed,
            depth_itrans=depth_itrans,
            heads_itrans=heads_itrans,
            dim_lstm=dim_lstm,
            depth_lstm=depth_lstm,
        )

    if model_name == "lstm":
        # Senior LSTM baseline (5 features, no IMFs)
        from models.LSTM import LSTM
        hidden_size  = int(model_cfg.get("hidden_size", 128))
        num_layers   = int(model_cfg.get("num_layers", 3))
        input_size   = 5  # baselines always use 5 raw features
        output_size  = cfg.horizon
        return LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
        )

    return None


# ---------------------------------------------------------------------------
# Helpers: loss criterion
# ---------------------------------------------------------------------------

def _build_criterion(train_cfg: dict) -> nn.Module:
    loss_name = train_cfg.get("loss", "mae").lower()
    if loss_name in ("mae", "l1"):
        return nn.L1Loss()
    if loss_name in ("mse", "l2"):
        return nn.MSELoss()
    raise ValueError(f"Unsupported loss '{loss_name}'. Use 'mae' or 'mse'.")


# ---------------------------------------------------------------------------
# Helpers: optimizer
# ---------------------------------------------------------------------------

def _build_optimizer(model: nn.Module, train_cfg: dict) -> torch.optim.Optimizer:
    lr = float(train_cfg.get("learning_rate", train_cfg.get("lr", 1e-4)))
    opt_name = train_cfg.get("optimizer", "adam").lower()
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    if opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    raise ValueError(f"Unsupported optimizer '{opt_name}'.")


# ---------------------------------------------------------------------------
# Helpers: scheduler
# ---------------------------------------------------------------------------

def _build_scheduler(optimizer: torch.optim.Optimizer, train_cfg: dict):
    sched_name = train_cfg.get("scheduler", "reduce_on_plateau").lower()
    if sched_name in ("reduce_on_plateau", "plateau"):
        factor   = float(train_cfg.get("scheduler_factor", 0.5))
        patience = int(train_cfg.get("scheduler_patience", 5))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience
        )
    if sched_name in ("none", ""):
        return None
    # If unrecognised, just use ReduceLROnPlateau as a safe default
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")


# ---------------------------------------------------------------------------
# Helpers: single epoch train / eval
# ---------------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0
    for x, y in loader:
        x = x.float().to(device)
        y = y.float().to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)
    return total_loss / n_samples if n_samples > 0 else float("nan")


def _eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (loss, all_preds, all_actuals) in *normalised* scale."""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_preds: list[np.ndarray] = []
    all_actuals: list[np.ndarray] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device)
            y = y.float().to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item() * x.size(0)
            n_samples += x.size(0)
            all_preds.append(pred.cpu().numpy())
            all_actuals.append(y.cpu().numpy())

    epoch_loss = total_loss / n_samples if n_samples > 0 else float("nan")
    preds_arr   = np.concatenate(all_preds,   axis=0)  # (N, horizon)
    actuals_arr = np.concatenate(all_actuals, axis=0)  # (N, horizon)
    return epoch_loss, preds_arr, actuals_arr


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def _validate_config(cfg: RunConfig) -> None:
    """Raise descriptive errors for invalid RunConfig values.

    Checks:
      1. horizon must be in [1, 24] (Requirement 3.6).
      2. lookback + horizon must not exceed any partition's sample count.
         The partition sizes are derived from the manifest if available,
         otherwise from heuristic defaults.
    """
    # --- horizon range check ------------------------------------------------
    if not (1 <= cfg.horizon <= 24):
        raise InvalidHorizonError(
            f"horizon must be in [1, 24]; got {cfg.horizon}."
        )

    # --- window size vs partition size check --------------------------------
    partition_path = cfg.dataset.get("partition_path")
    if partition_path and Path(partition_path).exists():
        try:
            indices = PartitionManifest.read(partition_path)
            for name, (start, end) in [
                ("train", indices.train),
                ("valid", indices.valid),
                ("test",  indices.test),
            ]:
                n_samples = end - start
                if cfg.lookback + cfg.horizon > n_samples:
                    raise InsufficientWindowError(
                        f"lookback ({cfg.lookback}) + horizon ({cfg.horizon}) = "
                        f"{cfg.lookback + cfg.horizon} exceeds the '{name}' "
                        f"partition size ({n_samples})."
                    )
        except (FileNotFoundError, KeyError):
            # If manifest can't be read, skip the partition-size check silently
            # (it will fail more explicitly when data loading is attempted).
            pass


# ---------------------------------------------------------------------------
# Main entry point: run
# ---------------------------------------------------------------------------

def run(cfg: RunConfig, progress_cb=None) -> RunRecord:
    """Execute a complete train/eval experiment and return a RunRecord.

    Parameters
    ----------
    cfg : RunConfig
        Fully specified run configuration.

    Returns
    -------
    RunRecord
        Populated record including metrics, file paths, env snapshot,
        timing information, and status='success'.

    Raises
    ------
    InvalidHorizonError
        If horizon is not in [1, 24].
    InsufficientWindowError
        If lookback + horizon exceeds any partition's sample count.
    UnknownModelError
        If cfg.model_name is not found in MODEL_REGISTRY or built-in factory.
    """
    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    # ── 0. Validate config ────────────────────────────────────────────────
    _validate_config(cfg)

    # ── 1. Reproducibility: seed ──────────────────────────────────────────
    deterministic = cfg.runtime.get("deterministic", True)
    set_global_seed(cfg.seed, deterministic=deterministic)

    # ── 2. Device selection ───────────────────────────────────────────────
    device = _resolve_device(cfg.runtime)

    # ── 3. Load data ──────────────────────────────────────────────────────
    data_train, data_valid, data_test, scaler = _load_data(cfg)

    # ── 4. Build datasets + loaders ───────────────────────────────────────
    batch_size = int(cfg.train.get("batch_size", 128))

    ds_train = WindowedSeriesDataset(data_train, cfg.lookback, cfg.horizon)
    ds_valid = WindowedSeriesDataset(data_valid, cfg.lookback, cfg.horizon)
    ds_test  = WindowedSeriesDataset(data_test,  cfg.lookback, cfg.horizon)

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  drop_last=False)
    loader_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=False, drop_last=False)
    loader_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, drop_last=False)

    # ── 5. Build model ────────────────────────────────────────────────────
    model = _build_model(cfg).to(device)

    # ── 6. Training setup ─────────────────────────────────────────────────
    criterion = _build_criterion(cfg.train)
    optimizer = _build_optimizer(model, cfg.train)
    scheduler = _build_scheduler(optimizer, cfg.train)

    epochs    = int(cfg.train.get("epochs", 150))
    patience  = int(cfg.train.get("early_stop_patience", cfg.train.get("patience", 10)))
    es_delta  = float(cfg.train.get("early_stop_delta", cfg.train.get("delta", 1e-4)))

    # Checkpoint path
    out_dir    = cfg.runtime.get("out_dir", ".")
    ckpt_dir   = Path("model_save") / "wind"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path  = str(ckpt_dir / f"{cfg.run_id}.pt")

    early_stopping = EarlyStopping(
        save_path=ckpt_path,
        patience=patience,
        verbose=False,   # 静默：不再每 epoch 打印，进度由 run_batch 统一展示
        delta=es_delta,
    )

    train_losses: list[float] = []
    valid_losses: list[float] = []

    # ── 7. Training loop ──────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        tr_loss = _train_epoch(model, loader_train, criterion, optimizer, device)
        va_loss, _, _ = _eval_epoch(model, loader_valid, criterion, device)

        train_losses.append(tr_loss)
        valid_losses.append(va_loss)

        # Scheduler step (ReduceLROnPlateau needs val loss)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(va_loss)
            else:
                scheduler.step()

        # Early stopping check + checkpoint save
        early_stopping(va_loss, model)
        # 进度回调：把当前 epoch / 训练损失 / 验证损失 交给外部展示（如进度条）
        if progress_cb is not None:
            progress_cb(epoch, tr_loss, va_loss, early_stopping.early_stop)
        if early_stopping.early_stop:
            break
    n_epochs_trained = epoch

    # ── 8. Load best checkpoint ───────────────────────────────────────────
    if Path(ckpt_path).exists():
        # EarlyStopping saves the full model via torch.save(model, path,
        # pickle_module=dill), so it must be loaded with torch.load (which
        # understands the zip wrapper), not dill.load directly.
        best_model = torch.load(ckpt_path, map_location=device,
                                pickle_module=dill, weights_only=False)
        best_model = best_model.to(device)
    else:
        # Fallback: use last state (no checkpoint saved means no improvement)
        best_model = model

    # ── 9. Evaluate on test set ───────────────────────────────────────────
    _, preds_norm, actuals_norm = _eval_epoch(
        best_model, loader_test, criterion, device
    )

    # ── 10. Compute validation metrics (normalised → kW) ──────────────────
    _, val_preds_norm, val_actuals_norm = _eval_epoch(
        best_model, loader_valid, criterion, device
    )

    # ── 11. Inverse-transform predictions to kW ───────────────────────────
    # preds_norm / actuals_norm shape: (N_test, horizon)
    preds_kw   = scaler.inverse_transform_target(preds_norm)
    actuals_kw = scaler.inverse_transform_target(actuals_norm)

    val_preds_kw   = scaler.inverse_transform_target(val_preds_norm)
    val_actuals_kw = scaler.inverse_transform_target(val_actuals_norm)

    # ── 12. Compute metrics ───────────────────────────────────────────────
    metrics     = compute_metrics(actuals_kw.flatten(), preds_kw.flatten())
    val_metrics = compute_metrics(val_actuals_kw.flatten(), val_preds_kw.flatten())

    # ── 13. Persist artefacts ─────────────────────────────────────────────
    runs_dir = Path(out_dir) / "outputs" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Predictions .npz
    preds_path = str(runs_dir / f"{cfg.run_id}_preds.npz")
    np.savez(preds_path, predictions=preds_kw, actuals=actuals_kw)

    # Losses .npz
    losses_path = str(runs_dir / f"{cfg.run_id}_losses.npz")
    np.savez(
        losses_path,
        train_losses=np.array(train_losses, dtype=np.float32),
        valid_losses=np.array(valid_losses, dtype=np.float32),
    )

    # Config YAML snapshot (best-effort)
    configs_dir = runs_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    config_yaml_path = str(configs_dir / f"{cfg.run_id}.yaml")
    _save_config_snapshot(cfg, config_yaml_path)

    # ── 14. Capture environment ───────────────────────────────────────────
    execution_location = cfg.runtime.get("execution_location", "local_cpu")
    env_snapshot = capture_environment(execution_location)

    # ── 15. Build and persist RunRecord ───────────────────────────────────
    finished_at = datetime.now(timezone.utc)
    wall_clock  = time.perf_counter() - t0

    record = RunRecord(
        run_id=cfg.run_id,
        model_name=cfg.model_name,
        horizon=cfg.horizon,
        lookback=cfg.lookback,
        seed=cfg.seed,
        metrics=metrics,
        val_metrics=val_metrics,
        train_losses=losses_path,
        checkpoint=ckpt_path,
        config_yaml=config_yaml_path,
        partition_path=cfg.dataset.get("partition_path", ""),
        scaler_path=cfg.dataset.get("scaler_path", ""),
        vmd_params_path=cfg.dataset.get("vmd", {}).get("params_path", None),
        env=env_snapshot,
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        wall_clock_seconds=wall_clock,
        status="success",
        failure_reason=None,
    )

    records_path = str(runs_dir / "run_records.jsonl")
    append_run_record(record, records_path)

    return record


# ---------------------------------------------------------------------------
# Helper: save config snapshot
# ---------------------------------------------------------------------------

def _save_config_snapshot(cfg: RunConfig, path: str) -> None:
    """Serialize the RunConfig to a YAML file.  Best-effort; never raises."""
    try:
        import yaml
        from dataclasses import asdict

        snapshot = {
            "run_id":     cfg.run_id,
            "model_name": cfg.model_name,
            "seed":       cfg.seed,
            "lookback":   cfg.lookback,
            "horizon":    cfg.horizon,
            "train":      cfg.train,
            "model":      cfg.model,
            "ablation":   cfg.ablation,
            "runtime":    cfg.runtime,
            "dataset":    cfg.dataset,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(snapshot, f, allow_unicode=True, default_flow_style=False)
    except Exception:
        # Never let config persistence failure abort a run
        pass

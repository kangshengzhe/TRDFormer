"""
V2 模型批量实验运行器
====================
专门跑 ProposedModelV2（趋势残差分解 + DWT + 深度残差分支）。

与 run_batch.py 的区别：
- 只跑 proposed_v2 一个模型（消融对比沿用 DWT 版 run_batch 的结果）
- 使用 ProposedModelV2 替代 UnifiedProposedModel
- 支持 trend_kernel, use_revin 等 V2 专有参数

用法：
    python scripts/run_batch_v2.py --horizons 1 6 12 24 --seeds 42 43 44 45 46 47 48 49 50 51
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import warnings
import traceback
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from experiments.runner import (
    RunConfig, _load_data, _build_criterion, _build_optimizer,
    _build_scheduler, _train_epoch, _eval_epoch, _validate_config,
    _resolve_device, _save_config_snapshot
)
from experiments.metrics import compute_metrics
from reproducibility.seeds import set_global_seed
from reproducibility.environment import capture_environment
from reproducibility.records import RunRecord, append_run_record
from data_pipeline.windowing import WindowedSeriesDataset
from utils.tools import EarlyStopping

import numpy as np
import torch
from torch.utils.data import DataLoader
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
def _fmt_hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _bar(frac: float, width: int = 28) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "\u2588" * filled + "\u2591" * (width - filled)


# ---------------------------------------------------------------------------
def _build_v2_model(cfg: RunConfig) -> torch.nn.Module:
    """Build ProposedModelV2 from config."""
    from models.proposed_v2 import ProposedModelV2

    ds_cfg = cfg.dataset
    vmd_cfg = ds_cfg.get("vmd", {})
    vmd_enabled = vmd_cfg.get("enabled", True)
    K = int(vmd_cfg.get("K", 5)) if vmd_enabled else 0

    model_cfg = cfg.model or {}
    ablation = cfg.ablation or {}

    return ProposedModelV2(
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        n_target_channels=1 + K,
        n_covariate_channels=4,
        trend_kernel=int(model_cfg.get("trend_kernel", 25)),
        use_revin=bool(model_cfg.get("use_revin", True)),
        use_itransformer=ablation.get("use_itransformer", True),
        use_lstm=ablation.get("use_lstm", True),
        fusion_type=ablation.get("fusion_type", "gated"),
        head_type=ablation.get("head_type", "kan"),
        dim_embed=int(model_cfg.get("dim_embed", 128)),
        depth_itrans=int(model_cfg.get("depth_itrans", 4)),
        heads_itrans=int(model_cfg.get("heads_itrans", 6)),
        dim_lstm=int(model_cfg.get("dim_lstm", 128)),
        depth_lstm=int(model_cfg.get("depth_lstm", 3)),
    )


def _dataset_cfg(horizon: int) -> dict:
    """Dataset config for V2 (always uses DWT/VMD)."""
    return {
        "csv_path": "data/wind/sdwpf_turb1_cleaned_final.csv",
        "features": ["Patv", "Wspd", "Wdir", "Etmp", "Itmp"],
        "scaler_path": "outputs/manifests/scaler.pkl",
        "partition_path": f"outputs/manifests/partition_indices_l144_h{horizon}.json",
        "vmd": {
            "enabled": True, "K": 5,
            "params_path": "outputs/manifests/vmd_params.json",
            "imf_path": "outputs/manifests/vmd_imfs.npz",
        },
        "cleaning": {"physical_rules": True},
    }


def _make_run_id(horizon: int, seed: int) -> str:
    return f"proposed_v2_h{horizon}_seed{seed}"


def _build_config(horizon, seed, epochs, device, out_dir) -> RunConfig:
    return RunConfig(
        run_id=_make_run_id(horizon, seed),
        model_name="proposed_v2",
        seed=seed,
        lookback=144,
        horizon=horizon,
        train={
            "batch_size": 128,
            "learning_rate": 1e-4,
            "optimizer": "adam",
            "epochs": epochs,
            "early_stop_patience": 10,
            "early_stop_delta": 1e-4,
            "scheduler": "reduce_on_plateau",
            "scheduler_factor": 0.5,
            "scheduler_patience": 5,
            "loss": "mae",
        },
        model={
            "dim_embed": 128, "depth_itrans": 4, "heads_itrans": 6,
            "dim_lstm": 128, "depth_lstm": 3,
            "trend_kernel": 25, "use_revin": False,
        },
        ablation={},
        runtime={
            "execution_location": "kaggle_gpu",
            "device": device,
            "deterministic": True,
            "out_dir": out_dir,
        },
        dataset=_dataset_cfg(horizon),
    )


def _completed_run_ids(records_path: Path) -> set:
    done = set()
    if not records_path.exists():
        return done
    with open(records_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("status") == "success":
                    done.add(rec.get("run_id"))
            except json.JSONDecodeError:
                continue
    return done


def _run_single(cfg: RunConfig, progress_cb=None) -> RunRecord:
    """Run a single V2 experiment end-to-end."""
    import dill

    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    set_global_seed(cfg.seed, deterministic=True)
    device = _resolve_device(cfg.runtime)
    out_dir = cfg.runtime.get("out_dir", ".")

    # Load data
    data_train, data_valid, data_test, scaler = _load_data(cfg)

    # Build datasets + loaders
    batch_size = int(cfg.train.get("batch_size", 128))
    ds_train = WindowedSeriesDataset(data_train, cfg.lookback, cfg.horizon)
    ds_valid = WindowedSeriesDataset(data_valid, cfg.lookback, cfg.horizon)
    ds_test = WindowedSeriesDataset(data_test, cfg.lookback, cfg.horizon)

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              drop_last=False, num_workers=0)
    loader_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=False,
                              drop_last=False, num_workers=0)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                             drop_last=False, num_workers=0)

    # Build V2 model
    model = _build_v2_model(cfg)
    model = model.to(device)

    # Training setup
    criterion = _build_criterion(cfg.train)
    optimizer = _build_optimizer(model, cfg.train)
    scheduler = _build_scheduler(optimizer, cfg.train)
    epochs = int(cfg.train.get("epochs", 150))

    # Checkpoint path
    ckpt_dir = Path(out_dir) / "model_save" / "wind"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(ckpt_dir / f"{cfg.run_id}.pt")

    patience = int(cfg.train.get("early_stop_patience", 10))
    delta = float(cfg.train.get("early_stop_delta", 1e-4))
    early_stopping = EarlyStopping(save_path=ckpt_path, patience=patience,
                                   delta=delta)

    train_losses = []
    valid_losses = []

    for epoch in range(1, epochs + 1):
        # Custom train step with gradient clipping for V2 stability
        model.train()
        total_loss = 0.0
        n_samples = 0
        for x, y in loader_train:
            x = x.float().to(device)
            y = y.float().to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            if loss.isnan():
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            n_samples += x.size(0)
        tr_loss = total_loss / n_samples if n_samples > 0 else float("nan")

        va_loss, _, _ = _eval_epoch(model, loader_valid, criterion, device)
        train_losses.append(tr_loss)
        valid_losses.append(va_loss)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(va_loss)
            else:
                scheduler.step()

        early_stopping(va_loss, model)
        if progress_cb is not None:
            progress_cb(epoch, tr_loss, va_loss, early_stopping.early_stop)
        if early_stopping.early_stop:
            break

    # Load best model
    if Path(ckpt_path).exists():
        best_model = torch.load(ckpt_path, map_location=device,
                                pickle_module=dill, weights_only=False)
        best_model = best_model.to(device)
    else:
        best_model = model

    # Evaluate
    _, preds_norm, actuals_norm = _eval_epoch(best_model, loader_test, criterion, device)
    _, val_preds_norm, val_actuals_norm = _eval_epoch(best_model, loader_valid, criterion, device)

    preds_kw = scaler.inverse_transform_target(preds_norm)
    actuals_kw = scaler.inverse_transform_target(actuals_norm)
    val_preds_kw = scaler.inverse_transform_target(val_preds_norm)
    val_actuals_kw = scaler.inverse_transform_target(val_actuals_norm)

    metrics = compute_metrics(actuals_kw.flatten(), preds_kw.flatten())
    val_metrics = compute_metrics(val_actuals_kw.flatten(), val_preds_kw.flatten())

    # Persist
    runs_dir = Path(out_dir) / "outputs" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    preds_path = str(runs_dir / f"{cfg.run_id}_preds.npz")
    np.savez(preds_path, predictions=preds_kw, actuals=actuals_kw)

    losses_path = str(runs_dir / f"{cfg.run_id}_losses.npz")
    np.savez(losses_path,
             train_losses=np.array(train_losses, dtype=np.float32),
             valid_losses=np.array(valid_losses, dtype=np.float32))

    configs_dir = runs_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    config_yaml_path = str(configs_dir / f"{cfg.run_id}.yaml")
    _save_config_snapshot(cfg, config_yaml_path)

    env_snapshot = capture_environment("kaggle_gpu")
    finished_at = datetime.now(timezone.utc)
    wall_clock = time.perf_counter() - t0

    record = RunRecord(
        run_id=cfg.run_id,
        model_name="proposed_v2",
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
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="V2 模型批量实验运行器")
    parser.add_argument("--horizons", nargs="+", type=int,
                        default=[1, 6, 12, 24])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    out_dir = args.out_dir
    records_path = Path(out_dir) / "outputs" / "runs" / "run_records.jsonl"
    done = _completed_run_ids(records_path)

    all_cells = [(h, s) for h in args.horizons for s in args.seeds]
    pending_all = [(h, s) for h, s in all_cells
                   if _make_run_id(h, s) not in done]

    # Sharding
    num_shards = max(1, args.num_shards)
    shard_index = max(0, min(args.shard_index, num_shards - 1))
    if num_shards > 1:
        pending_all = sorted(pending_all, key=lambda c: _make_run_id(*c))
        pending = [c for i, c in enumerate(pending_all)
                   if i % num_shards == shard_index]
    else:
        pending = pending_all

    line = "\u2550" * 70
    print(line)
    print("  V2 模型批量实验运行器 (趋势残差分解 + DWT + 深度残差)")
    print(f"  设备: {device:<8}  epochs: {args.epochs}")
    print(f"  步长: {args.horizons}   种子: {args.seeds}")
    if num_shards > 1:
        print(f"  分片: {shard_index}/{num_shards}")
    print(f"  矩阵总数: {len(all_cells)}   已完成: {len(done)}   "
          f"本次待运行: {len(pending)}")
    print(line)

    if not pending:
        print("  没有待运行的单元，全部已完成。")
        return 0

    t0 = time.perf_counter()
    n_ok = n_fail = 0

    for i, (horizon, seed) in enumerate(pending, 1):
        run_id = _make_run_id(horizon, seed)
        t_run = time.perf_counter()
        best_va = float("inf")

        def _cb(epoch, tr, va, stopped, _rid=run_id, _i=i):
            nonlocal best_va
            best_va = min(best_va, va)
            msg = (f"  [{_i}/{len(pending)}] {_rid:<30} "
                   f"ep {epoch:>3}/{args.epochs}  "
                   f"val={va:.4f} best={best_va:.4f}")
            print("\r" + msg + " " * 4, end="", flush=True)

        try:
            cfg = _build_config(horizon, seed, args.epochs, device, out_dir)
            rec = _run_single(cfg, progress_cb=_cb)
            m = rec.metrics
            dt = time.perf_counter() - t_run
            n_ok += 1
            print("\r" + " " * 90, end="\r")
            print(f"  OK [{i}/{len(pending)}] {run_id:<30} "
                  f"MAE={m['mae']:7.2f}  RMSE={m['rmse']:7.2f}  "
                  f"R2={m['r2']:.4f}  ({_fmt_hms(dt)})")
        except Exception as exc:
            n_fail += 1
            print("\r" + " " * 90, end="\r")
            print(f"  XX [{i}/{len(pending)}] {run_id:<30} FAIL: "
                  f"{type(exc).__name__}: {str(exc)[:80]}")
            traceback.print_exc()
            # Record failure
            runs_dir = Path(out_dir) / "outputs" / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            fail_rec = {
                "run_id": run_id, "model_name": "proposed_v2",
                "horizon": horizon, "seed": seed,
                "status": "failed", "failure_reason": repr(exc),
                "metrics": {}, "val_metrics": {},
            }
            with open(runs_dir / "run_records.jsonl", "a") as fh:
                fh.write(json.dumps(fail_rec, ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - t0
    print(f"\n{line}")
    print(f"  成功 {n_ok}   失败 {n_fail}   用时 {_fmt_hms(elapsed)}")
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

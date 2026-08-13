"""
V2 Multi-turbine generalization runner.
Runs proposed_v2, dlinear, itransformer, patchtst on 10 turbines.
Each turbine independently preprocessed with DWT (no cross-turbine leakage).
"""
from __future__ import annotations
import argparse, json, os, sys, time, warnings, traceback
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from experiments.runner import RunConfig, run
from scripts.run_batch import _default_train, _default_model
from scripts.run_batch_v2 import _run_single, _build_v2_model

TURBINES = [1, 2, 13, 55, 70, 83, 86, 88, 94, 99]
MODELS = ["proposed_v2", "dlinear", "itransformer", "patchtst"]
HORIZONS = [1, 6, 12, 24]
SEEDS = [42, 43, 44]
LOOKBACK = 144
OUT_DIR = "outputs/multiturb_v2"
CSV_TMPL = "data/wind/multiturb/sdwpf_turb{tid}_cleaned_final.csv"
MANIFEST_TMPL = "outputs/multiturb_v2/manifests/turb{tid}/h{h}"


def _run_id(tid, model, h, seed):
    safe = model.replace(":", "_")
    return f"t{tid}_{safe}_h{h}_seed{seed}"


def _completed(records_path):
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


def _dataset_cfg(tid, h, model_name):
    mdir = Path(MANIFEST_TMPL.format(tid=tid, h=h))
    vmd_enabled = model_name in ("proposed_v2", "proposed")
    return {
        "csv_path": CSV_TMPL.format(tid=tid),
        "features": ["Patv", "Wspd", "Wdir", "Etmp", "Itmp"],
        "scaler_path": str(mdir / "scaler.pkl"),
        "partition_path": str(mdir / f"partition_indices_l{LOOKBACK}_h{h}.json"),
        "vmd": {
            "enabled": vmd_enabled, "K": 5,
            "params_path": str(mdir / "vmd_params.json"),
            "imf_path": str(mdir / "vmd_imfs.npz"),
        },
        "cleaning": {"physical_rules": True},
    }


def _build_cfg(tid, model, h, seed, epochs, device):
    return RunConfig(
        run_id=_run_id(tid, model, h, seed),
        model_name=model,
        seed=seed,
        lookback=LOOKBACK,
        horizon=h,
        train=_default_train(epochs, False),
        model=_default_model(False),
        ablation={},
        runtime={
            "execution_location": "kaggle_gpu",
            "device": device,
            "deterministic": True,
            "out_dir": OUT_DIR,
        },
        dataset=_dataset_cfg(tid, h, model),
    )


def _run_v2_single(tid, h, seed, epochs, device):
    """Run proposed_v2 for a specific turbine."""
    import torch
    import numpy as np
    import dill
    from experiments.runner import (
        _load_data, _build_criterion, _build_optimizer,
        _build_scheduler, _eval_epoch, _resolve_device, _save_config_snapshot
    )
    from data_pipeline.windowing import WindowedSeriesDataset
    from torch.utils.data import DataLoader
    from reproducibility.seeds import set_global_seed
    from reproducibility.environment import capture_environment
    from reproducibility.records import RunRecord, append_run_record
    from experiments.metrics import compute_metrics
    from utils.tools import EarlyStopping
    from datetime import datetime, timezone
    from models.proposed_v2 import ProposedModelV2

    rid = _run_id(tid, "proposed_v2", h, seed)
    cfg = _build_cfg(tid, "proposed_v2", h, seed, epochs, device)

    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    set_global_seed(seed, deterministic=True)
    dev = torch.device(device)

    data_train, data_valid, data_test, scaler = _load_data(cfg)
    batch_size = 128
    ds_train = WindowedSeriesDataset(data_train, LOOKBACK, h)
    ds_valid = WindowedSeriesDataset(data_valid, LOOKBACK, h)
    ds_test = WindowedSeriesDataset(data_test, LOOKBACK, h)
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=False, num_workers=0)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ProposedModelV2(
        lookback=LOOKBACK, horizon=h, n_target_channels=6, n_covariate_channels=4,
        trend_kernel=25, use_revin=False, use_itransformer=True, use_lstm=True,
        fusion_type='gated', head_type='kan', dim_embed=128,
        depth_itrans=4, heads_itrans=6, dim_lstm=128, depth_lstm=3,
    ).to(dev)

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    ckpt_dir = Path(OUT_DIR) / "model_save" / "wind"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(ckpt_dir / f"{rid}.pt")
    early_stopping = EarlyStopping(save_path=ckpt_path, patience=10, delta=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for x, y in loader_train:
            x, y = x.float().to(dev), y.float().to(dev)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            if loss.isnan():
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)

        va_loss, _, _ = _eval_epoch(model, loader_valid, criterion, dev)
        scheduler.step(va_loss if not np.isnan(va_loss) else 1e9)
        early_stopping(va_loss, model)
        if early_stopping.early_stop:
            break

    # Load best
    if Path(ckpt_path).exists():
        best_model = torch.load(ckpt_path, map_location=dev, pickle_module=dill, weights_only=False).to(dev)
    else:
        best_model = model

    _, preds_norm, actuals_norm = _eval_epoch(best_model, loader_test, criterion, dev)
    preds_kw = scaler.inverse_transform_target(preds_norm)
    actuals_kw = scaler.inverse_transform_target(actuals_norm)
    metrics = compute_metrics(actuals_kw.flatten(), preds_kw.flatten())

    # Persist
    runs_dir = Path(OUT_DIR) / "outputs" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    wall_clock = time.perf_counter() - t0

    record = RunRecord(
        run_id=rid, model_name="proposed_v2", horizon=h, lookback=LOOKBACK,
        seed=seed, metrics=metrics, val_metrics={},
        train_losses="", checkpoint=ckpt_path, config_yaml="",
        partition_path=cfg.dataset.get("partition_path", ""),
        scaler_path=cfg.dataset.get("scaler_path", ""),
        vmd_params_path=None,
        env=capture_environment("kaggle_gpu"),
        started_at=started_at.isoformat(),
        finished_at=datetime.now(timezone.utc).isoformat(),
        wall_clock_seconds=wall_clock, status="success", failure_reason=None,
    )
    append_run_record(record, str(runs_dir / "run_records.jsonl"))
    return metrics


def main(argv=None):
    p = argparse.ArgumentParser(description="V2 Multi-turbine runner")
    p.add_argument("--turbines", nargs="+", type=int, default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--horizons", nargs="+", type=int, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--epochs", type=int, default=150)
    args = p.parse_args(argv)

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    turbines = args.turbines or TURBINES
    models = args.models or MODELS
    horizons = args.horizons or HORIZONS
    seeds = args.seeds or SEEDS
    epochs = args.epochs

    records_path = Path(OUT_DIR) / "outputs" / "runs" / "run_records.jsonl"
    done = _completed(records_path)

    print("=" * 70)
    print("  V2 Multi-turbine Generalization Runner")
    print(f"  Device: {device}  Epochs: {epochs}")
    print(f"  Turbines: {turbines}  Models: {models}")
    print(f"  Horizons: {horizons}  Seeds: {seeds}")
    print(f"  Already done: {len(done)}")
    print("=" * 70)

    n_ok = n_fail = n_skip = 0
    t0 = time.perf_counter()

    for tid in turbines:
        csv = Path(CSV_TMPL.format(tid=tid))
        if not csv.exists():
            print(f"  [turb{tid}] missing data, skip")
            continue

        for model in models:
            for h in horizons:
                for seed in seeds:
                    rid = _run_id(tid, model, h, seed)
                    if rid in done:
                        n_skip += 1
                        continue

                    t = time.perf_counter()
                    try:
                        if model == "proposed_v2":
                            m = _run_v2_single(tid, h, seed, epochs, device)
                        else:
                            cfg = _build_cfg(tid, model, h, seed, epochs, device)
                            rec = run(cfg)
                            m = rec.metrics
                        n_ok += 1
                        dt = time.perf_counter() - t
                        print(f"  OK {rid:<36} MAE={m['mae']:7.2f} ({dt:.0f}s)", flush=True)
                    except Exception as exc:
                        n_fail += 1
                        print(f"  XX {rid:<36} {type(exc).__name__}: {str(exc)[:60]}", flush=True)
                        runs_dir = Path(OUT_DIR) / "outputs" / "runs"
                        runs_dir.mkdir(parents=True, exist_ok=True)
                        fail_rec = {"run_id": rid, "model_name": model, "turbine": tid,
                                    "horizon": h, "seed": seed, "status": "failed",
                                    "failure_reason": repr(exc), "metrics": {}, "val_metrics": {}}
                        with open(runs_dir / "run_records.jsonl", "a") as fh:
                            fh.write(json.dumps(fail_rec, ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - t0
    print("=" * 70)
    print(f"  Done: OK={n_ok}  FAIL={n_fail}  SKIP={n_skip}  Time={elapsed/60:.1f} min")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

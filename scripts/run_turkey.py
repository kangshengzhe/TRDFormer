"""Retrain the case-study model on the second SCADA record (Turkey 2018).

WHAT QUESTION THIS ANSWERS
--------------------------
On SDWPF the paper shows three things: the offline sub-bands carry look-ahead,
delaying them 12 steps collapses the model onto the no-sub-band baseline, and
the architecture then contributes nothing measurable. The first of those was
replicated on this second record with a model-free probe (Section 5.3), but the
RETRAINING was SDWPF-only -- so a reviewer can fairly ask whether the accuracy
consequence transfers, not just the information-theoretic signature.

This script closes that gap by retraining three variants here:

    offline   [Patv, D1..A4, Wspd, Wdir]   sub-bands as the literature uses them
    lag12     same bands delayed 12 steps  frequency content identical, alignment broken
    no_dwt    [Patv, Wspd, Wdir]           no sub-bands at all

The decisive comparison is lag12 vs no_dwt. If they land on top of each other,
the sub-bands' contribution on this record is also alignment, not frequency
separation -- the same conclusion Table 4 reaches on SDWPF.

WHY IT IS A SEPARATE SCRIPT
---------------------------
experiments/runner.py assumes one contiguous series (it slices train/valid/test
by start:end). This record survives as 22 disjoint segments, so windows must be
segment-confined. Rather than change the runner -- which all 1,108 published
runs depend on -- this script reuses the runner's building blocks
(_build_model/_build_criterion/_build_optimizer/_build_scheduler/_train_epoch/
_eval_epoch) and supplies its own segment-aware datasets. Nothing SDWPF reads
is touched.

Usage
-----
    python scripts/prep_turkey.py                       # once
    python scripts/run_turkey.py --seeds 42 43 44 45 46 47 48 49 50 51
    python scripts/run_turkey.py --seeds 42 --epochs 2  # smoke test

Results append to outputs/runs/turkey_records.jsonl (one JSON per run).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_pipeline.segmented_windowing import SegmentedWindowedDataset
from models.unified_proposed import UnifiedProposedModel

# Only the training/eval loop helpers are reused. The runner's own model
# factory is NOT used: _builtin_model_factory hard-codes
# n_covariate_channels = 4 (SDWPF's Wspd/Wdir/Etmp/Itmp) and derives
# n_target_channels from dataset.vmd.K, neither of which fits a turbine with
# only Wspd and Wdir. Constructing UnifiedProposedModel directly keeps the
# hyper-parameters identical to the paper's while allowing the correct channel
# split -- and leaves runner.py untouched.
from experiments.runner import (
    RunConfig,
    _build_criterion,
    _build_optimizer,
    _build_scheduler,
    _eval_epoch,
    _resolve_device,
    _train_epoch,
)

MAN = Path("outputs/manifests")
RUNS = Path("outputs/runs")
# Each shard writes its OWN file and the summary globs them back together.
# Two processes appending to one file would usually be safe on Linux (a record
# is ~400 B, well under the atomic-append limit), but it is not guaranteed on
# every platform and a torn line would corrupt the whole record set.
JSONL_GLOB = "turkey_records*.jsonl"

# [Patv, (5 bands), Wspd, Wdir] -- this turbine has no temperature channels
VARIANTS = {
    "offline": {"imfs": "turkey_dwt_offline.npz", "n_target": 6},
    "lag12":   {"imfs": "turkey_dwt_lag12.npz",   "n_target": 6},
    "no_dwt":  {"imfs": None,                      "n_target": 1},
}
N_COVARIATE = 2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_matrix(variant: str):
    """Assemble [Patv, bands..., Wspd, Wdir] plus the segment bounds."""
    d = np.load(MAN / "turkey_data.npz")
    scaled = d["scaled"].astype(np.float32)            # [Patv, Wspd, Wdir]
    with open(MAN / "turkey_segments.json", encoding="utf-8") as fh:
        meta = json.load(fh)

    patv, cov = scaled[:, :1], scaled[:, 1:]
    spec = VARIANTS[variant]
    if spec["imfs"] is None:
        mat = np.concatenate([patv, cov], axis=1)
    else:
        bands = np.load(MAN / spec["imfs"])["all_imfs"].astype(np.float32)
        mat = np.concatenate([patv, bands, cov], axis=1)

    assert mat.shape[1] == spec["n_target"] + N_COVARIATE, (
        f"{variant}: assembled {mat.shape[1]} channels, expected "
        f"{spec['n_target'] + N_COVARIATE}")
    bounds = {k: [tuple(b) for b in v] for k, v in meta["bounds"].items()}
    return mat, bounds, meta


def make_cfg(variant: str, seed: int, horizon: int, epochs: int,
             patience: int, device: str) -> RunConfig:
    spec = VARIANTS[variant]
    return RunConfig(
        run_id=f"turkey_{variant}_h{horizon}_s{seed}",
        model_name="proposed",
        seed=seed,
        lookback=144,
        horizon=horizon,
        train={
            "batch_size": 128, "learning_rate": 1e-4, "epochs": epochs,
            "early_stop_patience": patience, "loss": "mae",
            "optimizer": "adam", "scheduler": "reduce_on_plateau",
            "scheduler_factor": 0.5, "scheduler_patience": 5,
            "grad_clip_norm": 1.0,
        },
        model={
            "n_target_channels": spec["n_target"],
            "n_covariate_channels": N_COVARIATE,
            "d_model": 128, "n_layers": 4, "n_heads": 6,
            "lstm_layers": 3, "lstm_hidden": 128,
        },
        ablation={"use_itransformer": True, "use_lstm": True,
                  "fusion_type": "gated", "head_type": "kan"},
        runtime={"device": device, "deterministic": True,
                 "out_dir": "outputs/runs"},
        dataset={},          # this script loads data itself
    )


def run_one(variant: str, seed: int, horizon: int, epochs: int,
            patience: int, device_str: str, out_jsonl: Path) -> dict:
    cfg = make_cfg(variant, seed, horizon, epochs, patience, device_str)
    device = _resolve_device(cfg.runtime)
    set_seed(seed)

    mat, bounds, meta = build_matrix(variant)
    lb, hz = cfg.lookback, cfg.horizon
    ds = {k: SegmentedWindowedDataset(mat, bounds[k], lb, hz)
          for k in ("train", "valid", "test")}
    ld = {
        "train": DataLoader(ds["train"], batch_size=128, shuffle=True),
        "valid": DataLoader(ds["valid"], batch_size=128, shuffle=False),
        "test":  DataLoader(ds["test"],  batch_size=128, shuffle=False),
    }

    m = cfg.model
    model = UnifiedProposedModel(
        lookback=lb,
        horizon=hz,
        n_target_channels=m["n_target_channels"],
        n_covariate_channels=m["n_covariate_channels"],
        use_itransformer=cfg.ablation["use_itransformer"],
        use_lstm=cfg.ablation["use_lstm"],
        fusion_type=cfg.ablation["fusion_type"],
        head_type=cfg.ablation["head_type"],
        dim_embed=m["d_model"],
        depth_itrans=m["n_layers"],
        heads_itrans=m["n_heads"],
        dim_lstm=m["lstm_hidden"],
        depth_lstm=m["lstm_layers"],
    ).to(device)
    criterion = _build_criterion(cfg.train)
    optimizer = _build_optimizer(model, cfg.train)
    scheduler = _build_scheduler(optimizer, cfg.train)

    sigma = float(meta["patv_sigma_kw"])
    best_val, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        _train_epoch(model, ld["train"], criterion, optimizer, device)
        val_loss, _, _ = _eval_epoch(model, ld["valid"], criterion, device)
        if scheduler is not None:
            scheduler.step(val_loss)
        if val_loss < best_val - 1e-6:
            best_val, bad = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    _, preds, actuals = _eval_epoch(model, ld["test"], criterion, device)

    # standardised -> kW. Both arrays are in the same standardised space, so a
    # single multiply by sigma is the correct conversion for an ERROR metric
    # (the mean offset cancels in a difference).
    mae_kw = float(np.abs(preds - actuals).mean()) * sigma
    rmse_kw = float(np.sqrt(((preds - actuals) ** 2).mean())) * sigma

    rec = {
        "run_id": cfg.run_id, "variant": variant, "seed": seed,
        "horizon": horizon, "epochs_ran": ep,
        "n_train_windows": len(ds["train"]),
        "n_test_windows": len(ds["test"]),
        "n_channels": int(mat.shape[1]),
        "test_mae_kw": mae_kw, "test_rmse_kw": rmse_kw,
        "best_valid_loss_std": best_val,
        "patv_sigma_kw": sigma,
        "seconds": round(time.time() - t0, 1),
        "device": str(device),
    }
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"  {variant:8s} seed {seed}  MAE {mae_kw:8.2f} kW  "
          f"({ep} epochs, {rec['seconds']}s)")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split the (variant, seed) task list across N processes")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="which shard this process runs (0-based)")
    ap.add_argument("--summary-only", action="store_true",
                    help="print the merged summary and exit, running nothing")
    a = ap.parse_args()

    if not (MAN / "turkey_data.npz").exists():
        raise SystemExit("run scripts/prep_turkey.py first")
    if not (0 <= a.shard_index < a.num_shards):
        raise SystemExit(
            f"--shard-index must be in [0, {a.num_shards - 1}]")

    # Round-robin over the flat task list rather than splitting the seed list.
    # no_dwt has 3 input channels against 8 for the other two and so trains
    # faster; interleaving keeps the fast and slow tasks spread evenly across
    # shards instead of loading one GPU with all the cheap ones.
    tasks = [(v, s) for v in a.variants for s in a.seeds]
    mine = tasks[a.shard_index::a.num_shards]

    suffix = f"_s{a.shard_index}" if a.num_shards > 1 else ""
    out_jsonl = RUNS / f"turkey_records{suffix}.jsonl"

    if not a.summary_only:
        print("=" * 66)
        print(f"Turkey retraining, h={a.horizon}")
        if a.num_shards > 1:
            print(f"shard {a.shard_index + 1}/{a.num_shards}: "
                  f"{len(mine)} of {len(tasks)} runs -> {out_jsonl.name}")
        else:
            print(f"{len(tasks)} runs -> {out_jsonl.name}")
        print("=" * 66)
        for v, s in mine:
            run_one(v, s, a.horizon, a.epochs, a.patience, a.device, out_jsonl)

    # ---- summary -------------------------------------------------------
    # Records are APPENDED, so a re-run (or an interrupted run resumed) leaves
    # more than one line per (variant, seed, horizon). Counting those as
    # independent samples would inflate n and shrink the standard deviation
    # toward zero, because the duplicates carry no extra variance. Keep only
    # the LAST record for each key, which is also what "re-run to overwrite"
    # should mean.
    files = sorted(RUNS.glob(JSONL_GLOB))
    if not files:
        raise SystemExit(f"no records found under {RUNS}/{JSONL_GLOB}")
    seen: dict[tuple, dict] = {}
    dupes = 0
    for f in files:
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["variant"], r["seed"], r["horizon"])
            if key in seen:
                dupes += 1
            seen[key] = r
    print(f"\nmerged {len(files)} record file(s): "
          f"{', '.join(f.name for f in files)}")
    if dupes:
        print(f"({dupes} duplicate record(s) collapsed; keeping the most "
              f"recent per variant/seed/horizon)")
    recs = [r for r in seen.values() if r["horizon"] == a.horizon]
    print()
    print("=" * 66)
    print(f"{'variant':<10}{'n':>3}{'MAE kW (mean+-std)':>26}")
    print("-" * 66)
    means = {}
    for v in a.variants:
        vals = np.array([r["test_mae_kw"] for r in recs if r["variant"] == v])
        if not len(vals):
            continue
        means[v] = vals.mean()
        sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
        print(f"{v:<10}{len(vals):>3}{vals.mean():>16.2f} +-{sd:>7.2f}")
    if "lag12" in means and "no_dwt" in means:
        gap = means["lag12"] - means["no_dwt"]
        print()
        print(f"DECISIVE: lag12 - no_dwt = {gap:+.2f} kW")
        print("  (near zero => the sub-bands' value was alignment, not frequency"
              " content -- same as SDWPF Table 4)")
    if "offline" in means and "no_dwt" in means:
        d = 100 * (means["no_dwt"] / means["offline"] - 1)
        print(f"Offline appears {d:+.1f}% better than no sub-bands "
              "(the inflated figure)")


if __name__ == "__main__":
    main()

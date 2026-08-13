"""
多风机泛化验证编排器（独立于 run_batch，不影响主实验）。

对 10 台代表风机(含 turb1)跑:
    models  = [proposed, dlinear, itransformer, patchtst]
    horizons= [1, 6, 12, 24]
    seeds   = [42, 43, 44]
每台每步长独立预处理(scaler/VMD 各自 fit),输出隔离到 outputs/multiturb/。

分片策略：按 **风机** 取模分配到多进程/多卡，保证每台的预处理与训练在同一进程内
完成，避免多进程同时写同一 manifest。断点续跑：读 outputs/multiturb 的 run_records.jsonl
中 status=success 的 run_id 跳过。

用法
----
本地冒烟(CPU、微型、1 台 1 模型 1 步长 1 种子 2 epoch):
    python -m scripts.run_multiturb --smoke

服务器单进程完整:
    python -m scripts.run_multiturb

服务器双卡(在 run_multiturb_dual_gpu.sh 里调用):
    CUDA_VISIBLE_DEVICES=0 python -m scripts.run_multiturb --num-shards 2 --shard-index 0
    CUDA_VISIBLE_DEVICES=1 python -m scripts.run_multiturb --num-shards 2 --shard-index 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from experiments.runner import RunConfig, run
from scripts.preprocess_cli import run_pipeline
from scripts.run_batch import _default_train, _default_model  # 复用主实验超参模板

# ── 多风机验证矩阵 ──────────────────────────────────────────────────────────
TURBINES = [1, 2, 13, 55, 70, 83, 86, 88, 94, 99]
MODELS = ["proposed", "dlinear", "itransformer", "patchtst"]
HORIZONS = [1, 6, 12, 24]
SEEDS = [42, 43, 44]

LOOKBACK = 144
VMD_K = 5
OUT_DIR = "outputs/multiturb"
CSV_TMPL = "data/wind/multiturb/sdwpf_turb{tid}_cleaned_final.csv"
MANIFEST_TMPL = "outputs/multiturb/manifests/turb{tid}/h{h}"


def _run_id(tid, model, h, seed):
    safe = model.replace(":", "_")
    return f"t{tid}_{safe}_h{h}_seed{seed}"


def _completed(records_path: Path):
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


def _preprocess_if_needed(tid, h, smoke):
    mdir = Path(MANIFEST_TMPL.format(tid=tid, h=h))
    partition = mdir / f"partition_indices_l{LOOKBACK}_h{h}.json"
    imf = mdir / "vmd_imfs.npz"
    scaler = mdir / "scaler.pkl"
    if partition.exists() and imf.exists() and scaler.exists():
        return mdir  # 已预处理，跳过
    mdir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "csv_path": CSV_TMPL.format(tid=tid),
        "manifest_dir": str(mdir),
        "lookback": LOOKBACK,
        "horizon": h,
        "vmd_k": VMD_K,
        "vmd_off": False,
        "cleaning": True,
        "vmd_alpha": 2000.0, "vmd_tau": 0.0, "vmd_DC": 0, "vmd_init": 1, "vmd_tol": 1e-7,
    }
    run_pipeline(cfg)
    return mdir


def _dataset_cfg(tid, h, mdir, model_name):
    vmd_enabled = (model_name == "proposed")
    return {
        "csv_path": CSV_TMPL.format(tid=tid),
        "features": ["Patv", "Wspd", "Wdir", "Etmp", "Itmp"],
        "scaler_path": str(mdir / "scaler.pkl"),
        "partition_path": str(mdir / f"partition_indices_l{LOOKBACK}_h{h}.json"),
        "vmd": {
            "enabled": vmd_enabled, "K": VMD_K,
            "params_path": str(mdir / "vmd_params.json"),
            "imf_path": str(mdir / "vmd_imfs.npz"),
        },
        "cleaning": {"physical_rules": True},
    }


def _build_cfg(tid, model, h, seed, mdir, epochs, smoke, device):
    return RunConfig(
        run_id=_run_id(tid, model, h, seed),
        model_name=model,
        seed=seed,
        lookback=LOOKBACK,
        horizon=h,
        train=_default_train(epochs, smoke),
        model=_default_model(smoke),
        ablation={},
        runtime={
            "execution_location": "kaggle_gpu",
            "device": device,
            "deterministic": True,
            "out_dir": OUT_DIR,
        },
        dataset=_dataset_cfg(tid, h, mdir, model),
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="多风机泛化验证编排器")
    p.add_argument("--turbines", nargs="+", type=int, default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--horizons", nargs="+", type=int, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    args = p.parse_args(argv)

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    smoke = args.smoke
    epochs = args.epochs if args.epochs is not None else (2 if smoke else 150)

    turbines = args.turbines or ([TURBINES[0]] if smoke else TURBINES)
    models = args.models or (["proposed"] if smoke else MODELS)
    horizons = args.horizons or ([6] if smoke else HORIZONS)
    seeds = args.seeds or ([42] if smoke else SEEDS)

    # 按风机分片
    ns = max(1, args.num_shards)
    si = max(0, min(args.shard_index, ns - 1))
    my_turbines = [t for i, t in enumerate(turbines) if i % ns == si]

    records_path = Path(OUT_DIR) / "outputs" / "runs" / "run_records.jsonl"
    done = _completed(records_path)

    line = "=" * 70
    print(line)
    print("  多风机泛化验证编排器")
    print(f"  设备:{device}  模式:{'冒烟' if smoke else '完整'}  epochs:{epochs}")
    print(f"  本进程风机:{my_turbines}  模型:{models}  步长:{horizons}  种子:{seeds}")
    if ns > 1:
        print(f"  分片:{si}/{ns}")
    print(f"  已完成(success):{len(done)}")
    print(line)

    n_ok = n_fail = n_skip = 0
    t0 = time.perf_counter()

    for tid in my_turbines:
        csv = Path(CSV_TMPL.format(tid=tid))
        if not csv.exists():
            print(f"  [turb{tid}] 缺少数据文件 {csv}，跳过")
            continue
        # 预处理该台所有步长
        mdirs = {}
        for h in horizons:
            print(f"  [turb{tid}] 预处理 h{h} …", flush=True)
            mdirs[h] = _preprocess_if_needed(tid, h, smoke)

        for model in models:
            for h in horizons:
                for seed in seeds:
                    rid = _run_id(tid, model, h, seed)
                    if rid in done:
                        n_skip += 1
                        continue
                    t = time.perf_counter()
                    try:
                        cfg = _build_cfg(tid, model, h, seed, mdirs[h], epochs, smoke, device)
                        rec = run(cfg)
                        m = rec.metrics
                        n_ok += 1
                        dt = time.perf_counter() - t
                        print(f"  OK {rid:<30} MAE={m['mae']:7.2f} RMSE={m['rmse']:7.2f} "
                              f"R2={m['r2']:.4f} ({dt:.0f}s)", flush=True)
                    except Exception as exc:
                        n_fail += 1
                        print(f"  XX {rid:<30} FAIL: {type(exc).__name__}: {str(exc)[:70]}", flush=True)
                        _append_failure(records_path, rid, tid, model, h, seed, repr(exc))

    print(line)
    print(f"  完成 OK={n_ok}  FAIL={n_fail}  SKIP={n_skip}  用时 {(time.perf_counter()-t0)/60:.1f} min")
    print(line)
    return 0


def _append_failure(records_path, rid, tid, model, h, seed, reason):
    records_path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"run_id": rid, "model_name": model, "turbine": tid,
           "horizon": h, "seed": seed, "status": "failed",
           "failure_reason": reason, "metrics": {}, "val_metrics": {}}
    with open(records_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())

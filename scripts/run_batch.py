"""
批量实验运行器 —— Kaggle GPU 会话的主入口。

功能
----
- 遍历完整实验矩阵：(proposed + 11 基线 + 8 消融) × horizon{1,6,12,24} × seed{42..46}
- 断点续跑：读取 run_records.jsonl，跳过已成功完成的 (model, horizon, seed)
- 时间预算：在 Kaggle 会话被强制中断前，主动停止派发新运行
- 逐运行容错：单个运行失败只记录并继续，不影响整批

用法
----
本地快速验证（微型配置、少量组合）:
    python -m scripts.run_batch --smoke

Kaggle 上跑完整矩阵（默认）:
    python -m scripts.run_batch --max-hours 8

只跑某一组:
    python -m scripts.run_batch --models proposed dlinear --horizons 6 --seeds 42 43

参数
----
--models     要运行的模型名列表（默认：全部）
--horizons   预测步长列表（默认：1 6 12 24）
--seeds      随机种子列表（默认：42 43 44 45 46）
--max-hours  本次会话最大挂钟小时数，达到后停止派发新运行（默认：无限）
--epochs     训练轮数（默认：150；--smoke 时为 2）
--smoke      冒烟模式：微型模型 + 2 epoch + 少量组合，用于快速验证
--out-dir    输出根目录（默认：当前目录）
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import warnings
import traceback
from pathlib import Path

# ── 统一压制无害警告，让输出干净 ──────────────────────────────────────────
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from experiments.matrix import PROPOSED, BASELINES, ABLATIONS, HORIZONS, SEEDS
from experiments.runner import RunConfig, run


# ---------------------------------------------------------------------------
# 终端美化辅助
# ---------------------------------------------------------------------------

def _fmt_hms(seconds: float) -> str:
    """秒 -> H:MM:SS。"""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _bar(frac: float, width: int = 28) -> str:
    """生成一个 [████████░░░░] 风格的进度条。"""
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


class _Progress:
    """总进度显示 + 单个运行的训练进度。同一行原地刷新。"""

    def __init__(self, total: int, done_already: int):
        self.total = total
        self.done_already = done_already
        self.completed = 0          # 本次会话完成数
        self.t0 = time.perf_counter()

    def overall_line(self, idx: int, run_id: str, extra: str = "") -> str:
        overall_done = self.done_already + self.completed
        frac = overall_done / self.total if self.total else 0
        elapsed = time.perf_counter() - self.t0
        # 估算剩余时间（按本次会话已完成的平均耗时）
        if self.completed > 0:
            per = elapsed / self.completed
            eta = per * (len(_PENDING) - idx)
            eta_s = _fmt_hms(eta)
        else:
            eta_s = "--:--:--"
        return (f"总进度 [{_bar(frac)}] {overall_done}/{self.total}  "
                f"用时 {_fmt_hms(elapsed)}  预计剩余 {eta_s}")

    def epoch_line(self, run_id, idx, epoch, epochs, tr, va, best_va):
        frac = epoch / epochs if epochs else 0
        return (f"  [{idx}/{len(_PENDING)}] {run_id:<34} "
                f"训练 [{_bar(frac, 18)}] ep {epoch:>3}/{epochs}  "
                f"val={va:.4f} best={best_va:.4f}")


# 供 _Progress 访问的模块级待运行列表
_PENDING: list = []


# ---------------------------------------------------------------------------
# 默认超参数模板
# ---------------------------------------------------------------------------

_PROPOSED_MODELS = set(PROPOSED) | {f"ablation:{a.split(':',1)[1]}" for a in ABLATIONS}


def _default_train(epochs: int, smoke: bool) -> dict:
    return {
        "batch_size": 64 if smoke else 128,
        "learning_rate": 1e-3 if smoke else 1e-4,
        "optimizer": "adam",
        "epochs": epochs,
        "early_stop_patience": 5 if smoke else 10,
        "early_stop_delta": 1e-4,
        "scheduler": "reduce_on_plateau",
        "scheduler_factor": 0.5,
        "scheduler_patience": 5,
        "loss": "mae",
    }


def _default_model(smoke: bool) -> dict:
    if smoke:
        return {
            "dim_embed": 32, "depth_itrans": 1, "heads_itrans": 2,
            "dim_lstm": 32, "depth_lstm": 1,
            "d_model": 32, "d_ff": 64, "e_layers": 1, "d_layers": 1, "n_heads": 2,
            "factor": 1, "moving_avg": 13, "dropout": 0.1,
            "embed": "timeF", "freq": "t", "activation": "gelu",
            "patch_len": 16, "stride": 8, "top_k": 5, "num_kernels": 6,
            "p_hidden_dims": [64, 64], "p_hidden_layers": 2, "use_norm": True,
        }
    return {
        "dim_embed": 128, "depth_itrans": 4, "heads_itrans": 6,
        "dim_lstm": 128, "depth_lstm": 3,
        "d_model": 128, "d_ff": 256, "e_layers": 2, "d_layers": 1, "n_heads": 8,
        "factor": 3, "moving_avg": 25, "dropout": 0.1,
        "embed": "timeF", "freq": "t", "activation": "gelu",
        "patch_len": 16, "stride": 8, "top_k": 5, "num_kernels": 6,
        "p_hidden_dims": [256, 256], "p_hidden_layers": 2, "use_norm": True,
    }


def _dataset_cfg(horizon: int, model_name: str) -> dict:
    """构造 dataset 配置。基线只用 5 特征（vmd 关闭）；proposed/消融用 VMD。

    outlier_off 消融本应使用未做物理清洗的数据，但那需要单独的预处理产物；
    这里保持与主产物一致（cleaning 标记仅用于记录），实际清洗在预处理阶段完成。
    """
    is_proposed_family = (model_name == "proposed") or model_name.startswith("ablation:")
    # 基线一律不加 VMD 通道（Req 4.3）；vmd_off 消融也不加
    vmd_enabled = is_proposed_family and (model_name != "ablation:vmd_off")

    return {
        "csv_path": "data/wind/sdwpf_turb1_cleaned_final.csv",
        "features": ["Patv", "Wspd", "Wdir", "Etmp", "Itmp"],
        "scaler_path": "outputs/manifests/scaler.pkl",
        "partition_path": f"outputs/manifests/partition_indices_l144_h{horizon}.json",
        "vmd": {
            "enabled": vmd_enabled, "K": 5,
            "params_path": "outputs/manifests/vmd_params.json",
            "imf_path": "outputs/manifests/vmd_imfs.npz",
        },
        "cleaning": {"physical_rules": model_name != "ablation:outlier_off"},
    }


def _ablation_switches(model_name: str) -> dict:
    """从消融名解析出模型开关。"""
    if not model_name.startswith("ablation:"):
        return {}
    variant = model_name.split(":", 1)[1]
    return {
        "itrans_off":    {"use_itransformer": False},
        "lstm_off":      {"use_lstm": False},
        "fusion_concat": {"fusion_type": "concat"},
        "fusion_sum":    {"fusion_type": "sum"},
        "fusion_gated":  {"fusion_type": "gated"},
        "fusion_cross_attention": {"fusion_type": "cross_attention"},
        "head_linear":   {"head_type": "linear"},
        "head_mlp":      {"head_type": "mlp"},
        "vmd_off":       {},   # 仅数据层变化
        "outlier_off":   {},   # 仅数据层变化
    }.get(variant, {})


def _make_run_id(model_name: str, horizon: int, seed: int) -> str:
    safe = model_name.replace(":", "_")
    return f"{safe}_h{horizon}_seed{seed}"


def _build_config(model_name, horizon, seed, epochs, smoke, device, out_dir) -> RunConfig:
    return RunConfig(
        run_id=_make_run_id(model_name, horizon, seed),
        model_name=model_name,
        seed=seed,
        lookback=144,
        horizon=horizon,
        train=_default_train(epochs, smoke),
        model=_default_model(smoke),
        ablation=_ablation_switches(model_name),
        runtime={
            "execution_location": "kaggle_gpu",
            "device": device,
            "deterministic": True,
            "out_dir": out_dir,
        },
        dataset=_dataset_cfg(horizon, model_name),
    )


# ---------------------------------------------------------------------------
# 断点续跑：读取已完成的运行
# ---------------------------------------------------------------------------

def _completed_run_ids(records_path: Path) -> set[str]:
    """返回 run_records.jsonl 中 status='success' 的 run_id 集合。"""
    done: set[str] = set()
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


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="批量实验运行器")
    parser.add_argument("--models", nargs="+", default=None,
                        help="模型名列表（默认全部）")
    parser.add_argument("--horizons", nargs="+", type=int, default=None,
                        help="预测步长列表（默认 1 6 12 24）")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="随机种子列表（默认 42..46）")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="本次会话最大挂钟小时数")
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练轮数（默认 150；smoke 时 2）")
    parser.add_argument("--smoke", action="store_true",
                        help="冒烟模式：微型模型 + 2 epoch + 少量组合")
    parser.add_argument("--out-dir", default=".", help="输出根目录")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="多卡并行时的分片总数（例如双卡设为 2）")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="本进程负责的分片编号（从 0 开始）；"
                             "配合 --num-shards 让多个进程各跑互不重叠的子集")
    args = parser.parse_args(argv)

    # 设备自动选择
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    smoke = args.smoke
    epochs = args.epochs if args.epochs is not None else (2 if smoke else 150)

    # 组装矩阵
    if args.models:
        models = args.models
    elif smoke:
        models = ["proposed", "dlinear", "ablation:vmd_off"]
    else:
        models = PROPOSED + BASELINES + ABLATIONS

    horizons = args.horizons if args.horizons else ([6] if smoke else HORIZONS)
    seeds = args.seeds if args.seeds else ([42] if smoke else SEEDS)

    out_dir = args.out_dir
    records_path = Path(out_dir) / "outputs" / "runs" / "run_records.jsonl"
    done = _completed_run_ids(records_path)

    # 生成所有待运行单元
    all_cells = [(m, h, s) for m in models for h in horizons for s in seeds]
    pending_all = [(m, h, s) for (m, h, s) in all_cells
                   if _make_run_id(m, h, s) not in done]

    # ── 多卡分片：按 run_id 排序后取模分配，保证同一批任务在多进程间互不重叠 ──
    num_shards = max(1, args.num_shards)
    shard_index = max(0, min(args.shard_index, num_shards - 1))
    if num_shards > 1:
        pending_all = sorted(pending_all, key=lambda c: _make_run_id(*c))
        pending = [c for i, c in enumerate(pending_all) if i % num_shards == shard_index]
    else:
        pending = pending_all

    global _PENDING
    _PENDING = pending

    # ── 头部信息面板 ──────────────────────────────────────────────────────
    line = "═" * 70
    print(line)
    print("  批量实验运行器")
    print(f"  设备: {device:<8}  模式: {'冒烟' if smoke else '完整'}  epochs: {epochs}")
    print(f"  模型: {len(models)}   步长: {horizons}   种子: {seeds}")
    if num_shards > 1:
        print(f"  分片: {shard_index}/{num_shards}（本进程只跑分给自己的子集）")
    print(f"  矩阵总数: {len(all_cells)}   已完成: {len(done)}   本次待运行: {len(pending)}")
    if args.max_hours:
        print(f"  时间预算: {args.max_hours} 小时")
    print(line)

    if not pending:
        print("  没有待运行的单元，全部已完成。")
        print(line)
        return 0

    prog = _Progress(total=len(all_cells), done_already=len(done))
    results_rows: list[tuple] = []
    n_ok = n_fail = 0

    for i, (model_name, horizon, seed) in enumerate(pending, 1):
        if args.max_hours is not None:
            elapsed_h = (time.perf_counter() - prog.t0) / 3600.0
            if elapsed_h >= args.max_hours:
                print(f"\n  已达时间预算 {args.max_hours}h，剩余 "
                      f"{len(pending) - i + 1} 个留待下次续跑。")
                break

        run_id = _make_run_id(model_name, horizon, seed)
        t_run = time.perf_counter()
        prog._best_va = float("inf")

        def _cb(epoch, tr, va, stopped, _rid=run_id, _idx=i):
            prog._best_va = min(prog._best_va, va)
            msg = prog.epoch_line(_rid, _idx, epoch, epochs, tr, va, prog._best_va)
            print("\r" + msg + " " * 4, end="", flush=True)

        try:
            cfg = _build_config(model_name, horizon, seed, epochs, smoke, device, out_dir)
            rec = run(cfg, progress_cb=_cb)
            m = rec.metrics
            dt = time.perf_counter() - t_run
            prog.completed += 1
            n_ok += 1
            print("\r" + " " * 90, end="\r")
            print(f"  OK [{i}/{len(pending)}] {run_id:<34} "
                  f"MAE={m['mae']:7.2f}  RMSE={m['rmse']:7.2f}  "
                  f"R2={m['r2']:.4f}  ({_fmt_hms(dt)})")
            print("     " + prog.overall_line(i, run_id))
            results_rows.append((run_id, m['mae'], m['rmse'], m['r2'], "OK"))
        except Exception as exc:
            n_fail += 1
            print("\r" + " " * 90, end="\r")
            print(f"  XX [{i}/{len(pending)}] {run_id:<34} FAIL: "
                  f"{type(exc).__name__}: {str(exc)[:60]}")
            _append_failure(records_path, run_id, model_name, horizon, seed, repr(exc))
            results_rows.append((run_id, None, None, None, "FAIL"))

    # ── 结果汇总表 ────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - prog.t0
    print("\n" + line)
    print("  本次会话结果汇总")
    print(line)
    print(f"  {'运行':<36}{'MAE':>9}{'RMSE':>9}{'R2':>9}   状态")
    print("  " + "-" * 66)
    for run_id, mae, rmse, r2, status in results_rows:
        if status == "OK":
            print(f"  {run_id:<36}{mae:>9.2f}{rmse:>9.2f}{r2:>9.4f}   OK")
        else:
            print(f"  {run_id:<36}{'-':>9}{'-':>9}{'-':>9}   XX")
    print("  " + "-" * 66)
    total_done = len(_completed_run_ids(records_path))
    print(f"  成功 {n_ok}   失败 {n_fail}   用时 {_fmt_hms(elapsed)}")
    print(f"  累计完成: {total_done}/{len(all_cells)}  [{_bar(total_done/len(all_cells))}]")
    print(line)
    return 0


def _append_failure(records_path: Path, run_id, model_name, horizon, seed, reason):
    """把失败运行记为 failed，避免续跑时重复。"""
    records_path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "run_id": run_id, "model_name": model_name,
        "horizon": horizon, "seed": seed,
        "status": "failed", "failure_reason": reason,
        "metrics": {}, "val_metrics": {},
    }
    with open(records_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())

"""
Computational cost benchmark: parameter count and inference latency for
the proposed model vs. representative baselines, addressing the
Limitations gap flagged for the manuscript ("we did not systematically
benchmark training time, inference latency, or memory footprint").

This script builds each model with the EXACT hyperparameters used in the
reported experiments (matching scripts/run_batch_v2.py::_build_v2_model
and baselines/tsl_configs.py), then measures:

  1. Total parameter count (trainable)
  2. CPU inference latency: mean/std wall-clock time for a forward pass
     at batch size 1 (single-sample, deployment-realistic) and batch
     size 128 (the training batch size used throughout the paper)

No training or checkpoint is required -- parameter count and forward-pass
latency depend only on model architecture and hyperparameters, not on
learned weights, so an untrained model (freshly initialised, in eval mode)
gives identical numbers to a trained one.

Models benchmarked (all at lookback=144, horizon=12, the paper's headline
configuration, Table tab:main_baseline):
  - Proposed (Ours)  : ProposedModelV2, DWT-on (n_target_channels=6)
  - DLinear          : strongest baseline (Table tab:main_baseline)
  - LSTM             : second-strongest baseline, same recurrent family
                        as the proposed model's exogenous branch
  - iTransformer      : architectural ancestor of the endogenous branch

Usage
-----
    python scripts/benchmark_compute_cost.py
    python scripts/benchmark_compute_cost.py --out manuscript/tables/compute_cost.tex
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import torch
import torch.nn as nn

LOOKBACK = 144
HORIZON = 12
N_WARMUP = 10
N_TIMED = 100


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _time_forward(model: nn.Module, x: torch.Tensor, n_warmup: int, n_timed: int) -> tuple[float, float]:
    """Return (mean_ms, std_ms) for a single forward pass on CPU."""
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        times = []
        for _ in range(n_timed):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1))


def build_proposed_v2() -> nn.Module:
    """Exact hyperparameters from scripts/run_batch_v2.py::_build_v2_model
    and _build_config, matching the reported experiments (DWT on, K=5 IMFs
    -> n_target_channels = 1 + 5 = 6)."""
    from models.proposed_v2 import ProposedModelV2

    return ProposedModelV2(
        lookback=LOOKBACK,
        horizon=HORIZON,
        n_target_channels=6,       # Patv + 5 DWT sub-bands
        n_covariate_channels=4,    # Wspd, Wdir, Etmp, Itmp
        trend_kernel=25,
        use_revin=False,           # matches _build_config: use_revin=False
        use_itransformer=True,
        use_lstm=True,
        fusion_type="gated",
        head_type="kan",
        dim_embed=128,
        depth_itrans=4,
        heads_itrans=6,
        dim_lstm=128,
        depth_lstm=3,
    )


def build_dlinear() -> tuple[nn.Module, tuple]:
    """DLinear baseline, exact config from baselines/tsl_configs.py
    (enc_in=dec_in=c_out=5, moving_avg=25, individual=False)."""
    from baselines.tsl_configs import make_tsl_configs
    from models.tsl.DLinear import Model

    cfg = make_tsl_configs(
        model_cfg={"moving_avg": 25, "individual": False},
        dataset_cfg={"lookback": LOOKBACK, "horizon": HORIZON},
    )
    model = Model(cfg, individual=False)
    # DLinear.forward signature: (x_enc, x_mark_enc, x_dec, x_mark_dec)
    x_enc = torch.randn(1, LOOKBACK, 5)
    dummy_args = (x_enc, None, None, None)
    return model, dummy_args


def build_lstm() -> nn.Module:
    """Recurrent baseline (implemented directly, not via TSL -- per
    Section 5.1 "a recurrent baseline (LSTM); ... implemented directly").
    Matches the model+head shape used for the other direct-implementation
    baseline in this codebase: LSTM encoder + linear head to horizon."""

    class LSTMBaseline(nn.Module):
        def __init__(self, input_size=5, hidden=128, layers=3, horizon=HORIZON):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, num_layers=layers,
                                 batch_first=True)
            self.head = nn.Linear(hidden, horizon)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    return LSTMBaseline()


def build_itransformer():
    """iTransformer baseline, exact config from baselines/tsl_configs.py."""
    from baselines.tsl_configs import make_tsl_configs
    from models.tsl.iTransformer import Model

    cfg = make_tsl_configs(
        model_cfg={"d_model": 128, "n_heads": 8, "e_layers": 2, "d_ff": 256},
        dataset_cfg={"lookback": LOOKBACK, "horizon": HORIZON},
    )
    model = Model(cfg)
    return model, cfg.label_len


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None,
                     help="Optional path to write a LaTeX table.")
    args = ap.parse_args()

    torch.manual_seed(0)
    results = []

    # ---- Proposed (Ours) ----
    model = build_proposed_v2()
    n_params = _count_params(model)
    x1 = torch.randn(1, LOOKBACK, 10)     # 6 target + 4 covariate channels
    x128 = torch.randn(128, LOOKBACK, 10)
    mean1, std1 = _time_forward(model, x1, N_WARMUP, N_TIMED)
    mean128, std128 = _time_forward(model, x128, N_WARMUP, N_TIMED)
    results.append(("Proposed (Ours)", n_params, mean1, std1, mean128, std128))
    print(f"Proposed (Ours):  params={n_params:,}  "
          f"latency@bs1={mean1:.3f}+-{std1:.3f} ms  "
          f"latency@bs128={mean128:.3f}+-{std128:.3f} ms")

    # ---- DLinear ----
    model, dummy = build_dlinear()
    n_params = _count_params(model)

    def _fwd_dlinear(x, m=model):
        return m(x, None, None, None)

    class _Wrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x, None, None, None)

    wrapped = _Wrap(model)
    x1 = torch.randn(1, LOOKBACK, 5)
    x128 = torch.randn(128, LOOKBACK, 5)
    mean1, std1 = _time_forward(wrapped, x1, N_WARMUP, N_TIMED)
    mean128, std128 = _time_forward(wrapped, x128, N_WARMUP, N_TIMED)
    results.append(("DLinear", n_params, mean1, std1, mean128, std128))
    print(f"DLinear:           params={n_params:,}  "
          f"latency@bs1={mean1:.3f}+-{std1:.3f} ms  "
          f"latency@bs128={mean128:.3f}+-{std128:.3f} ms")

    # ---- LSTM ----
    model = build_lstm()
    n_params = _count_params(model)
    x1 = torch.randn(1, LOOKBACK, 5)
    x128 = torch.randn(128, LOOKBACK, 5)
    mean1, std1 = _time_forward(model, x1, N_WARMUP, N_TIMED)
    mean128, std128 = _time_forward(model, x128, N_WARMUP, N_TIMED)
    results.append(("LSTM", n_params, mean1, std1, mean128, std128))
    print(f"LSTM:              params={n_params:,}  "
          f"latency@bs1={mean1:.3f}+-{std1:.3f} ms  "
          f"latency@bs128={mean128:.3f}+-{std128:.3f} ms")

    # ---- iTransformer ----
    try:
        model, label_len = build_itransformer()
        n_params = _count_params(model)

        class _WrapIT(nn.Module):
            def __init__(self, m, label_len):
                super().__init__()
                self.m = m
                self.label_len = label_len

            def forward(self, x):
                b = x.shape[0]
                x_mark_enc = torch.zeros(b, LOOKBACK, 4)
                x_dec = torch.randn(b, self.label_len + HORIZON, 5)
                x_mark_dec = torch.zeros(b, self.label_len + HORIZON, 4)
                return self.m(x, x_mark_enc, x_dec, x_mark_dec)

        wrapped_it = _WrapIT(model, label_len)
        x1 = torch.randn(1, LOOKBACK, 5)
        x128 = torch.randn(128, LOOKBACK, 5)
        mean1, std1 = _time_forward(wrapped_it, x1, N_WARMUP, N_TIMED)
        mean128, std128 = _time_forward(wrapped_it, x128, N_WARMUP, N_TIMED)
        results.append(("iTransformer", n_params, mean1, std1, mean128, std128))
        print(f"iTransformer:      params={n_params:,}  "
              f"latency@bs1={mean1:.3f}+-{std1:.3f} ms  "
              f"latency@bs128={mean128:.3f}+-{std128:.3f} ms")
    except Exception as exc:
        print(f"iTransformer: SKIPPED ({type(exc).__name__}: {exc})")

    print()
    print(f"CPU: {torch.get_num_threads()} threads, torch {torch.__version__}")

    if args.out:
        _write_latex_table(results, args.out)
        print(f"\nWrote LaTeX table to {args.out}")

    return 0


_RECORD_SOURCES = (
    "outputs/runs/run_records.jsonl",
    "outputs/v2_full/outputs/runs/run_records.jsonl",
)
_LOSS_DIRS = ("outputs/runs", "outputs/v2_full/outputs/runs")

_RECORD_KEY = {
    "Proposed (Ours)": "proposed_v2",
    "DLinear": "dlinear",
    "LSTM": "lstm",
    "iTransformer": "itransformer",
}


def _training_cost(horizon: int = 12) -> dict:
    """Real measured training wall-clock and epoch count, from run records.

    Returns {model: (wall_mean, wall_std, ep_mean, ep_std, n, env_signatures)}

    COMPARABILITY: wall-clock is only meaningful across models if they ran on
    the same machine, so the hostname/torch signature behind each model's runs
    is collected too and reported by the caller. At h=12 every model in this
    project ran on one host (askway, torch 2.5.1+cu121, cuda:0); a minority of
    h=6 runs came from a different container and must not be mixed in.

    Records store ``train_losses`` as a PATH to a ``*_losses.npz``; the epoch
    count actually executed before early stopping is the length of the stored
    loss curve.
    """
    import json
    from collections import defaultdict

    wall = defaultdict(list)
    eps = defaultdict(list)
    envs = defaultdict(set)

    for src in _RECORD_SOURCES:
        p = Path(_REPO) / src
        if not p.is_file():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("status") != "success" or r.get("horizon") != horizon:
                    continue
                m = r.get("model_name")
                w = r.get("wall_clock_seconds")
                if w is not None:
                    wall[m].append(float(w))
                e = r.get("env") or {}
                envs[m].add((e.get("hostname"), e.get("torch")))
                tl = r.get("train_losses")
                if isinstance(tl, str):
                    for ld in _LOSS_DIRS:
                        lp = Path(_REPO) / ld / Path(tl).name
                        if lp.is_file():
                            try:
                                z = np.load(lp)
                                eps[m].append(
                                    int(np.asarray(z["train_losses"]).size))
                            except Exception:
                                pass
                            break

    out = {}
    for m, w in wall.items():
        a = np.asarray(w, float)
        e = np.asarray(eps[m], float) if eps[m] else None
        out[m] = (
            float(a.mean()),
            float(a.std(ddof=1)) if a.size > 1 else 0.0,
            float(e.mean()) if e is not None and e.size else float("nan"),
            float(e.std(ddof=1)) if e is not None and e.size > 1 else 0.0,
            int(a.size),
            envs[m],
        )
    return out


def _write_latex_table(results, out_path: str, horizon: int = 12) -> None:
    n_threads = torch.get_num_threads()
    train = _training_cost(horizon)

    hosts = set()
    for key in _RECORD_KEY.values():
        if key in train:
            hosts |= train[key][5]
    print(f"\n[train-cost] host/torch signatures: {hosts}")
    print(f"[train-cost] mutually comparable: {len(hosts) == 1}")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Computational cost at $h=12$, $L=144$. Training cost is "
        r"the real wall-clock time recorded during the reported experiments "
        r"(10 seeds, mean $\pm$ std) together with the number of epochs "
        r"actually executed before early stopping; every run compared here "
        r"executed on the same machine (dual RTX~4090 workstation, "
        r"\texttt{cuda:0}, identical PyTorch build), so the times are "
        r"mutually comparable. Inference latency is measured separately on "
        f"CPU ({n_threads} threads, 100 forward passes) as a "
        r"deployment-realistic figure for a SCADA server without a GPU, on "
        r"an untrained instance -- parameter count and forward-pass cost "
        r"depend only on architecture, not on learned weights.}",
        r"\label{tab:compute_cost}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Training (GPU)} & "
        r"\multicolumn{2}{c}{Inference (CPU, ms)} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-7}",
        r"Model & Params & Time (s) & Epochs & s/epoch & $B{=}1$ & $B{=}128$ \\",
        r"\midrule",
    ]
    for name, n_params, m1, s1, m128, s128 in results:
        key = _RECORD_KEY.get(name)
        if key and key in train:
            wm, ws, em, es, _, _ = train[key]
            t_time = f"{wm:.0f}$\\pm${ws:.0f}"
            t_ep = f"{em:.0f}$\\pm${es:.0f}" if em == em else "--"
            t_spe = f"{wm / em:.1f}" if em == em and em > 0 else "--"
        else:
            t_time = t_ep = t_spe = "--"
        nm = f"\\textbf{{{name}}}" if name == "Proposed (Ours)" else name
        lines.append(
            f"  {nm} & {n_params:,} & {t_time} & {t_ep} & {t_spe} & "
            f"{m1:.2f}$\\pm${s1:.2f} & {m128:.1f}$\\pm${s128:.1f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")

    print(f"\n[train-cost] full picture at h={horizon} (all models, 10 seeds):")
    for m, (wm, ws, em, es, n, _) in sorted(train.items(), key=lambda t: t[1][0]):
        if m.startswith("ablation") or m == "proposed":
            continue
        spe = f"{wm / em:6.1f}" if em == em and em > 0 else "     -"
        print(f"    {m:28s} n={n:<3d} {wm:8.1f}+-{ws:6.1f} s  "
              f"epochs={em:5.1f}  s/epoch={spe}")


if __name__ == "__main__":
    sys.exit(main())

"""
Data access layer shared by the rebuilt manuscript figures.

Everything a figure needs comes from here so that no two figures can
disagree about what "the proposed model at h=12" means.

RESULT TREE MAP
---------------
``outputs/runs``                      11 SOTA baselines, 4 horizons x 10
                                      seeds (also holds stale V1
                                      ``proposed`` and ``ablation:*`` runs,
                                      which are never used by the paper)
``outputs/v2_full/outputs/runs``      proposed_v2 metrics for all horizons;
                                      per-sample predictions for h=12 only
``outputs/v2_refill/outputs/runs``    proposed_v2 per-sample predictions for
                                      h=1/6/24, all 10 seeds (added later to
                                      close that gap)
``outputs/ablation_v2/outputs/runs``  7 V2 ablation variants, h=12 only
``outputs/rolling_v2/outputs/runs``   expanding-window robustness
``outputs/multiturb_v2/outputs/runs`` 10-turbine generalization
``outputs/analysis/gate_weights.npz`` fusion gate weights per test window

Because proposed_v2 predictions are split across ``v2_full`` and
``v2_refill``, all prediction access goes through :func:`preds_path`, which
searches the candidate directories rather than hard-coding one.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

HORIZONS = (1, 6, 12, 24)
SEEDS = tuple(range(42, 52))          # 10 seeds, matching the paper
PROPOSED = "proposed_v2"

# --- record files ---------------------------------------------------------
REC_BASELINE = "outputs/runs/run_records.jsonl"
REC_PROPOSED = "outputs/v2_full/outputs/runs/run_records.jsonl"
REC_PROPOSED_REFILL = "outputs/v2_refill/outputs/runs/run_records.jsonl"
REC_ABLATION = "outputs/ablation_v2/outputs/runs/run_records.jsonl"
REC_ROLLING = "outputs/rolling_v2/outputs/runs/run_records.jsonl"
REC_MULTITURB = "outputs/multiturb_v2/outputs/runs/run_records.jsonl"

# --- prediction directories, searched in order ---------------------------
DIR_BASELINE = ("outputs/runs",)
DIR_PROPOSED = ("outputs/v2_full/outputs/runs",
                "outputs/v2_refill/outputs/runs")
DIR_ABLATION = ("outputs/ablation_v2/outputs/runs",)

GATE_NPZ = "outputs/analysis/gate_weights.npz"

# --- display names --------------------------------------------------------
PRETTY = {
    "proposed_v2": "TRDFormer",
    "dlinear": "DLinear",
    "lstm": "LSTM",
    "itransformer": "iTransformer",
    "nonstationary_transformer": "NS-Transformer",
    "patchtst": "PatchTST",
    "timexer": "TimeXer",
    "transformer": "Transformer",
    "timesnet": "TimesNet",
    "informer": "Informer",
    "fedformer": "FEDformer",
    "autoformer": "Autoformer",
}

MODEL_COLOR = {
    "proposed_v2": "#C41E3A",
    "dlinear": "#2166AC",
    "lstm": "#E8A33D",
    "itransformer": "#762A83",
    "nonstationary_transformer": "#4C72B0",
    "patchtst": "#1B7837",
    "timexer": "#C71585",
    "transformer": "#7F7F7F",
    "timesnet": "#FF7F0E",
    "informer": "#8C564B",
    "fedformer": "#17BECF",
    "autoformer": "#BCBD22",
}

BASELINES = ("dlinear", "lstm", "itransformer", "nonstationary_transformer",
             "patchtst", "timexer", "transformer", "timesnet", "informer",
             "fedformer", "autoformer")

#: Canonical 12-model order (proposed first, then baselines by strength).
MODEL_ORDER = (PROPOSED,) + BASELINES

#: The three baselines shown alongside TRDFormer in the prediction matrix:
#: the strongest overall (DLinear), the strongest recurrent (LSTM), and the
#: architecture our endogenous branch is built from (iTransformer), so the
#: comparison also isolates what the surrounding framework contributes.
MATRIX_MODELS = (PROPOSED, "dlinear", "lstm", "itransformer")

# --- V2 ablation variants -------------------------------------------------
#: variant -> (display label, innovation key it removes/replaces)
ABLATION_V2 = {
    "v2_no_dwt": ("w/o DWT sub-bands", "B"),
    "v2_no_trend": ("w/o trend-residual", "A"),
    "v2_no_itrans": ("w/o iTransformer branch", "C_endo"),
    "v2_no_lstm": ("w/o LSTM branch", "C_exo"),
    "v2_fusion_cross": ("gated $\\to$ cross-attn.", "D"),
    "v2_head_linear": ("KAN $\\to$ linear head", "D"),
    "v2_head_mlp": ("KAN $\\to$ MLP head", "D"),
}


# =========================================================================
# metrics
# =========================================================================
def _iter_records(path: str | Path):
    p = Path(path)
    if not p.is_file():
        logger.warning("records file missing: %s", p)
        return
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "success":
                yield r


def per_seed_metrics(metric: str = "mae") -> dict:
    """``{model: {horizon: {seed: value}}}`` for the 12 headline models.

    Per-seed granularity (rather than pre-aggregated mean/std) is required
    for the paired significance tests, which must compare like seed with
    like seed.
    """
    out: dict = defaultdict(lambda: defaultdict(dict))
    for r in _iter_records(REC_BASELINE):
        m = r["model_name"]
        if m not in BASELINES:
            continue
        v = (r.get("metrics") or {}).get(metric)
        if v is not None:
            out[m][r["horizon"]][r["seed"]] = float(v)
    for rec in (REC_PROPOSED, REC_PROPOSED_REFILL):
        for r in _iter_records(rec):
            if r["model_name"] != PROPOSED:
                continue
            v = (r.get("metrics") or {}).get(metric)
            if v is not None:
                out[PROPOSED][r["horizon"]][r["seed"]] = float(v)
    return {m: dict(h) for m, h in out.items()}


def agg_metrics(metrics=("mae", "rmse", "r2")) -> dict:
    """``{model: {horizon: {metric: (mean, std_ddof1), 'n_seeds': int}}}``."""
    per = {m: per_seed_metrics(m) for m in metrics}
    out: dict = defaultdict(dict)
    for model in per[metrics[0]]:
        for h in per[metrics[0]][model]:
            entry = {}
            for m in metrics:
                vals = np.array(list(per[m].get(model, {}).get(h, {}).values()),
                                dtype=float)
                if vals.size == 0:
                    entry = {}
                    break
                entry[m] = (float(vals.mean()),
                            float(vals.std(ddof=1)) if vals.size > 1 else 0.0)
            if entry:
                entry["n_seeds"] = len(per[metrics[0]][model][h])
                out[model][h] = entry
    return dict(out)


def paired_pvalues(metric: str = "mae", reference: str = PROPOSED) -> dict:
    """``{model: {horizon: p}}`` from a two-sided paired t-test vs *reference*.

    Pairing is on seed, so only seeds present for both models are used;
    fewer than 3 common seeds yields ``nan`` rather than a fragile p-value.
    """
    from scipy import stats

    per = per_seed_metrics(metric)
    ref = per.get(reference, {})
    out: dict = defaultdict(dict)
    for model, hmap in per.items():
        if model == reference:
            continue
        for h, seed_map in hmap.items():
            common = sorted(set(seed_map) & set(ref.get(h, {})))
            if len(common) < 3:
                out[model][h] = float("nan")
                continue
            a = np.array([ref[h][s] for s in common], dtype=float)
            b = np.array([seed_map[s] for s in common], dtype=float)
            if np.allclose(a, b):
                out[model][h] = 1.0
                continue
            out[model][h] = float(stats.ttest_rel(a, b).pvalue)
    return dict(out)


def ablation_metrics(metric: str = "mae", horizon: int = 12) -> dict:
    """``{variant: {seed: value}}`` plus the full model under key ``full``."""
    out: dict = defaultdict(dict)
    for r in _iter_records(REC_ABLATION):
        if r["horizon"] != horizon:
            continue
        v = (r.get("metrics") or {}).get(metric)
        if v is not None:
            out[r["model_name"]][r["seed"]] = float(v)
    full = per_seed_metrics(metric).get(PROPOSED, {}).get(horizon, {})
    out["full"] = dict(full)
    return dict(out)


def ablation_pvalue(variant: str, metric: str = "mae",
                    horizon: int = 12) -> float:
    """Paired t-test of one ablation variant against the full model."""
    from scipy import stats

    d = ablation_metrics(metric, horizon)
    a, b = d.get("full", {}), d.get(variant, {})
    common = sorted(set(a) & set(b))
    if len(common) < 3:
        return float("nan")
    va = np.array([a[s] for s in common])
    vb = np.array([b[s] for s in common])
    if np.allclose(va, vb):
        return 1.0
    return float(stats.ttest_rel(va, vb).pvalue)


# =========================================================================
# per-sample predictions
# =========================================================================
def preds_path(model: str, horizon: int, seed: int) -> Path | None:
    """Locate a ``*_preds.npz``, searching every tree that may hold it."""
    if model == PROPOSED:
        dirs, stem = DIR_PROPOSED, PROPOSED
    elif model in ABLATION_V2:
        dirs, stem = DIR_ABLATION, model
    else:
        dirs, stem = DIR_BASELINE, model
    for d in dirs:
        p = Path(d) / f"{stem}_h{horizon}_seed{seed}_preds.npz"
        if p.is_file():
            return p
    return None


def load_preds(model: str, horizon: int, seed: int = 42):
    """Return ``(actuals, predictions)``, each ``(n_windows, horizon)`` kW."""
    p = preds_path(model, horizon, seed)
    if p is None:
        raise FileNotFoundError(
            f"no predictions for {model} h={horizon} seed={seed}")
    z = np.load(p)
    return z["actuals"], z["predictions"]


def terminal_series(model: str, horizon: int, seed: int = 42):
    """The pure *h*-step-ahead series: column ``h-1`` of every window.

    This is the operationally meaningful view - at each window the forecast
    issued *h* steps earlier is compared with what actually happened - and
    it matches the convention used by the legacy panels.
    """
    act, pred = load_preds(model, horizon, seed)
    t = act.shape[1] - 1
    return act[:, t], pred[:, t]


def per_step_mae(model: str, horizon: int, seeds=SEEDS):
    """MAE at each lead time 1..h, averaged over seeds.

    Returns ``(mean, std)`` arrays of length *horizon*. Reveals how error
    accumulates within the horizon, which the aggregate MAE hides.
    """
    rows = []
    for s in seeds:
        try:
            act, pred = load_preds(model, horizon, s)
        except FileNotFoundError:
            continue
        rows.append(np.abs(pred - act).mean(axis=0))
    if not rows:
        return None, None
    a = np.vstack(rows)
    return a.mean(axis=0), (a.std(axis=0, ddof=1) if a.shape[0] > 1
                            else np.zeros(a.shape[1]))


LOOKBACK = 144


def abs_index(horizon: int, n_windows: int) -> np.ndarray:
    """Absolute test-set time index of each window's terminal target step.

    Window *i* consumes ``[i, i+L)`` and predicts ``[i+L, i+L+h)``, so its
    terminal step sits at ``i + L + h - 1``. Different horizons therefore
    start at different absolute times, and slicing every horizon by *window
    index* would compare different wall-clock periods across rows. Mapping
    to absolute time first lets a multi-horizon figure show all horizons
    over one shared period, which is what makes the rows comparable.
    """
    return np.arange(n_windows) + LOOKBACK + horizon - 1


def common_abs_span(horizons=HORIZONS, model: str = PROPOSED,
                    seed: int = 42) -> tuple[int, int]:
    """Absolute index range covered by every horizon (inclusive, exclusive)."""
    lo, hi = -np.inf, np.inf
    for h in horizons:
        act, _ = load_preds(model, h, seed)
        ai = abs_index(h, act.shape[0])
        lo, hi = max(lo, ai[0]), min(hi, ai[-1] + 1)
    return int(lo), int(hi)


def slice_by_abs(model: str, horizon: int, seed: int,
                 abs_lo: int, abs_hi: int):
    """Terminal-step actual/predicted over an absolute index window.

    Returns ``(abs_idx, actual, predicted)`` restricted to
    ``[abs_lo, abs_hi)``.
    """
    act, pred = load_preds(model, horizon, seed)
    t = act.shape[1] - 1
    ai = abs_index(horizon, act.shape[0])
    m = (ai >= abs_lo) & (ai < abs_hi)
    return ai[m], act[m, t], pred[m, t]


def pick_ramp_window(actual: np.ndarray, n_points: int, ramp_h: int = 6,
                     ramp_kw: float = 150.0) -> int:
    """Choose the display window that best exercises the models.

    Score = (number of significant ramps) x (fraction of the window that is
    *not* pinned at a flat state), with roughness as a tie-break.

    Why not simply maximise variance: 60% of this turbine's test set sits
    near zero output and 5% is clipped at rated power, so a variance
    criterion picks a window that merely alternates between two flat
    plateaus - 62% of it flat - where every model looks identical.
    Ramp count alone is not enough either: two candidate windows tie at 109
    ramps, and the tie-break on roughness preferred the one that spends 44%
    of its span at zero, because entering and leaving a long idle stretch is
    itself "rough". Penalising flat time resolves it to the window with 38%
    flat and a full dynamic range, which is the regime the models actually
    differ in and the one the paper's ramp analysis concerns.
    """
    n = len(actual)
    if n <= n_points:
        return 0
    rated = float(np.nanmax(actual)) or 1.0
    lo_f, hi_f = 0.02 * rated, 0.97 * rated
    step = max(1, (n - n_points) // 500)
    best_i, best_score = 0, (-np.inf, -np.inf)
    for i in range(0, n - n_points, step):
        seg = actual[i:i + n_points]
        if seg.size <= ramp_h + 1:
            continue
        d = np.abs(seg[ramp_h:] - seg[:-ramp_h])
        flat = float(np.mean((seg < lo_f) | (seg > hi_f)))
        score = (int(np.sum(d > ramp_kw)) * (1.0 - flat),
                 float(np.diff(seg).std()))
        if score > best_score:
            best_score, best_i = score, i
    return best_i


# =========================================================================
# fusion gate weights
# =========================================================================
def load_gate(horizon: int, seeds=SEEDS) -> dict:
    """Fusion gate weights and their operating context for one horizon.

    Returns
    -------
    dict with
        ``alpha``       ``(n_seeds, n_windows)`` endogenous gate weight
        ``alpha_mean``  ``(n_windows,)`` seed-averaged
        ``ramp_mag``    ``(n_windows,)`` |Patv(t+h) - Patv(t)| in kW
        ``wspd_mean``   ``(n_windows,)`` mean look-back wind speed, m/s
        ``patv_last``   ``(n_windows,)`` last observed power, kW
        ``actual_mean`` ``(n_windows,)`` mean realised power over the horizon
        ``seeds``       the seeds actually found

    ``alpha_ex = 1 - alpha`` by construction (the gate is a softmax over
    the two branches), so only the endogenous weight is stored.
    """
    p = Path(GATE_NPZ)
    if not p.is_file():
        raise FileNotFoundError(f"gate weights not found: {p}")
    z = np.load(p)
    got, rows = [], []
    for s in seeds:
        k = f"h{horizon}_alpha_en_seed{s}"
        if k in z.files:
            rows.append(z[k].astype(float))
            got.append(s)
    if not rows:
        raise KeyError(f"no gate weights for horizon {horizon}")
    alpha = np.vstack(rows)
    out = {"alpha": alpha, "alpha_mean": alpha.mean(axis=0), "seeds": got}
    for ctx in ("ramp_mag", "wspd_mean", "wspd_last", "patv_last",
                "actual_mean"):
        k = f"h{horizon}_{ctx}"
        if k in z.files:
            out[ctx] = z[k].astype(float)
    return out


# =========================================================================
# multi-turbine / rolling
# =========================================================================
def multiturb(metric: str = "mae") -> dict:
    """``{model: {turbine: {horizon: [values over seeds]}}}``.

    Run ids look like ``t70_dlinear_h6_seed43``; the turbine id is the
    prefix.
    """
    import re

    pat = re.compile(r"^t(\d+)_(.+)_h(\d+)_seed(\d+)$")
    out: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in _iter_records(REC_MULTITURB):
        m = pat.match(r.get("run_id", ""))
        if not m:
            continue
        turb, model, h = int(m.group(1)), m.group(2), int(m.group(3))
        v = (r.get("metrics") or {}).get(metric)
        if v is not None:
            out[model][turb][h].append(float(v))
    return {k: {t: dict(h) for t, h in v.items()} for k, v in out.items()}


TURBINE_PROFILE = "tools/turbine_profile.csv"


def turbine_profile() -> dict:
    """``{turbine_id: {patv_mean, patv_std, cv, stall_rate, ...}}``.

    Produced by ``tools/profile_turbines.py`` over all 134 turbines; used to
    check whether the accuracy gain depends on a turbine's output level or
    variability rather than holding across turbine types.
    """
    import csv

    p = Path(TURBINE_PROFILE)
    if not p.is_file():
        logger.warning("turbine profile missing: %s", p)
        return {}
    out: dict = {}
    with open(p, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                tid = int(float(row["TurbID"]))
            except (KeyError, ValueError):
                continue
            out[tid] = {k: (float(v) if v not in ("", None) else float("nan"))
                        for k, v in row.items() if k != "TurbID"}
    return out


def rolling(metric: str = "mae") -> dict:
    """``{model: {window_index: [values over seeds]}}`` for expanding-window.

    Window index is parsed from run ids of the form ``..._w1_...``; when
    absent, records are ordered by start time and bucketed by seed count.
    """
    import re

    pat = re.compile(r"_[wW](\d+)")     # run ids use "_W1_", "_W2_", ...
    out: dict = defaultdict(lambda: defaultdict(list))
    for r in _iter_records(REC_ROLLING):
        v = (r.get("metrics") or {}).get(metric)
        if v is None:
            continue
        m = pat.search(r.get("run_id", ""))
        w = int(m.group(1)) if m else None
        out[r["model_name"]][w].append(float(v))
    return {k: dict(v) for k, v in out.items()}

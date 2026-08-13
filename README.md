# TRDFormer

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21911236.svg)](https://doi.org/10.5281/zenodo.21911236)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code and result records for:

> **TRDFormer: Trend-residual decomposition with wavelet sub-band attention for
> short-term wind power forecasting**
> Shengzhe Kang, Xiping Zhu, Tao Yang
> School of Electrical and Information Engineering, Southwest Petroleum University
> *Under review, Electric Power Systems Research.*

The forecaster splits the task in two. A linear branch takes the strongly
auto-correlated trend — the component a single linear layer already predicts
almost perfectly, and the reason plain deep models often fail to beat DLinear
on wind power. A deep branch then models only the residual, which is handed to
it already separated into five db4 wavelet sub-bands, each entering a
variate-attention encoder as its own token. Meteorological covariates are
encoded separately by an LSTM, and a learned per-sample gate decides how much
of each branch to use.

## Headline results

SDWPF, turbine 1, 10-minute resolution, 10 seeds per cell. Horizons are
$h = 1, 6, 12, 24$ steps, i.e. 10 min to 4 h ahead.

| | TRDFormer | DLinear (strongest baseline) | |
|---|---|---|---|
| MAE, $h=1$ | **26.18 ± 1.20** kW | 30.93 ± 0.03 kW | −15.3 % |
| MAE, $h=12$ | **58.18 ± 1.13** kW | 70.37 ± 0.11 kW | −17.3 % |
| MAE, $h=24$ | 104.67 ± 10.61 kW | **100.21 ± 0.30** kW | +4.4 % |
| MAE, top decile of power ramps ($h=12$) | **157.04 ± 5.03** kW | 278.14 ± 0.06 kW | **−43.5 %** |

Ablation at $h=12$, quoted as the MAE increase when a component is removed:

| Component removed | MAE inflation |
|---|---|
| Partition-isolated DWT sub-bands | **+40.8 %** |
| Trend–residual decomposition | +31.0 % |
| Endogenous (iTransformer) branch | +28.3 % |
| Adaptive gate → fixed cross-attention | +9.4 % |
| Exogenous (LSTM) branch | +1.9 % |

Three results the paper reports because they are unfavourable, and which this
repository lets you check:

- **DLinear wins at $h=24$** under the canonical split. The $h=24$ ranking is
  split-dependent, not a stable property of the horizon.
- **The KAN head earns nothing.** It is within seed noise of a linear head
  despite 9.9× the parameters, and a two-layer MLP head is both smaller and
  more accurate. It is retained and reported, not quietly swapped out.
- **The DWT is not sample-wise causal.** Sub-bands are fitted independently
  *within* each partition, so no test value is informed by another partition —
  that is the leakage guarantee that matters for a fair comparison. But inside
  a partition `pywt.wavedec`/`waverec` reconstructs from the whole partition at
  once; a perturbation test shifts the reconstruction at a reference index by
  up to 2.3 standardised units. This suits the offline evaluation performed
  here, not a strictly online deployment.

## What is and is not in this repository

Tracked: all model and pipeline code, all experiment configurations, and the
**aggregate run records** (`run_records.jsonl`, per-epoch losses, fusion-gate
weights, scalers, partition manifests) for all 1,048 training runs.

Not tracked, and why:

| Excluded | Size | Why / how to get it |
|---|---|---|
| `data/` | 17 MB | SDWPF is public but redistribution terms are the publisher's. See below. |
| `outputs/**/*_preds.npz` | 511 MB | 917 per-sample prediction arrays. Regenerable by re-running, or available on request. |
| `manuscript/` | — | The article is under review; this is a code and results deposit. |

Consequently, from a fresh clone:

| Target | Reproducible as cloned? |
|---|---|
| Tables 1–3, Appendix Tables A.1–A.2 | **yes** |
| Fig. 4 (baseline comparison), Fig. 6 (ablation + gate), Fig. 8 (generalisation) | **yes** |
| Fig. 1, Fig. 3, graphical abstract | after downloading SDWPF (below) |
| Fig. 5 (prediction matrix), Fig. 7 (ramp cases), Fig. A.1 | needs the prediction arrays |

## Install

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tier 1 of `requirements.txt` is enough to regenerate every table and the
metric figures — no GPU and no PyTorch needed. Install Tier 2 as well only if
you intend to retrain.

## Data

SDWPF (134 turbines, 245 days, 10-minute SCADA) is released by Longyuan Power
Group via the Baidu KDD Cup 2022:

- Dataset: <https://aistudio.baidu.com/aistudio/competition/detail/152/0/datasets>
- Dataset paper: Zhou et al., *Scientific Data* **11**, 649 (2024),
  <https://doi.org/10.1038/s41597-024-03427-5>

Place the raw release under `data/wind/`, then run the preprocessing CLI, which
applies the three cleaning rules used in the paper (negative `Patv` clipped to
zero; rows with `Wspd` outside [0, 25] m/s dropped; rows with `Patv > 0` below
the 3 m/s cut-in dropped), interpolates the resulting gaps, fits the scaler and
the DWT **within each partition**, and writes the manifests:

```bash
python -m scripts.preprocess_cli --csv-path data/wind/<raw>.csv \
                                 --lookback 144 --horizon 12
python -m scripts.gen_partition_manifests        # remaining horizons
python -m scripts.gen_dwt_imfs                   # db4 sub-bands
```

Turbine 1 yields 35,279 valid samples. Pass `--help` to each for the full flag
list.

## Reproduce

### Figures and tables — no GPU, no training

```bash
python -m visualization.build_all_figures        # 7 in-article figures + graphical abstract
python tools/wordcount_epsr.py --sections --floats
```

Each figure script also writes a downscaled copy under `_preview/`; the print
assets are 640 dpi and exceed 3000 px on the long edge.

### Retrain

```bash
python -m scripts.run_batch     --horizons 1 6 12 24 --seeds 42 43 44 45 46 47 48 49 50 51
python -m scripts.run_batch_v2  --horizons 1 6 12 24 --seeds 42 43 44 45 46 47 48 49 50 51
python -m scripts.run_multiturb_v2 --horizons 1 6 12 24 --turbines 1 2 13 55 70 83 86 88 94 99
python -m scripts.benchmark_compute_cost --out outputs/analysis/compute_cost.json
```

`run_batch` covers the 11 baselines, `run_batch_v2` the proposed model and its
ablation variants. Both shard across GPUs via `--num-shards/--shard-index`; the
wrappers under `scripts/*.sh` show how the reported runs were launched. The
expanding-window study reuses `run_batch_v2` against alternative partition
manifests (`partition_W1/W2/W3.json`, run ids `proposed_v2_W{1,2,3}_seed*`).

The run budget behind the paper is 480 + 70 + 480 + 18 = **1,048** runs:
480 for the 12-model × 4-horizon × 10-seed main comparison, 70 for the
7-variant ablation, 480 for the 10-turbine generalisation study
(10 × 4 models × 4 horizons × 3 seeds), and 18 for the expanding-window
temporal robustness check.

## Layout

```
data_pipeline/    cleaning, partition-isolated scaling and DWT, windowing
layers/           DWT decomposition, adaptive gate, KAN, RevIN, attention
models/           TRDFormer and the 11 baselines
experiments/      library: run matrix, runner, metrics, significance tests
scripts/          CLI entry points (preprocessing, batch training, benchmarks)
visualization/    _style.py (one design system) + one script per figure
tools/            word-count / float-placement auditing
outputs/          run records, configs, gate weights, manifests
tests/            unit tests, incl. the DWT partition-isolation checks
```

`visualization/_style.py` is the single source of truth for figure geometry,
palette and DPI; the colour of a component is the same in the architecture
diagram and in every data figure.

## Citation

```bibtex
@article{kang2026trdformer,
  title   = {{TRDFormer}: Trend-residual decomposition with wavelet sub-band
             attention for short-term wind power forecasting},
  author  = {Kang, Shengzhe and Zhu, Xiping and Yang, Tao},
  journal = {Electric Power Systems Research},
  note    = {Under review},
  year    = {2026}
}

@misc{trdformer_code,
  title     = {{TRDFormer}: code and aggregate result records},
  author    = {Kang, Shengzhe and Zhu, Xiping and Yang, Tao},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21911236}
}
```

## Licence

MIT for the code and result records — see `LICENSE`. The SDWPF dataset is not
redistributed here and keeps its own terms. Baseline implementations adapted
from their authors' reference code retain the original licences.

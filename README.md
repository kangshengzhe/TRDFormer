# Look-ahead leakage in decomposition-based wind power forecasting

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21911236.svg)](https://doi.org/10.5281/zenodo.21911236)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code and result records for:

> **Look-ahead leakage in decomposition-based wind power forecasting:
> quantification and causal alternatives**
> Xiping Zhu, Shengzhe Kang, Tao Yang
> School of Electrical and Information Engineering, Southwest Petroleum University
> *Under review, Electric Power Systems Research.*

## What this repository shows

Decomposition-based hybrids dominate recent wind power forecasting: 44 of 50
papers we surveyed feed sub-bands from a signal decomposition (VMD, EMD,
wavelet, ...) into a deep network. Those decompositions are almost always
computed **offline over a whole data segment**, which makes the sub-band value
at time `t` a function of samples *after* `t` — even when the decomposition is
fitted independently within each train/validation/test partition, which is
the safeguard the field already applies against a different, cruder form of
leakage.

We built a complete trend-residual / wavelet hybrid (`TRDFormer` below) the way
this literature builds such models, then measured how much of its reported
accuracy comes from that look-ahead rather than from the architecture. Three
independent diagnostics agree: **essentially all of it.**

| Diagnostic | Offline decomposition | Strictly causal |
|---|---|---|
| Ridge probe, gain over persistence at lead 1 | −16.61 kW | −1.50 kW |
| Full model MAE, $h=12$ (10 seeds) | 58.18 ± 1.13 kW | 91.10 ± 4.37 kW |
| vs. training with no sub-bands at all (81.95 kW) | **−29.0 %** | **+11.2 %** |
| Ridge regression given the same sub-bands | 58.06 kW | — |

Reading the last row: a plain ridge regression handed the leaked sub-bands
matches the full deep model (58.06 vs. 58.18 kW). The architecture adds nothing
measurable once the leakage is available — though it is worth a genuine 24 kW
when the sub-bands are removed and it has to work with causally admissible
inputs only.

Delaying the offline sub-bands by 12 steps — changing nothing about the
decomposition itself, only how stale the features are — reproduces the
no-sub-band baseline to within 0.01 kW. That is the cleanest single number in
the repository: it isolates alignment from architecture and shows the entire
ablation effect is alignment.

### It is not a quirk of the wavelet either

Most of the 44 surveyed papers use VMD rather than a wavelet, so the probe was
repeated on a K=5 VMD of the same series (per partition, fixed parameters):

| Probe gain over persistence | lead 1 | lead 12 |
|---|---|---|
| Offline DWT | −16.61 kW | −32.39 kW |
| Offline VMD | −7.34 kW | **−68.33 kW** |
| VMD delayed 12 steps | **+55.93 kW** | −20.41 kW |

Both leak, with different profiles: the wavelet's finest detail band reveals
most about the *next* step, while VMD's narrow-band modes reveal the window's
*direction* and so dominate at long lead. Delaying the VMD modes by 12 steps
does not merely remove the gain, it **reverses** it — 12-step-old modes are
worse than persistence, which is what you expect from a feature whose value
came from time alignment rather than frequency content.

Note VMD is not exactly additive the way a DWT is: on the test partition
`|sum(modes) − y|` has mean 0.051 and max 1.004 standardised units, against
~1e-7 for the wavelet.

### It is not a quirk of one dataset

The two model-free diagnostics replicate on an independent single-turbine SCADA
record — different country, provider and year (Turkey, 2018), sharing only
timestamp, power and wind speed with SDWPF:

| Diagnostic (second dataset) | Offline | Strictly causal |
|---|---|---|
| Ridge probe, gain over persistence at lead 1 | −49.07 kW | −2.73 kW (**18.0×** gap) |
| Linear ceiling vs. raw lookback only (343.12 kW) | **−35.9 %** | **+1.0 %** |

`tools/cross_dataset_leakage_check.py` runs this end to end. It is
**segment-aware on purpose**: that record has 32 sampling gaps, three of them
spanning up to 4.3 days, and `pywt.wavedec` has no notion of a timestamp. So
the series is split at every discontinuity and each contiguous segment of
≥ 512 rows is decomposed independently (22 segments, 21,435 rows). Dropping the
missing rows and concatenating instead would let the filter bank manufacture a
step artefact at every splice, which the probe would then read as
"information" — a self-inflicted result. Interpolating is no better: a straight
line through a third of a week is invented data entering both the decomposition
and the evaluation.

Retraining results in this repository remain SDWPF-only. The second record's
segments (median 896 rows) are too short to train a 144-step-lookback model
without fabricating data across those gaps.

## The model used as the case study

The forecaster splits the task in two. A linear branch takes the strongly
auto-correlated trend — the component a single linear layer already predicts
almost perfectly, and the reason plain deep models often fail to beat DLinear
on wind power. A deep branch models the residual, handed to it pre-separated
into five db4 wavelet sub-bands, each entering a variate-attention encoder as
its own token. Meteorological covariates are encoded separately by an LSTM,
and a learned per-sample gate decides how much of each branch to use.

This is *not* proposed as a state-of-the-art forecaster. It is the vehicle used
to measure the leakage, assembled from components standard in this literature
specifically so that the measurement is a property of common practice.

## Causal alternative and diagnostic tools released here

- `scripts/gen_dwt_imfs_atrous.py` — a strictly causal, exactly additive
  undecimated filter bank (additivity `9e-16`, zero measured look-ahead by
  construction). Use this where a multi-scale decomposition is genuinely
  wanted and causality matters.
- `scripts/gen_dwt_imfs_causal.py`, `gen_dwt_imfs_variants.py`,
  `gen_dwt_imfs_lagged.py` — trailing-window and delayed-band variants used to
  separate "removing the sub-bands" from "removing the look-ahead".
- `tools/probe_subband_leakage.py` — the ridge probe: how much does a
  sub-band vector at `t` reveal about `y(t+h)`, beyond what `y(t)` already
  gives every model?
- `tools/probe_linear_ceiling.py` — the linear-ceiling check: does a plain
  ridge regression on the same inputs match the deep model? If yes, the
  architecture is not what the ablation is measuring.
- `tools/probe_perturbation.py` — inject a known perturbation after the
  forecast origin and confirm the decomposition should not, but does, move.
- `tools/cross_dataset_leakage_check.py` — both probes on a second,
  independent SCADA record, decomposing strictly inside contiguous segments
  so no filter crosses a sampling gap.
- `tools/probe_vmd_leakage.py` — the same probe on a K=5 VMD instead of the
  wavelet, since VMD is the commonest choice in the surveyed literature. It
  self-checks by first reproducing the paper's offline-DWT figure under the
  identical protocol, and aborts if it cannot.
- `tests/test_dwt_causality.py` — asserts both the partition-isolation
  guarantee and the sample-wise non-causality within a partition.

We recommend running the linear-ceiling check alongside any ablation of a
decomposition-based model: it costs one ridge fit and bounds what the
architecture can possibly be contributing.

## What is and is not in this repository

Tracked: all model and pipeline code, all experiment configurations, and the
**aggregate run records** (`run_records.jsonl`, per-epoch losses, fusion-gate
weights, scalers, partition manifests) for all 1,108 training runs.

Not tracked, and why:

| Excluded | Size | Why / how to get it |
|---|---|---|
| `data/` | 21 MB | SDWPF and the second SCADA record are both public, but redistribution terms are their publishers'. See [Data](#data). |
| `outputs/**/*_preds.npz` | 511 MB | Per-sample prediction arrays. Regenerable by re-running, or available on request. |
| `manuscript/` | — | The article is under review; this is a code and results deposit. |

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
python -m scripts.gen_dwt_imfs                   # offline db4 sub-bands
python -m scripts.gen_dwt_imfs_atrous             # causal alternative
```

Turbine 1 yields 35,279 valid samples. Pass `--help` to each for the full flag
list.

### Second dataset (cross-dataset check only)

A single-turbine SCADA record from Turkey, 2018, 10-minute resolution,
originally distributed via Kaggle as "Wind Turbine Scada Dataset" and mirrored
in `dominodatalab/reference-project-wind-turbine-scada`. Place `T1.csv` under
`data/wind_turkey/`; no preprocessing CLI is needed, the check script does its
own cleaning and segmentation. Columns used: `Date/Time`,
`LV ActivePower (kW)`, `Wind Speed (m/s)`, `Wind Direction (°)`.
`Theoretical_Power_Curve` is deliberately dropped — it is a manufacturer curve
evaluated on the same wind-speed column, not an independent sensor, so keeping
it would let the probe read wind speed by proxy.

## Reproduce

### The leakage diagnostics — no GPU, no training

```bash
python tools/probe_subband_leakage.py
python tools/probe_linear_ceiling.py
python tools/probe_perturbation.py
python tools/cross_dataset_leakage_check.py   # second dataset; needs data/wind_turkey/T1.csv
```

### Figures and tables — no GPU, no training

```bash
python -m visualization.build_all_figures
python tools/wordcount_epsr.py --sections --floats
python tools/verify_manuscript_numbers.py   # cross-checks every reported
                                             # number against run_records.jsonl
```

### Retrain

```bash
python -m scripts.run_batch     --horizons 1 6 12 24 --seeds 42 43 44 45 46 47 48 49 50 51
python -m scripts.run_batch_v2  --horizons 1 6 12 24 --seeds 42 43 44 45 46 47 48 49 50 51
python -m scripts.run_batch_v2  --horizons 12 --seeds 42 43 44 45 46 47 48 49 50 51 \
    --imf-path outputs/manifests/dwt_imfs_atrous.npz --out-dir outputs/causal_h12
python -m scripts.run_multiturb_v2 --horizons 1 6 12 24 --turbines 1 2 13 55 70 83 86 88 94 99
```

`run_batch` covers the 11 baselines, `run_batch_v2` the case-study model, its
ablation variants, and — via `--imf-path` — every causalisation variant
reported in the paper. Both shard across GPUs via `--num-shards/--shard-index`.

The run budget behind the paper is 480 + 70 + 60 + 480 + 18 = **1,108** runs:
480 for the 12-model × 4-horizon × 10-seed main comparison, 70 for the
7-variant ablation, 60 for the six causalisation variants (10 seeds each,
$h=12$), 480 for the 10-turbine generalisation study
(10 × 4 models × 4 horizons × 3 seeds), and 18 for the expanding-window
temporal robustness check.

## Layout

```
data_pipeline/    cleaning, partition-isolated scaling and DWT, windowing
layers/           DWT decomposition, adaptive gate, KAN, RevIN, attention
models/           the case-study model (TRDFormer) and the 11 baselines
experiments/      library: run matrix, runner, metrics, significance tests
scripts/          CLI entry points: preprocessing, batch training, benchmarks,
                  causal / delayed / trailing-window sub-band generators
tools/            leakage probes (incl. the cross-dataset check), word-count /
                  float-placement auditing, manuscript-number verification
visualization/    _style.py (one design system) + one script per figure
outputs/          run records, configs, gate weights, manifests
tests/            unit tests: cleaning rules, metrics against known values,
                  model/ablation forward shapes, run-matrix bookkeeping, and
                  test_dwt_causality.py, which asserts both the
                  partition-isolation guarantee and the sample-wise
                  non-causality within a partition that this paper measures
```

`visualization/_style.py` is the single source of truth for figure geometry,
palette and DPI; the colour of a component is the same in the architecture
diagram and in every data figure.

## Citation

```bibtex
@article{kang2026leakage,
  title   = {Look-ahead leakage in decomposition-based wind power forecasting:
             quantification and causal alternatives},
  author  = {Zhu, Xiping and Kang, Shengzhe and Yang, Tao},
  journal = {Electric Power Systems Research},
  note    = {Under review},
  year    = {2026}
}

@misc{trdformer_code,
  title     = {Code and aggregate result records for ``Look-ahead leakage in
               decomposition-based wind power forecasting''},
  author    = {Zhu, Xiping and Kang, Shengzhe and Yang, Tao},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21911236}
}
```

## Licence

MIT for the code and result records — see `LICENSE`. The SDWPF dataset is not
redistributed here and keeps its own terms. Baseline implementations adapted
from their authors' reference code retain the original licences.

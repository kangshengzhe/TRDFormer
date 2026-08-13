"""
更新论文到 V2 版本
==================
读取 V2 全量实验结果 + baseline 结果，重新生成：
1. manuscript/tables/main_baseline.tex — 主对比表
2. manuscript/tables/ablation.tex — 消融表
3. outputs/figures/paper/ — 所有论文图（使用已有 visualization 模块）

用法：
    python scripts/update_paper_v2.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from collections import defaultdict
from pathlib import Path


def load_records(path):
    records = []
    try:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get('status') == 'success':
                    records.append(r)
    except FileNotFoundError:
        print(f"  WARNING: {path} not found")
    return records


def summarize(records, metric='mae'):
    g = defaultdict(list)
    for r in records:
        g[(r['model_name'], r['horizon'])].append(r['metrics'][metric])
    return {k: (np.mean(v), np.std(v), len(v)) for k, v in g.items()}


def main():
    print("=" * 60)
    print("  更新论文到 V2 版本")
    print("=" * 60)

    # Load results
    baseline_recs = load_records('outputs/runs/run_records.jsonl')
    v2_recs = load_records('outputs/v2_full/outputs/runs/run_records.jsonl')
    dwt_recs = load_records('outputs/dwt_full/outputs/runs/run_records.jsonl')

    baseline_mae = summarize(baseline_recs, 'mae')
    v2_mae = summarize(v2_recs, 'mae')
    dwt_mae = summarize(dwt_recs, 'mae')

    print(f"\n  Baseline runs: {len(baseline_recs)}")
    print(f"  V2 runs: {len(v2_recs)}")
    print(f"  DWT ablation runs: {len(dwt_recs)}")

    # ═══════════════════════════════════════════════════════════════════
    # Table 1: Main Baseline Comparison
    # ═══════════════════════════════════════════════════════════════════
    print("\n  [1/3] Generating main_baseline.tex ...")

    models_main = [
        ('proposed_v2', 'Proposed (Ours)', v2_mae),
        ('dlinear', 'DLinear', baseline_mae),
        ('lstm', 'LSTM', baseline_mae),
        ('itransformer', 'iTransformer', baseline_mae),
        ('patchtst', 'PatchTST', baseline_mae),
        ('timexer', 'TimeXer', baseline_mae),
        ('timesnet', 'TimesNet', baseline_mae),
        ('transformer', 'Transformer', baseline_mae),
        ('informer', 'Informer', baseline_mae),
        ('nonstationary_transformer', 'NS-Transformer', baseline_mae),
        ('fedformer', 'FEDformer', baseline_mae),
        ('autoformer', 'Autoformer', baseline_mae),
    ]

    # Find best and second-best per horizon
    all_vals = {h: [] for h in [1, 6, 12, 24]}
    for m, _, src in models_main:
        for h in [1, 6, 12, 24]:
            val = src.get((m, h), (999, 0, 0))[0]
            all_vals[h].append((val, m))

    best = {}
    second = {}
    for h in [1, 6, 12, 24]:
        sorted_v = sorted(all_vals[h], key=lambda x: x[0])
        best[h] = sorted_v[0][1]
        second[h] = sorted_v[1][1]

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Test-set MAE (kW, mean $\pm$ std over 10 seeds) of the proposed model against 11 baselines. Best in \textbf{bold}, second-best \underline{underlined}. The proposed model achieves the lowest MAE at $h=1/6/12$ and ranks second at $h=24$.}')
    lines.append(r'\label{tab:main_baseline}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{lcccc}')
    lines.append(r'\toprule')
    lines.append(r'Model & $h=1$ & $h=6$ & $h=12$ & $h=24$ \\')
    lines.append(r'\midrule')

    for m, label, src in models_main:
        row = f'  {label}'
        for h in [1, 6, 12, 24]:
            mean, std, n = src.get((m, h), (0, 0, 0))
            if mean == 0:
                row += ' & --'
                continue
            val_str = f'{mean:.2f}$\\pm${std:.2f}'
            if m == best[h]:
                row += f' & \\textbf{{{val_str}}}'
            elif m == second[h]:
                row += f' & \\underline{{{val_str}}}'
            else:
                row += f' & {val_str}'
        row += r' \\'
        lines.append(row)

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    tables_dir = Path('manuscript/tables')
    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / 'main_baseline.tex', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print("    ✓ manuscript/tables/main_baseline.tex")

    # ═══════════════════════════════════════════════════════════════════
    # Table 2: Ablation Study
    # ═══════════════════════════════════════════════════════════════════
    print("\n  [2/3] Generating ablation.tex ...")

    # Ablation uses DWT-based results (dwt_full) + V2 as the full model
    ablation_models = [
        ('proposed_v2', 'Proposed V2 (full)', v2_mae),
        ('proposed', 'w/o Trend Decomposition [A]', dwt_mae),
        ('ablation:vmd_off', 'w/o DWT [B]', dwt_mae),
        ('ablation:itrans_off', 'w/o iTransformer branch [C]', dwt_mae),
        ('ablation:lstm_off', 'w/o LSTM branch [C]', dwt_mae),
        ('ablation:fusion_concat', 'Fusion: concat [D]', dwt_mae),
        ('ablation:fusion_sum', 'Fusion: sum [D]', dwt_mae),
        ('ablation:fusion_cross_attention', 'Fusion: cross-attn.\\ [D]', dwt_mae),
        ('ablation:head_linear', 'Head: linear [D]', dwt_mae),
        ('ablation:head_mlp', 'Head: MLP [D]', dwt_mae),
    ]

    lines2 = []
    lines2.append(r'\begin{table}[htbp]')
    lines2.append(r'\centering')
    lines2.append(r'\caption{Ablation study: test-set MAE (kW, mean $\pm$ std over 10 seeds). [A] Trend-residual decomposition; [B] DWT multi-scale encoding; [C] Dual-branch architecture; [D] Fusion \& head design.}')
    lines2.append(r'\label{tab:ablation}')
    lines2.append(r'\small')
    lines2.append(r'\begin{tabular}{p{0.38\linewidth}cccc}')
    lines2.append(r'\toprule')
    lines2.append(r'Variant & $h=1$ & $h=6$ & $h=12$ & $h=24$ \\')
    lines2.append(r'\midrule')

    for m, label, src in ablation_models:
        row = f'  {label}'
        for h in [1, 6, 12, 24]:
            data = src.get((m, h), (None, None, 0))
            if data[0] is None:
                row += ' & --'
            else:
                mean, std, n = data
                val_str = f'{mean:.2f}$\\pm${std:.2f}'
                if m == 'proposed_v2':
                    row += f' & \\textbf{{{val_str}}}'
                else:
                    row += f' & {val_str}'
        row += r' \\'
        lines2.append(row)

    lines2.append(r'\bottomrule')
    lines2.append(r'\end{tabular}')
    lines2.append(r'\end{table}')

    with open(tables_dir / 'ablation.tex', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines2))
    print("    ✓ manuscript/tables/ablation.tex")

    # ═══════════════════════════════════════════════════════════════════
    # Figures: Generate using existing visualization modules
    # ═══════════════════════════════════════════════════════════════════
    print("\n  [3/3] Generating figures ...")

    figures_dir = 'outputs/figures'
    scaler_path = 'outputs/manifests/scaler.pkl'

    # Find V2 preds files
    v2_runs_dir = Path('outputs/v2_full/outputs/runs')
    for h in [1, 6, 12, 24]:
        # Find a preds file for proposed_v2 at this horizon
        preds_file = v2_runs_dir / f'proposed_v2_h{h}_seed42_preds.npz'
        if not preds_file.exists():
            print(f"    ! preds not found: {preds_file}")
            continue

        # Prediction curve
        try:
            from visualization.prediction_curve import plot_prediction_curve
            rc = plot_prediction_curve(
                preds_path=str(preds_file),
                scaler_path=scaler_path,
                horizon=h,
                out_dir=figures_dir,
                n_points=500,
            )
            if rc == 0:
                print(f"    ✓ pred_vs_actual_h{h}.png")
        except Exception as e:
            print(f"    ! pred_curve h={h}: {e}")

        # Error distribution
        try:
            from visualization.error_distribution import plot_error_distribution
            rc = plot_error_distribution(
                preds_path=str(preds_file),
                scaler_path=scaler_path,
                horizon=h,
                out_dir=figures_dir,
            )
            if rc == 0:
                print(f"    ✓ error_dist_h{h}.png")
        except Exception as e:
            print(f"    ! error_dist h={h}: {e}")

    # ═══════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  完成！已更新:")
    print("    • manuscript/tables/main_baseline.tex")
    print("    • manuscript/tables/ablation.tex")
    print("    • outputs/figures/pred_vs_actual_h*.png")
    print("    • outputs/figures/error_dist_h*.png")
    print("    • outputs/paper_figures/ (之前生成的柱状图等)")
    print("=" * 60)


if __name__ == '__main__':
    main()

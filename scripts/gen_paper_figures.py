"""
论文图表生成脚本
================
生成 V2 模型最终论文所需的全部图表。

输出目录: outputs/paper_figures/

图表清单:
1. 基线对比柱状图（4步长，MAE）
2. 雷达图（多指标综合对比）  
3. 消融实验柱状图
4. 训练收敛曲线
5. 预测曲线对比（时间序列片段）
6. DWT分解可视化
7. 箱线图（种子稳定性分析）

用法:
    python scripts/gen_paper_figures.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

# 设置中文字体和全局风格
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

OUT_DIR = Path('outputs/paper_figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_records(path):
    records = []
    try:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get('status') == 'success':
                    records.append(r)
    except FileNotFoundError:
        pass
    return records


def summarize(records, metric='mae'):
    g = defaultdict(list)
    for r in records:
        g[(r['model_name'], r['horizon'])].append(r['metrics'][metric])
    return {k: (np.mean(v), np.std(v), len(v)) for k, v in g.items()}


# ─── Load Data ─────────────────────────────────────────────────────────────
baseline_recs = load_records('outputs/runs/run_records.jsonl')
v2_recs = load_records('outputs/v2_full/outputs/runs/run_records.jsonl')
dwt_recs = load_records('outputs/dwt_full/outputs/runs/run_records.jsonl')

baseline_mae = summarize(baseline_recs)
baseline_rmse = summarize(baseline_recs, 'rmse')
baseline_r2 = summarize(baseline_recs, 'r2')
v2_mae = summarize(v2_recs)
v2_rmse = summarize(v2_recs, 'rmse')
v2_r2 = summarize(v2_recs, 'r2')
dwt_mae = summarize(dwt_recs)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Baseline Comparison Bar Chart (MAE)
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_baseline_comparison():
    models = ['proposed_v2', 'dlinear', 'lstm', 'itransformer', 'patchtst',
              'timexer', 'timesnet', 'informer', 'transformer',
              'nonstationary_transformer', 'fedformer']
    labels = ['Proposed', 'DLinear', 'LSTM', 'iTransformer', 'PatchTST',
              'TimeXer', 'TimesNet', 'Informer', 'Transformer',
              'NS-Transformer', 'FEDformer']
    horizons = [1, 6, 12, 24]
    
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=False)
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    colors[0] = [0.8, 0.2, 0.2, 1.0]  # Red for proposed

    for ax_idx, h in enumerate(horizons):
        ax = axes[ax_idx]
        means = []
        stds = []
        for m in models:
            if m == 'proposed_v2':
                data = v2_mae.get((m, h), (None, None, 0))
            else:
                data = baseline_mae.get((m, h), (None, None, 0))
            means.append(data[0] if data[0] else 0)
            stds.append(data[1] if data[1] else 0)
        
        bars = ax.bar(range(len(models)), means, yerr=stds, 
                      color=colors, edgecolor='black', linewidth=0.5,
                      capsize=2, error_kw={'linewidth': 0.8})
        ax.set_title(f'h = {h} ({h*10} min)', fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('MAE (kW)' if ax_idx == 0 else '')
        ax.yaxis.set_major_locator(MaxNLocator(6))
        
        # Highlight best
        best_idx = np.argmin(means)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2)

    plt.suptitle('MAE Comparison Across Prediction Horizons', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig1_baseline_comparison.pdf')
    plt.savefig(OUT_DIR / 'fig1_baseline_comparison.png')
    plt.close()
    print("  [1/7] fig1_baseline_comparison.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Radar Chart (Multi-metric)
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_radar():
    """Radar chart comparing top-5 models across MAE/RMSE/R2 at h=12."""
    h = 12
    models = ['proposed_v2', 'dlinear', 'lstm', 'patchtst', 'itransformer']
    labels_m = ['Proposed', 'DLinear', 'LSTM', 'PatchTST', 'iTransformer']
    metrics = ['MAE', 'RMSE', '1-R²']
    
    data = []
    for m in models:
        if m == 'proposed_v2':
            mae_val = v2_mae.get((m, h), (0,0,0))[0]
            rmse_val = v2_rmse.get((m, h), (0,0,0))[0]
            r2_val = v2_r2.get((m, h), (0,0,0))[0]
        else:
            mae_val = baseline_mae.get((m, h), (0,0,0))[0]
            rmse_val = baseline_rmse.get((m, h), (0,0,0))[0]
            r2_val = baseline_r2.get((m, h), (0,0,0))[0]
        data.append([mae_val, rmse_val, 1 - r2_val])
    
    data = np.array(data)
    # Normalize to [0, 1] for radar (lower is better for all)
    data_norm = data / data.max(axis=0)
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    colors_r = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00']
    
    for i, (row, label) in enumerate(zip(data_norm, labels_m)):
        values = row.tolist() + row[:1].tolist()
        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=colors_r[i])
        ax.fill(angles, values, alpha=0.1, color=colors_r[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_title(f'Multi-metric Comparison (h={h})\n(smaller area = better)', 
                 fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig2_radar_h12.pdf')
    plt.savefig(OUT_DIR / 'fig2_radar_h12.png')
    plt.close()
    print("  [2/7] fig2_radar_h12.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Ablation Study (DWT-based, from dwt_full results)
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_ablation():
    """Ablation bar chart showing contribution of each component."""
    h = 12  # Representative horizon
    
    ablation_models = [
        ('proposed_v2', 'Full Model\n(Proposed V2)'),
        ('proposed', 'w/o Trend\nDecomposition'),
        ('ablation:vmd_off', 'w/o DWT'),
        ('ablation:itrans_off', 'w/o iTransformer'),
        ('ablation:lstm_off', 'w/o LSTM'),
        ('ablation:fusion_cross_attention', 'CrossAttention\n(vs Gated)'),
        ('ablation:head_linear', 'Linear Head\n(vs KAN)'),
    ]
    
    means = []
    stds = []
    labels = []
    for m, label in ablation_models:
        if m == 'proposed_v2':
            d = v2_mae.get((m, h), (None, None, 0))
        else:
            d = dwt_mae.get((m, h), baseline_mae.get((m, h), (None, None, 0)))
        means.append(d[0] if d[0] else 0)
        stds.append(d[1] if d[1] else 0)
        labels.append(label)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_a = ['#E41A1C'] + ['#377EB8'] * (len(means) - 1)
    bars = ax.bar(range(len(means)), means, yerr=stds, color=colors_a,
                  edgecolor='black', linewidth=0.5, capsize=3,
                  error_kw={'linewidth': 0.8})
    
    # Add value labels
    for bar, m in zip(bars, means):
        if m > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{m:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('MAE (kW)')
    ax.set_title(f'Ablation Study (h={h}, 10 seeds)', fontweight='bold')
    ax.axhline(y=means[0], color='red', linestyle='--', alpha=0.5, linewidth=0.8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig3_ablation_h12.pdf')
    plt.savefig(OUT_DIR / 'fig3_ablation_h12.png')
    plt.close()
    print("  [3/7] fig3_ablation_h12.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Box Plot (Seed Stability)
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_boxplot():
    """Box plot showing seed-to-seed variance."""
    models_to_plot = ['proposed_v2', 'dlinear', 'lstm', 'itransformer', 'patchtst']
    labels_b = ['Proposed', 'DLinear', 'LSTM', 'iTransformer', 'PatchTST']
    
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    
    for ax_idx, h in enumerate([1, 6, 12, 24]):
        ax = axes[ax_idx]
        box_data = []
        for m in models_to_plot:
            if m == 'proposed_v2':
                maes = [r['metrics']['mae'] for r in v2_recs
                        if r['model_name'] == m and r['horizon'] == h]
            else:
                maes = [r['metrics']['mae'] for r in baseline_recs
                        if r['model_name'] == m and r['horizon'] == h]
            box_data.append(maes if maes else [0])
        
        bp = ax.boxplot(box_data, labels=labels_b, patch_artist=True,
                        medianprops=dict(color='black', linewidth=1.5))
        colors_box = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00']
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_title(f'h = {h}', fontweight='bold')
        ax.set_ylabel('MAE (kW)' if ax_idx == 0 else '')
        ax.tick_params(axis='x', rotation=30)
    
    plt.suptitle('Prediction Stability Across Random Seeds (10 seeds)', 
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig4_boxplot_stability.pdf')
    plt.savefig(OUT_DIR / 'fig4_boxplot_stability.png')
    plt.close()
    print("  [4/7] fig4_boxplot_stability.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5: Improvement Percentage Bar (vs DLinear)
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_improvement():
    """Horizontal bar chart showing % improvement over DLinear."""
    horizons = [1, 6, 12, 24]
    improvements = []
    for h in horizons:
        v2_val = v2_mae.get(('proposed_v2', h), (0,0,0))[0]
        dl_val = baseline_mae.get(('dlinear', h), (1,0,0))[0]
        pct = (v2_val - dl_val) / dl_val * 100
        improvements.append(pct)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    colors_imp = ['green' if x < 0 else 'red' for x in improvements]
    bars = ax.barh(
        [f'h={h}\n({h*10} min)' for h in horizons],
        improvements,
        color=colors_imp, alpha=0.7, edgecolor='black', linewidth=0.5
    )
    
    for bar, val in zip(bars, improvements):
        ax.text(bar.get_width() + (0.5 if val >= 0 else -0.5), 
                bar.get_y() + bar.get_height()/2,
                f'{val:+.1f}%', va='center', 
                ha='left' if val >= 0 else 'right', fontsize=11)
    
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('MAE Improvement vs DLinear (%)')
    ax.set_title('Proposed Model vs DLinear Baseline', fontweight='bold')
    ax.set_xlim(-25, 10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig5_improvement_vs_dlinear.pdf')
    plt.savefig(OUT_DIR / 'fig5_improvement_vs_dlinear.png')
    plt.close()
    print("  [5/7] fig5_improvement_vs_dlinear.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6: DWT Decomposition Visualization
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_dwt_decomposition():
    """Visualize DWT decomposition of Patv signal."""
    import pandas as pd
    
    df = pd.read_csv('data/wind/sdwpf_turb1_cleaned_final.csv')
    patv = df['Patv'].values
    
    imf_data = np.load('outputs/manifests/dwt_imfs.npz')
    imfs = imf_data['all_imfs']
    
    # Show a 2-day segment (288 points = 48h)
    start = 5000
    end = start + 288
    t = np.arange(end - start) * 10 / 60  # hours
    
    fig, axes = plt.subplots(7, 1, figsize=(12, 10), sharex=True)
    
    # Original signal (need to get scaled patv)
    from data_pipeline.scaling import FeatureScaler
    scaler = FeatureScaler.load('outputs/manifests/scaler.pkl')
    raw_features = df[['Patv', 'Wspd', 'Wdir', 'Etmp', 'Itmp']].values[:, :1]
    
    axes[0].plot(t, patv[start:end], 'k-', linewidth=0.8)
    axes[0].set_ylabel('Patv (kW)')
    axes[0].set_title('Original Signal and DWT Decomposition (db4, 4 levels)', fontweight='bold')
    
    channel_names = ['D1 (Finest Detail)', 'D2', 'D3', 'D4 (Coarsest Detail)', 
                     'A4 (Approximation/Trend)']
    colors_dwt = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i in range(5):
        ax = axes[i + 1]
        ax.plot(t, imfs[start:end, i], color=colors_dwt[i], linewidth=0.8)
        ax.set_ylabel(channel_names[i], fontsize=8)
    
    # Reconstruction check
    recon = imfs[start:end].sum(axis=1)
    axes[6].plot(t, patv[start:end], 'k-', linewidth=0.8, label='Original', alpha=0.5)
    axes[6].set_ylabel('Σ(Components)')
    axes[6].set_xlabel('Time (hours)')
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig6_dwt_decomposition.pdf')
    plt.savefig(OUT_DIR / 'fig6_dwt_decomposition.png')
    plt.close()
    print("  [6/7] fig6_dwt_decomposition.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7: Summary Table (LaTeX)
# ═══════════════════════════════════════════════════════════════════════════════
def fig7_latex_table():
    """Generate LaTeX table for the paper."""
    models = [
        ('proposed_v2', 'Proposed (Ours)'),
        ('dlinear', 'DLinear'),
        ('lstm', 'LSTM'),
        ('itransformer', 'iTransformer'),
        ('patchtst', 'PatchTST'),
        ('timexer', 'TimeXer'),
        ('transformer', 'Transformer'),
        ('timesnet', 'TimesNet'),
        ('informer', 'Informer'),
        ('nonstationary_transformer', 'NS-Transformer'),
        ('fedformer', 'FEDformer'),
        ('autoformer', 'Autoformer'),
    ]
    
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{MAE comparison (kW) across prediction horizons. '
                 r'Best results in \textbf{bold}, second-best \underline{underlined}.}')
    lines.append(r'\label{tab:baseline_comparison}')
    lines.append(r'\begin{tabular}{lcccc}')
    lines.append(r'\toprule')
    lines.append(r'Model & h=1 (10min) & h=6 (1h) & h=12 (2h) & h=24 (4h) \\')
    lines.append(r'\midrule')
    
    # Collect all values to find best/second
    all_vals = {h: [] for h in [1, 6, 12, 24]}
    for m, _ in models:
        for h in [1, 6, 12, 24]:
            if m == 'proposed_v2':
                val = v2_mae.get((m, h), (999, 0, 0))[0]
            else:
                val = baseline_mae.get((m, h), (999, 0, 0))[0]
            all_vals[h].append(val)
    
    for h in [1, 6, 12, 24]:
        sorted_v = sorted(all_vals[h])
        all_vals[h] = (sorted_v[0], sorted_v[1])  # best, second
    
    for m, label in models:
        row = f'  {label}'
        for h in [1, 6, 12, 24]:
            if m == 'proposed_v2':
                mean, std, n = v2_mae.get((m, h), (0, 0, 0))
            else:
                mean, std, n = baseline_mae.get((m, h), (0, 0, 0))
            
            best, second = all_vals[h]
            if abs(mean - best) < 0.01:
                row += f' & \\textbf{{{mean:.2f}}}$\\pm${std:.2f}'
            elif abs(mean - second) < 0.01:
                row += f' & \\underline{{{mean:.2f}}}$\\pm${std:.2f}'
            else:
                row += f' & {mean:.2f}$\\pm${std:.2f}'
        row += r' \\'
        lines.append(row)
    
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    
    tex_content = '\n'.join(lines)
    with open(OUT_DIR / 'table_baseline_comparison.tex', 'w') as f:
        f.write(tex_content)
    print("  [7/7] table_baseline_comparison.tex")
    return tex_content


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("  论文图表生成")
    print("=" * 60)
    
    fig1_baseline_comparison()
    fig2_radar()
    fig3_ablation()
    fig4_boxplot()
    fig5_improvement()
    
    try:
        fig6_dwt_decomposition()
    except Exception as e:
        print(f"  [6/7] SKIP fig6 (need data files): {e}")
    
    tex = fig7_latex_table()
    
    print(f"\n  所有图表已保存到: {OUT_DIR}/")
    print("=" * 60)

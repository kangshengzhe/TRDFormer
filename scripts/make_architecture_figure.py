# -*- coding: utf-8 -*-
"""
本模型的框架结构图(Fig.1 风格)——纯 matplotlib 程序化绘制。
数据管道(物理清洗 + VMD) → 非对称双分支(iTransformer 目标分支 + LSTM 协变量分支)
→ 自适应门控融合(Softmax) → KAN 预测头 → 多步功率输出。
输出:outputs/figures/paper/fig0_architecture.png
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / "outputs" / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

# 配色(与全套图一致)
C_DATA   = "#DCE6F2"; E_DATA   = "#6C8EBF"   # 数据/输入 蓝
C_PREP   = "#FCE4D6"; E_PREP   = "#D79B62"   # 预处理 橙
C_ENDO   = "#E8DFF2"; E_ENDO   = "#7B4FA3"   # 目标分支 紫
C_EXO    = "#DFF0E3"; E_EXO    = "#2E8B57"   # 协变量分支 绿
C_FUSE   = "#FADBD8"; E_FUSE   = "#C0392B"   # 融合 红
C_HEAD   = "#FFF2CC"; E_HEAD   = "#C9A227"   # KAN头 金
C_OUT    = "#D6EAF8"; E_OUT    = "#2E86C1"   # 输出 亮蓝

fig, ax = plt.subplots(figsize=(16, 8.8))
ax.set_xlim(0, 16); ax.set_ylim(0, 8.8); ax.axis("off")


def zone(x, y, w, h, ec, title):
    """语义分区虚线框 + 左上角区块标题(学自顶刊图的分区手法)。"""
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                       fc="none", ec=ec, lw=1.6, ls=(0, (6, 4)), zorder=0)
    ax.add_patch(p)
    ax.text(x + 0.18, y + h - 0.28, title, ha="left", va="center",
            fontsize=10.5, fontweight="bold", color=ec, zorder=1)


def box(x, y, w, h, fc, ec, title, sub=None, title_fs=11, sub_fs=8.5, bold=True):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                       fc=fc, ec=ec, lw=1.6, zorder=3)
    ax.add_patch(p)
    cy = y + h/2 + (0.16 if sub else 0)
    ax.text(x + w/2, cy, title, ha="center", va="center",
            fontsize=title_fs, fontweight="bold" if bold else "normal",
            color="#222222", zorder=4)
    if sub:
        ax.text(x + w/2, y + h/2 - 0.28, sub, ha="center", va="center",
                fontsize=sub_fs, color="#444444", zorder=4, style="italic")
    return (x, y, w, h)


def arrow(p1, p2, color="#555555", lw=1.8, style="-|>"):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=16,
                        color=color, lw=lw, zorder=2,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)


def right(b): x, y, w, h = b; return (x + w, y + h/2)
def left(b):  x, y, w, h = b; return (x, y + h/2)
def top(b):   x, y, w, h = b; return (x + w/2, y + h)
def bottom(b):x, y, w, h = b; return (x + w/2, y)


# ── 标题 ────────────────────────────────────────────────────────────────
ax.text(8, 8.5, "Proposed VMD–Dual-Branch–Gated-Fusion–KAN Wind Power Forecasting Model",
        ha="center", va="center", fontsize=13.5, fontweight="bold")

# ── 语义分区虚线框(画在模块之下)──────────────────────────────────────────
zone(2.75, 3.15, 2.85, 3.75, E_PREP, "A · Decomposition")
zone(5.65, 1.35, 5.55, 5.55, E_ENDO, "B · Dual-branch encoder")
zone(11.25, 2.95, 4.55, 2.55, E_FUSE, "C · Fusion + KAN head")

# ── 1. 输入 ────────────────────────────────────────────────────────────
b_in = box(0.3, 3.3, 2.2, 1.6, C_DATA, E_DATA,
           "SDWPF Turb1",
           "Patv + Wspd,Wdir,\nEtmp,Itmp")

# ── 2. 预处理:物理清洗(中) → VMD(上);清洗直供协变量(下) ─────────────
b_clean = box(3.0, 3.45, 2.3, 1.3, C_PREP, E_PREP,
              "Physical-rule\ncleaning", title_fs=10)
b_vmd = box(3.0, 5.3, 2.3, 1.05, C_PREP, E_PREP,
            "VMD", "Patv → K IMFs", title_fs=10, sub_fs=8)

# ── 3. 双分支 ──────────────────────────────────────────────────────────
b_endo_in = box(5.9, 5.25, 2.1, 1.0, C_ENDO, E_ENDO,
                "Endogenous", "Patv + IMF$_{1..K}$", title_fs=9.5, sub_fs=8)
b_itrans = box(8.4, 5.25, 2.5, 1.0, C_ENDO, E_ENDO,
               "iTransformer", "variate attention", title_fs=10, sub_fs=8)

b_exo_in = box(5.9, 1.6, 2.1, 1.0, C_EXO, E_EXO,
               "Exogenous", "Wspd,Wdir,Etmp,Itmp", title_fs=9.5, sub_fs=7.5)
b_lstm = box(8.4, 1.6, 2.5, 1.0, C_EXO, E_EXO,
             "LSTM", "temporal encoding", title_fs=10, sub_fs=8)

# ── 4. 自适应门控融合 ───────────────────────────────────────────────────
b_fuse = box(11.4, 3.3, 2.4, 1.6, C_FUSE, E_FUSE,
             "Adaptive\nGated Fusion", title_fs=10.5)
ax.text(12.6, 3.62, r"$\alpha_{en}\,x_{1}+\alpha_{ex}\,x_{2}$",
        ha="center", va="center", fontsize=9, color="#7B241C", zorder=4)

# ── 5. KAN 头 ──────────────────────────────────────────────────────────
b_head = box(14.05, 3.5, 1.6, 1.2, C_HEAD, E_HEAD,
             "KAN\nhead", title_fs=10.5)

# ── 6. 输出 ────────────────────────────────────────────────────────────
b_out = box(13.6, 0.6, 2.1, 1.1, C_OUT, E_OUT,
            r"$\hat{Y}_{1:H}$", "multi-step power", title_fs=12, sub_fs=8)

# ── 箭头连线(无交叉:清洗→VMD 走上路供目标;清洗→协变量 走下路) ─────────
arrow(right(b_in), left(b_clean))          # 输入 → 清洗
arrow(top(b_clean), bottom(b_vmd))         # 清洗 → VMD(向上,串行)
arrow(right(b_vmd), left(b_endo_in))       # VMD(含清洗后Patv+IMF) → 目标输入
arrow(right(b_clean), left(b_exo_in), color=E_EXO)  # 清洗 → 协变量输入(向下)
arrow(right(b_endo_in), left(b_itrans))    # 目标 → iTransformer
arrow(right(b_exo_in), left(b_lstm))       # 协变量 → LSTM
arrow(right(b_itrans), (11.4, 4.35), color=E_ENDO)  # iTrans → 融合(上)
arrow(right(b_lstm), (11.4, 3.85), color=E_EXO)     # LSTM → 融合(下)
arrow(right(b_fuse), left(b_head), color=E_FUSE)    # 融合 → KAN头
arrow(bottom(b_head), (14.65, 1.7), color=E_HEAD)   # KAN头 → 输出

# ── 箭头图例(学自顶刊图的 notation 说明框)────────────────────────────────
lx, ly = 0.35, 2.55
ax.add_patch(FancyBboxPatch((lx - 0.15, ly - 2.05), 3.5, 2.15,
             boxstyle="round,pad=0.04,rounding_size=0.1",
             fc="#FBFBFB", ec="#BBBBBB", lw=1.2, zorder=1))
ax.text(lx + 1.6, ly - 0.12, "Legend", ha="center", va="center",
        fontsize=9.5, fontweight="bold", zorder=2)
_legend = [("#555555", "Data flow"),
           (E_ENDO, "Endogenous stream (target + IMFs)"),
           (E_EXO, "Exogenous stream (covariates)"),
           (E_FUSE, "Fused stream")]
for i, (col, lab) in enumerate(_legend):
    yy = ly - 0.5 - i * 0.42
    a = FancyArrowPatch((lx + 0.05, yy), (lx + 0.55, yy), arrowstyle="-|>",
                        mutation_scale=13, color=col, lw=2.2, zorder=2)
    ax.add_patch(a)
    ax.text(lx + 0.7, yy, lab, ha="left", va="center", fontsize=8.3,
            color="#333333", zorder=2)

fig.savefig(OUT / "fig0_architecture.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved:", OUT / "fig0_architecture.png")

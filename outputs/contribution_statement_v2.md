# 研究贡献声明（V2 定稿版）

**论文题目（拟）：** A Trend-Residual Decomposition Framework with DWT-Enhanced Dual-Branch Encoding and Adaptive Gated Fusion for Short-Term Wind Power Forecasting

**数据集：** SDWPF（国家电网大赛公开数据集），单机组 Turb1，10分钟分辨率

**模型名称：** ProposedModelV2（DWT-TrendRes-iTransformer-LSTM-GatedFusion-KAN）

---

## 一、最终实验结果（10种子，MAE kW）

| 步长 | Proposed V2 | DLinear | LSTM | iTransformer | PatchTST | 相对 DLinear |
|---|---|---|---|---|---|---|
| h=1 (10min) | **26.18 ± 1.14** | 30.93 ± 0.03 | 32.64 ± 0.41 | 36.06 ± 0.33 | 38.25 ± 1.52 | **-15.3%** |
| h=6 (1h) | **44.96 ± 0.82** | 52.45 ± 0.08 | 54.23 ± 0.29 | 59.77 ± 0.42 | 60.28 ± 1.11 | **-14.3%** |
| h=12 (2h) | **58.18 ± 1.07** | 70.37 ± 0.10 | 74.15 ± 2.04 | 80.94 ± 0.88 | 78.36 ± 1.29 | **-17.3%** |
| h=24 (4h) | 104.67 ± 10.06 | **100.21 ± 0.29** | 104.96 ± 2.03 | 115.95 ± 2.15 | 106.80 ± 1.87 | +4.4% |

**综合平均 MAE: 58.50 (Proposed) vs 63.49 (DLinear) — 总体领先 8%**

---

## 二、模型架构（V2 最终版）

```
输入: (B, L=144, F_in=10)
  通道布局: [Patv_scaled, D1, D2, D3, D4, A4, Wspd, Wdir, Etmp, Itmp]
       ↓
┌─────────────────────────────────────────────────────────────────┐
│ 趋势-残差分解 (Trend-Residual Decomposition)                     │
│                                                                 │
│  Patv → MovingAvg(kernel=25) → trend                           │
│  trend → Linear(144→H) → trend_prediction                      │
│  Patv - trend = seasonal_residual                              │
└─────────────────────────────────────────────────────────────────┘
       ↓ (residual + IMFs + covariates)
┌─────────────────────────────────────────────────────────────────┐
│ 非对称双分支编码器                                               │
│                                                                 │
│  Branch A: iTransformer(4层, 6头, dim=128)                      │
│    输入: [seasonal_residual, D1..D4, A4] → 6个变量token          │
│    输出: Patv_token (B, 1, 128)                                 │
│                                                                 │
│  Branch B: LSTM(3层, hidden=128)                                │
│    输入: [Wspd, Wdir, Etmp, Itmp] → 气象协变量序列              │
│    输出: (B, L, 128)                                            │
└─────────────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────────────┐
│ 自适应门控融合 (Adaptive Gated Fusion)                           │
│                                                                 │
│  [x_itrans ∥ x_lstm] → FC(256→2) → Softmax → (α_en, α_ex)    │
│  fused = α_en · x_itrans + α_ex · x_lstm                      │
└─────────────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────────────┐
│ KAN 预测头                                                       │
│  KAN([128, H]) — B样条基函数拟合非线性映射                       │
└─────────────────────────────────────────────────────────────────┘
       ↓
  residual_prediction (B, H)
       ↓
  final_output = trend_prediction + residual_prediction
```

---

## 三、四个创新点

### 创新点 A：趋势-残差分解学习（核心贡献）

**做了什么：** 在深度模型外层包裹一个趋势分解框架——用移动平均提取趋势分量，线性映射趋势做粗预测，深度模型只学习去趋势后的非线性残差。

**为什么有效：** DLinear 之所以强，是因为它直接捕获了 Patv 信号中的线性趋势结构（自相关 lag-1=0.974）。我们不与 DLinear 竞争趋势预测，而是让线性分支处理趋势（DLinear 的强项），让深度分支处理非线性残差（DLinear 的盲区）。两者相加 > 任何一个单独使用。

**文献依据：** DLinear (AAAI 2023) 的 series decomposition; N-BEATS (ICLR 2020) 的 trend-seasonality 分层设计; FiLM (NeurIPS 2022)。

**消融证据：** 去掉趋势分解后（仅 DWT + 双分支），h=12 MAE 从 58.18 升高至约 80（DWT-only proposed），**贡献 ΔAE ≈ -22 kW（-27%）**，是四个创新点中贡献最大的。

---

### 创新点 B：因果 DWT 多尺度频率分解（替代 VMD）

**做了什么：** 用离散小波变换（DWT, db4, 4 levels）对标准化 Patv 信号做多尺度分解，产生 5 个频率通道（D1-D4 细节 + A4 趋势近似），各作独立变量 token 输入 iTransformer。**各分区独立分解，严格无数据泄漏。**

**为什么有效：** DWT 天然因果（卷积操作只用当前和过去数据），避免了 VMD 的全局优化泄漏问题。多尺度频率分量让 iTransformer 的变量注意力能学习跨频率依赖（高频波动 vs 低频趋势）。

**文献依据：** CPLLM-WPF (Applied Energy 2025) 的 IMF 独立建模; iTransformer (ICLR 2024) 的变量 token 设计; 小波分解在时序预测中的广泛应用。

**消融证据：** 去掉 DWT 后（vmd_off），多步长 MAE 显著上升（见消融表）。

---

### 创新点 C：非对称双分支编码器

**做了什么：** 目标变量（Patv 残差 + DWT 分量）由 iTransformer 的变量注意力处理；气象协变量（Wspd, Wdir, Etmp, Itmp）由 LSTM 序列建模处理。两条分支独立编码不同物理属性的信号。

**为什么有效：** 内生变量（功率及其频率分量）和外生变量（气象）的物理驱动机制不同——前者需要频率间交叉注意力，后者需要时序动态建模。差异化处理比统一编码更高效。

**文献依据：** TimeXer (NeurIPS 2024) 的内生-外生分离范式。

---

### 创新点 D：自适应门控融合 + KAN 预测头

**做了什么：** 用 FC+Softmax 产生按样本自适应的融合权重 (α_en, α_ex)，替代固定结构的 CrossAttention；KAN 用 B 样条激活函数做最终预测映射。

**为什么有效：** 门控权重是输入依赖的——不同工况（高风速/低风速/切入切出）自动调整两分支的贡献比例。KAN 的分段多项式天然适配风功率曲线的非线性。

**文献依据：** GWS-STNet SCGF (Energy 2026); WD-SGformer (Energy 2025); KANformer (Energy 2025)。

---

## 四、与泄漏版的对比说明

本文明确指出并修正了 VMD 类风电预测论文中的常见数据泄漏陷阱：

| | 泄漏版 (VMD) | V2 (DWT) | 说明 |
|---|---|---|---|
| h=12 MAE | 54.8 | **58.18** | V2 略高但完全因果，无泄漏 |
| h=12 vs DLinear | -22% | **-17%** | 仍大幅领先，且结果真实可信 |
| 可复现性 | ✗（泄漏结果不可复现于实际部署） | ✓ | 可直接用于线上推理 |

---

## 五、实验规模

- 模型数量：1 proposed + 11 baselines + 消融变体
- 预测步长：4 个（h=1, 6, 12, 24）
- 随机种子：10 个（42-51）
- 总运行数：400+
- 训练设备：2× NVIDIA RTX 4090
- 每轮 epochs：150（early stopping patience=10）
- 评价指标：MAE, RMSE, R², MBE, SMAPE

---

## 六、论文关键图表清单

1. **Fig 1**: 基线对比柱状图（4步长 MAE）
2. **Fig 2**: 雷达图（多指标综合）
3. **Fig 3**: 消融实验柱状图
4. **Fig 4**: 箱线图（种子稳定性）
5. **Fig 5**: 相对 DLinear 提升百分比图
6. **Fig 6**: DWT 分解可视化
7. **Fig 7**: 模型架构示意图（需手绘或 TikZ）
8. **Table 1**: 基线对比表（LaTeX）
9. **Table 2**: 消融实验表

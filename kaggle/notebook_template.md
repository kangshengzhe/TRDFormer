# Kaggle GPU Notebook 模板

整个仓库现在是**自包含**的（TSL 模型已内置到 `models/tsl/` 和 `layers/`，
预处理产物已在 `outputs/manifests/`），所以**只需上传一个数据集**。

---

## 上传准备

1. 本地把整个 `iTansformer_LSTM_CA_KAN-master/` 文件夹压缩为 zip
2. Kaggle → Datasets → New Dataset → 上传 zip
3. 数据集命名：`itransformer-lstm-ca-kan-master`
   （挂载路径会是 `/kaggle/input/itransformer-lstm-ca-kan-master`）

> 每次改代码后，上传数据集的新版本即可。

---

## Notebook 设置

- Settings → Accelerator → **GPU**（推荐 T4 或 P100）
- Add Input → 选择你的数据集
- Settings → Persistence → 建议开启，方便跨会话保留 `/kaggle/working`

---

## Cell 1 — 初始化

```python
import shutil, os
# 把只读数据集复制到可写工作目录，使相对路径能正常读写
REPO_SRC = '/kaggle/input/itransformer-lstm-ca-kan-master'
REPO = '/kaggle/working/repo'
if not os.path.exists(REPO):
    shutil.copytree(REPO_SRC, REPO)
os.chdir(REPO)
%run kaggle/bootstrap.py
```

`bootstrap.py` 安装 5 个依赖（einops、fightingcv-attention、dill、PyYAML、
tabulate），设置环境变量，并把 REPO 加入 sys.path。
**注意：不再安装 vmdpy** —— VMD 已在本地完成，Kaggle 只读 `vmd_imfs.npz`。

---

## Cell 2 — 批量运行（断点续跑）

```python
import sys
sys.path.insert(0, '/kaggle/working/repo')
# 跑完整矩阵，限制本次会话最多 8 小时（Kaggle 单次 GPU 上限约 9-12h）
!python -m scripts.run_batch --max-hours 8 --out-dir /kaggle/working
```

批量运行器会：
- 遍历完整矩阵：(proposed + 11 基线 + 8 消融) × horizon{1,6,12,24} × seed{42..46}
- **断点续跑**：读 `run_records.jsonl`，跳过已完成的运行
- **时间预算**：到 8 小时主动停止，剩余留待下次会话
- **逐运行容错**：单个失败只记录并继续

### 先做一次 GPU 冒烟（强烈推荐第一次运行时用）

```python
!python -m scripts.run_batch --smoke --out-dir /kaggle/working
```

微型模型 + 2 epoch，跑 3 个代表性配置，几分钟内确认 GPU 环境没问题。

---

## 跨会话续跑

Kaggle 单次会话结束后，`/kaggle/working/outputs/runs/run_records.jsonl`
会记录已完成的运行。**下次新会话重复运行 Cell 2 即可自动接着跑**，
不会重复已完成的部分。

如果开启了 Persistence，`/kaggle/working` 会保留；否则每次会话开始时
先把上次下载的 `run_records.jsonl` 放回 `/kaggle/working/outputs/runs/`。

---

## 下载结果

会话结束后，从 **Output → `/kaggle/working`** 下载：

```
/kaggle/working/
├── model_save/wind/          # .pt 检查点
└── outputs/runs/
    ├── run_records.jsonl      # 核心：所有运行的指标记录
    ├── {run_id}_preds.npz     # 预测结果（画图用）
    └── {run_id}_losses.npz    # 训练损失（画训练曲线用）
```

把这些放回本地 `outputs/runs/`，然后本地跑聚合出表格和图：

```bash
python scripts/aggregate_cli.py --figures
```

---

## 只跑部分组合（可选）

```python
# 只跑 proposed 和某几个基线的 horizon=6
!python -m scripts.run_batch --models proposed dlinear timexer --horizons 6 --out-dir /kaggle/working

# 先跑完所有必选的 proposed + 基线，之后再跑消融
!python -m scripts.run_batch --models proposed lstm transformer informer fedformer dlinear patchtst itransformer timesnet autoformer nonstationary_transformer timexer --out-dir /kaggle/working
```

---

## 依赖版本

| 包 | 版本 | 用途 |
|---|---|---|
| einops | 0.8.0 | iTransformer 分支张量变换 |
| fightingcv-attention | 1.0.0 | CrossAttention 实现 |
| dill | 0.3.8 | 模型检查点序列化 |
| PyYAML | 6.0.2 | YAML 配置加载 |
| tabulate | 0.9.0 | LaTeX 表格导出 |

vmdpy **不在 Kaggle 安装列表**——VMD 只在本地 CPU 预处理阶段运行。

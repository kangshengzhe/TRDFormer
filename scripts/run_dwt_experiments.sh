#!/bin/bash
# ================================================================
# DWT 实验全流程：生成 DWT IMFs → 替换 → 跑全矩阵
# ================================================================
cd ~/windpower_model/iTansformer_LSTM_CA_KAN-master

echo "=== Step 1: 安装 PyWavelets ==="
.venv/bin/pip install PyWavelets -q

echo ""
echo "=== Step 2: 生成 DWT 分解 ==="
.venv/bin/python scripts/gen_dwt_imfs.py

echo ""
echo "=== Step 3: 备份旧 vmd_imfs.npz 并替换为 DWT 版 ==="
cp outputs/manifests/vmd_imfs.npz outputs/manifests/vmd_imfs_causal_backup.npz
cp outputs/manifests/dwt_imfs.npz outputs/manifests/vmd_imfs.npz
echo "已将 dwt_imfs.npz 复制为 vmd_imfs.npz（runner.py 读取此文件）"

echo ""
echo "=== Step 4: 启动双卡实验 ==="
MODELS="proposed ablation:fusion_concat ablation:fusion_sum ablation:fusion_cross_attention ablation:head_linear ablation:head_mlp ablation:itrans_off ablation:lstm_off ablation:vmd_off"
HORIZONS="1 6 12 24"
SEEDS="42 43 44 45 46 47 48 49 50 51"
EPOCHS=150
OUTDIR=outputs/dwt_full

mkdir -p $OUTDIR

# Shard 0 on GPU 0
CUDA_VISIBLE_DEVICES=0 nohup stdbuf -oL .venv/bin/python -u scripts/run_batch.py \
    --models $MODELS \
    --horizons $HORIZONS \
    --seeds $SEEDS \
    --epochs $EPOCHS \
    --out-dir $OUTDIR \
    --num-shards 2 --shard-index 0 \
    > /tmp/dwt_s0.log 2>&1 &
echo "Shard0 PID: $!"

# Shard 1 on GPU 1
CUDA_VISIBLE_DEVICES=1 nohup stdbuf -oL .venv/bin/python -u scripts/run_batch.py \
    --models $MODELS \
    --horizons $HORIZONS \
    --seeds $SEEDS \
    --epochs $EPOCHS \
    --out-dir $OUTDIR \
    --num-shards 2 --shard-index 1 \
    > /tmp/dwt_s1.log 2>&1 &
echo "Shard1 PID: $!"

sleep 5
echo ""
echo "=== 进程确认 ==="
ps aux | grep run_batch | grep -v grep
echo ""
echo "=== GPU 状态 ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo ""
echo "预计 2-3 小时完成。用 tail -f /tmp/dwt_s0.log 监控进度。"
